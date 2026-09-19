"""Stage `evaluate_gnn`: does relational structure actually add anything?

The GNN's own transaction-node score is **not** the number to compare against
the champion's 0.3190. Two things confound it:

1. **Trees beat neural nets on tabular data.** A GNN losing to XGBoost could
   simply be that, and would say nothing about structure.
2. **The training graph is under-sampled**, so a user node aggregates over a
   ~1-in-80 sample of its history at train time and a dense neighbourhood at
   eval time. The topology itself shifts between fit and score.

So the headline is an **ablation with the classifier held fixed**: the same
XGBoost, the same rows, the same hyperparameter search, differing only in
whether the 64-d user and merchant embeddings are appended.

    base                 -> AUC-PR
    base + embeddings    -> AUC-PR
    difference           -> what the graph contributed, and nothing else

Both heads train on the under-sampled training rows (the only ones with
embeddings) and are scored on the full val and test splits, so they are
comparable to each other exactly, and to the champion with the caveat that the
champion saw 20.6M training rows rather than 277k.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

from fraud.config import load_params, repo_path
from fraud.models.metrics import auc_pr, evaluate_scores, threshold_at_fpr

LABEL = "Fraud"
SPLITS = ("train", "val", "test")


def assemble(
    graph_dir: pathlib.Path,
    embed_dir: pathlib.Path,
    matrix_dir: pathlib.Path,
    split: str,
    years: list[int],
    with_embeddings: bool,
) -> tuple[pl.DataFrame, np.ndarray]:
    """Rows for one split: base features, optionally plus the embeddings."""
    nodes = pl.read_parquet(graph_dir / split / "transaction_nodes.parquet").select(
        "transaction_id", "txn_id", LABEL
    )
    matrix = (
        pl.scan_parquet(matrix_dir / "**/*.parquet", hive_partitioning=True)
        .filter(pl.col("Year").is_in(years))
        .drop("Year", LABEL)
        .collect(engine="streaming")
    )
    frame = nodes.join(matrix, on="txn_id", how="left")

    if with_embeddings:
        # transaction -> user / merchant comes from the edge lists; the node
        # tables themselves do not carry the mapping.
        ut = pl.read_parquet(graph_dir / split / "edges_user_transaction.parquet")
        tm = pl.read_parquet(graph_dir / split / "edges_transaction_merchant.parquet")
        users = pl.read_parquet(embed_dir / f"{split}_user_embeddings.parquet")
        merchants = pl.read_parquet(
            embed_dir / f"{split}_merchant_embeddings.parquet"
        )
        dim = len(users["embedding"][0])

        user_wide = users.select(
            "user_id",
            *[pl.col("embedding").list.get(i).alias(f"user_emb_{i}") for i in range(dim)],
        )
        merchant_wide = merchants.select(
            "merchant_id",
            *[
                pl.col("embedding").list.get(i).alias(f"merchant_emb_{i}")
                for i in range(dim)
            ],
        )
        frame = (
            frame.join(
                ut.rename({"src": "user_id", "dst": "transaction_id"}),
                on="transaction_id",
                how="left",
            )
            .join(
                tm.rename({"src": "transaction_id", "dst": "merchant_id"}),
                on="transaction_id",
                how="left",
            )
            .join(user_wide, on="user_id", how="left")
            .join(merchant_wide, on="merchant_id", how="left")
            .drop("user_id", "merchant_id")
        )

    labels = frame.get_column(LABEL).to_numpy().astype(np.int8)
    features = frame.drop("transaction_id", "txn_id", LABEL)
    return features, labels


def fit_head(
    x_train, y_train, x_val, y_val, cfg: dict, names: list[str]
) -> tuple[xgb.Booster, dict]:
    """XGBoost head, with scale_pos_weight swept on val (decision 15)."""
    dtrain = xgb.QuantileDMatrix(x_train, label=y_train, feature_names=names)
    dval = xgb.QuantileDMatrix(x_val, label=y_val, feature_names=names, ref=dtrain)

    best = {"auc_pr": -1.0}
    for spw in cfg["scale_pos_weight_grid"]:
        booster = xgb.train(
            {
                "objective": "binary:logistic",
                "tree_method": "hist",
                "max_depth": cfg["max_depth"],
                "learning_rate": cfg["learning_rate"],
                "subsample": cfg["subsample"],
                "colsample_bytree": cfg["colsample_bytree"],
                "scale_pos_weight": spw,
                "eval_metric": cfg["eval_metric"],
                "seed": cfg["random_state"],
            },
            dtrain,
            num_boost_round=cfg["n_estimators"],
            evals=[(dval, "val")],
            early_stopping_rounds=cfg["early_stopping_rounds"],
            verbose_eval=False,
        )
        scores = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
        value = auc_pr(y_val, scores)
        print(f"    spw={spw:<5} val AUC-PR {value:.4f}", flush=True)
        if value > best["auc_pr"]:
            best = {
                "auc_pr": value,
                "scale_pos_weight": spw,
                "booster": booster,
                "best_iteration": int(booster.best_iteration),
            }
    return best["booster"], {
        "scale_pos_weight": best["scale_pos_weight"],
        "best_iteration": best["best_iteration"],
        "val_auc_pr": best["auc_pr"],
    }


def run_variant(
    label: str, graph_dir, embed_dir, matrix_dir, wanted, params, with_embeddings
) -> dict:
    cfg = params["xgboost"]
    eval_cfg = params["evaluate"]
    print(f"\n{label}:")

    x_train, y_train = assemble(
        graph_dir, embed_dir, matrix_dir, "train", wanted["train"], with_embeddings
    )
    names = x_train.columns
    x_val, y_val = assemble(
        graph_dir, embed_dir, matrix_dir, "val", wanted["val"], with_embeddings
    )
    print(f"    {x_train.height:,} train rows x {len(names)} features")

    booster, info = fit_head(
        x_train.to_numpy().astype(np.float32),
        y_train,
        x_val.to_numpy().astype(np.float32),
        y_val,
        cfg,
        names,
    )
    del x_train

    # Score the test graph ONCE, then split 2019-only out with a year mask.
    test_years = wanted["test"]
    x_test, y_test = assemble(
        graph_dir, embed_dir, matrix_dir, "test", test_years, with_embeddings
    )
    test_scores = booster.predict(
        xgb.DMatrix(x_test.to_numpy().astype(np.float32), feature_names=names),
        iteration_range=(0, info["best_iteration"] + 1),
    )
    del x_test

    val_scores = booster.predict(
        xgb.DMatrix(x_val.to_numpy().astype(np.float32), feature_names=names),
        iteration_range=(0, info["best_iteration"] + 1),
    )
    threshold = threshold_at_fpr(
        y_val, val_scores, eval_cfg["target_false_positive_rate"]
    )
    del x_val

    ks = list(eval_cfg["precision_at_k"])
    fpr = eval_cfg["target_false_positive_rate"]
    result = {
        "n_features": len(names),
        "threshold": float(threshold),
        **info,
        "splits": {
            "val": evaluate_scores(y_val, val_scores, ks, fpr, threshold=threshold),
            "test_full": evaluate_scores(
                y_test, test_scores, ks, fpr, threshold=threshold
            ),
        },
    }

    # 2019-only: the headline, since 2020 contributes no positives.
    test_nodes = pl.read_parquet(
        graph_dir / "test" / "transaction_nodes.parquet"
    ).select("txn_id")
    year_2019 = (
        pl.scan_parquet(matrix_dir / "**/*.parquet", hive_partitioning=True)
        .filter(pl.col("Year") == min(test_years))
        .select("txn_id")
        .collect(engine="streaming")
        .get_column("txn_id")
    )
    mask = test_nodes.get_column("txn_id").is_in(year_2019).to_numpy()
    result["splits"]["test_2019_only"] = evaluate_scores(
        y_test[mask], test_scores[mask], ks, fpr, threshold=threshold
    )

    return result


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default=params["paths"]["graph"])
    ap.add_argument("--embeddings", default=params["paths"]["embeddings"])
    ap.add_argument("--matrix", default=params["paths"]["feature_matrix"])
    ap.add_argument("--output", default=params["paths"]["metrics_gnn"])
    args = ap.parse_args(argv)

    from fraud.data.split import split_years

    processed = repo_path(params["paths"]["processed"])
    years = sorted(int(p.name.split("=")[1]) for p in processed.glob("Year=*"))
    wanted = split_years(params, years)

    graph_dir = repo_path(args.graph)
    embed_dir = repo_path(args.embeddings)
    matrix_dir = repo_path(args.matrix)

    started = time.perf_counter()
    report = {
        "comparison": "XGBoost head held fixed; only the embeddings differ",
        "variants": {
            "base": run_variant(
                "base features only", graph_dir, embed_dir, matrix_dir,
                wanted, params, with_embeddings=False,
            ),
            "base_plus_embeddings": run_variant(
                "base + GNN embeddings", graph_dir, embed_dir, matrix_dir,
                wanted, params, with_embeddings=True,
            ),
        },
    }

    base = report["variants"]["base"]["splits"]["test_2019_only"]["auc_pr"]
    withemb = report["variants"]["base_plus_embeddings"]["splits"]["test_2019_only"][
        "auc_pr"
    ]
    report["headline"] = {
        "base_auc_pr": base,
        "base_plus_embeddings_auc_pr": withemb,
        "delta": withemb - base,
        "relative": (withemb - base) / base if base else None,
        "verdict": "embeddings help" if withemb > base else "embeddings do not help",
    }
    report["seconds"] = round(time.perf_counter() - started, 1)

    out = repo_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n=== headline: test 2019 AUC-PR ===")
    print(f"  base features only   {base:.4f}")
    print(f"  + GNN embeddings     {withemb:.4f}")
    print(f"  delta                {withemb - base:+.4f} -> {report['headline']['verdict']}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
