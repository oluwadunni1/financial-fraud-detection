"""Stage `fm_head`: does sequential structure add anything -- and does context?

The structure is upstream's notebook 05: PCA the 512-d embeddings to 64, then
fit XGBoost three ways -- base features, embeddings only, and both combined.
It runs once per embedding arm:

    isolated     NVIDIA's extraction, each transaction encoded alone
    contextual   ours, each transaction read in its card's history

so the table answers two questions with the classifier held fixed:

    combined - base              what the foundation model adds at all
    contextual - isolated        what sequence CONTEXT adds over the
                                 transaction-alone embedding NVIDIA ships

The evaluation is OURS, not upstream's (decision 16). They train on a
balanced 1M sample, score 100k stratified subsets and leave scale_pos_weight
at 1. Here every head trains on the real 2016-2017 distribution, sweeps
scale_pos_weight (decision 15), picks its threshold on 2018 and reports 2019
as the headline (decision 12) -- the same footing as the champion and the GNN.

PCA is fitted on TRAIN rows only and shipped with the head, for the same
reason the encoder is: re-fitting it elsewhere changes what every column means.
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
from sklearn.decomposition import PCA

from fraud.config import load_params, repo_path
from fraud.models.evaluate_gnn import fit_head
from fraud.models.metrics import evaluate_scores, threshold_at_fpr

LABEL = "Fraud"
ARMS = ("isolated", "contextual")
CONFIGS = ("embeddings_only", "combined")


def load_matrix(matrix_dir: pathlib.Path, years: list[int]) -> pl.DataFrame:
    """Base feature rows for `years`, sorted by txn_id, with Year kept."""
    return (
        pl.scan_parquet(matrix_dir / "**/*.parquet", hive_partitioning=True)
        .filter(pl.col("Year").is_in(years))
        .collect(engine="streaming")
        .sort("txn_id")
    )


class EmbeddingStore:
    """Memmapped (txn_ids, float16 embeddings) as `fm_embed` writes them."""

    def __init__(self, directory: pathlib.Path):
        self.txn_ids = np.load(directory / "txn_ids.npy")
        self.embeddings = np.load(directory / "embeddings.npy", mmap_mode="r")

    def rows_for(self, txn_ids: np.ndarray) -> np.ndarray:
        rows = np.searchsorted(self.txn_ids, txn_ids)
        rows = np.minimum(rows, len(self.txn_ids) - 1)
        if not np.array_equal(self.txn_ids[rows], txn_ids):
            missing = int((self.txn_ids[rows] != txn_ids).sum())
            raise RuntimeError(
                f"{missing:,} transactions have no embedding. The embedding "
                f"stage and the feature matrix cover different rows."
            )
        return rows


def fit_pca(
    store: EmbeddingStore, train_rows: np.ndarray, dim: int, fit_rows: int, seed: int
) -> PCA:
    """PCA on a random sample of TRAIN rows -- 3.4M x 512 fp32 is 7 GB, more
    than this machine holds next to everything else; 500k rows pin 64
    components just as well."""
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(train_rows, size=min(fit_rows, len(train_rows)),
                              replace=False))
    return PCA(n_components=dim, random_state=seed).fit(
        np.asarray(store.embeddings[pick], dtype=np.float32)
    )


def project(store: EmbeddingStore, rows: np.ndarray, pca: PCA,
            chunk: int = 500_000) -> np.ndarray:
    out = np.empty((len(rows), pca.n_components_), dtype=np.float32)
    for start in range(0, len(rows), chunk):
        block = np.asarray(store.embeddings[rows[start : start + chunk]],
                           dtype=np.float32)
        out[start : start + chunk] = pca.transform(block)
    return out


def score_variant(
    label: str,
    x: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    names: list[str],
    test_2019: np.ndarray,
    cfg: dict,
    eval_cfg: dict,
) -> tuple[xgb.Booster, dict]:
    print(f"\n{label}: {x['train'].shape[0]:,} train rows x {len(names)} features",
          flush=True)
    booster, info = fit_head(x["train"], y["train"], x["val"], y["val"], cfg, names)
    upto = (0, info["best_iteration"] + 1)

    def predict(split: str) -> np.ndarray:
        return booster.predict(
            xgb.DMatrix(x[split], feature_names=names), iteration_range=upto
        )

    val_scores, test_scores = predict("val"), predict("test")
    fpr = eval_cfg["target_false_positive_rate"]
    ks = list(eval_cfg["precision_at_k"])
    threshold = threshold_at_fpr(y["val"], val_scores, fpr)
    result = {
        "n_features": len(names),
        "threshold": float(threshold),
        **info,
        "splits": {
            "val": evaluate_scores(y["val"], val_scores, ks, fpr, threshold=threshold),
            "test_full": evaluate_scores(
                y["test"], test_scores, ks, fpr, threshold=threshold
            ),
            "test_2019_only": evaluate_scores(
                y["test"][test_2019], test_scores[test_2019], ks, fpr,
                threshold=threshold,
            ),
        },
    }
    print(f"    -> test 2019 AUC-PR {result['splits']['test_2019_only']['auc_pr']:.4f}",
          flush=True)
    return booster, result


def run(params: dict, matrix_dir: pathlib.Path, stores: dict[str, pathlib.Path],
        model_dir: pathlib.Path) -> dict:
    from fraud.data.split import split_years

    processed = repo_path(params["paths"]["processed"])
    years = sorted(int(p.name.split("=")[1]) for p in processed.glob("Year=*"))
    wanted = split_years(params, years)
    fm = params["fm"]
    head = fm["head"]
    # The embeddings cover 2016+ only (fm.min_year), so every head -- the base
    # one included -- trains on the same 2016-2017 rows. The lift is then the
    # embeddings' alone, not a difference in how much history was seen.
    wanted["train"] = [yr for yr in wanted["train"] if yr >= fm["min_year"]]

    cfg = dict(params["xgboost"])
    cfg["device"] = head["device"]
    eval_cfg = params["evaluate"]

    frames = {split: load_matrix(matrix_dir, wanted[split])
              for split in ("train", "val", "test")}
    test_2019 = (frames["test"].get_column("Year") == min(wanted["test"])).to_numpy()
    y = {s: f.get_column(LABEL).to_numpy().astype(np.int8) for s, f in frames.items()}
    txn = {s: f.get_column("txn_id").to_numpy().astype(np.int64)
           for s, f in frames.items()}
    base = {s: f.drop("txn_id", LABEL, "Year").to_numpy().astype(np.float32)
            for s, f in frames.items()}
    base_names = frames["train"].drop("txn_id", LABEL, "Year").columns
    del frames

    model_dir.mkdir(parents=True, exist_ok=True)
    variants: dict[str, dict] = {}

    booster, variants["base"] = score_variant(
        "base features only", base, y, base_names, test_2019, cfg, eval_cfg
    )
    booster.save_model(model_dir / "base.json")

    for arm in ARMS:
        store = EmbeddingStore(stores[arm])
        rows = {s: store.rows_for(txn[s]) for s in txn}
        pca = fit_pca(store, rows["train"], head["pca_dim"], head["pca_fit_rows"],
                      cfg["random_state"])
        np.savez(model_dir / f"pca_{arm}.npz", components=pca.components_,
                 mean=pca.mean_, explained_variance_ratio=pca.explained_variance_ratio_)
        emb = {s: project(store, rows[s], pca) for s in rows}
        emb_names = [f"fm_{arm}_pc{i}" for i in range(pca.n_components_)]
        print(f"\n[{arm}] PCA {store.embeddings.shape[1]} -> {pca.n_components_}: "
              f"{pca.explained_variance_ratio_.sum():.1%} variance", flush=True)

        for config in CONFIGS:
            if config == "embeddings_only":
                x, names = emb, emb_names
            else:
                x = {s: np.hstack([base[s], emb[s]]) for s in base}
                names = [*base_names, *emb_names]
            key = f"{arm}_{config}"
            booster, variants[key] = score_variant(
                f"{arm} / {config}", x, y, names, test_2019, cfg, eval_cfg
            )
            variants[key]["pca_explained_variance"] = float(
                pca.explained_variance_ratio_.sum()
            )
            booster.save_model(model_dir / f"{key}.json")
            del x
        del emb

    headline = {k: v["splits"]["test_2019_only"]["auc_pr"] for k, v in variants.items()}
    b = headline["base"]
    return {
        "comparison": (
            "XGBoost head held fixed (upstream notebook 05's three configs, our "
            "evaluation); only the features differ"
        ),
        "train_years": wanted["train"],
        "variants": variants,
        "headline_test_2019_auc_pr": headline,
        "lift_over_base": {k: v - b for k, v in headline.items() if k != "base"},
        "context_effect": {
            config: headline[f"contextual_{config}"] - headline[f"isolated_{config}"]
            for config in CONFIGS
        },
    }


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    paths = params["paths"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default=paths["feature_matrix"])
    ap.add_argument("--isolated", default=paths["fm_embeddings_isolated"])
    ap.add_argument("--contextual", default=paths["fm_embeddings"])
    ap.add_argument("--models", default=paths["fm_head"])
    ap.add_argument("--output", default=paths["fm_metrics"])
    args = ap.parse_args(argv)

    started = time.perf_counter()
    report = run(
        params, repo_path(args.matrix),
        {"isolated": repo_path(args.isolated), "contextual": repo_path(args.contextual)},
        repo_path(args.models),
    )
    report["seconds"] = round(time.perf_counter() - started, 1)

    out = repo_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n=== headline: test 2019 AUC-PR ===")
    for k, v in report["headline_test_2019_auc_pr"].items():
        print(f"  {k:<30} {v:.4f}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
