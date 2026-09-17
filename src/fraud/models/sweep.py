"""Choose `scale_pos_weight` on validation instead of guessing it.

`params.yaml` carried `scale_pos_weight: 50` with the comment "tuned on val,
not guessed" -- which it was not. This module makes the comment true.

Two reasons the value cannot be reasoned out from prevalence alone:

- Full inverse prevalence is 819 (1 positive per 819 rows). That maximises
  recall and usually wrecks precision, which is the metric an alert queue
  actually feels.
- 2017, the last training year, has a fraud rate 14x below its neighbours and
  sits directly against the val boundary. Any weight derived from training
  prevalence is calibrated to an anomaly.

Runs on a stratified subsample with a shorter boosting budget: selecting a
hyperparameter does not need all 20.6M rows, and the winner is refit on the
full split afterwards. That keeps the sweep to minutes on CPU rather than the
hour-plus a full-data sweep would cost.
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
from fraud.data.split import split_years
from fraud.models.metrics import auc_pr
from fraud.models.xgb import LABEL, feature_names, load_split_matrix, matrix_files


def _subsample(
    matrix: pathlib.Path, years: list[int], names: list[str], fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform subsample that **preserves the class ratio**.

    Deliberately not stratified. Keeping every positive while thinning negatives
    would raise training prevalence ~6.7x, and `scale_pos_weight` is defined
    relative to that ratio -- so the winning value would not transfer back to a
    full-data fit. The first version of this sweep did exactly that and pointed
    at 10 when the equivalent full-data weight was about 66.

    Validation is never subsampled, so a uniform training sample keeps the
    train/val relationship intact and the chosen weight transfers directly.

    The sampling happens **inside the lazy plan**, not after collecting. An
    earlier version collected the whole training split first and then sampled it,
    materialising ~7 GB to keep 1 GB, and was OOM-killed. Only the sampled rows
    are ever read.

    Selection is `txn_id % 100 < fraction*100` rather than a random draw: it
    streams, needs no seed, and is reproducible across runs and machines.
    `txn_id` is assignment order in the source file, which is independent of the
    label, so this is unbiased with respect to fraud.
    """
    keep = max(1, round(fraction * 100))
    lf = pl.scan_parquet(matrix / "**/*.parquet", hive_partitioning=True).filter(
        pl.col("Year").is_in(years)
    )
    sampled = lf.filter((pl.col("txn_id") % 100) < keep)
    frame = sampled.select([*names, LABEL]).collect(engine="streaming")
    return (
        frame.select(names).to_numpy().astype(np.float32),
        frame.get_column(LABEL).to_numpy().astype(np.int8),
    )


def sweep(matrix: pathlib.Path, params: dict) -> dict:
    cfg = params["xgboost"]
    years = sorted(int(p.name.split("=")[1]) for p in matrix.glob("Year=*"))
    splits = split_years(params, years)
    names = feature_names(matrix_files(matrix, splits["train"])[0])

    x_train, y_train = _subsample(
        matrix, splits["train"], names, cfg["sweep_sample_fraction"]
    )
    x_val, y_val = load_split_matrix(matrix, splits["val"], names)
    dtrain = xgb.QuantileDMatrix(x_train, label=y_train, feature_names=names)
    dval = xgb.QuantileDMatrix(x_val, label=y_val, feature_names=names, ref=dtrain)

    results = []
    for candidate in cfg["scale_pos_weight_grid"]:
        started = time.perf_counter()
        booster = xgb.train(
            {
                "objective": "binary:logistic",
                "tree_method": "hist",
                "max_depth": cfg["max_depth"],
                "learning_rate": cfg["learning_rate"],
                "subsample": cfg["subsample"],
                "colsample_bytree": cfg["colsample_bytree"],
                "scale_pos_weight": candidate,
                "eval_metric": cfg["eval_metric"],
                "seed": cfg["random_state"],
            },
            dtrain,
            num_boost_round=cfg["sweep_n_estimators"],
            evals=[(dval, "val")],
            early_stopping_rounds=cfg["early_stopping_rounds"],
            verbose_eval=False,
        )
        scores = booster.predict(
            dval, iteration_range=(0, booster.best_iteration + 1)
        )
        entry = {
            "scale_pos_weight": candidate,
            "val_auc_pr": auc_pr(y_val, scores),
            "best_iteration": int(booster.best_iteration),
            "seconds": round(time.perf_counter() - started, 1),
        }
        results.append(entry)
        print(
            f"  spw={candidate:<5} val AUC-PR {entry['val_auc_pr']:.4f} "
            f"(best_iter {entry['best_iteration']}, {entry['seconds']}s)",
            flush=True,
        )

    best = max(results, key=lambda r: r["val_auc_pr"])
    return {
        "grid": results,
        "best": best,
        "selected_on": "val AUC-PR",
        "train_rows_sampled": int(len(y_train)),
        "sample_fraction": cfg["sweep_sample_fraction"],
        "note": (
            "Uniform subsample, so training prevalence matches full data and the "
            "winning scale_pos_weight transfers directly. Validation is scored "
            "in full."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default=params["paths"]["feature_matrix"])
    ap.add_argument("--output", default="reports/sweep_scale_pos_weight.json")
    args = ap.parse_args(argv)

    matrix_path: str = args.matrix
    output_path: str = args.output

    result = sweep(repo_path(matrix_path), params)
    out = repo_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(
        f"\nbest scale_pos_weight = {result['best']['scale_pos_weight']} "
        f"(val AUC-PR {result['best']['val_auc_pr']:.4f}) -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
