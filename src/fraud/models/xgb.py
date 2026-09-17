"""Stage `train_xgb`: the baseline champion.

The training matrix is 20.6M rows by 86 float32 features -- about 7 GB if held
whole, on a box with ~11 GB free. So it never is: `ParquetBatchIter` feeds
Parquet row groups to `xgb.QuantileDMatrix`, which compresses each batch into
histogram bins and discards the raw rows.

Measured, not estimated: **peak RSS 5.8 GB, 40 minutes** on 4 CPU cores. That is
well above the ~2 GB a naive reading of "streaming" suggests, and the reasons are
worth knowing before tuning this:

- `QuantileDMatrix` walks the iterator **4 times** (sketching, then filling), so
  the Parquet is decoded four times over. It is the bulk of the wall clock.
- The quantised bins for 20.6M x 86 stay resident for the whole fit.
- The validation split is materialised whole (1.7M x 86 float32, ~590 MB) plus
  its own DMatrix, because early stopping scores it every round.

If this needs to shrink, `ExtMemQuantileDMatrix` spills to disk, and a GPU
(`device="cuda"`) would cut the wall clock several-fold -- but per CLAUDE.md the
Lightning GPU budget is reserved for Phase 3, and a booster trained on GPU still
serves on CPU, so this is a credits trade rather than a correctness one.

Early stopping runs on the 2018 validation split. Nothing here ever looks at
test -- the threshold and the stopping point are both chosen on val, and test is
scored once, at the end, by the evaluate stage.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import resource
import sys
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import xgboost as xgb

from fraud.config import load_params, repo_path
from fraud.data.split import split_years
from fraud.models.metrics import auc_pr

LABEL = "Fraud"
KEY = ("txn_id", "Year")


def matrix_files(matrix: pathlib.Path, years: list[int]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for year in years:
        files += sorted((matrix / f"Year={year}").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no feature files for years {years} under {matrix}")
    return files


def feature_names(matrix_file: pathlib.Path) -> list[str]:
    schema = pq.ParquetFile(matrix_file).schema_arrow
    return [n for n in schema.names if n not in (LABEL, *KEY)]


class ParquetBatchIter(xgb.DataIter):
    """Feeds row groups to XGBoost so the matrix is never resident at once."""

    def __init__(self, files: list[pathlib.Path], names: list[str]):
        self._names = names
        # (file, row_group) pairs, in order.
        self._batches: list[tuple[pathlib.Path, int]] = []
        for path in files:
            for group in range(pq.ParquetFile(path).num_row_groups):
                self._batches.append((path, group))
        self._index = 0
        self._pass_rows = 0
        self._completed_rows = 0
        super().__init__()

    @property
    def rows(self) -> int:
        """Rows in one pass over the data -- not the sum across passes.

        XGBoost walks the iterator several times (QuantileDMatrix: 4 -- sketching
        then filling), and it calls `reset` both between passes and after the
        last one. Summing naively reports 4x the true size; zeroing on reset
        reports 0. Both have happened here. So: accumulate within a pass, and
        keep the last completed pass's total.
        """
        return self._completed_rows or self._pass_rows

    def reset(self) -> None:
        if self._pass_rows:
            self._completed_rows = self._pass_rows
        self._pass_rows = 0
        self._index = 0

    def next(self, input_data) -> int:
        if self._index == len(self._batches):
            return 0
        path, group = self._batches[self._index]
        table = pq.ParquetFile(path).read_row_group(
            group, columns=[*self._names, LABEL]
        )
        # from_arrow is typed as DataFrame | Series; a Table always yields a
        # DataFrame, but narrow it so the type checker and a 1-column edge case
        # both behave.
        frame = pl.from_arrow(table)
        if isinstance(frame, pl.Series):
            frame = frame.to_frame()
        features = frame.select(self._names).to_numpy().astype(np.float32)
        labels = frame.get_column(LABEL).to_numpy().astype(np.int8)
        self._pass_rows += len(labels)
        input_data(data=features, label=labels, feature_names=self._names)
        self._index += 1
        return 1


def load_split_matrix(
    matrix: pathlib.Path, years: list[int], names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Materialise a split. Only used for val/test, which are ~1.7-2.1M rows."""
    lf = pl.scan_parquet(matrix / "**/*.parquet", hive_partitioning=True).filter(
        pl.col("Year").is_in(years)
    )
    frame = lf.select([*names, LABEL]).collect(engine="streaming")
    return (
        frame.select(names).to_numpy().astype(np.float32),
        frame[LABEL].to_numpy().astype(np.int8),
    )


def train(
    matrix: pathlib.Path, params: dict, scale_pos_weight: float | None = None
) -> tuple[xgb.Booster, dict]:
    years = sorted(int(p.name.split("=")[1]) for p in matrix.glob("Year=*"))
    splits = split_years(params, years)
    names = feature_names(matrix_files(matrix, splits["train"])[0])

    cfg = dict(params["xgboost"])
    spw = scale_pos_weight if scale_pos_weight is not None else cfg["scale_pos_weight"]

    started = time.perf_counter()
    train_iter = ParquetBatchIter(matrix_files(matrix, splits["train"]), names)
    # feature_names rides on each batch (see ParquetBatchIter.next): passing
    # it here as well is rejected when the input is an iterator.
    dtrain = xgb.QuantileDMatrix(train_iter)

    x_val, y_val = load_split_matrix(matrix, splits["val"], names)
    # ref=dtrain reuses the training bin edges, so val is quantised identically.
    dval = xgb.QuantileDMatrix(x_val, label=y_val, feature_names=names, ref=dtrain)

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
            "nthread": 0,
        },
        dtrain,
        num_boost_round=cfg["n_estimators"],
        evals=[(dval, "val")],
        early_stopping_rounds=cfg["early_stopping_rounds"],
        verbose_eval=50,
    )

    val_scores = booster.predict(
        dval, iteration_range=(0, booster.best_iteration + 1)
    )
    info = {
        "scale_pos_weight": float(spw),
        "best_iteration": int(booster.best_iteration),
        "n_features": len(names),
        "train_rows": train_iter.rows,
        "val_rows": int(len(y_val)),
        "val_auc_pr": auc_pr(y_val, val_scores),
        "seconds": round(time.perf_counter() - started, 1),
        "feature_names": names,
    }
    return booster, info


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default=params["paths"]["feature_matrix"])
    ap.add_argument("--output", default=params["paths"]["models"])
    ap.add_argument(
        "--scale-pos-weight",
        type=float,
        default=None,
        help="override params.yaml (used by the sweep)",
    )
    args = ap.parse_args(argv)

    matrix_path: str = args.matrix
    output_path: str = args.output
    spw: float | None = args.scale_pos_weight

    booster, info = train(repo_path(matrix_path), params, scale_pos_weight=spw)

    out = repo_path(output_path) / "xgb"
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(out / "model.json")
    (out / "train_info.json").write_text(json.dumps(info, indent=2))

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(
        f"trained on {info['train_rows']:,} rows x {info['n_features']} features "
        f"| best_iter {info['best_iteration']} | val AUC-PR {info['val_auc_pr']:.4f} "
        f"({info['seconds']}s, peak RSS {peak_mb:.0f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
