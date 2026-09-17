"""Stage `features`: transactions + velocity -> the encoded feature matrix.

Fits the encoder on the **training split only** and applies it unchanged to val
and test. That ordering is the whole point: the encoder is part of the model, so
fitting it on anything the model is later scored against leaks.

Written one year partition at a time. The full training matrix is ~20.6M rows
wide enough that materialising it would not fit alongside anything else, and the
training stage streams it back in row groups for the same reason.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import resource
import shutil
import sys
import time

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds

from fraud.config import load_params, repo_path
from fraud.data.split import split_years
from fraud.features.encoders import Encoder
from fraud.features.velocity import velocity_columns

# Carried through unencoded: the key, the label, and the partition column.
PASSTHROUGH = ["txn_id", "Fraud", "Year"]


def _joined(
    processed: pathlib.Path, velocity: pathlib.Path, years: list[int] | None = None
) -> pl.LazyFrame:
    """Transactions with their velocity features attached, by txn_id."""
    txns = pl.scan_parquet(processed / "**/*.parquet", hive_partitioning=True)
    vel = pl.scan_parquet(velocity / "**/*.parquet", hive_partitioning=True)
    if years is not None:
        txns = txns.filter(pl.col("Year").is_in(years))
        vel = vel.filter(pl.col("Year").is_in(years))
    return txns.join(vel.drop("Year"), on="txn_id", how="left")


def build(
    processed: pathlib.Path,
    velocity: pathlib.Path,
    out_dir: pathlib.Path,
    encoder_path: pathlib.Path,
    params: dict,
) -> dict:
    cfg = params["features"]
    windows = params["velocity"]["windows_hours"]
    categorical = list(cfg["categorical"])
    # Velocity features are numerics too, and are scaled with the rest.
    numeric = list(cfg["numeric"]) + velocity_columns(windows)

    years = sorted(
        int(p.name.split("=")[1]) for p in processed.glob("Year=*")
    )
    if not years:
        raise FileNotFoundError(f"no Year=* partitions under {processed}")
    train_years = split_years(params, years)["train"]

    started = time.perf_counter()

    # --- fit on train only ------------------------------------------------
    encoder = Encoder.fit(
        _joined(processed, velocity, train_years),
        categorical=categorical,
        numeric=numeric,
        one_hot_max_cardinality=cfg["one_hot_max_cardinality"],
    )
    encoder.to_json(encoder_path)
    names = encoder.feature_names()

    # --- transform every year --------------------------------------------
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = pa.schema(
        [("txn_id", pa.int64()), ("Fraud", pa.int8()), ("Year", pa.int32())]
        + [(n, pa.float32()) for n in names]
    )

    rows = 0
    for year in years:
        frame = _joined(processed, velocity, [year]).collect(engine="streaming")
        encoded = frame.select(PASSTHROUGH).hstack(encoder.transform(frame))
        # One dtype for every feature: XGBoost will coerce anyway, and doing it
        # here keeps the on-disk matrix uniform for the batch loader.
        encoded = encoded.with_columns(
            [pl.col(n).cast(pl.Float32) for n in names]
        )
        rows += encoded.height
        ds.write_dataset(
            encoded.to_arrow().cast(schema),
            base_dir=out_dir,
            format="parquet",
            partitioning=ds.partitioning(
                pa.schema([("Year", pa.int32())]), flavor="hive"
            ),
            existing_data_behavior="overwrite_or_ignore",
            basename_template="part-{i}.parquet",
            min_rows_per_group=min(200_000, params["ingest"]["row_group_size"]),
            max_rows_per_group=params["ingest"]["row_group_size"],
        )

    return {
        "rows": rows,
        "n_features": len(names),
        "train_years": [train_years[0], train_years[-1]],
        "one_hot": {c: len(v) for c, v in encoder.one_hot.items()},
        "binary": {c: len(v) for c, v in encoder.binary.items()},
        "seconds": round(time.perf_counter() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", default=params["paths"]["processed"])
    ap.add_argument("--velocity", default=params["paths"]["velocity"])
    ap.add_argument("--output", default=params["paths"]["feature_matrix"])
    ap.add_argument("--encoder", default=params["paths"]["encoder"])
    args = ap.parse_args(argv)

    result = build(
        repo_path(args.processed),
        repo_path(args.velocity),
        repo_path(args.output),
        repo_path(args.encoder),
        params,
    )
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(
        f"features: {result['rows']:,} rows x {result['n_features']} features "
        f"| encoder fitted on {result['train_years'][0]}-{result['train_years'][1]} "
        f"({result['seconds']}s, peak RSS {peak_mb:.0f} MB)"
    )
    print("  one-hot:", json.dumps(result["one_hot"]))
    print("  binary: ", json.dumps(result["binary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
