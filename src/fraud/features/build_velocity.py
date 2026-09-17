"""Stage `velocity`: compute velocity features over the full history.

The work is in `fraud.features.velocity.compute_velocity`, which is pure and
shared with the serving path. This module is only the offline driver: it decides
what to feed that function and where to put the answer.

Why user chunks rather than year partitions: a velocity window crosses year
boundaries -- a 7-day lookback on 2018-01-02 needs December 2017 -- so computing
per year partition would silently zero out every window at the start of each
year. A window never spans two users, so chunking by user is safe where chunking
by year is not.

Output is written back partitioned by year so the `features` stage can join it
to the transactions one year at a time.
"""

from __future__ import annotations

import argparse
import pathlib
import resource
import shutil
import sys
import time

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds

from fraud.config import load_params, repo_path
from fraud.features.velocity import compute_velocity, velocity_columns

# Only what compute_velocity needs, plus Year to partition the output.
_READ_COLUMNS = ["txn_id", "User", "ts", "Amount", "Merchant", "State", "Year"]


def _compact(df: pl.DataFrame, windows: list[int]) -> pl.DataFrame:
    """Counts fit in u32, money and seconds in f32. Halves the artifact."""
    casts = []
    for w in windows:
        casts += [
            pl.col(f"velocity_count_{w}h").cast(pl.UInt32),
            pl.col(f"velocity_merchants_{w}h").cast(pl.UInt32),
            pl.col(f"velocity_states_{w}h").cast(pl.UInt32),
            pl.col(f"velocity_amount_{w}h").cast(pl.Float32),
        ]
    casts.append(pl.col("velocity_seconds_since_last").cast(pl.Float32))
    return df.with_columns(casts)


def build(
    processed: pathlib.Path,
    out_dir: pathlib.Path,
    windows: list[int],
    users_per_chunk: int,
) -> dict:
    lf = pl.scan_parquet(processed / "**/*.parquet", hive_partitioning=True)
    users = sorted(
        lf.select(pl.col("User").unique()).collect(engine="streaming")["User"].to_list()
    )
    if not users:
        raise FileNotFoundError(f"no transactions under {processed}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = [
        users[i : i + users_per_chunk] for i in range(0, len(users), users_per_chunk)
    ]
    started = time.perf_counter()
    rows_out = 0

    for index, chunk in enumerate(chunks):
        # A chunk pulls each user's FULL history across every year, which is
        # exactly what makes the cross-year windows correct.
        frame = (
            lf.filter(pl.col("User").is_in(chunk))
            .select(_READ_COLUMNS)
            .collect(engine="streaming")
        )
        result = compute_velocity(frame.drop("Year"), windows)
        # Year comes back by txn_id rather than positionally: compute_velocity
        # sorts internally, so positions do not survive.
        result = result.join(frame.select("txn_id", "Year"), on="txn_id", how="left")
        result = _compact(result, windows)
        rows_out += result.height

        ds.write_dataset(
            result.to_arrow(),
            base_dir=out_dir,
            format="parquet",
            partitioning=ds.partitioning(
                pa.schema([("Year", pa.int32())]), flavor="hive"
            ),
            existing_data_behavior="overwrite_or_ignore",
            basename_template=f"chunk{index:03d}-{{i}}.parquet",
        )
        print(
            f"  chunk {index + 1}/{len(chunks)}: {len(chunk)} users, "
            f"{result.height:,} rows",
            flush=True,
        )

    return {
        "rows": rows_out,
        "users": len(users),
        "chunks": len(chunks),
        "columns": velocity_columns(windows),
        "seconds": round(time.perf_counter() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=params["paths"]["processed"])
    ap.add_argument("--output", default=params["paths"]["velocity"])
    args = ap.parse_args(argv)

    result = build(
        repo_path(args.input),
        repo_path(args.output),
        windows=params["velocity"]["windows_hours"],
        users_per_chunk=params["velocity"]["users_per_chunk"],
    )
    # ru_maxrss is KiB on Linux. Reported because the chunking exists purely to
    # keep this bounded -- if it creeps up, users_per_chunk is the dial.
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(
        f"velocity: {result['rows']:,} rows | {result['users']:,} users in "
        f"{result['chunks']} chunks | {len(result['columns'])} features "
        f"({result['seconds']}s, peak RSS {peak_mb:.0f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
