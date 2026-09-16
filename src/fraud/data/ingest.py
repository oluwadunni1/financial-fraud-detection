"""Stage `ingest`: raw TabFormer CSV -> year-partitioned Parquet.

One streaming pass. The 2.35 GB / 24,386,900-row CSV is never materialised
whole: PyArrow reads it in batches, each batch is transformed with polars, and
PyArrow routes it to the right per-year Parquet writer. Peak memory stays flat,
which matters on a 14 GB machine.

Why PyArrow rather than polars for the write: this polars build's
`sink_parquet` exposes no partitioning, so partitioned output would need one
filtered pass per year -- 30 passes over 2.35 GB. `write_dataset` does it in one.

A single sequential pass is required in any case, because `txn_id` is the row's
index in the original file. TabFormer has no natural key: (User, Card, Year,
Month, Day, Time) collides on 142,010 rows, and 66 rows are exact duplicates --
so a content hash would not be unique either.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import time

import pandera.errors
import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.dataset as ds

from fraud.config import load_params, repo_path
from fraud.data.schemas import (
    COLUMN_RENAME,
    RAW_COLUMNS,
    UNKNOWN_STRING_MARKER,
    UNKNOWN_ZIP_CODE,
    RawTransactionSchema,
)

# Explicit raw types. Never rely on inference: on a file this size chunked
# inference silently yields mixed/object columns that fail far from the cause.
RAW_ARROW_TYPES = {
    "User": pa.int64(),
    "Card": pa.int64(),
    "Year": pa.int64(),
    "Month": pa.int64(),
    "Day": pa.int64(),
    "Time": pa.string(),
    "Amount": pa.string(),
    "Use Chip": pa.string(),
    "Merchant Name": pa.int64(),
    "Merchant City": pa.string(),
    "Merchant State": pa.string(),
    "Zip": pa.float64(),
    "MCC": pa.int64(),
    "Errors?": pa.string(),
    "Is Fraud?": pa.string(),
}

# Output column order, and the contract the validate stage checks.
OUTPUT_SCHEMA = pa.schema(
    [
        ("txn_id", pa.int64()),
        ("ts", pa.timestamp("us")),
        ("User", pa.int64()),
        ("Card", pa.int64()),
        ("Year", pa.int32()),
        ("Month", pa.int8()),
        ("Day", pa.int8()),
        ("Time", pa.int16()),
        ("Amount", pa.float64()),
        # String, not int: the hashed id spans nearly the full int64 range and
        # must never round-trip through a float.
        ("Merchant", pa.string()),
        ("City", pa.string()),
        ("State", pa.string()),
        ("Zip", pa.int32()),
        ("MCC", pa.int32()),
        ("Chip", pa.string()),
        ("Errors", pa.string()),
        ("Fraud", pa.int8()),
    ]
)

OUTPUT_COLUMNS = [f.name for f in OUTPUT_SCHEMA]


def transform_batch(df: pl.DataFrame, start_id: int) -> pl.DataFrame:
    """Raw batch -> clean batch. Pure, so tests exercise it directly."""
    hours = pl.col("Time").str.split(":").list.get(0).cast(pl.Int32)
    minutes = pl.col("Time").str.split(":").list.get(1).cast(pl.Int32)

    return df.rename(COLUMN_RENAME).with_columns(
        # Row index in the original file -- stable across re-ingests.
        txn_id=pl.int_range(start_id, start_id + df.height, dtype=pl.Int64),
        ts=pl.datetime(pl.col("Year"), pl.col("Month"), pl.col("Day"), hours, minutes),
        # Refunds are "$-292.00": the minus is INSIDE the dollar sign, so
        # dropping "$" leaves a parseable signed decimal.
        Amount=pl.col("Amount").str.replace("$", "", literal=True).cast(pl.Float64),
        # Minutes since midnight.
        Time=(hours * 60 + minutes).cast(pl.Int16),
        Fraud=(pl.col("Fraud") == "Yes").cast(pl.Int8),
        Merchant=pl.col("Merchant").cast(pl.String),
        City=pl.col("City").fill_null(UNKNOWN_STRING_MARKER),
        State=pl.col("State").fill_null(UNKNOWN_STRING_MARKER),
        # The trailing comma is part of the raw value ("Bad PIN,"). Strip it and
        # keep the whole string as one category -- that is the 23 distinct
        # values the feature-width budget assumes.
        Errors=pl.col("Errors")
        .str.strip_chars_end(",")
        .fill_null(UNKNOWN_STRING_MARKER),
        Zip=pl.col("Zip").fill_null(UNKNOWN_ZIP_CODE).cast(pl.Int32),
        Year=pl.col("Year").cast(pl.Int32),
        Month=pl.col("Month").cast(pl.Int8),
        Day=pl.col("Day").cast(pl.Int8),
        MCC=pl.col("MCC").cast(pl.Int32),
    ).select(OUTPUT_COLUMNS)


def _validate_raw(df: pl.DataFrame) -> None:
    """Run RawTransactionSchema on a batch, pre-rename.

    Per batch rather than on a sample: every one of the 24.4M rows gets checked,
    and a schema break stops the build instead of surfacing three stages later.
    """
    pdf = df.to_pandas()
    for col, dtype in pdf.dtypes.items():
        if dtype == "object":
            pdf[col] = pdf[col].where(pdf[col].notna(), None)
    RawTransactionSchema.validate(pdf, lazy=True)


def _batches(reader: pacsv.CSVStreamingReader, validate_raw: bool, state: dict):
    """Yield transformed RecordBatches, tracking the running row index."""
    for raw in reader:
        df = pl.from_arrow(raw)
        if isinstance(df, pl.Series):  # single-column edge case
            df = df.to_frame()
        if validate_raw:
            _validate_raw(df)

        out = transform_batch(df, state["rows"])
        state["rows"] += out.height
        state["batches"] += 1

        table = out.to_arrow().cast(OUTPUT_SCHEMA)
        yield from table.to_batches()


def ingest(
    raw_csv: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    row_group_size: int = 1_000_000,
    batch_rows: int = 500_000,
    validate_raw: bool = True,
) -> dict:
    """Run the ingest stage. Returns counters for the caller to report."""
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"{raw_csv} not found. It is DVC-tracked, not committed: "
            "run `dvc checkout` or re-download it."
        )

    # Remove stale partitions wholesale: a previous run with different years
    # would otherwise leave orphans that every later stage would happily read.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = pacsv.open_csv(
        raw_csv,
        read_options=pacsv.ReadOptions(
            # block_size is bytes, not rows; ~100 B/row is close enough.
            block_size=batch_rows * 100,
        ),
        convert_options=pacsv.ConvertOptions(
            column_types=RAW_ARROW_TYPES,
            # Only a genuinely empty field is null. PyArrow's default list would
            # also swallow "NA"/"null", which are plausible string values.
            null_values=[""],
            strings_can_be_null=True,
        ),
    )

    # Always on, even when per-batch validation is skipped: if IBM changes the
    # file's shape we stop here rather than writing a subtly wrong dataset.
    header = reader.schema.names
    if header != RAW_COLUMNS:
        raise ValueError(
            f"CSV header does not match the expected TabFormer columns.\n"
            f"  expected: {RAW_COLUMNS}\n  found:    {header}"
        )

    state = {"rows": 0, "batches": 0}
    started = time.perf_counter()

    ds.write_dataset(
        pa.RecordBatchReader.from_batches(
            OUTPUT_SCHEMA, _batches(reader, validate_raw, state)
        ),
        base_dir=out_dir,
        format="parquet",
        partitioning=ds.partitioning(
            pa.schema([("Year", pa.int32())]), flavor="hive"
        ),
        existing_data_behavior="delete_matching",
        basename_template="part-{i}.parquet",
        # Bounds per-partition buffering: ~30 open writers on a 14 GB box.
        min_rows_per_group=min(200_000, row_group_size),
        max_rows_per_group=row_group_size,
    )

    return {
        "rows": state["rows"],
        "batches": state["batches"],
        "seconds": round(time.perf_counter() - started, 1),
        "partitions": sorted(p.name for p in out_dir.glob("Year=*")),
    }


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=params["paths"]["raw_csv"])
    ap.add_argument("--output", default=params["paths"]["processed"])
    ap.add_argument(
        "--no-validate-raw",
        action="store_true",
        help="skip per-batch raw validation (faster; loses the guarantee)",
    )
    args = ap.parse_args(argv)

    try:
        result = ingest(
            repo_path(args.input),
            repo_path(args.output),
            row_group_size=params["ingest"]["row_group_size"],
            batch_rows=params["ingest"]["batch_rows"],
            validate_raw=params["ingest"]["validate_raw"] and not args.no_validate_raw,
        )
    except pandera.errors.SchemaErrors as exc:
        # Pandera's full dump is thousands of lines; the useful part is which
        # check failed and a few offending values.
        print("\nRAW VALIDATION FAILED -- the CSV does not match "
              "RawTransactionSchema:\n", file=sys.stderr)
        for _, row in exc.failure_cases.drop_duplicates(
            subset=["column", "check"]
        ).head(10).iterrows():
            print(f"  {row['column']}: {row['check']} "
                  f"(e.g. {row['failure_case']!r})", file=sys.stderr)
        return 1
    print(
        f"ingested {result['rows']:,} rows in {result['batches']} batches "
        f"({result['seconds']}s) -> {len(result['partitions'])} year partitions"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
