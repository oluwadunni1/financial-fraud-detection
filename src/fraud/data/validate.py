"""Stage `validate`: the gate between ingest and everything downstream.

Runs CleanTransactionSchema over every year partition, then dataset-level
checks that no single partition could catch (global txn_id uniqueness, the
fraud-rate band, year coverage). Writes a JSON report so the numbers are a
tracked DVC output rather than console noise that scrolls away.

Failing here is the point. A corrupted row must stop the build, not train
quietly -- see ARCHITECTURE.md section 9.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandera.errors
import polars as pl

from fraud.config import load_params, repo_path
from fraud.data.schemas import CleanTransactionSchema, check_fraud_rate

# Columns that came from the raw file, i.e. everything except what ingest
# minted. Duplicate detection runs over these -- txn_id is unique by
# construction and would mask genuine duplicates.
_SYNTHETIC = ("txn_id", "ts")


def _partitions(processed: pathlib.Path) -> list[pathlib.Path]:
    parts = sorted(processed.glob("Year=*"), key=lambda p: int(p.name.split("=")[1]))
    if not parts:
        raise FileNotFoundError(
            f"no Year=* partitions under {processed}. Run the ingest stage first."
        )
    return parts


def validate(processed: pathlib.Path, params: dict) -> dict:
    cfg = params["validation"]
    report: dict = {"partitions": {}, "checks": {}}
    failures: list[str] = []

    # --- per-partition: schema + partition-key consistency -----------------
    for part in _partitions(processed):
        year = int(part.name.split("=")[1])
        # Each year is at most ~1.72M rows, so the existing pandas-backed
        # schema is reused unchanged rather than ported to another backend.
        pdf = pl.read_parquet(part).with_columns(Year=pl.lit(year, pl.Int32)).to_pandas()
        try:
            CleanTransactionSchema.validate(pdf, lazy=True)
        except pandera.errors.SchemaErrors as exc:
            # Record the failure in the report rather than dumping thousands of
            # lines: the report is a tracked output and is where you look first.
            for _, row in (
                exc.failure_cases.drop_duplicates(subset=["column", "check"])
                .head(5)
                .iterrows()
            ):
                failures.append(
                    f"Year={year}: column {row['column']!r} failed "
                    f"{row['check']} (e.g. {row['failure_case']!r})"
                )

        ts_years = pdf["ts"].dt.year
        if not (ts_years == year).all():
            failures.append(f"Year={year}: ts year disagrees with the partition key")

        report["partitions"][str(year)] = {
            "rows": int(len(pdf)),
            "fraud": int(pdf["Fraud"].sum()),
        }

    # --- dataset-level ------------------------------------------------------
    lf = pl.scan_parquet(processed / "**/*.parquet", hive_partitioning=True)
    agg = lf.select(
        pl.len().alias("rows"),
        pl.col("Fraud").sum().alias("fraud"),
        pl.col("txn_id").n_unique().alias("txn_id_unique"),
        pl.col("txn_id").min().alias("txn_id_min"),
        pl.col("txn_id").max().alias("txn_id_max"),
        pl.col("ts").min().alias("ts_min"),
        pl.col("ts").max().alias("ts_max"),
    ).collect(engine="streaming")

    rows = int(agg["rows"][0])
    fraud = int(agg["fraud"][0])
    report["rows"] = rows
    report["fraud"] = fraud
    report["fraud_rate"] = fraud / rows if rows else 0.0
    report["ts_min"] = str(agg["ts_min"][0])
    report["ts_max"] = str(agg["ts_max"][0])

    # txn_id must be a contiguous 0..n-1 range: that is what makes it stable
    # across re-ingests and safe as a Postgres primary key.
    checks = report["checks"]
    checks["txn_id_unique"] = int(agg["txn_id_unique"][0]) == rows
    checks["txn_id_contiguous"] = (
        int(agg["txn_id_min"][0]) == 0 and int(agg["txn_id_max"][0]) == rows - 1
    )

    checks["row_count_min"] = rows >= cfg["expected_row_count_min"]

    # Reuse the existing gate rather than re-deriving the band here.
    try:
        check_fraud_rate(
            lf.select("Fraud").collect(engine="streaming").to_pandas(),
            cfg["fraud_rate_min"],
            cfg["fraud_rate_max"],
        )
        checks["fraud_rate_in_band"] = True
    except ValueError as exc:
        checks["fraud_rate_in_band"] = False
        failures.append(str(exc))

    # No nulls anywhere: ingest applies markers, so a null means a transform
    # silently failed.
    nulls = lf.null_count().collect(engine="streaming").row(0, named=True)
    report["null_counts"] = {k: int(v) for k, v in nulls.items() if v}
    checks["no_nulls"] = not report["null_counts"]

    # Duplicates are expected (66) and harmless once txn_id is synthetic, but
    # they stay visible rather than being silently dropped.
    payload = [c for c in lf.collect_schema().names() if c not in _SYNTHETIC]
    distinct = (
        lf.select(payload).unique().select(pl.len().alias("n")).collect(engine="streaming")
    )
    report["duplicate_rows"] = rows - int(distinct["n"][0])

    expected_years = set(range(2020 - 29, 2021))  # 1991..2020
    found_years = {int(y) for y in report["partitions"]}
    checks["all_years_present"] = expected_years <= found_years
    if not checks["all_years_present"]:
        failures.append(f"missing years: {sorted(expected_years - found_years)}")

    for name, passed in checks.items():
        if not passed and name != "fraud_rate_in_band":
            failures.append(f"check failed: {name}")

    report["passed"] = not failures
    report["failures"] = failures
    return report


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=params["paths"]["processed"])
    ap.add_argument("--report", default=params["paths"]["validation_report"])
    args = ap.parse_args(argv)

    report = validate(repo_path(args.input), params)

    out = repo_path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(
        f"{report['rows']:,} rows | {report['fraud']:,} fraud "
        f"({report['fraud_rate']:.5f}) | {report['duplicate_rows']} duplicate rows"
    )
    if not report["passed"]:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for f in report["failures"]:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("all checks passed ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
