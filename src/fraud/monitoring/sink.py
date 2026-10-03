"""Monitoring results into the shapes production stores them in.

`drift_metrics` and `labels` already exist in Supabase (sql/001_init.sql;
`model` added by sql/004_monitoring.sql). The notebook writes Parquet by
default; writing to the database is a deliberate, separate call, because the
free tier is 500 MB and a full year of labels is most of it.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import psycopg

DRIFT_COLUMNS = ("model", "metric", "feature", "value", "lower_bound",
                 "upper_bound", "alert", "chunk_start", "chunk_end")


def _bound(column: str | None) -> pl.Expr:
    return (pl.col(column) if column else pl.lit(None)).cast(pl.Float64)


def drift_rows(frame: pl.DataFrame, model: str) -> pl.DataFrame:
    """Any estimate/drift frame from fraud.monitoring.estimate -> drift_metrics rows.

    Bounds are the alert band when the frame has one (thresholds), otherwise
    the confidence band; `feature` is null for model-level metrics. Rows with
    no value (a month with no positives) are dropped -- `value` is NOT NULL.
    """
    columns = frame.columns
    lower = ("lower_threshold" if "lower_threshold" in columns
             else "lower_bound" if "lower_bound" in columns else None)
    upper = ("upper_threshold" if "upper_threshold" in columns
             else "upper_bound" if "upper_bound" in columns else None)
    out = frame.select(
        pl.lit(model).alias("model"),
        pl.col("metric"),
        (pl.col("feature") if "feature" in columns else pl.lit(None, pl.String)).alias("feature"),
        pl.col("value").cast(pl.Float64),
        _bound(lower).alias("lower_bound"),
        _bound(upper).alias("upper_bound"),
        (pl.col("alert") if "alert" in columns else pl.lit(False)).alias("alert"),
        pl.col("chunk_start"),
        pl.col("chunk_end"),
    )
    return out.filter(pl.col("value").is_not_nan() & pl.col("value").is_not_null())


def _copy(conn: psycopg.Connection, table: str, columns: tuple[str, ...],
          rows: list[tuple[Any, ...]]) -> int:
    sql = f"copy {table} ({', '.join(columns)}) from stdin"
    with conn.cursor() as cur, cur.copy(sql) as copy:
        for row in rows:
            copy.write_row(row)
    return len(rows)


def write_drift_metrics(conn: psycopg.Connection, rows: pl.DataFrame) -> int:
    """Append drift_rows() output. The caller commits."""
    return _copy(conn, "drift_metrics", DRIFT_COLUMNS,
                 rows.select(DRIFT_COLUMNS).rows())


def write_labels(conn: psycopg.Connection, rows: pl.DataFrame) -> int:
    """Append labels.label_rows() output. The caller commits."""
    return _copy(conn, "labels", ("txn_id", "is_fraud", "labeled_at"),
                 rows.select("txn_id", "is_fraud", "labeled_at").rows())
