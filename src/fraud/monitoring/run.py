"""The production monitoring job: one run per model, for the months closed so far.

    .venv-monitoring/bin/python -m fraud.monitoring.run --as-of 2019-08-01 [--write]

What notebook 01 explored, as the job Phase 6 will schedule. For each model:

1. score months that have CLOSED by `--as-of` -- an open month is a partial
   chunk, and a partial month's AUC-PR is not comparable to a full one;
2. estimate performance with CBPE (no labels) and flag degradation against
   the reference level (decision 25);
3. raise an alert only when the flag PERSISTS for `persistence_months`
   consecutive months -- single months swing by ±0.1 on noise alone;
4. add the realized metric for every month whose labels are complete by
   `--as-of`, so the estimate and the truth sit side by side once it exists;
5. record input drift, freshness, and a watermark in `job_watermarks`.

Historical data has no meaningful "now", so the clock is `--as-of`. In
production it is simply the run time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from typing import Any

import polars as pl

from fraud.config import load_params
from fraud.monitoring import estimate as E
from fraud.monitoring import labels as L
from fraud.monitoring import sink

MODELS = ("xgboost", "graphsage")


def persistent(flags: list[bool], months: int) -> list[bool]:
    """True where this month and the `months - 1` before it are all flagged."""
    out, streak = [], 0
    for flag in flags:
        streak = streak + 1 if flag else 0
        out.append(streak >= months)
    return out


def closed_before(frame: pl.DataFrame, as_of: dt.datetime) -> pl.DataFrame:
    """Rows from calendar months that ended before `as_of`."""
    month_start = as_of.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return frame.filter(pl.col("ts") < month_start)


def load_frames(params: dict, model: str,
                sample: int | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(reference, analysis with label arrival) -- load once, monitor many times."""
    cfg = params["monitoring"]
    reference = E.scored_frame(params, model, "reference", sample=sample)
    analysis = L.label_arrival(
        E.scored_frame(params, model, "analysis", sample=sample), cfg["labels"])
    return reference, analysis


def monitor_model(params: dict, model: str, as_of: dt.datetime,
                  frames: tuple[pl.DataFrame, pl.DataFrame] | None = None) -> dict[str, Any]:
    cfg = params["monitoring"]
    reference, analysis = frames or load_frames(params, model)
    analysis = closed_before(analysis, as_of)
    if analysis.is_empty():
        return {"model": model, "months": 0, "rows": pl.DataFrame()}

    ref_level = E.realized_performance(reference, params).filter(
        pl.col("metric") == "realized_average_precision")["value"].mean()
    estimates = E.estimate_performance(reference, analysis, params)
    ap = E.flag_degradation(
        estimates.filter(pl.col("metric") == "estimated_average_precision").sort("chunk_start"),
        ref_level, cfg["degradation_tolerance"])
    ap = ap.with_columns(pl.Series(
        "alert", persistent(ap["degraded"].to_list(), cfg["persistence_months"])))

    # The truth, only where it exists by now: months whose labels are complete.
    ready = E.first_label_based_view(analysis, params).filter(
        pl.col("labels_ready_at") <= as_of)
    realized = (E.realized_performance(analysis, params, as_of=as_of)
                .join(ready.select("chunk_start"), on="chunk_start", how="semi")
                .with_columns(pl.lit(False).alias("alert")))

    drift = E.univariate_drift(reference, analysis, params)
    recon = E.multivariate_drift(reference, analysis, params)

    rows = pl.concat([
        sink.drift_rows(ap, model),
        sink.drift_rows(estimates.filter(pl.col("metric") != "estimated_average_precision"), model),
        sink.drift_rows(realized, model),
        sink.drift_rows(drift, model),
        sink.drift_rows(recon, model),
    ])
    latest = ap.row(-1, named=True)
    return {
        "model": model,
        "months": ap.height,
        "reference_level": ref_level,
        "latest_month": str(latest["chunk_start"].date()),
        "latest_estimate": latest["value"],
        "latest_relative_change": latest["relative_change"],
        "alert": bool(latest["alert"]),
        "months_flagged": int(ap["degraded"].sum()),
        "months_alerting": int(ap["alert"].sum()),
        "label_ready_months": ready.height,
        "drifting_features": sorted(
            drift.filter(pl.col("alert") & (pl.col("chunk_start") == latest["chunk_start"]))
            ["feature"].to_list()),
        "cursor": latest["chunk_end"],
        "analysis_rows": analysis.height,
        "rows": rows,
    }


def freshness(conn, as_of: dt.datetime, params: dict) -> pl.DataFrame:
    """Freshness SLIs, in hours behind `as_of`, as drift_metrics rows.

    hot store   the newest row the serving path can see -- what the staleness
                experiment turns into an SLA for ingest
    labels      the newest label that has landed
    """
    limits = params["monitoring"]["freshness"]
    with conn.cursor() as cur:
        cur.execute("select max(ts) as ts from transaction_events")
        store_ts = cur.fetchone()["ts"]
        cur.execute("select max(labeled_at) as ts from labels")
        label_ts = cur.fetchone()["ts"]

    def lag(ts) -> float | None:
        if ts is None:
            return None
        ts = ts.replace(tzinfo=None) if ts.tzinfo else ts
        return (as_of - ts).total_seconds() / 3600

    rows = []
    for metric, ts, limit in (
        ("freshness_hot_store_hours", store_ts, limits["hot_store_max_hours"]),
        ("freshness_labels_hours", label_ts, limits["labels_max_hours"]),
    ):
        value = lag(ts)
        if value is not None:
            rows.append({"metric": metric, "value": value, "upper_threshold": float(limit),
                         "alert": value > limit, "chunk_start": as_of, "chunk_end": as_of})
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def write(conn, result: dict, job: str, as_of: dt.datetime) -> int:
    written = sink.write_drift_metrics(conn, result["rows"]) if result["rows"].height else 0
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into job_watermarks (job_name, last_run_at, last_cursor, rows_processed)
            values (%(job)s, now(), %(cursor)s, %(rows)s)
            on conflict (job_name) do update
               set last_run_at = excluded.last_run_at,
                   last_cursor = excluded.last_cursor,
                   rows_processed = excluded.rows_processed
            """,
            {"job": job, "cursor": result.get("cursor", as_of),
             "rows": result.get("analysis_rows", 0)},
        )
    return written


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", required=True, help="the job's clock, e.g. 2019-08-01")
    ap.add_argument("--model", choices=[*MODELS, "all"], default="all")
    ap.add_argument("--write", action="store_true", help="append to Supabase")
    args = ap.parse_args(argv)

    as_of = dt.datetime.fromisoformat(args.as_of)
    models = MODELS if args.model == "all" else (args.model,)
    results = [monitor_model(params, m, as_of) for m in models]

    conn = None
    if args.write:
        from fraud.api import store
        conn = store.connect()
        fresh = freshness(conn, as_of, params)
        for r in results:
            job = f"monitoring:{r['model']}"
            print(f"{r['model']}: wrote {write(conn, r, job, as_of)} rows")
        if fresh.height:
            sink.write_drift_metrics(conn, sink.drift_rows(fresh, "platform"))
        conn.commit()
        conn.close()

    summary = [{k: v for k, v in r.items() if k != "rows"} for r in results]
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
