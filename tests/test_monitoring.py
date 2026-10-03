"""Monitoring: late labels, label-free estimates, and where results land.

The NannyML-backed tests skip in .venv (it is not installed there by design)
and run under .venv-monitoring:

    .venv-monitoring/bin/python -m pytest tests/test_monitoring.py
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from fraud.monitoring import labels as L
from fraud.monitoring import sink

CFG = {"fraud_delay_median_days": 45, "fraud_delay_sigma": 0.5,
       "fraud_delay_min_days": 7, "fraud_delay_max_days": 120,
       "confirm_after_days": 90, "seed": 42}
PARAMS = {"monitoring": {"chunk_period": "M", "min_label_coverage": 0.99}}


def frame(n: int = 4000, fraud_every: int = 20, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = dt.datetime(2019, 1, 1)
    ts = sorted(start + dt.timedelta(minutes=float(m)) for m in rng.uniform(0, 60 * 24 * 90, n))
    label = (np.arange(n) % fraud_every == 0).astype(np.int8)
    return pl.DataFrame({
        "txn_id": rng.permutation(n).astype(np.int64) * 7 + 3,
        "ts": ts,
        "label": label,
        "score": np.clip(label * 0.6 + rng.uniform(0, 0.5, n), 0, 1),
    })


# --- delayed labels ---------------------------------------------------------------

def test_legitimate_rows_are_confirmed_when_the_dispute_window_closes():
    out = L.label_arrival(frame(), CFG)
    legit = out.filter(pl.col("label") == 0)
    assert (legit["labeled_at"] - legit["ts"]).dt.total_days().unique().to_list() == [90]


def test_fraud_labels_arrive_within_the_clipped_delay():
    out = L.label_arrival(frame(n=20_000, fraud_every=2), CFG)
    days = ((out.filter(pl.col("label") == 1)["labeled_at"]
             - out.filter(pl.col("label") == 1)["ts"]).dt.total_seconds() / 86_400)
    assert days.min() >= 7 and days.max() <= 120
    assert 40 < days.median() < 50          # lognormal, median 45


def test_a_transaction_gets_the_same_delay_however_the_frame_is_ordered():
    """Draws are made in txn_id order, so a resorted or filtered frame cannot
    hand a transaction a different chargeback date."""
    base = frame()
    a = L.label_arrival(base, CFG).select("txn_id", "labeled_at").sort("txn_id")
    b = L.label_arrival(base.sample(fraction=1.0, shuffle=True, seed=9), CFG)
    assert a.equals(b.select("txn_id", "labeled_at").sort("txn_id"))


def test_nothing_is_known_before_its_label_arrives():
    out = L.label_arrival(frame(), CFG)
    as_of = dt.datetime(2019, 2, 15)
    known = L.label_rows(out, as_of)
    assert known["labeled_at"].max() <= as_of
    assert known.columns == ["txn_id", "is_fraud", "labeled_at"]
    assert known.height == int(L.known_at(out, as_of).sum())


# --- realized and label-readiness ---------------------------------------------------

def test_known_label_metrics_report_their_coverage():
    from fraud.monitoring.estimate import realized_performance

    out = L.label_arrival(frame(), CFG)
    hindsight = realized_performance(out, PARAMS)
    assert hindsight["label_coverage"].min() == 1.0
    early = realized_performance(out, PARAMS, as_of=dt.datetime(2019, 3, 1))
    assert early["label_coverage"].max() < 0.2   # only early frauds exist yet


def test_a_month_is_label_ready_about_one_dispute_window_after_it_closes():
    from fraud.monitoring.estimate import first_label_based_view

    view = first_label_based_view(L.label_arrival(frame(), CFG), PARAMS)
    assert view["days_after_month_end"].drop_nulls().is_between(60, 91).all()


# --- sink -----------------------------------------------------------------------

def test_drift_rows_prefer_the_alert_band_and_drop_empty_values():
    est = pl.DataFrame({
        "chunk_start": [dt.datetime(2019, 1, 1)] * 2,
        "chunk_end": [dt.datetime(2019, 1, 31)] * 2,
        "metric": ["estimated_average_precision", "realized_average_precision"],
        "value": [0.3, float("nan")],
        "lower_bound": [0.1, 0.1], "upper_bound": [0.5, 0.5],
        "lower_threshold": [0.2, 0.2], "upper_threshold": [0.9, 0.9],
        "alert": [True, False],
    })
    rows = sink.drift_rows(est, "xgboost")
    assert rows.height == 1
    assert rows.columns == list(sink.DRIFT_COLUMNS)
    row = rows.row(0, named=True)
    assert (row["lower_bound"], row["upper_bound"]) == (0.2, 0.9)
    assert row["feature"] is None and row["model"] == "xgboost" and row["alert"]


def _db():
    from fraud.api import store
    try:
        store.connection_string()
        return store.connect()
    except Exception as exc:  # noqa: BLE001 -- no database is a skip, not a failure
        pytest.skip(f"database unavailable: {exc}")


def test_drift_rows_round_trip_through_the_table():
    """Into a TEMP shadow of drift_metrics, rolled back: the live table is
    never touched, and the test passes before or after sql/004 is applied."""
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute("create temp table drift_metrics "
                        "(like public.drift_metrics including all)")
            cur.execute("alter table drift_metrics add column if not exists model text")
        rows = sink.drift_rows(pl.DataFrame({
            "chunk_start": [dt.datetime(2019, 1, 1)], "chunk_end": [dt.datetime(2019, 1, 31)],
            "metric": ["jensen_shannon"], "feature": ["Amount"], "value": [0.04],
            "upper_threshold": [0.1], "alert": [False],
        }), "graphsage")
        assert sink.write_drift_metrics(conn, rows) == 1
        with conn.cursor() as cur:
            cur.execute("select model, metric, feature, alert from drift_metrics")
            assert cur.fetchone() == {"model": "graphsage", "metric": "jensen_shannon",
                                      "feature": "Amount", "alert": False}
    finally:
        conn.rollback()
        conn.close()


# --- environment hygiene ---------------------------------------------------------------

def test_monitoring_never_imports_a_model_library():
    """It reads scores; it never loads a model. If it ever imported xgboost
    or torch it could not run in .venv-monitoring, whose XGBoost is the 2.1
    NannyML pins rather than the 3.4 the boosters were saved with."""
    code = ("import sys; import fraud.monitoring.labels, fraud.monitoring.estimate, "
            "fraud.monitoring.sink; "
            "bad = [m for m in ('xgboost', 'torch') if m in sys.modules]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


# --- CBPE, on a toy where the answer is known ----------------------------------------

def test_cbpe_estimates_without_seeing_analysis_labels():
    pytest.importorskip("nannyml")
    from fraud.monitoring.estimate import estimate_performance

    rng = np.random.default_rng(1)

    def period(start: dt.datetime, n: int) -> pl.DataFrame:
        label = (rng.uniform(size=n) < 0.05).astype(np.int8)
        score = np.clip(np.where(label == 1, rng.beta(5, 2, n), rng.beta(2, 5, n)), 0, 1)
        minutes = rng.uniform(0, 60 * 24 * 120, n)
        ts = sorted(start + dt.timedelta(minutes=float(m)) for m in minutes)
        return pl.DataFrame({"txn_id": np.arange(n), "ts": ts, "score": score, "label": label})

    params = {"monitoring": {"chunk_period": "M", "target_false_positive_rate": 0.01}}
    ref, ana = period(dt.datetime(2018, 1, 1), 40_000), period(dt.datetime(2019, 1, 1), 40_000)
    est = estimate_performance(ref, ana.drop("label"), params)
    ap = est.filter(pl.col("metric") == "estimated_average_precision")
    assert ap.height >= 4
    assert ap["value"].is_between(0, 1).all()


def test_degradation_is_judged_against_the_reference_level():
    from fraud.monitoring.estimate import flag_degradation

    est = pl.DataFrame({"value": [0.50, 0.44, 0.30]})
    out = flag_degradation(est, reference_level=0.5, tolerance=0.15)
    assert out["degraded"].to_list() == [False, False, True]     # 0.425 is the line
    assert out["relative_change"].to_list() == pytest.approx([0.0, -0.12, -0.4])
