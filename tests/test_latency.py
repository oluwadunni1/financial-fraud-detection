"""The latency replay's moving parts, held to what the numbers depend on.

A latency figure is only as good as its accounting: the header must carry what
the server measured, the projection must be plain arithmetic on it, and the
one-round-trip read must return exactly what the two-round-trip read did --
otherwise a faster number is a faster number for a different model input.
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib

import numpy as np
import polars as pl
import pytest

from fraud.api import store
from fraud.api.timing import Spans, format_server_timing, parse_server_timing
from fraud.features.velocity import velocity_columns
from fraud.jobs.latency import analyse

# --- Server-Timing ---------------------------------------------------------

def test_server_timing_round_trips_through_the_header():
    header = format_server_timing({"fetch": 210.5, "score": 4.25},
                                  {"fetch": 3, "score": 0})
    assert header == "fetch;dur=210.500;rt=3, score;dur=4.250;rt=0"
    assert parse_server_timing(header) == {
        "fetch": {"dur": 210.5, "rt": 3.0},
        "score": {"dur": 4.25, "rt": 0.0},
    }


def test_parse_ignores_parameters_it_does_not_know():
    parsed = parse_server_timing('db;desc="pg";dur=1.5, cache;dur=0.2')
    assert parsed == {"db": {"dur": 1.5, "rt": 0.0}, "cache": {"dur": 0.2, "rt": 0.0}}


def test_spans_record_duration_and_declared_round_trips():
    spans = Spans()
    with spans.span("fetch", round_trips=3):
        pass
    with spans.span("score"):
        pass
    parsed = parse_server_timing(spans.header())
    assert list(parsed) == ["fetch", "score"]
    assert parsed["fetch"]["rt"] == 3 and parsed["score"]["rt"] == 0
    assert all(v["dur"] >= 0 for v in parsed.values())


# --- the projection is arithmetic, nothing more ----------------------------

def test_projection_swaps_measured_rtt_for_the_target():
    # Two requests, 6 round trips each at a measured 70 ms RTT.
    wall = np.array([430.0, 450.0])
    spans = {"fetch": np.array([212.0, 215.0]), "score": np.array([5.0, 6.0]),
             "write": np.array([210.0, 220.0])}
    trips = {"fetch": np.array([3, 3]), "score": np.array([0, 0]),
             "write": np.array([3, 3])}
    out = analyse(wall, spans, trips, rtt_ms=70.0, target_rtt_ms=[1.0])

    # wall - 6 x (70 - 1)
    projected = out["projected_same_region_ms"]["rtt_1ms"]
    assert projected["p50"] == pytest.approx(np.median([430 - 414, 450 - 414]))
    # server-side DB time = span - round trips x RTT, summed over DB spans
    assert out["decomposition_ms"]["db_server"]["mean"] == pytest.approx(
        np.mean([(212 - 210) + (210 - 210), (215 - 210) + (220 - 210)])
    )
    # what is left of wall time outside every span is HTTP overhead
    assert out["decomposition_ms"]["http_overhead"]["mean"] == pytest.approx(
        np.mean([430 - 427, 450 - 441])
    )
    assert out["round_trips_per_request"]["p50"] == 6


# --- the store returns every encoder input ---------------------------------

def _selected_columns() -> set[str]:
    names = {c.strip() for c in store._HISTORY_COLUMNS.replace("\n", " ").split(",")}
    return {store._DB_TO_CANONICAL.get(n, n) for n in names if n}


ENCODER = pathlib.Path("data/features/encoder.json")


@pytest.mark.skipif(not ENCODER.exists(), reason="encoder absent -- dvc pull")
def test_history_reads_return_every_encoder_input():
    """The first HTTP replay with a populated store failed on `MCC`: the
    history SELECT never carried it, and every unit test built neighbour
    frames in memory, so nothing had ever asked the database for one."""
    from fraud.features.encoders import Encoder

    encoder = Encoder.from_json(ENCODER)
    required = {*encoder.one_hot, *encoder.binary, *encoder.numeric}
    missing = required - _selected_columns()
    assert not missing, f"history reads omit encoder inputs: {sorted(missing)}"


def test_stored_columns_match_the_insert():
    """Bulk-loaded seed rows and live inserts must fill the same columns."""
    for column in store.STORED_COLUMNS:
        assert f"%({column})s" in store._INSERT_SQL


# --- seeding only the slice's keys changes nothing they can see ------------

def _write_partitioned(frame: pl.DataFrame, root: pathlib.Path) -> None:
    for (year,), part in frame.group_by("Year"):
        out = root / f"Year={year}"
        out.mkdir(parents=True)
        part.drop("Year").write_parquet(out / "part.parquet")


def test_filtered_seed_gives_the_same_neighbourhoods(tmp_path: pathlib.Path):
    from fraud.jobs.replay import (
        CausalHistory,
        row_velocity,
        seed_rows,
        transaction_from_row,
    )

    rng = np.random.default_rng(3)
    n = 400
    start = dt.datetime(2019, 3, 1)
    ts = sorted(start - dt.timedelta(hours=float(h))
                for h in rng.uniform(1, 24 * 40, n))
    frame = pl.DataFrame({
        "txn_id": np.arange(n, dtype=np.int64),
        "ts": ts,
        "User": rng.integers(0, 12, n),
        "Card": np.zeros(n, dtype=np.int64),
        "Amount": rng.uniform(1, 200, n).round(2),
        "Merchant": [str(m) for m in rng.integers(100, 115, n)],
        "MCC": np.full(n, 5411),
        "City": ["X"] * n, "State": ["CA"] * n, "Zip": np.full(n, 90210),
        "Errors": [""] * n, "Chip": ["Chip"] * n,
        "Time": np.zeros(n, dtype=np.int64), "Month": np.ones(n, dtype=np.int64),
        "Day": np.ones(n, dtype=np.int64), "Fraud": np.zeros(n, dtype=np.int8),
    }).with_columns(pl.col("ts").dt.year().alias("Year"))
    names = velocity_columns([1, 24, 168])
    velocity = frame.select("txn_id", "Year", *[pl.lit(1.0).alias(c) for c in names])
    _write_partitioned(frame, tmp_path / "processed")
    _write_partitioned(velocity, tmp_path / "velocity")

    params = {
        "paths": {"processed": str(tmp_path / "processed"),
                  "velocity": str(tmp_path / "velocity")},
        "serving": {"neighbours": 3},
    }
    graph_since = start - dt.timedelta(days=30)
    users, merchants = {1, 4, 7}, {"103", "110"}

    def seeded(**keys) -> CausalHistory:
        history = CausalHistory(window_hours=168, merchant_cap=3)
        for r in seed_rows(params, start, 168, graph_since, **keys).iter_rows(named=True):
            history.add(transaction_from_row(r), row_velocity(r, names))
        return history

    full, slim = seeded(), seeded(users=users, merchants=merchants)
    for user in users:
        for merchant in merchants:
            probe = {"User": user, "Merchant": merchant, "ts": start}
            a, b = full.neighbourhood(probe), slim.neighbourhood(probe)
            assert a.user_history.equals(b.user_history)
            assert a.merchant_history.equals(b.merchant_history)


# --- one round trip reads exactly what two did -----------------------------

def _database_available() -> bool:
    try:
        store.connection_string()
    except RuntimeError:
        return False
    return not os.getenv("FRAUD_SKIP_DB_TESTS")


@pytest.mark.skipif(not _database_available(), reason="no DATABASE_URL")
def test_merged_read_returns_what_the_two_reads_return():
    """Run against a TEMP table that shadows the real one inside a rolled-back
    transaction, so the live store is never touched."""
    try:
        conn = store.connect()
    except Exception as exc:  # noqa: BLE001 -- an unreachable DB is a skip
        pytest.skip(f"database unreachable: {exc}")
    now = dt.datetime(2019, 6, 3, 12, 0)
    rng = np.random.default_rng(5)
    try:
        with conn.cursor() as cur:
            cur.execute("create temp table transaction_events "
                        "(like public.transaction_events including all)")
            rows = []
            for i in range(120):
                ts = now - dt.timedelta(hours=float(rng.uniform(-5, 400)))
                ts = ts.replace(second=0, microsecond=0)  # force ties
                rows.append(store.stored_row(
                    {"txn_id": i, "User": int(rng.integers(0, 3)), "ts": ts,
                     "Amount": 10.0, "MCC": 5411,
                     "Merchant": str(int(rng.integers(0, 3))),
                     "City": "X", "State": "CA", "Zip": 90210, "Errors": "",
                     "Chip": "Chip", "Time": 0, "Month": 6, "Day": 3},
                    {c: 1.0 for c in velocity_columns([1, 24, 168])},
                ))
            cols = ", ".join(store.STORED_COLUMNS)
            with cur.copy(f"copy transaction_events ({cols}) from stdin") as copy:
                for row in rows:
                    copy.write_row(tuple(row[c] for c in store.STORED_COLUMNS))
        for user in range(3):
            for merchant in range(3):
                args = {"user_id": user, "merchant_id": merchant, "now": now,
                        "history_hours": 168, "graph_rows": 10}
                two = store.fetch_neighbourhood(conn, **args)
                one = store.fetch_neighbourhood_once(conn, **args)
                assert two.user_history.equals(one.user_history)
                assert two.merchant_history.equals(one.merchant_history)
    finally:
        conn.rollback()
        conn.close()


def test_equivalence_counts_only_real_divergence():
    from fraud.jobs.latency import equivalence

    ref_ids, ref_scores = [1, 2, 3, 4], [0.10, 0.20, 0.30, 0.40]
    # float32 storage noise on 1 and 2; a real divergence on 3; 9 unknown
    out = equivalence([1, 2, 3, 9], [0.1000001, 0.1999999, 0.34, 0.5],
                      ref_ids, ref_scores)
    assert out["compared"] == 3
    assert out["mismatches_over_tolerance"] == 1
    assert out["max_abs_diff"] == pytest.approx(0.04)
