"""The replay must not be able to see the future.

A leaky replay is indistinguishable from an honest one by inspection: same code,
same queries, plausible latencies, and a metric that simply looks better than it
should. So the invariant is asserted rather than reviewed.

`CausalHistory` is the whole mechanism. Everything it returns must be strictly
earlier than the transaction asking, and a row only enters it *after* that
transaction has been scored.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from fraud.jobs.replay import CausalHistory, shard_boundaries

D = dt.datetime
VELOCITY = {"velocity_count_1h": 0.0, "velocity_seconds_since_last": -1.0}


def txn(txn_id: int, user: int, ts: dt.datetime, merchant: str = "m1") -> dict:
    return {
        "txn_id": txn_id, "User": user, "Card": 0, "ts": ts, "Amount": 1.0,
        "Merchant": merchant, "MCC": 5411, "City": "X", "State": "CA",
        "Zip": 1, "Errors": "XX", "Chip": "Swipe", "Time": 0, "Month": 1,
        "Day": 1,
    }


def history() -> CausalHistory:
    return CausalHistory(window_hours=168, merchant_cap=10)


def test_history_starts_empty():
    h = history()
    n = h.neighbourhood(txn(1, 7, D(2019, 1, 1, 10, 0)))
    assert n.user_history.is_empty()
    assert n.cold_start_user


def test_a_transaction_cannot_see_itself():
    """The ordering that matters: score, then add."""
    h = history()
    first = txn(1, 7, D(2019, 1, 1, 10, 0))
    assert h.neighbourhood(first).user_history.is_empty()
    h.add(first, VELOCITY)
    # Only now is it visible, and only to something later.
    later = txn(2, 7, D(2019, 1, 1, 11, 0))
    assert h.neighbourhood(later).user_history.height == 1


def test_a_later_transaction_is_invisible_to_an_earlier_one():
    """The leak this whole design exists to prevent."""
    h = history()
    h.add(txn(1, 7, D(2019, 1, 1, 10, 0)), VELOCITY)
    h.add(txn(2, 7, D(2019, 1, 1, 15, 0)), VELOCITY)

    # Ask as of 11:00 -- the 15:00 row must not appear.
    seen = h.neighbourhood(txn(99, 7, D(2019, 1, 1, 11, 0))).user_history
    assert seen.height == 1
    assert seen["txn_id"].to_list() == [1]


def test_same_timestamp_rows_are_invisible():
    """Ties, the case that broke first.

    Offline uses closed="left" and the store queries `ts < :now`, so a row
    sharing the arriving timestamp must not be returned here either. 142,010
    rows share a user and a minute.
    """
    h = history()
    h.add(txn(1, 7, D(2019, 1, 1, 10, 0)), VELOCITY)
    h.add(txn(2, 7, D(2019, 1, 1, 10, 30)), VELOCITY)

    seen = h.neighbourhood(txn(3, 7, D(2019, 1, 1, 10, 30))).user_history
    assert seen["txn_id"].to_list() == [1], "a tied row leaked into history"


def test_other_users_are_never_returned():
    h = history()
    h.add(txn(1, 8, D(2019, 1, 1, 10, 0)), VELOCITY)
    seen = h.neighbourhood(txn(2, 7, D(2019, 1, 1, 11, 0))).user_history
    assert seen.is_empty()


def test_old_rows_are_kept_while_the_graph_still_needs_them():
    """The velocity window must not bound the GRAPH.

    Beyond 168h a row affects no velocity feature -- `compute_velocity` rolls
    over the window and ignores it. But the offline user node aggregates
    transactions sampled across the whole split, so a card whose last activity
    was a month ago must still present neighbours. Pruning these was worth
    0.5339 -> 0.5804 AUC-PR on a June 2019 slice.
    """
    h = history()
    h.add(txn(1, 7, D(2019, 1, 1, 10, 0)), VELOCITY)
    h.add(txn(2, 7, D(2019, 1, 5, 10, 0)), VELOCITY)

    # 10 days later both are outside the 168h window, and both are still here.
    seen = h.neighbourhood(txn(3, 7, D(2019, 1, 11, 10, 0))).user_history
    assert seen["txn_id"].to_list() == [2, 1]


def test_old_rows_are_dropped_once_newer_ones_replace_them():
    """Retention is bounded: the cap is row count, not unlimited memory."""
    h = CausalHistory(window_hours=168, merchant_cap=3)
    h.add(txn(1, 7, D(2019, 1, 1, 10, 0)), VELOCITY)          # ancient
    for i in range(3):
        h.add(txn(10 + i, 7, D(2019, 6, 1, 10 + i, 0)), VELOCITY)

    seen = h.neighbourhood(txn(99, 7, D(2019, 6, 2, 10, 0))).user_history
    assert seen["txn_id"].to_list() == [12, 11, 10], "the ancient row should age out"


def test_merchant_history_is_shared_across_users():
    """A merchant node's neighbours come from every card that used it."""
    h = history()
    h.add(txn(1, 7, D(2019, 1, 1, 10, 0), merchant="shop"), VELOCITY)
    h.add(txn(2, 8, D(2019, 1, 1, 11, 0), merchant="shop"), VELOCITY)

    n = h.neighbourhood(txn(3, 9, D(2019, 1, 1, 12, 0), merchant="shop"))
    assert n.merchant_history.height == 2
    assert n.user_history.is_empty()      # user 9 has no history of its own
    assert not n.cold_start_merchant


def test_merchant_history_also_excludes_the_future():
    h = history()
    h.add(txn(1, 7, D(2019, 1, 1, 15, 0), merchant="shop"), VELOCITY)
    n = h.neighbourhood(txn(2, 8, D(2019, 1, 1, 10, 0), merchant="shop"))
    assert n.merchant_history.is_empty()


def test_most_recent_rows_come_first():
    """The graph takes the nearest neighbours, so ordering is load-bearing."""
    h = history()
    for i in range(5):
        h.add(txn(i, 7, D(2019, 1, 1, 10 + i, 0)), VELOCITY)
    seen = h.neighbourhood(txn(99, 7, D(2019, 1, 1, 20, 0))).user_history
    assert seen["txn_id"].to_list() == [4, 3, 2, 1, 0]


@pytest.mark.parametrize("hours", [1, 24, 168])
def test_window_boundary_is_inclusive_of_exactly_the_window(hours):
    h = CausalHistory(window_hours=hours, merchant_cap=10)
    base = D(2019, 6, 1, 12, 0)
    h.add(txn(1, 7, base), VELOCITY)
    just_inside = h.neighbourhood(
        txn(2, 7, base + dt.timedelta(hours=hours) - dt.timedelta(minutes=1))
    )
    assert just_inside.user_history.height == 1


# --- sharding must not change what any transaction can see ---------------
# A shard is seeded with everything before it, so the parallel replay equals
# the sequential one. That holds only if a boundary never lands inside a group
# of tied timestamps: seeding uses `ts < start`, so a tied row left in the
# previous shard would be invisible to ALL of the next one rather than just to
# its twin -- and 142,010 rows share a user and a minute.

def frame_with_ties(pattern: list[int]) -> pl.DataFrame:
    """`pattern` gives how many rows share each successive timestamp."""
    rows, base = [], D(2019, 1, 1, 0, 0)
    for step, count in enumerate(pattern):
        for _ in range(count):
            rows.append({"ts": base + dt.timedelta(minutes=step)})
    return pl.DataFrame(rows)


def test_boundaries_cover_every_row_exactly_once():
    frame = frame_with_ties([1] * 40)
    ranges = shard_boundaries(frame, 4)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == frame.height
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:], strict=False))


def test_a_boundary_never_splits_tied_timestamps():
    # 10 distinct timestamps, the middle one shared by 12 rows -- a boundary
    # computed purely by row count would land inside it.
    frame = frame_with_ties([2, 2, 2, 2, 12, 2, 2, 2, 2, 2])
    ts = frame.get_column("ts").to_list()
    for shards in (2, 3, 4, 5):
        for lo, _ in shard_boundaries(frame, shards)[1:]:
            assert ts[lo] != ts[lo - 1], (
                f"{shards} shards: boundary at {lo} splits a tied timestamp"
            )


def test_single_shard_is_the_whole_frame():
    frame = frame_with_ties([1] * 10)
    assert shard_boundaries(frame, 1) == [(0, 10)]


def test_all_rows_tied_collapses_to_one_shard():
    """Nothing can be split, so asking for 4 shards must still be correct."""
    frame = frame_with_ties([20])
    assert shard_boundaries(frame, 4) == [(0, 20)]


def test_a_same_minute_row_does_not_cost_a_merchant_neighbour():
    """The bug the HTTP latency replay found. With `deque(maxlen=10)`, a
    same-minute transaction at the merchant evicted the 10th strictly-earlier
    row on append and was then filtered out itself, so the model saw 9
    neighbours where Postgres's `ts < now ... limit 10` returns 10."""
    history = CausalHistory(window_hours=168, merchant_cap=10)
    base = dt.datetime(2019, 6, 3, 6, 0)
    for i in range(10):
        history.add(txn(i, user=100 + i, ts=base + dt.timedelta(minutes=i)), {})
    now = base + dt.timedelta(minutes=30)
    history.add(txn(50, user=200, ts=now), {})       # scored earlier this minute
    seen = history.neighbourhood(txn(51, user=201, ts=now)).merchant_history
    assert seen.height == 10
    assert 50 not in seen["txn_id"].to_list()        # its twin stays invisible


def test_a_same_minute_row_does_not_cost_a_user_neighbour():
    """Same flaw on the card side: a quiet card's pruning counted the
    invisible same-minute row toward its 10 graph neighbours."""
    history = CausalHistory(window_hours=1, merchant_cap=10)
    base = dt.datetime(2019, 1, 1)
    for i in range(10):
        history.add(txn(i, user=7, ts=base + dt.timedelta(days=i)), {})
    now = base + dt.timedelta(days=60)
    history.add(txn(50, user=7, ts=now), {})
    seen = history.neighbourhood(txn(51, user=7, ts=now)).user_history
    assert sorted(seen["txn_id"].to_list()) == list(range(10))


def test_a_visibility_lag_hides_exactly_the_recent_rows():
    """The staleness experiment's mechanism: with a 6 h lag, a row is visible
    only once it is 6 h old -- what a store whose ingest runs 6 h behind shows."""
    h = CausalHistory(window_hours=168, merchant_cap=10, visibility_lag_hours=6)
    base = D(2019, 6, 1, 0, 0)
    for i, hours in enumerate([1, 3, 5, 7, 9]):
        h.add(txn(i, user=7, ts=base - dt.timedelta(hours=hours)), {})
    seen = h.neighbourhood(txn(99, user=7, ts=base)).user_history
    assert sorted(seen["txn_id"].to_list()) == [3, 4]          # only the 7 h and 9 h rows


def test_a_lagged_merchant_still_gets_its_full_neighbour_cap():
    """Rows inside the lag are invisible, so they must not evict the older
    rows that ARE visible -- the deque(maxlen) bug again, at a larger scale."""
    h = CausalHistory(window_hours=168, merchant_cap=3, visibility_lag_hours=24)
    base = D(2019, 6, 10)
    for i in range(3):                                          # old, visible
        h.add(txn(i, user=100 + i, ts=base - dt.timedelta(days=5 - i)), {})
    for i in range(3, 9):                                       # recent, lagged
        h.add(txn(i, user=100 + i, ts=base - dt.timedelta(hours=9 - i)), {})
    seen = h.neighbourhood(txn(50, user=200, ts=base)).merchant_history
    assert sorted(seen["txn_id"].to_list()) == [0, 1, 2]


def test_zero_lag_is_serving_as_built():
    a = CausalHistory(window_hours=168, merchant_cap=10)
    b = CausalHistory(window_hours=168, merchant_cap=10, visibility_lag_hours=0)
    base = D(2019, 6, 1)
    for i in range(15):
        row = txn(i, user=7, ts=base - dt.timedelta(hours=(14 - i) * 7))
        a.add(row, {})
        b.add(row, {})
    probe = txn(99, user=7, ts=base)
    assert a.neighbourhood(probe).user_history.equals(b.neighbourhood(probe).user_history)
    assert a.neighbourhood(probe).merchant_history.equals(b.neighbourhood(probe).merchant_history)


@pytest.mark.parametrize("mode", ["graph", "velocity"])
def test_a_lag_can_be_applied_to_one_consumer_only(mode):
    """The decomposition behind the staleness result: lag the graph's view or
    velocity's view, never both, and leave the other fresh."""
    h = CausalHistory(window_hours=168, merchant_cap=10, visibility_lag_hours=6,
                      lag_applies_to=mode)
    base = D(2019, 6, 1)
    for i, hours in enumerate([1, 3, 9]):
        h.add(txn(i, user=7, ts=base - dt.timedelta(hours=hours)), {})
    nb = h.neighbourhood(txn(99, user=7, ts=base))
    lagged, fresh = [2], [0, 1, 2]
    graph, velocity = sorted(nb.user_history["txn_id"].to_list()), \
        sorted(nb.velocity_history["txn_id"].to_list())
    assert (graph, velocity) == ((lagged, fresh) if mode == "graph" else (fresh, lagged))
    merchant = sorted(nb.merchant_history["txn_id"].to_list())
    assert merchant == (lagged if mode == "graph" else fresh)


def test_the_graph_off_ablation_keeps_velocity_and_drops_every_neighbour():
    h = CausalHistory(window_hours=168, merchant_cap=10, drop_neighbours=True)
    base = D(2019, 6, 1)
    for i in range(4):
        h.add(txn(i, user=7, ts=base - dt.timedelta(hours=i + 1)), {})
    nb = h.neighbourhood(txn(99, user=7, ts=base))
    assert nb.user_history.is_empty() and nb.merchant_history.is_empty()
    assert sorted(nb.velocity_history["txn_id"].to_list()) == [0, 1, 2, 3]
