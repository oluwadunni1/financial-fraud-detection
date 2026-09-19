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
