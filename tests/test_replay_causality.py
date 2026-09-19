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

import pytest

from fraud.jobs.replay import CausalHistory

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


def test_rows_outside_the_velocity_window_are_dropped():
    """Beyond 168h nothing can affect a feature, and serving would not fetch it."""
    h = history()
    h.add(txn(1, 7, D(2019, 1, 1, 10, 0)), VELOCITY)
    h.add(txn(2, 7, D(2019, 1, 5, 10, 0)), VELOCITY)

    # 10 days later: the first row has aged out, the second has not.
    seen = h.neighbourhood(txn(3, 7, D(2019, 1, 11, 10, 0))).user_history
    assert seen["txn_id"].to_list() == [2]


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
