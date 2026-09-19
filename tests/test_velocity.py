"""Velocity feature tests.

These guard the project's stated top risk: train/serve skew. The expected values
below are hand-counted, not captured from a run -- a test that records whatever
the code currently does would happily lock in a leak.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from fraud.features.velocity import (
    NO_PRIOR_TRANSACTION,
    REQUIRED_COLUMNS,
    SECONDS_SINCE_LAST,
    compute_velocity,
    velocity_columns,
)

D = dt.datetime


def frame(rows: list[tuple]) -> pl.DataFrame:
    """rows: (txn_id, User, ts, Amount, Merchant, State)."""
    return pl.DataFrame(
        {
            "txn_id": pl.Series([r[0] for r in rows], dtype=pl.Int64),
            "User": pl.Series([r[1] for r in rows], dtype=pl.Int64),
            "ts": pl.Series([r[2] for r in rows], dtype=pl.Datetime("us")),
            "Amount": pl.Series([r[3] for r in rows], dtype=pl.Float64),
            "Merchant": [r[4] for r in rows],
            "State": [r[5] for r in rows],
        }
    )


# One user, four transactions inside an hour, then one the next day.
BASIC = frame(
    [
        (1, 7, D(2020, 1, 1, 10, 0), 10.0, "m1", "CA"),
        (2, 7, D(2020, 1, 1, 10, 15), 20.0, "m2", "CA"),
        (3, 7, D(2020, 1, 1, 10, 30), 30.0, "m1", "NY"),
        (4, 7, D(2020, 1, 2, 9, 0), 40.0, "m3", "TX"),
    ]
)


def by_id(result: pl.DataFrame) -> dict[int, dict]:
    return {row["txn_id"]: row for row in result.to_dicts()}


def test_counts_only_earlier_transactions():
    """The defining property: a transaction never sees itself or the future."""
    r = by_id(compute_velocity(BASIC, [1]))
    assert r[1]["velocity_count_1h"] == 0  # nothing before it
    assert r[2]["velocity_count_1h"] == 1  # sees txn 1
    assert r[3]["velocity_count_1h"] == 2  # sees txns 1 and 2
    assert r[4]["velocity_count_1h"] == 0  # next day, window empty


def test_amount_sums_exclude_current_row():
    r = by_id(compute_velocity(BASIC, [1]))
    assert r[1]["velocity_amount_1h"] == 0.0
    assert r[2]["velocity_amount_1h"] == 10.0
    assert r[3]["velocity_amount_1h"] == 30.0  # 10 + 20, NOT its own 30
    assert r[4]["velocity_amount_1h"] == 0.0


def test_window_length_is_respected():
    """24h sees the whole first day; 1h does not."""
    r = by_id(compute_velocity(BASIC, [1, 24]))
    assert r[4]["velocity_count_1h"] == 0
    assert r[4]["velocity_count_24h"] == 3
    assert r[4]["velocity_amount_24h"] == 60.0


def test_distinct_merchants_and_states():
    r = by_id(compute_velocity(BASIC, [24]))
    # txn 4 looks back at m1, m2, m1 -> 2 distinct; CA, CA, NY -> 2 distinct.
    assert r[4]["velocity_merchants_24h"] == 2
    assert r[4]["velocity_states_24h"] == 2


def test_same_timestamp_transactions_are_mutually_invisible():
    """The tie rule. 142,010 real rows share a user and a minute.

    Both must see only the earlier transaction -- never each other -- or the
    online `ts < :now` query disagrees with training.
    """
    tied = frame(
        [
            (1, 7, D(2020, 1, 1, 10, 0), 10.0, "m1", "CA"),
            (2, 7, D(2020, 1, 1, 10, 30), 20.0, "m2", "CA"),
            (3, 7, D(2020, 1, 1, 10, 30), 30.0, "m3", "NY"),
        ]
    )
    r = by_id(compute_velocity(tied, [1]))
    assert r[2]["velocity_count_1h"] == 1
    assert r[3]["velocity_count_1h"] == 1
    assert r[2]["velocity_amount_1h"] == 10.0
    assert r[3]["velocity_amount_1h"] == 10.0


def test_users_never_see_each_other():
    two = frame(
        [
            (1, 7, D(2020, 1, 1, 10, 0), 10.0, "m1", "CA"),
            (2, 8, D(2020, 1, 1, 10, 30), 20.0, "m2", "CA"),
            (3, 7, D(2020, 1, 1, 10, 45), 30.0, "m3", "NY"),
        ]
    )
    r = by_id(compute_velocity(two, [24]))
    assert r[2]["velocity_count_24h"] == 0  # user 8's first, ignores user 7
    assert r[3]["velocity_count_24h"] == 1  # user 7 sees only its own txn 1


def test_seconds_since_last():
    # 24h window: the 22.5h gap to txn 4 has to be inside it to be reported at
    # all -- the feature is bounded by the largest window so that serving, which
    # fetches finite history, can compute the same thing.
    r = by_id(compute_velocity(BASIC, [1, 24]))
    assert r[1][SECONDS_SINCE_LAST] == NO_PRIOR_TRANSACTION
    assert r[2][SECONDS_SINCE_LAST] == 15 * 60
    assert r[4][SECONDS_SINCE_LAST] == pytest.approx((22 * 60 + 30) * 60)


def test_seconds_since_last_skips_tied_rows():
    """A tied row is not "the previous transaction" -- it is invisible.

    `closed="left"` excludes same-timestamp rows from every window, so this
    feature must exclude them too or one vector carries two different notions of
    "previous". It used to report 0.0 here, which disagreed with what serving
    computes from `ts < :now`. See tests/test_skew.py.
    """
    tied = frame(
        [
            (1, 7, D(2020, 1, 1, 10, 0), 10.0, "m1", "CA"),
            (2, 7, D(2020, 1, 1, 10, 0), 20.0, "m2", "CA"),
        ]
    )
    r = by_id(compute_velocity(tied, [1]))
    # Both rows tie, so neither has a strictly-earlier predecessor.
    assert r[1][SECONDS_SINCE_LAST] == NO_PRIOR_TRANSACTION
    assert r[2][SECONDS_SINCE_LAST] == NO_PRIOR_TRANSACTION


def test_seconds_since_last_is_bounded_by_the_largest_window():
    """Serving fetches finite history, so an unbounded lookback is unservable."""
    far_apart = frame(
        [
            (1, 7, D(2020, 1, 1, 10, 0), 10.0, "m1", "CA"),
            (2, 7, D(2020, 3, 1, 10, 0), 20.0, "m2", "CA"),  # ~60 days later
        ]
    )
    r = by_id(compute_velocity(far_apart, [1, 24, 168]))
    assert r[2][SECONDS_SINCE_LAST] == NO_PRIOR_TRANSACTION


def test_input_order_does_not_matter():
    """Offline gets bulk rows; online gets one user's recent rows. Same answer."""
    shuffled = BASIC.sample(fraction=1.0, shuffle=True, seed=7)
    a = compute_velocity(BASIC, [1, 24]).sort("txn_id")
    b = compute_velocity(shuffled, [1, 24]).sort("txn_id")
    assert a.equals(b)


def test_is_pure_and_repeatable():
    """Phase 4's skew test leans on this: no hidden state, no I/O."""
    a = compute_velocity(BASIC, [1, 24, 168])
    b = compute_velocity(BASIC, [1, 24, 168])
    assert a.equals(b)
    # The input must not be mutated.
    assert BASIC["txn_id"].to_list() == [1, 2, 3, 4]


def test_serving_shape_matches_bulk():
    """A single user's slice must give that user identical numbers.

    This is the offline/online equivalence in miniature: the API will call this
    with one user's recent rows, not the whole dataset.
    """
    many = frame(
        [
            (1, 7, D(2020, 1, 1, 10, 0), 10.0, "m1", "CA"),
            (2, 8, D(2020, 1, 1, 10, 5), 99.0, "m9", "FL"),
            (3, 7, D(2020, 1, 1, 10, 30), 20.0, "m2", "CA"),
            (4, 8, D(2020, 1, 1, 10, 40), 99.0, "m9", "FL"),
        ]
    )
    bulk = by_id(compute_velocity(many, [1]))
    slice_ = by_id(compute_velocity(many.filter(pl.col("User") == 7), [1]))
    for txn in (1, 3):
        assert bulk[txn]["velocity_count_1h"] == slice_[txn]["velocity_count_1h"]
        assert bulk[txn]["velocity_amount_1h"] == slice_[txn]["velocity_amount_1h"]


def test_output_columns_match_declared_names():
    windows = [1, 24, 168]
    result = compute_velocity(BASIC, windows)
    assert result.columns == ["txn_id", *velocity_columns(windows)]


def test_one_row_per_input_row():
    result = compute_velocity(BASIC, [1, 24])
    assert result.height == BASIC.height
    assert sorted(result["txn_id"].to_list()) == [1, 2, 3, 4]


def test_no_nulls_in_output():
    """Nulls would reach XGBoost as missing and silently change splits."""
    result = compute_velocity(BASIC, [1, 24, 168])
    assert result.null_count().sum_horizontal().item() == 0


@pytest.mark.parametrize("column", [c for c in REQUIRED_COLUMNS if c != "txn_id"])
def test_missing_required_column_raises(column):
    with pytest.raises(ValueError, match="missing required columns"):
        compute_velocity(BASIC.drop(column), [1])


def test_empty_windows_raises():
    with pytest.raises(ValueError, match="windows_hours is empty"):
        compute_velocity(BASIC, [])
