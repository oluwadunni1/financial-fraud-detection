"""Train/serve skew: the project's stated top risk.

CLAUDE.md: "They must be computed by the same code offline and online -- one
module in src/fraud/features/velocity.py, two callers, with a test asserting
they agree on fixed input."

This is that test. It matters more than it looks: skew here produces no error
and no warning. The model is simply fed numbers that differ from its training
distribution, and every downstream metric stays plausible while being wrong.

The strategy is to simulate serving honestly -- walk a fixed history one
transaction at a time, computing each one's velocity from only what preceded it,
and assert the result equals what the bulk offline path produced for that same
row in one pass.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from fraud.features.velocity import REQUIRED_COLUMNS, compute_velocity, velocity_columns
from fraud.features.velocity_online import velocity_for_transaction

D = dt.datetime
WINDOWS = [1, 24, 168]


def history(rows: list[tuple]) -> pl.DataFrame:
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


# Deliberately awkward: two users interleaved, a same-minute tie, a refund, a
# gap that falls outside the 1h window but inside 24h, and a merchant repeat.
FIXTURE = history(
    [
        (1, 7, D(2020, 1, 1, 10, 0), 10.0, "m1", "CA"),
        (2, 8, D(2020, 1, 1, 10, 5), 99.0, "m9", "FL"),
        (3, 7, D(2020, 1, 1, 10, 30), 20.0, "m2", "CA"),
        (4, 7, D(2020, 1, 1, 10, 30), -5.0, "m3", "NY"),   # tie with txn 3
        (5, 7, D(2020, 1, 1, 14, 0), 30.0, "m1", "CA"),    # >1h after, <24h
        (6, 8, D(2020, 1, 1, 15, 0), 40.0, "m9", "FL"),
        (7, 7, D(2020, 1, 3, 9, 0), 50.0, "m4", "TX"),     # >24h, <168h
    ]
)


def offline() -> dict[int, dict]:
    result = compute_velocity(FIXTURE, WINDOWS)
    return {row["txn_id"]: row for row in result.to_dicts()}


def online(txn_id: int) -> dict[str, float]:
    """Serve one transaction from only what strictly preceded it."""
    arriving = FIXTURE.filter(pl.col("txn_id") == txn_id).row(0, named=True)
    prior = FIXTURE.filter(
        (pl.col("User") == arriving["User"]) & (pl.col("ts") < arriving["ts"])
    )
    return velocity_for_transaction(arriving, prior, WINDOWS)


@pytest.mark.parametrize("txn_id", [1, 2, 3, 4, 5, 6, 7])
def test_offline_and_online_agree_on_every_row(txn_id):
    """The guarantee. Every feature, every row, exactly equal."""
    expected = offline()[txn_id]
    actual = online(txn_id)
    for name in velocity_columns(WINDOWS):
        assert actual[name] == pytest.approx(expected[name]), (
            f"SKEW on txn {txn_id}, feature {name}: "
            f"online {actual[name]} != offline {expected[name]}"
        )


def test_a_tied_row_in_history_is_rejected():
    """The exact input that caused the skew this test suite was written to find.

    Offline used to treat a same-timestamp row as "the previous transaction"
    while its window aggregates ignored it. Serving queries `ts < :now` and so
    never sees it. The online path now refuses history containing such a row
    rather than silently producing a different number than training did.
    """
    tied = FIXTURE.filter(pl.col("txn_id") == 4).row(0, named=True)
    with_tie = FIXTURE.filter(
        (pl.col("User") == tied["User"])
        & (pl.col("ts") <= tied["ts"])
        & (pl.col("txn_id") != 4)
    )
    assert with_tie.height > 0, "fixture no longer contains the tie this guards"
    with pytest.raises(ValueError, match="at or after"):
        velocity_for_transaction(tied, with_tie, WINDOWS)


# --- the contract the store must honour ----------------------------------

def test_future_rows_in_history_are_rejected():
    """Serving must never be handed a row from the future, even by accident."""
    arriving = FIXTURE.filter(pl.col("txn_id") == 3).row(0, named=True)
    contaminated = FIXTURE.filter(pl.col("User") == 7)  # includes later rows
    with pytest.raises(ValueError, match="at or after"):
        velocity_for_transaction(arriving, contaminated, WINDOWS)


def test_empty_history_is_a_normal_case():
    """A card's first-ever transaction must score, not raise."""
    arriving = FIXTURE.filter(pl.col("txn_id") == 1).row(0, named=True)
    result = velocity_for_transaction(arriving, FIXTURE.head(0), WINDOWS)
    assert result["velocity_count_1h"] == 0.0
    assert result["velocity_seconds_since_last"] == -1.0


def test_missing_columns_fail_loudly():
    arriving = FIXTURE.filter(pl.col("txn_id") == 3).row(0, named=True)
    del arriving["State"]
    with pytest.raises(ValueError, match="missing columns"):
        velocity_for_transaction(arriving, FIXTURE.head(0), WINDOWS)


def test_output_covers_exactly_the_declared_features():
    assert set(online(5)) == set(velocity_columns(WINDOWS))


def test_other_users_history_does_not_leak_in():
    """User 8's rows must never affect user 7's velocity."""
    arriving = FIXTURE.filter(pl.col("txn_id") == 5).row(0, named=True)
    own = FIXTURE.filter(
        (pl.col("User") == 7) & (pl.col("ts") < arriving["ts"])
    )
    everyone = FIXTURE.filter(pl.col("ts") < arriving["ts"])
    assert velocity_for_transaction(
        arriving, own, WINDOWS
    ) == velocity_for_transaction(arriving, everyone, WINDOWS)


def test_required_columns_are_the_stores_contract():
    """If this changes, sql/003_serving.sql has to change with it."""
    assert set(REQUIRED_COLUMNS) == {
        "txn_id", "User", "ts", "Amount", "Merchant", "State",
    }
