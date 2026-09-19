"""Velocity features -- the shared offline/online module.

This is the file the project's top risk lives in. Velocity aggregates must be
computed by *the same code* offline (training) and online (serving); if the two
drift apart, the model is fed different numbers than it was trained on and
nothing else in the system notices. One module, two callers.

`compute_velocity` is therefore deliberately pure: a frame in, a frame out. No
config reads, no database, no file I/O. The Phase 4 API hands it the arriving
transaction plus that user's recent rows and gets numbers identical to the ones
the training matrix was built from, which is what the skew test asserts.

Causality
---------
At serving, the arriving transaction has NOT yet been written to
`transaction_events`, so its aggregates cover only strictly-earlier
transactions. Offline must match exactly:

    offline  rolling(..., closed="left")   <->   online  WHERE ts < :now

`closed="left"` excludes the current row *and* every row sharing its timestamp.
That is not a corner case here: TabFormer timestamps are minute-resolution and
142,010 rows share a (User, Card, minute) with another row. Two transactions in
the same minute are mutually invisible, and the online query must use a strict
`<` so it agrees. A `<=` online would let the later of the pair see the earlier
while the earlier saw nothing -- asymmetric, and undetectable without this test.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

# What a caller must supply. Kept explicit so the online caller fails loudly
# rather than silently producing nulls for a column it forgot.
REQUIRED_COLUMNS = ("txn_id", "User", "ts", "Amount", "Merchant", "State")

# Returned when a user has no earlier transaction at all. A sentinel rather than
# null: the model needs a number, and 0 would read as "just transacted", which
# is the opposite of the truth.
NO_PRIOR_TRANSACTION = -1.0

SECONDS_SINCE_LAST = "velocity_seconds_since_last"


def velocity_columns(windows_hours: Sequence[int]) -> list[str]:
    """Feature names produced for these windows, in output order."""
    names: list[str] = []
    for w in windows_hours:
        names += [
            f"velocity_count_{w}h",
            f"velocity_amount_{w}h",
            f"velocity_merchants_{w}h",
            f"velocity_states_{w}h",
        ]
    return [*names, SECONDS_SINCE_LAST]


def compute_velocity(
    txns: pl.DataFrame, windows_hours: Sequence[int]
) -> pl.DataFrame:
    """Per-user aggregates over strictly-earlier transactions.

    Args:
        txns: must contain REQUIRED_COLUMNS. Order does not matter -- the frame
            is sorted internally, so offline (bulk, arbitrary order) and online
            (one user, recent rows) give the same answer.
        windows_hours: lookback windows, e.g. [1, 24, 168].

    Returns:
        One row per input row, keyed by `txn_id`, with `velocity_columns()`.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in txns.columns]
    if missing:
        raise ValueError(
            f"compute_velocity is missing required columns: {missing}. "
            f"Required: {list(REQUIRED_COLUMNS)}"
        )
    if not windows_hours:
        raise ValueError("windows_hours is empty; nothing to compute")

    # Sorting is part of the contract, not an optimisation: `rolling` requires a
    # sorted index column, and the online caller cannot be trusted to pre-sort.
    frame = txns.sort("User", "ts")
    out = frame.select("txn_id")

    for window in windows_hours:
        # closed="left" == strictly-earlier. See the module docstring.
        agg = frame.rolling(
            index_column="ts",
            period=f"{window}h",
            group_by="User",
            closed="left",
        ).agg(
            pl.len().alias(f"velocity_count_{window}h"),
            pl.col("Amount").sum().alias(f"velocity_amount_{window}h"),
            pl.col("Merchant").n_unique().alias(f"velocity_merchants_{window}h"),
            pl.col("State").n_unique().alias(f"velocity_states_{window}h"),
        )
        # `rolling` emits one row per input row in the sorted input's order, so
        # a positional hstack is safe (asserted in tests/test_velocity.py).
        out = out.hstack(agg.drop("User", "ts"))

    # Seconds since the most recent STRICTLY EARLIER transaction, within the
    # largest window.
    #
    # Not `ts.diff().over("User")`. That reads the previous row in sort order,
    # which includes rows sharing this timestamp -- so a tied row reported 0.0
    # while the window aggregates beside it, using closed="left", had already
    # excluded that same row. Two different notions of "previous" in one feature
    # vector. Serving queries `ts < :now` and would therefore have disagreed with
    # training on all 142,010 same-user-same-minute rows, silently.
    #
    # Bounding it by the largest window matters just as much: the online caller
    # fetches a finite slice of history, so an unbounded lookback is not
    # computable at serving time. Beyond the window the answer is the
    # no-prior-transaction sentinel, which is what serving would produce anyway.
    #
    # Caught by tests/test_skew.py on its first run.
    previous = frame.rolling(
        index_column="ts",
        period=f"{max(windows_hours)}h",
        group_by="User",
        closed="left",
    ).agg(pl.col("ts").max().alias("_previous_ts"))

    gap = frame.select("ts").hstack(previous.select("_previous_ts")).select(
        (pl.col("ts") - pl.col("_previous_ts"))
        .dt.total_seconds()
        .cast(pl.Float64)
        .fill_null(NO_PRIOR_TRANSACTION)
        .alias(SECONDS_SINCE_LAST)
    )
    out = out.hstack(gap)

    # An empty window sums to null rather than 0 in polars; the model needs a
    # number. Counts are already 0.
    amount_cols = [f"velocity_amount_{w}h" for w in windows_hours]
    return out.with_columns(
        [pl.col(c).fill_null(0.0).cast(pl.Float64) for c in amount_cols]
    )
