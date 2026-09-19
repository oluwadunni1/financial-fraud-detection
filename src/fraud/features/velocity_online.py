"""The online velocity caller -- the second half of the project's top risk.

CLAUDE.md names train/serve skew on velocity features as the single most likely
way this system goes quietly wrong: the model is fed numbers that differ from
the ones it was trained on, nothing errors, and every metric stays plausible.

The mitigation is structural rather than careful. This module contains **no
aggregation logic at all**. It arranges the arriving transaction and the rows
that precede it into one frame and hands that to
`fraud.features.velocity.compute_velocity` -- the identical function the offline
stage calls -- then picks out the arriving row. If the offline definition of a
window ever changes, this changes with it, because there is only one definition.

Causality
---------
`compute_velocity` uses `closed="left"`: strictly-earlier rows only, and rows
sharing a timestamp are mutually invisible. The caller must therefore hand over
only rows with `ts < transaction.ts`, which is what
`store.fetch_user_history` queries. The two halves have to agree on *strictly
before*, and 142,010 rows in this dataset share a user and a minute, so a `<=`
on either side would diverge on real traffic rather than on an edge case.

`tests/test_skew.py` asserts the two paths produce identical vectors on fixed
input. Without that test this module is a promise; with it, it is a guarantee.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

from fraud.features.velocity import (
    REQUIRED_COLUMNS,
    compute_velocity,
    velocity_columns,
)


def _as_frame(transaction: dict[str, Any]) -> pl.DataFrame:
    """One arriving transaction, typed to match the history frame."""
    missing = [c for c in REQUIRED_COLUMNS if c not in transaction]
    if missing:
        raise ValueError(
            f"transaction is missing columns velocity needs: {missing}. "
            f"Required: {list(REQUIRED_COLUMNS)}"
        )
    return pl.DataFrame(
        {
            "txn_id": pl.Series([transaction["txn_id"]], dtype=pl.Int64),
            "User": pl.Series([transaction["User"]], dtype=pl.Int64),
            "ts": pl.Series([transaction["ts"]], dtype=pl.Datetime("us")),
            "Amount": pl.Series([float(transaction["Amount"])], dtype=pl.Float64),
            "Merchant": pl.Series([str(transaction["Merchant"])], dtype=pl.String),
            "State": pl.Series([str(transaction["State"])], dtype=pl.String),
        }
    )


def velocity_for_transaction(
    transaction: dict[str, Any],
    history: pl.DataFrame,
    windows_hours: Sequence[int],
) -> dict[str, float]:
    """Velocity features for one arriving transaction.

    Args:
        transaction: the arriving row; needs `REQUIRED_COLUMNS`.
        history: that user's rows with `ts < transaction["ts"]`. May be empty --
            a first-ever transaction is a normal case, not an error.
        windows_hours: from `params.yaml:velocity.windows_hours`, so the online
            path cannot drift to a different set of windows than training used.

    Returns:
        `{feature_name: value}` for `velocity_columns(windows_hours)`.
    """
    arriving = _as_frame(transaction)

    if history.is_empty():
        frame = arriving
    else:
        missing = [c for c in REQUIRED_COLUMNS if c not in history.columns]
        if missing:
            raise ValueError(f"history is missing columns: {missing}")
        # Guard the caller's contract rather than trusting it: a single row at
        # or after the arriving timestamp would make this prediction see its own
        # present, and `closed="left"` would not catch a row from the future.
        future = history.filter(pl.col("ts") >= transaction["ts"])
        if not future.is_empty():
            raise ValueError(
                f"history contains {future.height} row(s) at or after the "
                f"arriving transaction's timestamp. Serving must only ever see "
                f"the past -- query with `ts < :now`, strictly."
            )
        frame = pl.concat(
            [history.select(REQUIRED_COLUMNS), arriving], how="vertical_relaxed"
        )

    # The one call. Everything above is arranging inputs; nothing computes.
    result = compute_velocity(frame, windows_hours)
    row = result.filter(pl.col("txn_id") == transaction["txn_id"])
    if row.height != 1:
        raise RuntimeError(
            f"expected exactly one velocity row for txn_id "
            f"{transaction['txn_id']}, got {row.height}"
        )
    values = row.row(0, named=True)
    return {name: float(values[name]) for name in velocity_columns(windows_hours)}
