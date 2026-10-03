"""Simulated chargebacks: ground truth that arrives late, the way it really does.

TabFormer hands us every label at once. Production never has that: a fraud is
known when the cardholder disputes it and the chargeback lands, typically 30-90+
days after the transaction, and a legitimate transaction is never positively
confirmed -- it is simply assumed good once the dispute window closes with
nothing filed. So for the most recent months there is almost nothing to compute
a metric on, which is exactly the gap label-free estimation has to cover.

Delays are drawn per transaction, deterministically: rows are put in `txn_id`
order before drawing, so the same transaction always gets the same delay no
matter how the caller sorted or filtered the frame.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl


def label_arrival(frame: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """`frame` (txn_id, ts, label) plus `labeled_at`: when the label is known.

    Fraud: ts + lognormal delay (median `fraud_delay_median_days`), clipped to
    [`fraud_delay_min_days`, `fraud_delay_max_days`].
    Legitimate: ts + `confirm_after_days`, when the dispute window closes.
    """
    ordered = frame.select("txn_id").with_row_index("_pos").sort("txn_id")
    rng = np.random.default_rng(cfg["seed"])
    draws = rng.standard_normal(ordered.height)
    delay_fraud = np.clip(
        np.exp(np.log(cfg["fraud_delay_median_days"]) + cfg["fraud_delay_sigma"] * draws),
        cfg["fraud_delay_min_days"],
        cfg["fraud_delay_max_days"],
    )
    drawn = ordered.with_columns(pl.Series("_fraud_delay_days", delay_fraud))
    out = frame.with_row_index("_pos").join(
        drawn.select("_pos", "_fraud_delay_days"), on="_pos"
    ).sort("_pos")

    delay_days = (
        pl.when(pl.col("label") == 1)
        .then(pl.col("_fraud_delay_days"))
        .otherwise(float(cfg["confirm_after_days"]))
    )
    return out.with_columns(
        (pl.col("ts") + pl.duration(seconds=(delay_days * 86_400).cast(pl.Int64)))
        .alias("labeled_at")
    ).drop("_pos", "_fraud_delay_days")


def known_at(frame: pl.DataFrame, as_of: dt.datetime) -> pl.Series:
    """Which rows have a label by `as_of`. Needs `labeled_at` from label_arrival."""
    return frame.get_column("labeled_at") <= as_of


def label_rows(frame: pl.DataFrame, as_of: dt.datetime | None = None) -> pl.DataFrame:
    """Rows in the shape of the `labels` table (txn_id, is_fraud, labeled_at).

    With `as_of`, only the labels that exist by then -- what a nightly job
    would have inserted so far.
    """
    rows = frame
    if as_of is not None:
        rows = rows.filter(known_at(rows, as_of))
    return rows.select(
        "txn_id",
        (pl.col("label") == 1).alias("is_fraud"),
        "labeled_at",
    )
