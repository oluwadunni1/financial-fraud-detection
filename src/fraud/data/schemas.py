"""Pandera schemas for the TabFormer transaction data.

Two schemas, one per pipeline boundary:

  RawTransactionSchema    -- what IBM ships in card_transaction.v1.csv, warts
                             and all ("$123.45" strings, "Yes"/"No" labels).
  CleanTransactionSchema  -- post-ingest, after type coercion and missing-value
                             markers are applied.

Column names below are the literal CSV headers. They contain spaces and a
question mark; that is deliberate, not a typo. Renaming happens in ingest.

Measured against the real file (24,386,900 rows, 2026-09-16) -- see
docs/ARCHITECTURE.md section 4.4. Every constraint here was checked against all
24M rows, not inferred from a sample.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

# Raw CSV headers, in file order. Verified to match the shipped file exactly.
RAW_COLUMNS = [
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "Merchant Name",
    "Merchant City",
    "Merchant State",
    "Zip",
    "MCC",
    "Errors?",
    "Is Fraud?",
]

# Raw header -> clean single-word name. Matches the blueprint's mapping so the
# ported feature logic stays comparable.
COLUMN_RENAME = {
    "Merchant Name": "Merchant",
    "Merchant State": "State",
    "Merchant City": "City",
    "Errors?": "Errors",
    "Use Chip": "Chip",
    "Is Fraud?": "Fraud",
}

UNKNOWN_STRING_MARKER = "XX"
UNKNOWN_ZIP_CODE = 0

# The sign sits INSIDE the dollar sign: refunds are "$-292.00", not "-$292.00".
# 1,244,689 rows (5.1%) are negative and every one uses this form. Getting this
# wrong fails validation on every refund in the dataset.
AMOUNT_PATTERN = r"^\$-?\d+\.\d{2}$"


class RawTransactionSchema(pa.DataFrameModel):
    """Validates the CSV exactly as read, before any coercion.

    All fifteen columns are declared so that `strict = True` can do its job:
    if IBM adds, drops or renames a column, the pipeline stops here instead of
    silently training on a different file.
    """

    User: Series[int] = pa.Field(ge=0)
    Card: Series[int] = pa.Field(ge=0)
    Year: Series[int] = pa.Field(ge=1990, le=2030)
    Month: Series[int] = pa.Field(ge=1, le=12)
    Day: Series[int] = pa.Field(ge=1, le=31)

    # "HH:MM", 24-hour.
    Time: Series[str] = pa.Field(str_matches=r"^\d{1,2}:\d{2}$")

    # "$123.45" or "$-123.45" -- refunds put the minus after the dollar sign.
    Amount: Series[str] = pa.Field(str_matches=AMOUNT_PATTERN)

    # Three values: Swipe / Chip / Online Transaction.
    chip: Series[str] = pa.Field(alias="Use Chip", nullable=False)

    # Hashed merchant id; large and frequently negative, so no lower bound.
    # Spans nearly the full int64 range (-9.22e18 .. 9.22e18). 100,343 distinct.
    merchant_name: Series[int] = pa.Field(alias="Merchant Name")

    city: Series[str] = pa.Field(alias="Merchant City", nullable=False)

    # Null for online transactions -- 2,720,821 rows (11.2%).
    state: Series[str] = pa.Field(alias="Merchant State", nullable=True)

    # Nullable in the source file (2,878,135 rows, 11.8%). Read as float
    # because of the nulls, which drops leading zeros on the 1,555,116 zips
    # below 10000 -- harmless, since Zip is used as a categorical code.
    Zip: Series[float] = pa.Field(nullable=True, ge=0)

    # Merchant *category* code: 4-digit, 109 distinct, range 1711..9402.
    MCC: Series[int] = pa.Field(ge=0)

    # Null 98.4% of the time (23,998,469 rows); 23 distinct non-null values,
    # each a trailing-comma-separated list like "Bad PIN,Technical Glitch,".
    errors: Series[str] = pa.Field(alias="Errors?", nullable=True)

    fraud: Series[str] = pa.Field(alias="Is Fraud?", isin=["Yes", "No"])

    class Config:
        name = "RawTransaction"
        # Extra/renamed columns are a hard error -- if IBM changes the file we
        # want the pipeline to stop, not silently train on the wrong thing.
        strict = True
        coerce = False


class CleanTransactionSchema(pa.DataFrameModel):
    """Validates the post-ingest frame: typed, renamed, markers applied."""

    # Synthetic primary key: the row's index in the original CSV. TabFormer has
    # no natural key -- (User, Card, Year, Month, Day, Time) collides on 142,010
    # rows because timestamps are minute-resolution, and 66 rows are exact
    # duplicates. `predictions`, `labels` and `transaction_events` all key on
    # this (see sql/001_init.sql), so it must be stable across re-ingests.
    txn_id: Series[int] = pa.Field(ge=0, unique=True)

    # Full event timestamp assembled from Year/Month/Day/Time. Every one of the
    # 24,386,900 rows yields a valid datetime (verified), spanning
    # 1991-01-02 07:10 to 2020-02-28 23:58. Velocity windows depend on it.
    ts: Series[pd.Timestamp] = pa.Field(nullable=False)

    User: Series[int] = pa.Field(ge=0)
    Card: Series[int] = pa.Field(ge=0)
    Year: Series[int] = pa.Field(ge=1990, le=2030)
    Month: Series[int] = pa.Field(ge=1, le=12)
    Day: Series[int] = pa.Field(ge=1, le=31)

    # Minutes since midnight.
    Time: Series[int] = pa.Field(ge=0, lt=1440)

    Amount: Series[float] = pa.Field(nullable=False)
    # Kept as str after rename: the id exceeds float64's exact-integer range,
    # so it must never round-trip through a float.
    Merchant: Series[str] = pa.Field(nullable=False)
    City: Series[str] = pa.Field(nullable=False)
    State: Series[str] = pa.Field(nullable=False)
    Zip: Series[int] = pa.Field(ge=0, nullable=False)
    MCC: Series[int] = pa.Field(ge=0)
    Chip: Series[str] = pa.Field(nullable=False)
    Errors: Series[str] = pa.Field(nullable=False)
    Fraud: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        name = "CleanTransaction"
        strict = False
        coerce = True


def check_fraud_rate(df: pd.DataFrame, lo: float, hi: float) -> float:
    """Dataset-level sanity gate.

    A fraud rate outside [lo, hi] almost always means the Yes/No mapping broke
    or the wrong file was loaded -- both of which produce a model that trains
    happily and is completely wrong. Fail the stage instead.

    Measured whole-file rate is 0.00122 (29,757 of 24,386,900). Note this is
    per-file: individual years range from 0.0 (2020) to 0.0030 (2008), so a
    per-partition caller needs a wider band than the dataset-level one.
    """
    rate = float(df["Fraud"].mean())
    if not lo <= rate <= hi:
        raise ValueError(
            f"Fraud rate {rate:.6f} outside expected band [{lo}, {hi}]. "
            "Check the Is Fraud? label mapping and the source file."
        )
    return rate
