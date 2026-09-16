"""Pandera schemas for the TabFormer transaction data.

Two schemas, one per pipeline boundary:

  RawTransactionSchema    -- what IBM ships in card_transaction.v1.csv, warts
                             and all ("$123.45" strings, "Yes"/"No" labels).
  CleanTransactionSchema  -- post-ingest, after type coercion and missing-value
                             markers are applied.

Column names below are the literal CSV headers. They contain spaces and a
question mark; that is deliberate, not a typo. Renaming happens in ingest.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

# Raw CSV headers, in file order.
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


class RawTransactionSchema(pa.DataFrameModel):
    """Validates the CSV exactly as read, before any coercion."""

    User: Series[int] = pa.Field(ge=0)
    Card: Series[int] = pa.Field(ge=0)
    Year: Series[int] = pa.Field(ge=1990, le=2030)
    Month: Series[int] = pa.Field(ge=1, le=12)
    Day: Series[int] = pa.Field(ge=1, le=31)

    # "HH:MM", 24-hour.
    Time: Series[str] = pa.Field(str_matches=r"^\d{1,2}:\d{2}$")

    # "$123.45" or "-$123.45" -- refunds are negative.
    Amount: Series[str] = pa.Field(str_matches=r"^-?\$\d+\.\d{2}$")

    # Hashed merchant id; large, may be negative.
    MCC: Series[int] = pa.Field(ge=0)

    # Nullable in the source file.
    Zip: Series[float] = pa.Field(nullable=True)

    class Config:
        name = "RawTransaction"
        # Extra/renamed columns are a hard error -- if IBM changes the file we
        # want the pipeline to stop, not silently train on the wrong thing.
        strict = False
        coerce = False

    @pa.check("Is Fraud?", name="fraud_label_is_yes_no")
    def fraud_is_yes_no(cls, s: Series[str]) -> Series[bool]:
        return s.isin(["Yes", "No"])


class CleanTransactionSchema(pa.DataFrameModel):
    """Validates the post-ingest frame: typed, renamed, markers applied."""

    User: Series[int] = pa.Field(ge=0)
    Card: Series[int] = pa.Field(ge=0)
    Year: Series[int] = pa.Field(ge=1990, le=2030)
    Month: Series[int] = pa.Field(ge=1, le=12)
    Day: Series[int] = pa.Field(ge=1, le=31)

    # Minutes since midnight.
    Time: Series[int] = pa.Field(ge=0, lt=1440)

    Amount: Series[float] = pa.Field(nullable=False)
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
    """
    rate = float(df["Fraud"].mean())
    if not lo <= rate <= hi:
        raise ValueError(
            f"Fraud rate {rate:.6f} outside expected band [{lo}, {hi}]. "
            "Check the Is Fraud? label mapping and the source file."
        )
    return rate
