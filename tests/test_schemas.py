"""Schema tests.

These are the first half of the guarantee ARCHITECTURE.md section 9 asks for --
"corrupt a row -> Pandera fails the stage". Phase 1 completes it by wiring the
schema into a DVC stage; here we only prove the schema itself bites.

Constraints encoded here were measured against all 24,386,900 rows of
card_transaction.v1.csv, not inferred from a sample.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pandera.errors
import pytest

from fraud.data.schemas import (
    RAW_COLUMNS,
    CleanTransactionSchema,
    RawTransactionSchema,
    check_fraud_rate,
)

RAW_CSV = "data/TabFormer/raw/card_transaction.v1.csv"

# A blind `Exception` here would also swallow a typo in the test itself.
SCHEMA_ERRORS = (pandera.errors.SchemaError, pandera.errors.SchemaErrors)


def raw_frame(n: int = 4, **overrides) -> pd.DataFrame:
    """A minimal frame shaped exactly like the shipped CSV."""
    df = pd.DataFrame(
        {
            "User": pd.Series([0, 0, 1, 1][:n], dtype="int64"),
            "Card": pd.Series([0, 1, 0, 0][:n], dtype="int64"),
            "Year": pd.Series([2002, 2013, 2018, 2019][:n], dtype="int64"),
            "Month": pd.Series([9, 1, 12, 6][:n], dtype="int64"),
            "Day": pd.Series([1, 31, 25, 15][:n], dtype="int64"),
            "Time": ["06:21", "23:59", "0:05", "12:00"][:n],
            # Refunds put the minus INSIDE the dollar sign.
            "Amount": ["$134.09", "$-292.00", "$0.00", "$38.48"][:n],
            "Use Chip": [
                "Swipe Transaction",
                "Chip Transaction",
                "Online Transaction",
                "Swipe Transaction",
            ][:n],
            # Hashed id: int64-wide and frequently negative.
            "Merchant Name": pd.Series(
                [3527213246127876953, -727612092139916043, 1, -9222899435637403521][:n],
                dtype="int64",
            ),
            "Merchant City": ["La Verne", "Monterey Park", "ONLINE", "Merrimack"][:n],
            # Null for online transactions.
            "Merchant State": ["CA", "CA", None, "NH"][:n],
            "Zip": pd.Series([91750.0, 91754.0, None, 3054.0][:n], dtype="float64"),
            "MCC": pd.Series([5300, 5411, 4121, 5814][:n], dtype="int64"),
            # Null 98.4% of the time; trailing comma is part of the value.
            "Errors?": [None, "Bad PIN,", None, "Technical Glitch,"][:n],
            "Is Fraud?": ["No", "No", "Yes", "No"][:n],
        }
    )
    return df.assign(**overrides)


def test_valid_raw_frame_passes():
    RawTransactionSchema.validate(raw_frame(), lazy=True)


def test_column_order_matches_shipped_header():
    assert list(raw_frame().columns) == RAW_COLUMNS


@pytest.mark.parametrize(
    "name,mutate",
    [
        # strict=True: the file changing shape must stop the pipeline, not be
        # silently tolerated.
        ("extra_column", lambda df: df.assign(Surprise=1)),
        ("dropped_column", lambda df: df.drop(columns=["MCC"])),
        ("renamed_column", lambda df: df.rename(columns={"Zip": "ZipCode"})),
        # Label mapping breaking is the failure that trains happily and is
        # completely wrong.
        ("bad_fraud_label", lambda df: df.assign(**{"Is Fraud?": "Maybe"})),
        # "-$12.00" is the US convention; TabFormer does NOT use it. Accepting
        # it would mean the parser silently disagrees with the file.
        ("us_style_refund", lambda df: df.assign(Amount="-$12.00")),
        ("amount_without_currency", lambda df: df.assign(Amount="134.09")),
        ("amount_with_thousands_sep", lambda df: df.assign(Amount="$1,234.56")),
        ("month_out_of_range", lambda df: df.assign(Month=13)),
        ("day_out_of_range", lambda df: df.assign(Day=0)),
        ("year_out_of_range", lambda df: df.assign(Year=1899)),
        ("negative_user", lambda df: df.assign(User=-1)),
        ("malformed_time", lambda df: df.assign(Time="6.21")),
        ("null_city", lambda df: df.assign(**{"Merchant City": None})),
    ],
)
def test_corrupt_raw_frame_is_rejected(name, mutate):
    with pytest.raises(SCHEMA_ERRORS):
        RawTransactionSchema.validate(mutate(raw_frame()), lazy=True)


def test_nullable_columns_accept_nulls():
    """State, Zip and Errors? are null in 11.2%, 11.8% and 98.4% of rows."""
    df = raw_frame().assign(
        **{
            "Merchant State": None,
            "Zip": pd.Series([None] * 4, dtype="float64"),
            "Errors?": None,
        }
    )
    RawTransactionSchema.validate(df, lazy=True)


def test_negative_merchant_id_accepted():
    """Merchant Name spans nearly the full int64 range and is often negative."""
    df = raw_frame().assign(
        **{"Merchant Name": pd.Series([-9222899435637403521] * 4, dtype="int64")}
    )
    RawTransactionSchema.validate(df, lazy=True)


def clean_frame(n: int = 4, **overrides) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            # Minted by ingest: the row's index in the original CSV, plus the
            # timestamp assembled from Year/Month/Day/Time.
            "txn_id": pd.Series([0, 1, 2, 3][:n], dtype="int64"),
            "ts": pd.to_datetime(
                [
                    "2002-09-01 06:21",
                    "2013-01-31 23:59",
                    "2018-12-25 00:05",
                    "2019-06-15 12:00",
                ][:n]
            ),
            "User": pd.Series([0, 0, 1, 1][:n], dtype="int64"),
            "Card": pd.Series([0, 1, 0, 0][:n], dtype="int64"),
            "Year": pd.Series([2002, 2013, 2018, 2019][:n], dtype="int64"),
            "Month": pd.Series([9, 1, 12, 6][:n], dtype="int64"),
            "Day": pd.Series([1, 31, 25, 15][:n], dtype="int64"),
            "Time": pd.Series([381, 1439, 5, 720][:n], dtype="int64"),
            "Amount": pd.Series([134.09, -292.0, 0.0, 38.48][:n], dtype="float64"),
            "Merchant": ["3527213246127876953", "-727612092139916043", "1", "-9"][:n],
            "City": ["La Verne", "Monterey Park", "ONLINE", "Merrimack"][:n],
            "State": ["CA", "CA", "XX", "NH"][:n],
            "Zip": pd.Series([91750, 91754, 0, 3054][:n], dtype="int64"),
            "MCC": pd.Series([5300, 5411, 4121, 5814][:n], dtype="int64"),
            "Chip": ["Swipe", "Chip", "Online", "Swipe"][:n],
            "Errors": ["XX", "Bad PIN", "XX", "Technical Glitch"][:n],
            "Fraud": pd.Series([0, 0, 1, 0][:n], dtype="int64"),
        }
    )
    return df.assign(**overrides)


def test_valid_clean_frame_passes():
    CleanTransactionSchema.validate(clean_frame(), lazy=True)


@pytest.mark.parametrize(
    "name,mutate",
    [
        ("time_past_midnight", lambda df: df.assign(Time=1440)),
        ("fraud_not_binary", lambda df: df.assign(Fraud=2)),
        ("unfilled_state", lambda df: df.assign(State=None)),
        ("negative_zip", lambda df: df.assign(Zip=-1)),
    ],
)
def test_corrupt_clean_frame_is_rejected(name, mutate):
    with pytest.raises(SCHEMA_ERRORS):
        CleanTransactionSchema.validate(mutate(clean_frame()), lazy=True)


# --- fraud rate gate -------------------------------------------------------
# Whole-file measured rate is 0.00122 (29,757 of 24,386,900).

def test_check_fraud_rate_returns_rate_in_band():
    df = pd.DataFrame({"Fraud": [0] * 999 + [1]})
    assert check_fraud_rate(df, 0.0005, 0.005) == pytest.approx(0.001)


@pytest.mark.parametrize("frauds", [0, 500])
def test_check_fraud_rate_rejects_out_of_band(frauds):
    """0% means the label mapping broke; 50% means the wrong file loaded."""
    df = pd.DataFrame({"Fraud": [1] * frauds + [0] * (1000 - frauds)})
    with pytest.raises(ValueError, match="outside expected band"):
        check_fraud_rate(df, 0.0005, 0.005)


# --- guard against the shipped file changing under us ----------------------

@pytest.mark.skipif(
    not pathlib.Path(RAW_CSV).exists(),
    reason="raw CSV not present (it is DVC-tracked, not committed)",
)
def test_shipped_csv_header_matches_raw_columns():
    header = pd.read_csv(RAW_CSV, nrows=0).columns.tolist()
    assert header == RAW_COLUMNS


def test_clean_frame_rejects_duplicate_txn_id():
    """txn_id is a primary key downstream -- duplicates must not pass."""
    df = clean_frame()
    df.loc[1, "txn_id"] = df.loc[0, "txn_id"]
    with pytest.raises(SCHEMA_ERRORS):
        CleanTransactionSchema.validate(df, lazy=True)


def test_clean_frame_rejects_null_timestamp():
    df = clean_frame()
    df.loc[0, "ts"] = pd.NaT
    with pytest.raises(SCHEMA_ERRORS):
        CleanTransactionSchema.validate(df, lazy=True)
