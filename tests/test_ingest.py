"""Ingest transform tests.

`transform_batch` is pure, so it is tested directly on hand-built frames rather
than through the DVC stage. The constants encoded here were measured against
all 24,386,900 rows -- see docs/ARCHITECTURE.md section 2.1.
"""

from __future__ import annotations

import polars as pl
import pytest

from fraud.data.ingest import OUTPUT_COLUMNS, OUTPUT_SCHEMA, transform_batch
from fraud.data.schemas import UNKNOWN_STRING_MARKER, UNKNOWN_ZIP_CODE

# The largest-magnitude merchant ids in the file. These are the precision trap:
# both exceed float64's exact-integer range (2^53).
MERCHANT_MIN = -9222899435637403521
MERCHANT_MAX = 9223291803303717674


def raw_batch(**overrides) -> pl.DataFrame:
    """A raw batch shaped exactly as PyArrow hands it over, pre-rename."""
    df = pl.DataFrame(
        {
            "User": pl.Series([0, 0, 1], dtype=pl.Int64),
            "Card": pl.Series([0, 1, 0], dtype=pl.Int64),
            "Year": pl.Series([2002, 2013, 2019], dtype=pl.Int64),
            "Month": pl.Series([9, 1, 6], dtype=pl.Int64),
            "Day": pl.Series([1, 31, 15], dtype=pl.Int64),
            "Time": ["06:21", "23:59", "0:05"],
            # Refund in the middle: the minus is INSIDE the dollar sign.
            "Amount": ["$134.09", "$-292.00", "$0.00"],
            "Use Chip": ["Swipe Transaction", "Chip Transaction", "Online Transaction"],
            "Merchant Name": pl.Series(
                [MERCHANT_MAX, MERCHANT_MIN, 1], dtype=pl.Int64
            ),
            "Merchant City": ["La Verne", "Monterey Park", "ONLINE"],
            "Merchant State": ["CA", None, None],
            "Zip": pl.Series([91750.0, None, None], dtype=pl.Float64),
            "MCC": pl.Series([5300, 5411, 4121], dtype=pl.Int64),
            "Errors?": [None, "Bad PIN,", "Bad Card Number,Bad Expiration,"],
            "Is Fraud?": ["No", "No", "Yes"],
        }
    )
    return df.with_columns(**overrides) if overrides else df


def test_output_columns_and_order():
    out = transform_batch(raw_batch(), 0)
    assert out.columns == OUTPUT_COLUMNS
    # Must survive the cast the writer performs.
    out.to_arrow().cast(OUTPUT_SCHEMA)


def test_amount_parses_refunds():
    """`$-292.00`, not `-$292.00` -- 5.1% of the real file are refunds."""
    out = transform_batch(raw_batch(), 0)
    assert out["Amount"].to_list() == [134.09, -292.0, 0.0]


def test_time_becomes_minutes_since_midnight():
    out = transform_batch(raw_batch(), 0)
    assert out["Time"].to_list() == [6 * 60 + 21, 23 * 60 + 59, 5]


def test_timestamp_assembled_from_parts():
    out = transform_batch(raw_batch(), 0)
    ts = out["ts"].to_list()
    assert (ts[0].year, ts[0].month, ts[0].day, ts[0].hour, ts[0].minute) == (
        2002, 9, 1, 6, 21,
    )
    assert ts[1].hour == 23 and ts[1].minute == 59


def test_fraud_label_becomes_binary():
    out = transform_batch(raw_batch(), 0)
    assert out["Fraud"].to_list() == [0, 0, 1]


def test_merchant_id_survives_as_exact_string():
    """The int64 precision trap: via float64 these would both be wrong."""
    out = transform_batch(raw_batch(), 0)
    assert out["Merchant"].to_list()[:2] == [str(MERCHANT_MAX), str(MERCHANT_MIN)]
    assert int(out["Merchant"][0]) == MERCHANT_MAX
    assert int(out["Merchant"][1]) == MERCHANT_MIN


def test_missing_value_markers_applied():
    out = transform_batch(raw_batch(), 0)
    assert out["State"].to_list() == ["CA", UNKNOWN_STRING_MARKER, UNKNOWN_STRING_MARKER]
    assert out["Zip"].to_list() == [91750, UNKNOWN_ZIP_CODE, UNKNOWN_ZIP_CODE]
    assert out["Errors"][0] == UNKNOWN_STRING_MARKER
    assert out.null_count().sum_horizontal().item() == 0


def test_errors_trailing_comma_stripped():
    """23 distinct values depend on the trailing comma being removed."""
    out = transform_batch(raw_batch(), 0)
    assert out["Errors"].to_list()[1:] == [
        "Bad PIN",
        "Bad Card Number,Bad Expiration",
    ]


def test_txn_id_continues_from_offset():
    """Batches are numbered by a running counter, so ids never restart."""
    first = transform_batch(raw_batch(), 0)
    second = transform_batch(raw_batch(), first.height)
    assert first["txn_id"].to_list() == [0, 1, 2]
    assert second["txn_id"].to_list() == [3, 4, 5]


def test_txn_id_is_deterministic():
    """Same input, same ids -- it is a Postgres primary key downstream."""
    assert (
        transform_batch(raw_batch(), 100)["txn_id"].to_list()
        == transform_batch(raw_batch(), 100)["txn_id"].to_list()
    )


@pytest.mark.parametrize("year", [2002, 2013, 2019])
def test_year_preserved_for_partitioning(year):
    out = transform_batch(raw_batch(), 0)
    assert year in out["Year"].to_list()
