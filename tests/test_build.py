"""Feature-build integration tests.

Builds a miniature year-partitioned dataset on disk and runs the real stage
against it. The property that matters most: the encoder must be fitted on the
training years only, so a category that appears solely in val or test stays
unknown rather than quietly getting its own code.
"""

from __future__ import annotations

import datetime as dt
import json

import polars as pl
import pytest

from fraud.features.build import PASSTHROUGH, build
from fraud.features.encoders import Encoder

PARAMS = {
    "split": {"train_end_year": 2017, "val_year": 2018},
    "features": {
        "one_hot_max_cardinality": 8,
        "categorical": ["Merchant", "State", "Chip"],
        "numeric": ["Amount", "Time"],
    },
    "velocity": {"windows_hours": [1]},
    "ingest": {"row_group_size": 1_000_000},
}


def write_dataset(root, rows):
    """rows: (txn_id, Year, Merchant, State, Chip, Amount, Time, Fraud)."""
    processed = root / "processed"
    velocity = root / "velocity"
    for year in sorted({r[1] for r in rows}):
        year_rows = [r for r in rows if r[1] == year]
        (processed / f"Year={year}").mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "txn_id": pl.Series([r[0] for r in year_rows], dtype=pl.Int64),
                "ts": pl.Series(
                    [dt.datetime(r[1], 6, 1, 12, 0) for r in year_rows],
                    dtype=pl.Datetime("us"),
                ),
                "Merchant": [r[2] for r in year_rows],
                "State": [r[3] for r in year_rows],
                "Chip": [r[4] for r in year_rows],
                "Amount": pl.Series([r[5] for r in year_rows], dtype=pl.Float64),
                "Time": pl.Series([r[6] for r in year_rows], dtype=pl.Int16),
                "Fraud": pl.Series([r[7] for r in year_rows], dtype=pl.Int8),
            }
        ).write_parquet(processed / f"Year={year}" / "part-0.parquet")

        (velocity / f"Year={year}").mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "txn_id": pl.Series([r[0] for r in year_rows], dtype=pl.Int64),
                "velocity_count_1h": pl.Series(
                    [0] * len(year_rows), dtype=pl.UInt32
                ),
                "velocity_amount_1h": pl.Series(
                    [0.0] * len(year_rows), dtype=pl.Float32
                ),
                "velocity_merchants_1h": pl.Series(
                    [0] * len(year_rows), dtype=pl.UInt32
                ),
                "velocity_states_1h": pl.Series(
                    [0] * len(year_rows), dtype=pl.UInt32
                ),
                "velocity_seconds_since_last": pl.Series(
                    [-1.0] * len(year_rows), dtype=pl.Float32
                ),
                "Year": pl.Series([year] * len(year_rows), dtype=pl.Int32),
            }
        ).write_parquet(velocity / f"Year={year}" / "part-0.parquet")
    return processed, velocity


ROWS = [
    # train years
    (1, 2016, "m_train", "CA", "Swipe", 10.0, 600, 0),
    (2, 2017, "m_train", "CA", "Chip", 20.0, 700, 1),
    (3, 2017, "m_also_train", "NY", "Swipe", 30.0, 800, 0),
    # val year -- introduces a merchant train never saw
    (4, 2018, "m_val_only", "CA", "Swipe", 40.0, 900, 0),
    # test year -- likewise
    (5, 2019, "m_test_only", "TX", "Online", 50.0, 1000, 1),
]


@pytest.fixture
def built(tmp_path):
    processed, velocity = write_dataset(tmp_path, ROWS)
    out = tmp_path / "matrix"
    encoder_path = tmp_path / "encoder.json"
    result = build(processed, velocity, out, encoder_path, PARAMS)
    return result, out, encoder_path


def test_encoder_is_fitted_on_training_years_only(built):
    """The leakage guard: val/test-only categories must not be in the mapping."""
    _, _, encoder_path = built
    payload = json.loads(encoder_path.read_text())
    mapping = payload["binary"].get("Merchant") or payload["one_hot"].get("Merchant")
    known = set(mapping if isinstance(mapping, list) else mapping.keys())
    assert "m_train" in known and "m_also_train" in known
    assert "m_val_only" not in known
    assert "m_test_only" not in known


def test_val_and_test_only_categories_encode_as_unknown(built):
    _, out, encoder_path = built
    encoder = Encoder.from_json(encoder_path)
    matrix = pl.read_parquet(out / "**/*.parquet", hive_partitioning=True)
    bits = [c for c in matrix.columns if c.startswith("Merchant_")]
    unseen_rows = matrix.filter(pl.col("txn_id").is_in([4, 5]))
    for row in unseen_rows.select(bits).rows():
        assert set(row) == {0.0}
    assert encoder  # encoder round-trips from the written artifact


def test_train_years_reported(built):
    result, _, _ = built
    assert result["train_years"] == [2016, 2017]


def test_every_row_survives(built):
    result, out, _ = built
    matrix = pl.read_parquet(out / "**/*.parquet", hive_partitioning=True)
    assert result["rows"] == len(ROWS) == matrix.height
    assert sorted(matrix["txn_id"].to_list()) == [1, 2, 3, 4, 5]


def test_passthrough_columns_are_kept(built):
    _, out, _ = built
    matrix = pl.read_parquet(out / "**/*.parquet", hive_partitioning=True)
    for column in PASSTHROUGH:
        assert column in matrix.columns
    # The label must survive unencoded, or training has nothing to learn from.
    assert sorted(matrix["Fraud"].to_list()) == [0, 0, 0, 1, 1]


def test_matrix_columns_match_encoder_feature_names(built):
    """If these drift the model is fed columns in an order it never saw."""
    _, out, encoder_path = built
    encoder = Encoder.from_json(encoder_path)
    matrix = pl.read_parquet(out / "**/*.parquet", hive_partitioning=True)
    assert [c for c in matrix.columns if c not in PASSTHROUGH] == (
        encoder.feature_names()
    )


def test_velocity_features_are_joined_in(built):
    _, out, _ = built
    matrix = pl.read_parquet(out / "**/*.parquet", hive_partitioning=True)
    assert "velocity_count_1h" in matrix.columns
    assert "velocity_seconds_since_last" in matrix.columns


def test_no_nulls_reach_the_model(built):
    _, out, _ = built
    matrix = pl.read_parquet(out / "**/*.parquet", hive_partitioning=True)
    assert matrix.null_count().sum_horizontal().item() == 0
