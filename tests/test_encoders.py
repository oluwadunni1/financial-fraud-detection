"""Encoder tests.

The encoder is part of the model: serving must apply the identical mapping, so
these check determinism, round-tripping, and above all that an unseen category
degrades to the reserved code instead of raising or shifting other columns.
"""

from __future__ import annotations

import polars as pl
import pytest

from fraud.features.encoders import (
    UNKNOWN_ORDINAL,
    Encoder,
    binary_width,
)

TRAIN = pl.DataFrame(
    {
        # 3 categories -> below the threshold -> one-hot
        "Chip": ["Swipe", "Chip", "Online", "Swipe", "Chip"],
        # 5 categories -> at/above a threshold of 4 -> binary
        "Merchant": ["m1", "m2", "m3", "m4", "m5"],
        "Amount": [10.0, 20.0, 30.0, 40.0, 50.0],
    }
)


def fitted(threshold: int = 4) -> Encoder:
    return Encoder.fit(
        TRAIN,
        categorical=["Chip", "Merchant"],
        numeric=["Amount"],
        one_hot_max_cardinality=threshold,
    )


# --- width ---------------------------------------------------------------

@pytest.mark.parametrize(
    "n,expected",
    [
        (1, 1),        # ordinals 0..1
        (2, 2),        # 0..2 needs 2 bits
        (3, 2),
        (4, 3),
        (23, 5),       # Errors
        (109, 7),      # MCC
        (223, 8),      # State
        (13_429, 14),  # City
        (27_322, 15),  # Zip
        (100_343, 17), # Merchant
    ],
)
def test_binary_width_matches_measured_cardinalities(n, expected):
    """These are the widths the architecture doc's feature budget assumes."""
    assert binary_width(n) == expected


# --- fitting -------------------------------------------------------------

def test_low_cardinality_goes_one_hot_high_goes_binary():
    enc = fitted()
    assert "Chip" in enc.one_hot and "Chip" not in enc.binary
    assert "Merchant" in enc.binary and "Merchant" not in enc.one_hot


def test_threshold_is_respected():
    """With a high enough threshold everything is one-hot."""
    enc = fitted(threshold=99)
    assert set(enc.one_hot) == {"Chip", "Merchant"}
    assert enc.binary == {}


def test_ordinals_start_at_one_leaving_zero_for_unknown():
    enc = fitted()
    assert UNKNOWN_ORDINAL == 0
    assert min(enc.binary["Merchant"].values()) == 1


# --- transform -----------------------------------------------------------

def test_feature_names_match_transform_output():
    """If these drift, the model is fed columns in an order it never saw."""
    enc = fitted()
    out = enc.transform(TRAIN)
    assert out.columns == enc.feature_names()


def test_one_hot_is_exactly_one_per_row():
    enc = fitted()
    out = enc.transform(TRAIN)
    oh = [c for c in out.columns if c.startswith("Chip_oh_")]
    assert out.select(pl.sum_horizontal(oh)).to_series().to_list() == [1] * 5


def test_binary_encodes_distinct_values_distinctly():
    enc = fitted()
    out = enc.transform(TRAIN)
    bits = [c for c in out.columns if c.startswith("Merchant_bin_")]
    codes = {tuple(row) for row in out.select(bits).rows()}
    assert len(codes) == 5  # five merchants, five codes


def test_unknown_category_encodes_to_all_zeros():
    """The cold-start path: an unseen merchant must not raise or shift columns."""
    enc = fitted()
    unseen = pl.DataFrame(
        {"Chip": ["Swipe"], "Merchant": ["m_never_seen"], "Amount": [10.0]}
    )
    out = enc.transform(unseen)
    bits = [c for c in out.columns if c.startswith("Merchant_bin_")]
    assert out.select(bits).row(0) == tuple([0] * len(bits))


def test_unknown_one_hot_category_is_all_zeros_too():
    """One-hot and binary must agree on what 'unknown' looks like."""
    enc = fitted()
    unseen = pl.DataFrame({"Chip": ["Tap"], "Merchant": ["m1"], "Amount": [10.0]})
    out = enc.transform(unseen)
    oh = [c for c in out.columns if c.startswith("Chip_oh_")]
    assert out.select(oh).row(0) == tuple([0] * len(oh))


def test_unknown_rows_do_not_disturb_known_rows():
    enc = fitted()
    mixed = pl.DataFrame(
        {
            "Chip": ["Swipe", "Tap"],
            "Merchant": ["m1", "nope"],
            "Amount": [10.0, 10.0],
        }
    )
    out = enc.transform(mixed)
    known = enc.transform(TRAIN.head(1))
    bits = [c for c in out.columns if c.startswith("Merchant_bin_")]
    assert out.select(bits).row(0) == known.select(bits).row(0)


def test_transform_is_deterministic():
    enc = fitted()
    assert enc.transform(TRAIN).equals(enc.transform(TRAIN))


def test_missing_column_raises():
    enc = fitted()
    with pytest.raises(ValueError, match="missing encoded columns"):
        enc.transform(TRAIN.drop("Merchant"))


# --- numeric scaling -----------------------------------------------------

def test_median_maps_to_zero():
    enc = fitted()
    out = enc.transform(TRAIN)
    # Amount median is 30.0
    assert out["Amount"].to_list()[2] == pytest.approx(0.0)


def test_scaling_uses_the_iqr():
    enc = fitted()
    stats = enc.numeric["Amount"]
    assert stats["median"] == pytest.approx(30.0)
    assert stats["scale"] == pytest.approx(20.0)  # q3 40 - q1 20


def test_constant_column_does_not_divide_by_zero():
    df = pl.DataFrame({"Chip": ["a"], "Merchant": ["m"], "Amount": [5.0]})
    enc = Encoder.fit(df, categorical=["Chip"], numeric=["Amount"])
    out = enc.transform(df)
    assert out["Amount"].to_list() == [0.0]


# --- persistence ---------------------------------------------------------

def test_json_round_trip(tmp_path):
    enc = fitted()
    path = tmp_path / "encoder.json"
    enc.to_json(path)
    reloaded = Encoder.from_json(path)
    assert reloaded.feature_names() == enc.feature_names()
    assert reloaded.transform(TRAIN).equals(enc.transform(TRAIN))


def test_reloaded_encoder_still_handles_unknowns(tmp_path):
    """Serving loads from JSON, so the cold-start path must survive the trip."""
    enc = fitted()
    path = tmp_path / "encoder.json"
    enc.to_json(path)
    reloaded = Encoder.from_json(path)
    unseen = pl.DataFrame({"Chip": ["x"], "Merchant": ["y"], "Amount": [1.0]})
    out = reloaded.transform(unseen)
    bits = [c for c in out.columns if c.startswith("Merchant_bin_")]
    assert out.select(bits).row(0) == tuple([0] * len(bits))


def test_fit_sees_only_the_rows_it_is_given():
    """Leakage guard: categories only present in val/test must stay unknown."""
    train = TRAIN
    val = pl.DataFrame(
        {"Chip": ["Swipe"], "Merchant": ["val_only_merchant"], "Amount": [10.0]}
    )
    enc = Encoder.fit(
        train, categorical=["Chip", "Merchant"], numeric=["Amount"],
        one_hot_max_cardinality=4,
    )
    assert "val_only_merchant" not in enc.binary["Merchant"]
    bits = [
        c for c in enc.transform(val).columns if c.startswith("Merchant_bin_")
    ]
    assert enc.transform(val).select(bits).row(0) == tuple([0] * len(bits))
