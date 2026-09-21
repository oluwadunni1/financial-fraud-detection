"""The tokenizer must agree with the checkpoint, or the embeddings are noise.

Every failure mode here is silent. A token one bucket off does not raise --
the model looks up a vector trained to mean something else and returns a
confident number. There is no exception to catch, only a metric that is quietly
worse. So the contract is pinned rather than reviewed:

* the vocabulary is exactly the 6,251 the checkpoint's `config.json` declares
* the two upstream quirks are reproduced, not tidied away
* a fixed set of rows produces a fixed set of ids

The one thing these tests CANNOT settle is whether cuDF's `hash_values()` is
standard MurmurHash3 x86_32 over the UTF-8 bytes. `test_merchant_hash_matches_reference_vectors`
proves our implementation is real murmur3; confirming cuDF agrees needs a
RAPIDS environment and is a gate on the GPU run, not on this suite.
"""

from __future__ import annotations

import datetime as dt

import mmh3
import polars as pl
import pytest

from fraud.features.fm_tokenizer import (
    CAT_LABELS,
    CHIP_LABELS,
    EXPECTED_VOCAB_SIZE,
    MCC_LABELS,
    MERCHANT_HASH_SIZE,
    SPECIAL_TOKENS,
    STATE_LABELS,
    TOKEN_COLUMNS,
    TOKENS_PER_TRANSACTION,
    VOCAB,
    build_vocab,
    local_indices,
    merchant_bucket,
    tokenize,
)


def rows(**overrides) -> pl.DataFrame:
    base = {
        "txn_id": 1, "User": 0, "Card": 0,
        "ts": dt.datetime(2016, 1, 3, 10, 48),
        "Amount": 66.48, "Merchant": "-3345936507911876459",
        "MCC": 5411, "Zip": 91750, "Chip": "Chip Transaction", "State": "CA",
    }
    return pl.DataFrame([base | overrides])


# --- the vocabulary must match the checkpoint ----------------------------

def test_vocab_size_matches_the_checkpoint():
    """config.json declares vocab_size 6251. Anything else is unrecoverable."""
    assert VOCAB.size == EXPECTED_VOCAB_SIZE


def test_component_sizes_sum_to_the_vocab():
    assert len(SPECIAL_TOKENS) + sum(VOCAB.sizes.values()) == EXPECTED_VOCAB_SIZE


def test_label_tables_have_the_sizes_the_fitted_pipeline_reported():
    # Measured by running upstream's own fit(): CAT 14, MCC 110, CHIP 4, STATE 58.
    assert len(CAT_LABELS) == 14
    assert len(MCC_LABELS) == 110
    assert len(CHIP_LABELS) == 4
    assert len(STATE_LABELS) == 58


def test_offsets_are_contiguous_after_the_special_tokens():
    cursor = len(SPECIAL_TOKENS)
    for name, size in ((n, VOCAB.sizes[n]) for n in TOKEN_COLUMNS):
        assert VOCAB.offsets[name] == cursor
        cursor += size


# --- the two upstream quirks, reproduced on purpose ----------------------

def test_month_and_card_collide_exactly_as_upstream_does():
    """MONTH_12 and CARD_0 share id 2179.

    FixedVocabTokenizer keys its table by VALUE, and MONTH is the only step
    with min_val=1, so it runs one past its own block. Verified against
    upstream's fitted pipeline. "Fixing" this would desynchronise every token
    after MONTH from the pretrained embedding matrix.
    """
    december = VOCAB.offsets["month"] + 12
    card_zero = VOCAB.offsets["card"] + 0
    assert december == card_zero == 2179


def test_december_tokenises_to_the_colliding_id():
    tokens = tokenize(rows(ts=dt.datetime(2016, 12, 25, 10, 48)))
    assert tokens["month"][0] == 2179


def test_merchant_hash_is_unsigned():
    """A signed modulo picks a different bucket -- 1001 vs 297 on this row."""
    merchant = "-3345936507911876459"
    assert merchant_bucket(merchant) == mmh3.hash(merchant, 0, signed=False) % MERCHANT_HASH_SIZE
    assert merchant_bucket(merchant) != mmh3.hash(merchant, 0, signed=True) % MERCHANT_HASH_SIZE


def test_merchant_hash_matches_reference_vectors():
    """Proves the algorithm is real murmur3; cuDF agreement is a GPU gate."""
    for text, seed, expected in [
        ("", 0, 0), ("", 1, 0x514E28B7),
        ("aaaa", 0x9747B28C, 0x5A97808A),
        ("Hello, world!", 0x9747B28C, 0x24884CBA),
    ]:
        assert mmh3.hash(text, seed, signed=False) == expected


# --- shape and range ------------------------------------------------------

def test_every_id_is_inside_the_embedding_matrix():
    """An id >= 6251 would index past the checkpoint's embedding table."""
    frame = pl.concat([
        rows(txn_id=1),
        rows(txn_id=2, MCC=9402, Zip=0, State="ZZ", Chip="Online Transaction",
             Amount=12345.0, User=2999, Card=9, ts=dt.datetime(2019, 12, 31, 23, 59)),
        rows(txn_id=3, MCC=-1, Zip=3054, State=None, Amount=-292.0,
             ts=dt.datetime(2016, 6, 1, 0, 0)),
    ])
    tokens = tokenize(frame)
    for name in TOKEN_COLUMNS:
        column = tokens[name]
        assert column.min() >= len(SPECIAL_TOKENS)
        assert column.max() < EXPECTED_VOCAB_SIZE


def test_twelve_tokens_per_transaction_in_a_fixed_order():
    assert TOKENS_PER_TRANSACTION == 12
    assert tokenize(rows()).columns == ["txn_id", *TOKEN_COLUMNS]


# --- the derivations that are easy to get one step wrong ------------------

def test_day_of_week_uses_the_cudf_convention():
    """polars weekday() is 1-7 Mon=1; cuDF dayofweek is 0-6 Mon=0."""
    monday = local_indices(rows(ts=dt.datetime(2016, 1, 4, 12, 0)))
    sunday = local_indices(rows(ts=dt.datetime(2016, 1, 10, 12, 0)))
    assert monday["dow"][0] == 0
    assert sunday["dow"][0] == 6


@pytest.mark.parametrize(
    "amount,expected",
    [(0.0, 0), (9.99, 0), (10.0, 1), (49.99, 1), (50.0, 2), (100.0, 3),
     (500.0, 4), (1000.0, 5), (5000.0, 6), (99999.0, 6), (-292.0, 0)],
)
def test_amount_buckets_count_thresholds_crossed(amount, expected):
    """Refunds are negative and must land in bucket 0, not underflow."""
    assert local_indices(rows(Amount=amount))["amt_val"][0] == expected


def test_zip_takes_the_first_three_characters_not_the_first_three_digits():
    """A 4-digit zip yields '305', matching upstream's string slice."""
    assert local_indices(rows(Zip=3054))["zip3"][0] == 305
    assert local_indices(rows(Zip=91750))["zip3"][0] == 917
    assert local_indices(rows(Zip=0))["zip3"][0] == 0


def test_unknown_categories_fall_back_to_their_defaults():
    unknown = local_indices(rows(State="ZZ", MCC=1234, Chip="Tap"))
    assert unknown["state_clean"][0] == STATE_LABELS.index("XX")
    assert unknown["mcc_str"][0] == MCC_LABELS.index("-1")
    assert unknown["chip_upper"][0] == CHIP_LABELS.index("UNK")
    # 1234 is inside the AGRICULTURAL range, so CAT is a real label.
    assert unknown["mcc_int"][0] == CAT_LABELS.index("AGRICULTURAL")


def test_category_comes_from_the_mcc_range_table():
    assert local_indices(rows(MCC=5411))["mcc_int"][0] == CAT_LABELS.index("RETAIL")
    assert local_indices(rows(MCC=9402))["mcc_int"][0] == CAT_LABELS.index("GOVERNMENT")
    assert local_indices(rows(MCC=-1))["mcc_int"][0] == CAT_LABELS.index("GENERAL")


def test_build_vocab_is_deterministic():
    assert build_vocab() == VOCAB
