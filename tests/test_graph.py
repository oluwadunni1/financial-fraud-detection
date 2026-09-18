"""Graph construction tests.

The properties worth guarding are structural, not numeric: node ids must be
self-consistent per split, edges must point at nodes that exist, the identity
columns must have moved off the transaction and onto the id nodes, and
under-sampling must never drop a fraud row.
"""

from __future__ import annotations

import polars as pl
import pytest

from fraud.features.encoders import Encoder
from fraud.features.graph import (
    DEDUPE_SUBSET,
    LABEL,
    NODE_ID_COLUMNS,
    build_split,
    combined_card_id,
    select_training_ids,
)

N_CARD_VALUES = 3  # cards per user in the fixture


def transactions(rows: list[tuple]) -> pl.DataFrame:
    """rows: (txn_id, User, Card, Merchant, MCC, Fraud)."""
    return pl.DataFrame(
        {
            "txn_id": pl.Series([r[0] for r in rows], dtype=pl.Int64),
            "User": pl.Series([r[1] for r in rows], dtype=pl.Int64),
            "Card": pl.Series([r[2] for r in rows], dtype=pl.Int64),
            "Merchant": [r[3] for r in rows],
            "MCC": pl.Series([r[4] for r in rows], dtype=pl.Int32),
            "City": ["c"] * len(rows),
            "State": ["s"] * len(rows),
            "Zip": pl.Series([1] * len(rows), dtype=pl.Int32),
            "Errors": ["XX"] * len(rows),
            "Chip": ["Swipe"] * len(rows),
            LABEL: pl.Series([r[5] for r in rows], dtype=pl.Int8),
        }
    )


def matrix_for(txn_ids: list[int]) -> pl.DataFrame:
    """A stand-in feature matrix with the id columns the graph must drop."""
    n = len(txn_ids)
    return pl.DataFrame(
        {
            "txn_id": pl.Series(txn_ids, dtype=pl.Int64),
            "Year": pl.Series([2017] * n, dtype=pl.Int32),
            LABEL: pl.Series([0] * n, dtype=pl.Int8),
            "Amount": pl.Series([1.0] * n, dtype=pl.Float32),
            "velocity_count_1h": pl.Series([0.0] * n, dtype=pl.Float32),
            # These must NOT survive onto the transaction node.
            "Merchant_bin_0": pl.Series([1] * n, dtype=pl.Float32),
            "MCC_bin_0": pl.Series([1] * n, dtype=pl.Float32),
        }
    )


ROWS = [
    (10, 0, 0, "m1", 5411, 0),
    (11, 0, 1, "m2", 5300, 1),
    (12, 1, 0, "m1", 5411, 0),
    (13, 1, 0, "m3", 4121, 0),
]

ENCODER = Encoder(
    binary={"Merchant": {"m1": 1, "m2": 2, "m3": 3}, "MCC": {"5411": 1, "5300": 2}}
)
CARD_MAPPING = {"0": 1, "1": 2, "3": 3}  # card_id = User * 3 + Card


@pytest.fixture
def built():
    txns = transactions(ROWS)
    return build_split(
        txns,
        matrix_for([r[0] for r in ROWS]),
        ENCODER,
        CARD_MAPPING,
        N_CARD_VALUES,
    )


# --- card identity --------------------------------------------------------

def test_combined_card_id_matches_the_blueprint_formula():
    df = transactions(ROWS).with_columns(combined_card_id(N_CARD_VALUES))
    # User * n_cards + Card
    assert df["card_id"].to_list() == [0, 1, 3, 3]


def test_same_user_different_cards_are_different_nodes():
    """A 'user' node is really a card -- fraud follows the card, not the person."""
    df = transactions(ROWS).with_columns(combined_card_id(N_CARD_VALUES))
    user0 = df.filter(pl.col("User") == 0)["card_id"].to_list()
    assert len(set(user0)) == 2


# --- node ids -------------------------------------------------------------

def test_transaction_ids_are_contiguous_from_zero(built):
    ids = built["transaction_nodes"]["transaction_id"].to_list()
    assert ids == list(range(len(ROWS)))


def test_node_ids_are_dense_per_type(built):
    for key, column in (
        ("user_nodes", "user_id"),
        ("merchant_nodes", "merchant_id"),
    ):
        ids = built[key][column].to_list()
        assert ids == list(range(len(ids)))


def test_txn_id_is_kept_for_joining_back(built):
    """Node ids are per-split; txn_id is the global key the matrix joins on."""
    assert sorted(built["transaction_nodes"]["txn_id"].to_list()) == [10, 11, 12, 13]


# --- edges ----------------------------------------------------------------

def test_one_edge_per_transaction_per_direction(built):
    assert built["edges_user_transaction"].height == len(ROWS)
    assert built["edges_transaction_merchant"].height == len(ROWS)


def test_edge_endpoints_reference_existing_nodes(built):
    n_txn = built["transaction_nodes"].height
    n_user = built["user_nodes"].height
    n_merchant = built["merchant_nodes"].height

    ut = built["edges_user_transaction"]
    tm = built["edges_transaction_merchant"]
    assert ut["src"].max() < n_user and ut["dst"].max() < n_txn
    assert tm["src"].max() < n_txn and tm["dst"].max() < n_merchant
    assert ut["src"].min() >= 0 and tm["dst"].min() >= 0


def test_edges_connect_the_right_transaction_to_the_right_merchant(built):
    """m1 is shared by two transactions, so both must point at the same node."""
    nodes = built["merchant_nodes"]
    m1_id = nodes.filter(pl.col("Merchant") == "m1")["merchant_id"].item()
    txn = built["transaction_nodes"]
    tm = built["edges_transaction_merchant"]
    def merchant_of(txn_id: int) -> int:
        node = txn.filter(pl.col("txn_id") == txn_id)["transaction_id"].item()
        return tm.filter(pl.col("src") == node)["dst"].item()

    shared = [merchant_of(t) for t in (10, 12)]
    assert shared == [m1_id, m1_id]


# --- the feature split ----------------------------------------------------

def test_identity_columns_are_removed_from_transaction_features(built):
    """Merchant and MCC belong to the graph now, not the feature vector."""
    columns = built["transaction_nodes"].columns
    for prefix in NODE_ID_COLUMNS:
        assert not [c for c in columns if c.startswith(f"{prefix}_bin_")]


def test_transaction_keeps_its_own_features_and_label(built):
    columns = built["transaction_nodes"].columns
    assert "Amount" in columns and "velocity_count_1h" in columns
    assert LABEL in columns


def test_user_nodes_carry_their_card_identity_code(built):
    columns = built["user_nodes"].columns
    assert [c for c in columns if c.startswith("card_bin_")]
    assert not [c for c in columns if c.startswith("Merchant_bin_")]


def test_merchant_nodes_carry_merchant_and_mcc_codes(built):
    columns = built["merchant_nodes"].columns
    assert [c for c in columns if c.startswith("Merchant_bin_")]
    assert [c for c in columns if c.startswith("MCC_bin_")]


def test_unseen_merchant_gets_the_all_zero_cold_start_code():
    """Fitted on train only, so a merchant first seen in test is unidentifiable."""
    rows = [*ROWS, (14, 1, 0, "m_never_seen_in_train", 5411, 0)]
    built = build_split(
        transactions(rows),
        matrix_for([r[0] for r in rows]),
        ENCODER,
        CARD_MAPPING,
        N_CARD_VALUES,
    )
    nodes = built["merchant_nodes"]
    bits = [c for c in nodes.columns if c.startswith("Merchant_bin_")]
    unseen = nodes.filter(pl.col("Merchant") == "m_never_seen_in_train")
    assert set(unseen.select(bits).row(0)) == {0}


# --- under-sampling -------------------------------------------------------

def big_frame(n_fraud: int, n_clean: int) -> pl.LazyFrame:
    rows = [(i, i % 5, i % 3, f"m{i % 40}", 5000 + (i % 7), 1) for i in range(n_fraud)]
    rows += [
        (n_fraud + i, i % 5, i % 3, f"m{i % 40}", 5000 + (i % 7), 0)
        for i in range(n_clean)
    ]
    return transactions(rows).lazy()


def test_under_sampling_keeps_every_fraud():
    """Non-negotiable: dropping positives would handicap the comparison."""
    ids = select_training_ids(big_frame(50, 5_000), fraud_ratio=0.1, seed=42)
    assert set(range(50)) <= set(ids)


def test_under_sampling_hits_the_blueprint_ratio():
    """fraud_ratio=0.1 means non_fraud = fraud/0.1, i.e. 1/1.1 = 9.09% fraud."""
    ids = select_training_ids(big_frame(50, 5_000), fraud_ratio=0.1, seed=42)
    fraud_kept = len([i for i in ids if i < 50])
    assert fraud_kept == 50
    assert len(ids) == pytest.approx(50 + 500, rel=0.25)


def test_under_sampling_is_deterministic():
    a = select_training_ids(big_frame(50, 5_000), 0.1, 42)
    b = select_training_ids(big_frame(50, 5_000), 0.1, 42)
    assert a == b


def test_under_sampling_cannot_ask_for_more_clean_rows_than_exist():
    ids = select_training_ids(big_frame(50, 10), fraud_ratio=0.1, seed=42)
    assert len(ids) <= 60


def test_dedupe_subset_avoids_the_derived_card_id():
    """It must run in the lazy plan, before card_id is computed."""
    assert "card_id" not in DEDUPE_SUBSET
    assert "User" in DEDUPE_SUBSET and "Card" in DEDUPE_SUBSET
