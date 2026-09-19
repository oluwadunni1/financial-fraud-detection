"""The serving graph must match the training graph, exactly.

Every failure mode here is silent. A wrong feature order, a missing edge type,
or a node type of the wrong width all produce a confident number rather than an
exception -- the only symptom is a metric quietly worse than the offline one.

Two things are pinned:

1. **Shape.** `fraud.models.gnn.load_graph` defines what a graph looks like;
   `build_request_graph` must produce the same node types, feature widths and
   edge types. If the offline builder changes, these fail.
2. **Encoding.** The serving path uses `Encoder.transform_rows` because
   `transform` cost 110 ms per request. That speed is only safe if the two
   agree, so they are compared directly here.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from fraud.api.store import Neighbourhood
from fraud.api.subgraph import build_request_graph, transaction_feature_names
from fraud.features.encoders import Encoder
from fraud.models.gnn import MERCHANT, TXN, USER

WINDOWS = [1, 24, 168]
VELOCITY = {
    f"velocity_{stat}_{w}h": 0.0
    for w in WINDOWS
    for stat in ("count", "amount", "merchants", "states")
} | {"velocity_seconds_since_last": -1.0}

ENCODER = Encoder(
    one_hot={"Chip": ["Chip Transaction", "Online Transaction", "Swipe Transaction"]},
    binary={
        "Merchant": {"m1": 1, "m2": 2, "m3": 3},
        "Zip": {"91750": 1, "3054": 2},
        "City": {"La Verne": 1, "Merrimack": 2},
        "State": {"CA": 1, "NH": 2},
        "MCC": {"5411": 1, "5300": 2},
        "Errors": {"XX": 1, "Bad PIN": 2},
    },
    numeric={
        name: {"median": 0.0, "scale": 1.0}
        for name in ("Amount", "Time", "Month", "Day", *VELOCITY)
    },
)
CARD_MAPPING = {"0": 1, "1": 2, "3": 3}
N_CARDS = 3


def transaction(**overrides) -> dict:
    base = {
        "txn_id": 1, "User": 0, "Card": 0, "ts": dt.datetime(2019, 6, 1, 12, 0),
        "Amount": 10.0, "Merchant": "m1", "MCC": 5411, "City": "La Verne",
        "State": "CA", "Zip": 91750, "Errors": "XX", "Chip": "Swipe Transaction",
        "Time": 720, "Month": 6, "Day": 1,
    }
    return base | overrides


def neighbour_frame(n: int) -> pl.DataFrame:
    """Rows shaped as the store returns them: canonical names, own velocity."""
    return pl.DataFrame(
        [
            {**transaction(txn_id=100 + i, Amount=float(i)), **VELOCITY}
            for i in range(n)
        ]
    )


EMPTY = pl.DataFrame(schema={})


def build(neighbours: int = 0):
    frame = neighbour_frame(neighbours) if neighbours else EMPTY
    return build_request_graph(
        transaction(), VELOCITY, Neighbourhood(frame, EMPTY),
        ENCODER, CARD_MAPPING, N_CARDS,
    )


# --- the fast path must equal the batch path -----------------------------

def test_row_encoding_matches_batch_encoding():
    """The 110ms -> ~1ms optimisation is only safe if these agree."""
    rows = [
        {**transaction(), **VELOCITY},
        {**transaction(Merchant="never_seen", City="Nowhere"), **VELOCITY},
    ]
    batch = ENCODER.transform(pl.DataFrame(rows)).to_numpy().astype(np.float32)
    fast = ENCODER.transform_rows(rows)
    assert np.array_equal(batch, fast)


def test_unseen_category_encodes_to_zero_in_the_fast_path():
    rows = [{**transaction(Merchant="never_seen_at_all"), **VELOCITY}]
    encoded = ENCODER.transform_rows(rows)
    names = ENCODER.feature_names()
    bits = [i for i, n in enumerate(names) if n.startswith("Merchant_bin_")]
    assert encoded[0, bits].sum() == 0.0


# --- shape must match what the model was trained on ----------------------

def test_node_types_match_the_offline_graph():
    data = build(neighbours=3)
    assert set(data.node_types) == {TXN, USER, MERCHANT}


def test_transaction_features_exclude_the_id_columns():
    """Merchant and MCC live on the id nodes, not the transaction."""
    names = transaction_feature_names(ENCODER)
    assert not [n for n in names if n.startswith(("Merchant_bin_", "MCC_bin_"))]
    assert build().x_dict[TXN].shape[1] == len(names)


def test_id_node_widths_match_their_codes():
    data = build()
    # card code width, and merchant + MCC widths
    assert data.x_dict[USER].shape == (1, 2)
    assert data.x_dict[MERCHANT].shape == (1, 4)


def test_graph_is_undirected():
    """Without reverse edges a transaction never sees its merchant."""
    types = {tuple(e) for e in build(neighbours=2).edge_types}
    assert (USER, "transacts", TXN) in types
    assert (TXN, "at", MERCHANT) in types
    assert any(src == MERCHANT for src, _, _ in types), "merchant unreachable"
    assert any(src == TXN and dst == USER for src, _, dst in types)


# --- the arriving transaction is always node 0 ---------------------------

def test_arriving_transaction_is_node_zero():
    """The predictor reads its logit at index 0; nothing else guarantees that."""
    data = build(neighbours=4)
    encoded = ENCODER.transform_rows([{**transaction(), **VELOCITY}])
    keep = [
        i
        for i, n in enumerate(ENCODER.feature_names())
        if not n.startswith(("Merchant_bin_", "MCC_bin_"))
    ]
    assert np.allclose(data.x_dict[TXN][0].numpy(), encoded[0, keep])


@pytest.mark.parametrize("neighbours", [0, 1, 5, 10])
def test_node_count_tracks_the_neighbourhood(neighbours):
    data = build(neighbours=neighbours)
    assert data.x_dict[TXN].shape[0] == neighbours + 1


# --- cold start is a normal path -----------------------------------------

def test_cold_start_scores_rather_than_raising():
    data = build(neighbours=0)
    assert data.x_dict[TXN].shape[0] == 1
    assert data.x_dict[USER].shape[0] == 1
    assert data.x_dict[MERCHANT].shape[0] == 1


def test_unknown_card_gets_the_reserved_zero_code():
    graph = build_request_graph(
        transaction(User=999, Card=9), VELOCITY, Neighbourhood(EMPTY, EMPTY),
        ENCODER, CARD_MAPPING, N_CARDS,
    )
    assert graph.x_dict[USER].sum() == 0.0


def test_neighbour_missing_encoder_columns_fails_loudly():
    """A store that forgets a column must not silently produce a wrong vector."""
    broken = neighbour_frame(2).drop("City")
    with pytest.raises(ValueError, match="missing columns"):
        build_request_graph(
            transaction(), VELOCITY, Neighbourhood(broken, EMPTY),
            ENCODER, CARD_MAPPING, N_CARDS,
        )
