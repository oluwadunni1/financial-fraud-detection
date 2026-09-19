"""Assemble one request's neighbourhood into the graph the model expects.

This is the riskiest file in the serving path, because every way it can be
wrong is silent. The model will happily consume a graph with the wrong feature
order, a missing edge type, or a node type carrying the wrong width, and return
a confident number. There is no exception to catch -- only a metric that is
quietly worse than the offline one.

So the shape is not described here twice. `fraud.models.gnn.load_graph` defines
what a graph looks like; this module reproduces it and
`tests/test_subgraph.py` asserts the two agree on node types, feature widths and
edge types. If the offline builder changes, that test fails rather than serving
silently drifting.

Cold start is a normal path, not an error. An unseen card or merchant still gets
a node -- with the reserved all-zero identity code, exactly as the offline
encoder produces for a category it never saw during fit. 11.55% of test-period
merchants are in that position, so this path runs constantly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData

from fraud.api.store import Neighbourhood
from fraud.features.encoders import UNKNOWN_ORDINAL, Encoder, binary_width
from fraud.features.graph import NODE_ID_COLUMNS
from fraud.models.gnn import MERCHANT, TXN, USER

# The transaction node's own features come from the encoder, minus the columns
# the blueprint moves onto the id nodes.
_DROPPED_PREFIXES = tuple(f"{c}_bin_" for c in NODE_ID_COLUMNS)


def transaction_feature_names(encoder: Encoder) -> list[str]:
    """Exactly the 62 columns a transaction node carries, in order."""
    return [
        name
        for name in encoder.feature_names()
        if not name.startswith(_DROPPED_PREFIXES)
    ]


def transaction_feature_indices(encoder: Encoder) -> np.ndarray:
    """Which columns of the encoder's output a transaction node keeps."""
    return np.array(
        [
            i
            for i, name in enumerate(encoder.feature_names())
            if not name.startswith(_DROPPED_PREFIXES)
        ],
        dtype=np.int64,
    )


def _binary_code(value: str, mapping: dict[str, int]) -> list[float]:
    """One categorical value's binary code. Unseen -> all zeros (cold start)."""
    ordinal = mapping.get(str(value), UNKNOWN_ORDINAL)
    return [
        float((ordinal >> bit) & 1) for bit in range(binary_width(len(mapping)))
    ]


def build_request_graph(
    transaction: dict[str, Any],
    velocity: dict[str, float],
    neighbourhood: Neighbourhood,
    encoder: Encoder,
    card_mapping: dict[str, int],
    n_card_values: int,
    max_neighbours: int = 10,
) -> HeteroData:
    """One arriving transaction plus its past, as the model's graph.

    Node 0 of the transaction type is always the transaction being scored, so
    the caller reads its logit at index 0. Neighbours follow.
    """
    # Serving encodes a handful of rows, so the row-wise path is used here --
    # `transform` builds 86 polars expressions (one holding 93,298 merchant
    # codes) and cost 110 ms per request. `transform_rows` is bit-identical and
    # ~100x faster at this size; tests/test_subgraph.py binds them together.
    keep = transaction_feature_indices(encoder)

    required = (*encoder.one_hot, *encoder.binary, *encoder.numeric)
    rows = [{**transaction, **velocity}]
    # Kept apart, because they attach to different nodes. The card's recent
    # transactions happened at OTHER merchants, and the merchant's recent
    # transactions belong to OTHER cards -- wiring both groups to both id nodes
    # asserts edges that are false and pollutes each node's aggregate with
    # roughly 50% foreign rows.
    counts = []
    for frame in (neighbourhood.user_history, neighbourhood.merchant_history):
        if frame.is_empty():
            counts.append(0)
            continue
        # The store returns the whole velocity window because velocity counts
        # all of it; the GRAPH only wants the nearest few, matching the offline
        # second-hop fanout. Trimming here rather than in the query keeps
        # velocity correct and the subgraph small.
        frame = frame.head(max_neighbours)
        # A store that silently drops a column would otherwise surface as a
        # KeyError deep in the encoder, or worse, as a plausible wrong number.
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(
                f"neighbour rows are missing columns the encoder needs: "
                f"{missing}. The store must return every encoder input."
            )
        rows.extend(frame.to_dicts())
        counts.append(frame.height)
    n_user_rows, n_merchant_rows = counts

    encoded = encoder.transform_rows(rows)[:, keep]
    x_txn = torch.from_numpy(np.ascontiguousarray(encoded))

    # --- id nodes -----------------------------------------------------------
    # The same formula combined_card_id() encodes, evaluated directly -- a
    # one-row polars frame is not worth building for two integers.
    card_id = int(transaction["User"]) * n_card_values + int(transaction["Card"])
    merchant = str(transaction["Merchant"])

    data = HeteroData()
    data[TXN].x = x_txn
    data[USER].x = torch.tensor(
        [_binary_code(str(card_id), card_mapping)], dtype=torch.float32
    )
    data[MERCHANT].x = torch.tensor(
        [
            _binary_code(merchant, encoder.binary["Merchant"])
            + _binary_code(str(int(transaction["MCC"])), encoder.binary["MCC"])
        ],
        dtype=torch.float32,
    )

    # The model is 2-layer, so the arriving transaction's representation is
    # f(its own features, user_node^(1), merchant_node^(1)) -- and those two are
    # aggregates over the RAW features of their own transactions. Nothing deeper
    # reaches node 0. Two things therefore have to be right:
    #
    #   which transactions hang off which id node
    #       user node      <- the CARD's recent transactions
    #       merchant node  <- the MERCHANT's recent transactions
    #     Attaching every row to both makes each aggregate ~50% rows that were
    #     never there.
    #
    #   the arriving transaction must NOT be inside its own id nodes' aggregates
    #     It has to RECEIVE from them, but offline it is only one candidate
    #     among the ~280 a card has in the split, and NeighborLoader usually
    #     does not draw it. Letting ToUndirected make that edge bidirectional
    #     puts a copy of the seed into both aggregates every single time, and
    #     the self-echo measured 0.6945 -> 0.6427 AUC-PR on a June 2019 slice.
    #
    # So the four edge types are written out rather than derived: direction is
    # the thing being controlled, and ToUndirected controls it wrongly here.
    def zeros(n: int) -> torch.Tensor:
        return torch.zeros(n, dtype=torch.long)

    one = torch.zeros(1, dtype=torch.long)
    user_neighbours = torch.arange(1, 1 + n_user_rows)
    merchant_neighbours = torch.arange(
        1 + n_user_rows, 1 + n_user_rows + n_merchant_rows
    )

    # hop 1: the id nodes send to the arriving transaction.
    data[USER, "transacts", TXN].edge_index = torch.stack([one, one])
    data[MERCHANT, "rev_at", TXN].edge_index = torch.stack([one, one])
    # hop 2: the neighbours send to the id nodes -- the arriving one does not.
    data[TXN, "rev_transacts", USER].edge_index = torch.stack(
        [user_neighbours, zeros(n_user_rows)]
    )
    data[TXN, "at", MERCHANT].edge_index = torch.stack(
        [merchant_neighbours, zeros(n_merchant_rows)]
    )
    return data
