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
from torch_geometric.transforms import ToUndirected

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
    for frame in (neighbourhood.user_history, neighbourhood.merchant_history):
        if frame.is_empty():
            continue
        # A store that silently drops a column would otherwise surface as a
        # KeyError deep in the encoder, or worse, as a plausible wrong number.
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(
                f"neighbour rows are missing columns the encoder needs: "
                f"{missing}. The store must return every encoder input."
            )
        rows.extend(frame.to_dicts())

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

    # Every transaction node hangs off the one card and the one merchant. That
    # is a faithful miniature of the offline graph from this request's point of
    # view: two hops of the card's and merchant's recent activity.
    n_txn = x_txn.shape[0]
    data[USER, "transacts", TXN].edge_index = torch.stack(
        [torch.zeros(n_txn, dtype=torch.long), torch.arange(n_txn)]
    )
    data[TXN, "at", MERCHANT].edge_index = torch.stack(
        [torch.arange(n_txn), torch.zeros(n_txn, dtype=torch.long)]
    )

    # Without this the merchant side is unreachable from a transaction node --
    # the same trap the offline builder documents.
    return ToUndirected()(data)
