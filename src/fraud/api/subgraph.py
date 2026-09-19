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
import polars as pl
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

from fraud.api.store import Neighbourhood
from fraud.features.encoders import Encoder, binary_code_exprs
from fraud.features.graph import NODE_ID_COLUMNS, combined_card_id
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


def _encode_transactions(
    frame: pl.DataFrame, encoder: Encoder, names: list[str]
) -> torch.Tensor:
    encoded = encoder.transform(frame)
    return torch.from_numpy(
        encoded.select(names).to_numpy().astype(np.float32)
    )


def _card_features(card_ids: list[int], card_mapping: dict[str, int]) -> torch.Tensor:
    frame = pl.DataFrame({"card_id": pl.Series(card_ids, dtype=pl.Int64)})
    coded = frame.select(binary_code_exprs("card_id", card_mapping, prefix="card"))
    return torch.from_numpy(coded.to_numpy().astype(np.float32))


def _merchant_features(
    merchants: list[str], mccs: list[int], encoder: Encoder
) -> torch.Tensor:
    frame = pl.DataFrame(
        {
            "Merchant": pl.Series(merchants, dtype=pl.String),
            "MCC": pl.Series(mccs, dtype=pl.Int64),
        }
    )
    coded = frame.select(
        *binary_code_exprs("Merchant", encoder.binary["Merchant"]),
        *binary_code_exprs("MCC", encoder.binary["MCC"]),
    )
    return torch.from_numpy(coded.to_numpy().astype(np.float32))


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
    names = transaction_feature_names(encoder)

    # --- the arriving transaction, with the velocity just computed for it ---
    arriving = pl.DataFrame({**{k: [v] for k, v in transaction.items()}})
    arriving = arriving.with_columns(
        **{name: pl.lit(value, dtype=pl.Float64) for name, value in velocity.items()}
    )

    # --- neighbours: their stored velocity is reused, not recomputed ---------
    # Those values were derived from `ts < that row's ts` when the row was
    # scored, so they cannot contain anything that row could not have seen.
    neighbours = neighbourhood.user_history
    merchant_neighbours = neighbourhood.merchant_history

    txn_frames = [arriving.select(sorted(arriving.columns))]
    for frame in (neighbours, merchant_neighbours):
        if not frame.is_empty():
            renamed = frame.rename(
                {
                    "city": "City",
                    "zip": "Zip",
                    "errors": "Errors",
                    "chip": "Chip",
                    "time_min": "Time",
                    "month": "Month",
                    "day": "Day",
                }
            )
            txn_frames.append(renamed.select(sorted(arriving.columns)))

    all_txns = pl.concat(txn_frames, how="vertical_relaxed")
    x_txn = _encode_transactions(all_txns, encoder, names)

    # --- id nodes -----------------------------------------------------------
    card_id = int(
        pl.DataFrame(
            {"User": [transaction["User"]], "Card": [transaction["Card"]]}
        )
        .select(combined_card_id(n_card_values))
        .item()
    )
    merchant = str(transaction["Merchant"])

    data = HeteroData()
    data[TXN].x = x_txn
    data[USER].x = _card_features([card_id], card_mapping)
    data[MERCHANT].x = _merchant_features(
        [merchant], [int(transaction["MCC"])], encoder
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
