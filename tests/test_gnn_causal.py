"""Vectorised batches must be the graphs `causal_graph.assemble` builds --
which `test_causal_graph.py` holds equal to serving. Skips until the
`causal_graph` stage has run."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
import torch
from torch_geometric.data import Batch

from fraud.config import load_params, repo_path

PARAMS = load_params()


@pytest.fixture(scope="module")
def parts():
    from fraud.features.causal_graph import load
    from fraud.features.encoders import Encoder

    out_dir = repo_path(PARAMS["gnn_causal"]["out_dir"])
    if not (out_dir / "val_examples.parquet").exists():
        pytest.skip("run `dvc repro causal_graph` first")
    encoder = Encoder.from_json(repo_path(PARAMS["paths"]["encoder"]))
    cards = json.loads((repo_path(PARAMS["paths"]["graph"]) / "card_mapping.json").read_text())
    val = load(PARAMS, "val")
    # Frauds, legit, a cold merchant and a short cardholder history, if present.
    sample = pl.concat([val.filter(pl.col("Fraud") == 1).head(5), val.head(5),
                        val.filter(pl.col("merchant_nbrs").list.len() == 0).head(2),
                        val.filter(pl.col("user_nbrs").list.len() < 10).head(2)])
    ids = set(sample["txn_id"].to_list())
    for col in ("user_nbrs", "merchant_nbrs"):
        ids |= set(sample[col].explode(empty_as_null=True).drop_nulls().to_list())
    nodes = (pl.scan_parquet(out_dir / "nodes.parquet")
             .filter(pl.col("txn_id").is_in(sorted(ids))).collect().sort("txn_id"))
    return sample, nodes, encoder, cards


def _equal(a, b):
    for t in b.node_types:
        torch.testing.assert_close(a[t].x, b[t].x, rtol=0, atol=0)
    for e in b.edge_types:
        assert torch.equal(a[e].edge_index, b[e].edge_index), e


def test_batches_equal_batch_from_data_list_of_assemble(parts):
    from fraud.features.causal_graph import NodeLookup, assemble
    from fraud.models.gnn_causal import CausalBatches

    sample, nodes, encoder, cards = parts
    data = CausalBatches(sample, nodes, encoder, cards["mapping"], cards["n_card_values"])
    lookup = NodeLookup(nodes)
    ex = np.array([3, 0, 7, len(sample) - 1, 5])         # shuffled, like training
    got, centre = data.batch(ex)
    rows = sample.to_dicts()
    expected = Batch.from_data_list([
        assemble(rows[i], lookup, encoder, cards["mapping"], cards["n_card_values"])
        for i in ex])
    _equal(got, expected)
    assert torch.equal(centre, expected["transaction"].ptr[:-1])
    assert torch.equal(data.y[ex], torch.tensor([rows[i]["Fraud"] for i in ex],
                                                dtype=torch.float32))


def test_no_cardholder_history_empties_only_the_user_side(parts):
    from fraud.models.gnn_causal import CausalBatches

    sample, nodes, encoder, cards = parts
    data = CausalBatches(sample, nodes, encoder, cards["mapping"], cards["n_card_values"],
                         cardholder_history=False)
    graph, _ = data.batch(np.arange(len(sample)))
    assert graph["transaction", "rev_transacts", "user"].edge_index.shape[1] == 0
    assert graph["transaction", "at", "merchant"].edge_index.shape[1] == \
        sample["merchant_nbrs"].list.len().sum()


def test_masked_short_velocity_is_zero_everywhere_else_untouched(parts):
    from fraud.api.subgraph import transaction_feature_names
    from fraud.models.gnn_causal import CausalBatches, short_velocity_columns

    sample, nodes, encoder, cards = parts
    names = transaction_feature_names(encoder)
    masked = [names.index(c) for c in short_velocity_columns(PARAMS)]
    assert len(masked) == 5
    plain = CausalBatches(sample, nodes, encoder, cards["mapping"], cards["n_card_values"])
    hidden = CausalBatches(sample, nodes, encoder, cards["mapping"], cards["n_card_values"],
                           masked=masked)
    ex = np.arange(len(sample))
    a, b = plain.batch(ex)[0]["transaction"].x, hidden.batch(ex)[0]["transaction"].x
    assert torch.all(b[:, masked] == 0)
    keep = [i for i in range(len(names)) if i not in masked]
    assert torch.equal(a[:, keep], b[:, keep])
    assert torch.any(a[:, masked] != 0)          # the mask had something to remove
    assert torch.all(plain.x[:, masked].abs().sum() > 0)   # and did not mutate the source
