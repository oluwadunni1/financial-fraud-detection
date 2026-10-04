"""The causal training graphs must be the graphs serving builds.

Three layers, cheapest first:
  1. `walk` selects exactly what `CausalHistory.neighbourhood` selects, truncated
     as `build_request_graph` truncates -- synthetic rows, no data needed.
  2. `assemble` (node table + ids) equals `build_request_graph` (rows + encoder)
     for the same inputs -- needs only the encoder.
  3. On the BUILT examples: sampled val examples, re-derived through the explain /
     gate path (`neighbourhood_for` + live velocity + `build_request_graph`),
     must equal what training will read. Skips until `dvc repro causal_graph`.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest
import torch

from fraud.config import load_params, repo_path
from fraud.features.causal_graph import NodeLookup, assemble, walk
from fraud.jobs.replay import CausalHistory

D = dt.datetime
PARAMS = load_params()


def _rows():
    base = D(2017, 3, 1)
    rows = []
    for i in range(30):   # user 1 and 2 alternate; merchants m0..m2
        rows.append((i, base + dt.timedelta(minutes=10 * i), 1 + i % 2, f"m{i % 3}"))
    rows.append((30, rows[-1][1], 1, "m0"))   # same minute as 29: mutually invisible
    return rows


def test_walk_matches_the_replay_history_lookup():
    rows = _rows()
    got = walk(rows, CausalHistory(168, 10), {5, 29, 30}, max_neighbours=10)

    reference = CausalHistory(168, 10)
    expected = {}
    for txn_id, ts, user, merchant in rows:
        txn = {"txn_id": txn_id, "ts": ts, "User": user, "Merchant": merchant}
        if txn_id in {5, 29, 30}:
            nb = reference.neighbourhood(txn)
            expected[txn_id] = (nb.user_history.head(10)["txn_id"].to_list(),
                                nb.merchant_history.head(10)["txn_id"].to_list())
        reference.add(txn, {})
    assert got == expected


def test_walk_is_causal_recent_first_and_capped():
    got = walk(_rows(), CausalHistory(168, 3), {29, 30}, max_neighbours=3)
    user_29, merchant_29 = got[29]
    assert user_29 == [27, 25, 23]                 # user 2's, most recent first, capped
    assert merchant_29 == [26, 23, 20]             # m2's
    assert 29 not in got[30][0] and 29 not in got[30][1]   # same minute: invisible
    assert all(i < 29 for i in user_29 + merchant_29)


def test_walk_refuses_unsorted_rows():
    rows = _rows()
    with pytest.raises(RuntimeError, match="backwards"):
        walk([rows[3], rows[1]], CausalHistory(168, 10), set(), 10)


# --- 2. assembly == the serving builder ----------------------------------------------

def _serving_parts():
    from fraud.features.encoders import Encoder

    enc_path = repo_path(PARAMS["paths"]["encoder"])
    maps = repo_path(PARAMS["paths"]["graph"]) / "card_mapping.json"
    if not (enc_path.exists() and maps.exists()):
        pytest.skip("needs the encoder and card mapping (dvc pull)")
    cards = json.loads(maps.read_text())
    return Encoder.from_json(enc_path), cards["mapping"], cards["n_card_values"]


def _txn(i, user, merchant, ts):
    return {"txn_id": i, "User": user, "Card": 0, "ts": ts, "Amount": 12.5 + i,
            "Merchant": merchant, "MCC": 5411, "City": "Rome", "State": "Italy",
            "Zip": 0, "Errors": "", "Chip": "Chip Transaction", "Time": 600 + i,
            "Month": 3, "Day": 1}


def _graphs_equal(a, b):
    assert a.node_types == b.node_types and a.edge_types == b.edge_types
    for t in a.node_types:
        torch.testing.assert_close(a[t].x, b[t].x, rtol=0, atol=1e-6)
    for e in a.edge_types:
        assert torch.equal(a[e].edge_index, b[e].edge_index), e


def test_assemble_equals_build_request_graph():
    from fraud.api.store import Neighbourhood
    from fraud.api.subgraph import build_request_graph, transaction_feature_indices
    from fraud.features.velocity import velocity_columns

    encoder, mapping, n_card = _serving_parts()
    names = velocity_columns(PARAMS["velocity"]["windows_hours"])
    vel = {n: float(k) for k, n in enumerate(names)}
    base = D(2018, 3, 1)
    target = _txn(100, 7, "123", base)
    users = [_txn(i, 7, "999", base - dt.timedelta(hours=i)) for i in (1, 2, 3)]
    merchants = [_txn(i, 8, "123", base - dt.timedelta(hours=i)) for i in (4, 5)]
    nb = Neighbourhood(user_history=pl.DataFrame([{**r, **vel} for r in users]),
                       merchant_history=pl.DataFrame([{**r, **vel} for r in merchants]))
    serving = build_request_graph(target, vel, nb, encoder, mapping, n_card)

    rows = [target, *users, *merchants]
    keep = transaction_feature_indices(encoder)
    x = encoder.transform_rows([{**r, **vel} for r in rows])[:, keep]
    nodes = pl.DataFrame({"txn_id": [r["txn_id"] for r in rows]}).hstack(
        pl.DataFrame(x.astype(np.float32), schema=[f"f{i}" for i in range(x.shape[1])]))
    example = {**target, "user_nbrs": [1, 2, 3], "merchant_nbrs": [4, 5]}
    _graphs_equal(assemble(example, NodeLookup(nodes), encoder, mapping, n_card), serving)


# --- 3. the built examples == what serving builds ------------------------------------

@pytest.mark.parametrize("seed", [0])
def test_built_val_examples_equal_the_serving_graph(seed):
    from fraud.api.predictor import transaction_from_row
    from fraud.api.subgraph import build_request_graph
    from fraud.explain.graph import neighbourhood_for
    from fraud.features.causal_graph import load
    from fraud.features.velocity_online import velocity_for_transaction

    out_dir = repo_path(PARAMS["gnn_causal"]["out_dir"])
    if not (out_dir / "val_examples.parquet").exists():
        pytest.skip("run `dvc repro causal_graph` first")
    encoder, mapping, n_card = _serving_parts()
    val = load(PARAMS, "val")
    # Frauds and legit both, including a cold merchant if there is one.
    sample = pl.concat([val.filter(pl.col("Fraud") == 1).sample(4, seed=seed),
                        val.filter(pl.col("Fraud") == 0).sample(4, seed=seed),
                        val.filter(pl.col("merchant_nbrs").list.len() == 0).head(1)])
    lookup = NodeLookup(pl.read_parquet(out_dir / "nodes.parquet"))
    rows = (pl.scan_parquet(repo_path(PARAMS["paths"]["processed"]) / "**/*.parquet",
                            hive_partitioning=True)
            .filter(pl.col("txn_id").is_in(sample["txn_id"].to_list())).collect())
    by_id = {r["txn_id"]: r for r in rows.iter_rows(named=True)}
    hours = PARAMS["serving"]["history_hours"]
    since = D(PARAMS["split"]["val_year"], 1, 1) - dt.timedelta(hours=hours)
    for example in sample.iter_rows(named=True):
        txn = transaction_from_row(by_id[example["txn_id"]])
        nb = neighbourhood_for(PARAMS, txn, since)
        velocity = velocity_for_transaction(txn, nb.user_history,
                                            PARAMS["velocity"]["windows_hours"])
        serving = build_request_graph(txn, velocity, nb, encoder, mapping, n_card,
                                      max_neighbours=PARAMS["serving"]["neighbours"])
        _graphs_equal(assemble(example, lookup, encoder, mapping, n_card), serving)
