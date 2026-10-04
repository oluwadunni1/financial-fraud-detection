"""Stage `causal_graph`: training examples with the neighbourhood SERVING sees.

Why this exists
---------------
GraphSAGE v3 was trained on `build_graph`'s graph and served on
`subgraph.build_request_graph`'s, and the two disagree in three ways:

=====================  ===================================  ==============================
                       training graph (`build_graph`)       serving (`/predict`)
=====================  ===================================  ==============================
neighbour pool         the UNDER-SAMPLED rows only: every   the full history, 0.12% fraud
                       fraud + ~1% of legit (9% fraud)
which neighbours       sampled across 1991-2017, future     the 10 most recent, strictly
                       ones included                        earlier
"card" history         one card's rows (card_id)            the CARDHOLDER's rows (User),
                                                            all their cards
=====================  ===================================  ==============================

So the model learned what a card's history looks like inside a fraud-enriched,
time-scrambled graph and was served the real thing -- the likeliest reason the
served number trails offline by 17% (decision 18) and borrowing ANOTHER card's
history scores higher (decision 31).

The fix is the velocity rule applied to the graph: one code path, two callers.
Every example here gets the neighbourhood `CausalHistory` -- the replay's, and
through the HTTP check the API's -- produces for it, walking the FULL history in
time order. Under-sampling now decides only which transactions are trained ON,
never what their neighbourhoods contain.

Outputs (`gnn_causal.out_dir`)
------------------------------
  train_examples.parquet / val_examples.parquet
      one row per example: txn_id, Fraud, weight, the id-node inputs (User, Card,
      Merchant, MCC) and the neighbour txn_ids in serving order
      (`user_nbrs` = the cardholder's most recent first, `merchant_nbrs` likewise)
  nodes.parquet
      the 62 transaction-node features for every txn_id referenced, from the
      feature matrix -- the same numbers `encoder.transform_rows` gives serving
      (tests/test_causal_graph.py asserts the assembled graph equals
      `build_request_graph`'s)

Neighbour selection depends only on timestamps, order and the two keys, so the
walk feeds `CausalHistory` slim rows (txn_id, ts, User, Merchant): the selection
code is the serving code, and 24M rows fit in minutes rather than hours.

    python -m fraud.features.causal_graph
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import time
from collections.abc import Iterable

import numpy as np
import polars as pl
import torch
from torch_geometric.data import HeteroData

from fraud.api.subgraph import assemble_request_graph, id_node_features, transaction_feature_names
from fraud.config import load_params, repo_path
from fraud.data.split import split_years
from fraud.features.encoders import Encoder
from fraud.features.graph import LABEL, select_training_ids

KEYS = ("txn_id", "ts", "User", "Merchant")
ID_INPUTS = ("User", "Card", "Merchant", "MCC")


def walk(
    rows: Iterable[tuple],
    history,
    targets: set[int],
    max_neighbours: int,
) -> dict[int, tuple[list[int], list[int]]]:
    """Replay `rows` (txn_id, ts, User, Merchant) in order, as the replay does:
    look up the neighbourhood, then make the row visible. Records the neighbour
    ids for the rows in `targets`, truncated exactly as `build_request_graph`
    truncates (`frame.head(max_neighbours)`)."""
    out: dict[int, tuple[list[int], list[int]]] = {}
    previous = None
    for txn_id, ts, user, merchant in rows:
        if previous is not None and ts < previous:
            raise RuntimeError(f"walk went backwards at txn {txn_id}: rows must be sorted")
        previous = ts
        txn = {"txn_id": txn_id, "ts": ts, "User": user, "Merchant": merchant}
        if txn_id in targets:
            nb = history.neighbourhood(txn)
            out[txn_id] = (_ids(nb.user_history, max_neighbours),
                           _ids(nb.merchant_history, max_neighbours))
        history.add(txn, {})
    return out


def _ids(frame: pl.DataFrame, n: int) -> list[int]:
    return [] if frame.is_empty() else frame.head(n)["txn_id"].to_list()


def _scan(params: dict) -> pl.LazyFrame:
    return pl.scan_parquet(repo_path(params["paths"]["processed"]) / "**/*.parquet",
                           hive_partitioning=True)


def _year_keys(params: dict, year: int) -> Iterable[tuple]:
    """One year's (txn_id, ts, User, Merchant) in replay order."""
    frame = (_scan(params).filter(pl.col("Year") == year)
             .select("txn_id", "ts", "User", pl.col("Merchant").cast(pl.String))
             .collect(engine="streaming").sort("ts", "txn_id"))
    return zip(*(frame[c].to_list() for c in KEYS), strict=True)


def val_targets(txns: pl.LazyFrame, fraction: float, seed: int) -> pl.DataFrame:
    """Every fraud plus a `fraction` sample of legit, weighted 1/fraction so a
    weighted AUC-PR estimates the full-year number."""
    frame = txns.select("txn_id", LABEL).collect(engine="streaming").sort("txn_id")
    frauds = frame.filter(pl.col(LABEL) == 1)
    legit = frame.filter(pl.col(LABEL) == 0)
    legit = legit.sample(n=int(legit.height * fraction), seed=seed, shuffle=False)
    return pl.concat([frauds.with_columns(weight=pl.lit(1.0)),
                      legit.with_columns(weight=pl.lit(1.0 / fraction))]).sort("txn_id")


def node_table(params: dict, ids: list[int], encoder: Encoder) -> pl.DataFrame:
    """The transaction-node features of `ids`, columns in serving order."""
    names = transaction_feature_names(encoder)
    matrix = pl.scan_parquet(repo_path(params["paths"]["feature_matrix"]) / "**/*.parquet",
                             hive_partitioning=True)
    missing = [c for c in names if c not in matrix.collect_schema().names()]
    if missing:
        raise ValueError(f"feature matrix lacks transaction-node columns: {missing}")
    table = (matrix.filter(pl.col("txn_id").is_in(ids))
             .select("txn_id", *[pl.col(c).cast(pl.Float32) for c in names])
             .collect(engine="streaming").sort("txn_id"))
    if table.height != len(ids):
        raise ValueError(f"feature matrix has {table.height:,} of {len(ids):,} referenced rows")
    return table


def _examples(targets: pl.DataFrame, nbrs: dict, params: dict) -> pl.DataFrame:
    ids = targets["txn_id"].to_list()
    lost = [i for i in ids if i not in nbrs]
    if lost:
        raise RuntimeError(f"{len(lost):,} targets never reached by the walk, e.g. {lost[:3]}")
    inputs = (_scan(params).filter(pl.col("txn_id").is_in(ids))
              .select("txn_id", "User", "Card", pl.col("Merchant").cast(pl.String), "MCC")
              .collect(engine="streaming"))
    return (targets.join(inputs, on="txn_id", how="left")
            .with_columns(
                user_nbrs=pl.Series([nbrs[i][0] for i in ids], dtype=pl.List(pl.Int64)),
                merchant_nbrs=pl.Series([nbrs[i][1] for i in ids], dtype=pl.List(pl.Int64)))
            .sort("txn_id"))


def build(params: dict) -> dict:
    from fraud.jobs.replay import CausalHistory, seed_rows

    cfg, gnn = params["gnn_causal"], params["gnn"]
    hours, cap = params["serving"]["history_hours"], params["serving"]["neighbours"]
    out_dir = repo_path(cfg["out_dir"])
    encoder = Encoder.from_json(repo_path(params["paths"]["encoder"]))
    processed = repo_path(params["paths"]["processed"])
    years = sorted(int(p.name.split("=")[1]) for p in processed.glob("Year=*"))
    wanted = split_years(params, years)
    started = time.perf_counter()
    summary: dict = {"splits": {}}

    # --- train: the v3 target set, neighbourhoods from the FULL history -------
    train_ids = select_training_ids(
        _scan(params).filter(pl.col("Year").is_in(wanted["train"])),
        gnn["fraud_ratio"], gnn["random_state"], gnn["dedupe_non_fraud"])
    train_targets = (_scan(params).filter(pl.col("txn_id").is_in(train_ids))
                     .select("txn_id", LABEL).collect(engine="streaming")
                     .with_columns(weight=pl.lit(1.0)).sort("txn_id"))
    history = CausalHistory(hours, cap)
    nbrs: dict = {}
    wanted_ids = set(train_ids)
    for year in wanted["train"]:
        t = time.perf_counter()
        nbrs |= walk(_year_keys(params, year), history, wanted_ids, cap)
        print(f"  train {year}: {len(nbrs):>7,} examples so far "
              f"({time.perf_counter() - t:.0f}s)", flush=True)
    splits = {"train": _examples(train_targets, nbrs, params)}

    # --- val: seeded exactly as the 2019 replay seeds its year ----------------
    (val_year,) = wanted["val"]
    start = dt.datetime(val_year, 1, 1)
    since = start - dt.timedelta(hours=hours)
    val_t = val_targets(_scan(params).filter(pl.col("Year") == val_year),
                        cfg["val_legit_fraction"], gnn["random_state"])
    history = CausalHistory(hours, cap)
    seed = seed_rows(params, start, hours, graph_since=since)
    for txn_id, ts, user, merchant in zip(
            *(seed.with_columns(pl.col("Merchant").cast(pl.String))[c].to_list()
              for c in KEYS), strict=True):
        history.add({"txn_id": txn_id, "ts": ts, "User": user, "Merchant": merchant}, {})
    nbrs = walk(_year_keys(params, val_year), history, set(val_t["txn_id"].to_list()), cap)
    splits["val"] = _examples(val_t, nbrs, params)
    print(f"  val   {val_year}: {splits['val'].height:,} examples "
          f"(seeded {seed.height:,} rows from {since})", flush=True)

    # --- one node table for everything referenced -----------------------------
    referenced: set[int] = set()
    for frame in splits.values():
        referenced |= set(frame["txn_id"].to_list())
        for col in ("user_nbrs", "merchant_nbrs"):
            referenced |= set(frame[col].explode(empty_as_null=True).drop_nulls().to_list())
    nodes = node_table(params, sorted(referenced), encoder)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for name, frame in splits.items():
        frame.write_parquet(out_dir / f"{name}_examples.parquet", compression="zstd")
        n_user = frame["user_nbrs"].list.len()
        n_merchant = frame["merchant_nbrs"].list.len()
        summary["splits"][name] = {
            "examples": frame.height,
            "frauds": int(frame[LABEL].sum()),
            "weighted_rows": float(frame["weight"].sum()),
            "mean_user_neighbours": float(n_user.mean()),
            "mean_merchant_neighbours": float(n_merchant.mean()),
            "cold_user": int((n_user == 0).sum()),
            "cold_merchant": int((n_merchant == 0).sum()),
            # Fraud among the neighbours: ~9% in v3's training graph; here it
            # must be near the real rate, or the fix did not take.
            "neighbour_fraud_rate": _neighbour_fraud_rate(params, frame),
        }
    nodes.write_parquet(out_dir / "nodes.parquet", compression="zstd")
    summary.update(nodes=nodes.height, node_features=nodes.width - 1,
                   seconds=round(time.perf_counter() - started, 1))
    return summary


def _neighbour_fraud_rate(params: dict, frame: pl.DataFrame) -> float:
    ids = (pl.concat([frame["user_nbrs"].explode(empty_as_null=True),
                      frame["merchant_nbrs"].explode(empty_as_null=True)])
           .drop_nulls().unique())
    labels = (_scan(params).filter(pl.col("txn_id").is_in(ids.to_list()))
              .select(LABEL).collect(engine="streaming")[LABEL])
    return float(labels.mean()) if labels.len() else 0.0


# --- assembly: example -> the graph serving would build -------------------------

class NodeLookup:
    """txn_id -> row of the node table, as one float32 tensor."""

    def __init__(self, nodes: pl.DataFrame):
        ids = nodes["txn_id"].to_numpy()
        self._index = dict(zip(ids.tolist(), range(len(ids)), strict=True))
        self.x = torch.from_numpy(np.ascontiguousarray(
            nodes.drop("txn_id").to_numpy().astype(np.float32)))

    def rows(self, ids: list[int]) -> torch.Tensor:
        return self.x[[self._index[i] for i in ids]]


def assemble(example: dict, lookup: NodeLookup, encoder: Encoder,
             card_mapping: dict[str, int], n_card_values: int) -> HeteroData:
    """One example as the request graph -- `subgraph`'s own two functions."""
    user_ids, merchant_ids = example["user_nbrs"], example["merchant_nbrs"]
    x_txn = lookup.rows([example["txn_id"], *user_ids, *merchant_ids])
    user_x, merchant_x = id_node_features(example, encoder, card_mapping, n_card_values)
    return assemble_request_graph(x_txn, user_x, merchant_x, len(user_ids), len(merchant_ids))


def load(params: dict, split: str) -> pl.DataFrame:
    return pl.read_parquet(repo_path(params["gnn_causal"]["out_dir"]) / f"{split}_examples.parquet")


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    summary = build(params)
    out = repo_path(params["gnn_causal"]["summary"])
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
