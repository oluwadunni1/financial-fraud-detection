"""Stage `gnn_embeddings`: extract the 64-d user and merchant embeddings.

These are the artifact Phase 4 actually serves. The trained GraphSAGE encoder is
run over each split's graph and the hidden representations for the *id* nodes
are kept; the transaction-node head is not used here.

Node ids are translated back to global keys on the way out -- `card_id` for
users, the hashed merchant id for merchants -- because the per-split node
indices mean nothing outside their own graph, while `node_embeddings` in
Postgres is keyed globally and read by the serving path.

One honesty note on the test embeddings
---------------------------------------
A user's embedding for the test split aggregates over that split's transactions,
including the one being scored. At serving time the arriving transaction is not
yet in the store, so this is mildly optimistic. It matches the architecture's
design -- embeddings are precomputed on a schedule and served frozen (§3.1), and
§6.2's staleness experiment exists precisely to quantify what that costs -- but
it is not the strictly-causal treatment velocity features get.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import polars as pl
import torch
from torch_geometric.loader import NeighborLoader

from fraud.config import load_params, repo_path
from fraud.models.gnn import (
    MERCHANT,
    TXN,
    USER,
    FraudGNN,
    load_graph,
)

SPLITS = ("train", "val", "test")


@torch.no_grad()
def embed_node_type(
    model: FraudGNN,
    data,
    node_type: str,
    cfg: dict,
    device: torch.device,
    workers: int = 3,
) -> np.ndarray:
    """Embeddings for every node of one type, in node-id order."""
    model.eval()
    loader = NeighborLoader(
        data,
        num_neighbors=cfg["num_neighbors"],
        input_nodes=node_type,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    chunks = []
    for batch in loader:
        batch = batch.to(device)
        seeds = batch[node_type].batch_size
        out = model.embed(batch.x_dict, batch.edge_index_dict)[node_type][:seeds]
        chunks.append(out.cpu().numpy())
    return np.concatenate(chunks)


def extract(
    graph_dir: pathlib.Path,
    model_dir: pathlib.Path,
    out_dir: pathlib.Path,
    params: dict,
    workers: int = 3,
) -> dict:
    cfg = params["gnn"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild the architecture from the train graph's metadata, then load weights.
    train_data = load_graph(graph_dir / "train")
    model = FraudGNN(train_data.metadata(), cfg)
    with torch.no_grad():  # materialise the lazy layers before loading state
        warm = NeighborLoader(
            train_data,
            num_neighbors=cfg["num_neighbors"],
            input_nodes=TXN,
            batch_size=16,
            shuffle=False,
        )
        batch = next(iter(warm))
        model(batch.x_dict, batch.edge_index_dict)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model = model.to(device)

    info = json.loads((model_dir / "train_info.json").read_text())
    model_version = f"gnn-v1-epoch{info['best_epoch']}"

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summary: dict = {"model_version": model_version, "splits": {}}

    for split in SPLITS:
        data = train_data if split == "train" else load_graph(graph_dir / split)
        users = pl.read_parquet(graph_dir / split / "user_nodes.parquet")
        merchants = pl.read_parquet(graph_dir / split / "merchant_nodes.parquet")

        user_emb = embed_node_type(model, data, USER, cfg, device, workers)
        merchant_emb = embed_node_type(model, data, MERCHANT, cfg, device, workers)

        dim = user_emb.shape[1]
        # Translate per-split node indices back to global keys.
        user_frame = users.select("user_id", "card_id").with_columns(
            pl.Series("embedding", [row.tolist() for row in user_emb])
        )
        merchant_frame = merchants.select("merchant_id", "Merchant").with_columns(
            pl.Series("embedding", [row.tolist() for row in merchant_emb])
        )
        user_frame.write_parquet(out_dir / f"{split}_user_embeddings.parquet")
        merchant_frame.write_parquet(
            out_dir / f"{split}_merchant_embeddings.parquet"
        )

        summary["splits"][split] = {
            "users": int(user_emb.shape[0]),
            "merchants": int(merchant_emb.shape[0]),
            "dim": int(dim),
            "user_norm_mean": float(np.linalg.norm(user_emb, axis=1).mean()),
            "merchant_norm_mean": float(np.linalg.norm(merchant_emb, axis=1).mean()),
            "any_nan": bool(np.isnan(user_emb).any() or np.isnan(merchant_emb).any()),
        }
        print(
            f"  {split:<5} {user_emb.shape[0]:>6,} users + "
            f"{merchant_emb.shape[0]:>7,} merchants @ {dim}-d",
            flush=True,
        )

    summary["seconds"] = round(time.perf_counter() - started, 1)
    return summary


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default=params["paths"]["graph"])
    ap.add_argument("--model", default=params["paths"]["models"])
    ap.add_argument("--output", default=params["paths"]["embeddings"])
    ap.add_argument("--summary", default=params["paths"]["embeddings_summary"])
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)

    summary = extract(
        repo_path(args.graph),
        repo_path(args.model) / "gnn",
        repo_path(args.output),
        params,
        workers=args.workers,
    )

    out = repo_path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"embeddings extracted in {summary['seconds']}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
