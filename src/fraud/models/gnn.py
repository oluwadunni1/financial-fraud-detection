"""Stage `train_gnn`: heterogeneous GraphSAGE over the tri-partite graph.

Three node types with different feature widths (transaction 62, user 13,
merchant 24), so this is `HeteroData` and `to_hetero`, not a homogeneous graph.

Why the graph is made undirected
--------------------------------
The blueprint writes edges in one direction only: `user -> transaction` and
`transaction -> merchant`. Message passing flows along edge direction, so with
those edges alone a transaction node can aggregate from its user but **never
from its merchant** -- the merchant signal would be silently unreachable, and
the model would look like it was using the graph while ignoring half of it.
`ToUndirected` adds the reverse edge types so a transaction sees both.

Prediction is on transaction nodes. The same trained encoder also produces the
user and merchant embeddings that Phase 4 serves from Postgres, which is why the
final layer width is `embedding_dim` rather than 1: the classification head sits
on top and the embeddings underneath are the reusable artifact.
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
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.transforms import ToUndirected

from fraud.config import load_params, repo_path
from fraud.models.metrics import auc_pr

LABEL = "Fraud"
TXN, USER, MERCHANT = "transaction", "user", "merchant"

# Columns that identify a row rather than describe it.
_NON_FEATURES = {"transaction_id", "txn_id", LABEL, "user_id", "card_id",
                 "merchant_id", "Merchant"}


def _features(frame: pl.DataFrame) -> torch.Tensor:
    columns = [c for c in frame.columns if c not in _NON_FEATURES]
    return torch.from_numpy(
        frame.select(columns).to_numpy().astype(np.float32)
    )


def load_graph(split_dir: pathlib.Path) -> HeteroData:
    """Read one split's parquet frames into a PyG heterogeneous graph."""
    txn = pl.read_parquet(split_dir / "transaction_nodes.parquet")
    users = pl.read_parquet(split_dir / "user_nodes.parquet")
    merchants = pl.read_parquet(split_dir / "merchant_nodes.parquet")
    ut = pl.read_parquet(split_dir / "edges_user_transaction.parquet")
    tm = pl.read_parquet(split_dir / "edges_transaction_merchant.parquet")

    data = HeteroData()
    data[TXN].x = _features(txn)
    data[TXN].y = torch.from_numpy(txn.get_column(LABEL).to_numpy().astype(np.float32))
    data[TXN].txn_id = torch.from_numpy(
        txn.get_column("txn_id").to_numpy().astype(np.int64)
    )
    data[USER].x = _features(users)
    data[MERCHANT].x = _features(merchants)

    data[USER, "transacts", TXN].edge_index = torch.from_numpy(
        np.stack([ut.get_column("src").to_numpy(), ut.get_column("dst").to_numpy()])
    ).long()
    data[TXN, "at", MERCHANT].edge_index = torch.from_numpy(
        np.stack([tm.get_column("src").to_numpy(), tm.get_column("dst").to_numpy()])
    ).long()

    # Without this the merchant side of the graph is unreachable from a
    # transaction node. See the module docstring.
    return ToUndirected()(data)


class SAGE(nn.Module):
    """Homogeneous GraphSAGE; `to_hetero` specialises it per edge type."""

    def __init__(
        self, hidden_channels: int, out_channels: int, num_layers: int, dropout: float
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            # (-1, -1) lets PyG infer each node type's input width lazily, which
            # is what makes one module work across three different widths.
            width = out_channels if layer == num_layers - 1 else hidden_channels
            self.convs.append(SAGEConv((-1, -1), width))
        # nn.Dropout, NOT F.dropout(training=self.training). `to_hetero` traces
        # the module with FX, which bakes a functional call's `training` flag in
        # as a constant -- leaving dropout active during eval and quietly
        # corrupting every validation score. A module carries its own flag.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for index, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if index < len(self.convs) - 1:
                x = self.dropout(x.relu())
        return x


class FraudGNN(nn.Module):
    """Hetero encoder + a binary head on transaction nodes."""

    def __init__(self, metadata, cfg: dict):
        super().__init__()
        self.encoder = to_hetero(
            SAGE(
                cfg["hidden_channels"],
                cfg["embedding_dim"],
                cfg["num_layers"],
                cfg["dropout"],
            ),
            metadata,
            aggr="sum",
        )
        self.head = nn.Linear(cfg["embedding_dim"], 1)

    def embed(self, x_dict, edge_index_dict) -> dict[str, torch.Tensor]:
        return self.encoder(x_dict, edge_index_dict)

    def forward(self, x_dict, edge_index_dict) -> torch.Tensor:
        return self.head(self.embed(x_dict, edge_index_dict)[TXN]).squeeze(-1)


def make_loader(
    data: HeteroData, cfg: dict, *, shuffle: bool, workers: int
) -> NeighborLoader:
    """A seeded loader.

    `torch.manual_seed` does NOT cover NeighborLoader's shuffling or its worker
    processes, so without an explicit generator two runs of the same config
    diverge: epoch 17 scored 0.0963 in one run and 0.1435 in another. Seeding
    here makes `dvc repro` mean something for this stage.
    """
    generator = torch.Generator()
    generator.manual_seed(cfg["random_state"])
    return NeighborLoader(
        data,
        num_neighbors=cfg["num_neighbors"],
        input_nodes=TXN,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def predict(model: FraudGNN, loader: NeighborLoader, device: torch.device):
    """Score every transaction node, in loader order."""
    model.eval()
    scores, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        seeds = batch[TXN].batch_size
        logits = model(batch.x_dict, batch.edge_index_dict)[:seeds]
        scores.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch[TXN].y[:seeds].cpu().numpy())
    return np.concatenate(scores), np.concatenate(labels)


def train(
    graph_dir: pathlib.Path, params: dict, epochs: int | None = None, workers: int = 3
) -> tuple[FraudGNN, dict]:
    cfg = params["gnn"]
    torch.manual_seed(cfg["random_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device, training on CPU", file=sys.stderr)

    train_data = load_graph(graph_dir / "train")
    val_data = load_graph(graph_dir / "val")

    train_loader = make_loader(train_data, cfg, shuffle=True, workers=workers)
    val_loader = make_loader(val_data, cfg, shuffle=False, workers=workers)

    model = FraudGNN(train_data.metadata(), cfg)
    # Lazy (-1, -1) layers need one forward pass to materialise their weights
    # before the optimiser can see any parameters.
    with torch.no_grad():
        warmup = next(iter(train_loader))
        model(warmup.x_dict, warmup.edge_index_dict)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    # Under-sampling already lifted training prevalence to ~9.1%, so the class
    # weight here is NOT the champion's 10 -- it is derived from this graph.
    # Decision 15: sweep per model, never inherit.
    y_train = train_data[TXN].y
    prevalence = float(y_train.mean())
    pos_weight = cfg.get("pos_weight") or (1 - prevalence) / prevalence
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    total_epochs = epochs if epochs is not None else cfg["epochs"]
    history, best = [], {"val_auc_pr": -1.0, "epoch": -1, "state": None}
    started = time.perf_counter()

    for epoch in range(total_epochs):
        model.train()
        epoch_start = time.perf_counter()
        total_loss = seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            seeds = batch[TXN].batch_size
            optimizer.zero_grad()
            logits = model(batch.x_dict, batch.edge_index_dict)[:seeds]
            loss = criterion(logits, batch[TXN].y[:seeds])
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * seeds
            seen += seeds

        scores, labels = predict(model, val_loader, device)
        val = auc_pr(labels, scores)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / seen,
                "val_auc_pr": val,
                "seconds": round(time.perf_counter() - epoch_start, 1),
            }
        )
        print(
            f"  epoch {epoch:>2} loss {total_loss / seen:.4f} "
            f"val AUC-PR {val:.4f} ({history[-1]['seconds']}s)",
            flush=True,
        )
        if val > best["val_auc_pr"]:
            best = {
                "val_auc_pr": val,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    info = {
        "device": str(device),
        "epochs_run": total_epochs,
        "best_epoch": best["epoch"],
        "val_auc_pr": best["val_auc_pr"],
        "pos_weight": pos_weight,
        "train_prevalence": prevalence,
        "train_nodes": int(train_data[TXN].num_nodes),
        "val_nodes": int(val_data[TXN].num_nodes),
        "feature_widths": {
            TXN: int(train_data[TXN].x.shape[1]),
            USER: int(train_data[USER].x.shape[1]),
            MERCHANT: int(train_data[MERCHANT].x.shape[1]),
        },
        "edge_types": [list(e) for e in train_data.edge_types],
        "seconds": round(time.perf_counter() - started, 1),
        "history": history,
    }
    return model, info


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default=params["paths"]["graph"])
    ap.add_argument("--output", default=params["paths"]["models"])
    ap.add_argument("--epochs", type=int, default=None, help="override params.yaml")
    ap.add_argument("--workers", type=int, default=3, help="NeighborLoader workers")
    args = ap.parse_args(argv)

    model, info = train(
        repo_path(args.graph), params, epochs=args.epochs, workers=args.workers
    )

    out = repo_path(args.output) / "gnn"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "train_info.json").write_text(json.dumps(info, indent=2))

    peak = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )
    print(
        f"\ntrained {info['epochs_run']} epochs on {info['device']} | "
        f"best epoch {info['best_epoch']} | val AUC-PR {info['val_auc_pr']:.4f} "
        f"({info['seconds']}s, peak VRAM {peak:.0f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
