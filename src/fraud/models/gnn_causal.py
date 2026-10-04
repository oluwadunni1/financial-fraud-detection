"""Stage `train_gnn_causal`: GraphSAGE trained on the neighbourhood it is served.

Same `FraudGNN`, same optimiser, batch size and epochs as v3 (`gnn` params) --
what changes is the input. Every example is the request graph serving builds
(`fraud.features.causal_graph`, decision 33), so the validation number estimates
the served one directly instead of an offline graph's.

Arms (`gnn_causal.arms`), all selected on weighted 2018 AUC-PR:

  full                    the served graph as is
  no_cardholder_history   the user node aggregates nothing (decision 31: borrowing
                          ANOTHER card's history scored higher -- does the
                          cardholder path help once it is trained honestly?)
  no_short_velocity       the 1 h window and seconds-since-last zeroed on every
                          transaction node (decision 27: those drifted)

`pos_weight` is swept on `full` and the best value reused (decision 15). v3 is
scored on the same val examples, and its weighted estimate is checked against its
full 2018 causal replay -- the evidence that the 5% sample measures the year.

Batches are assembled vectorised (`CausalBatches`): every example has the same
shape -- one transaction, <= 10 cardholder rows, <= 10 merchant rows, two id
nodes -- so a batch is index arithmetic, not 4,096 HeteroData objects.
`tests/test_gnn_causal.py` asserts it equals `Batch.from_data_list` of
`causal_graph.assemble`, which is in turn tested equal to serving.

    python -m fraud.models.gnn_causal [--arms full] [--epochs 2]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import polars as pl
import torch
from torch import nn
from torch_geometric.data import HeteroData

from fraud.api.subgraph import id_node_features, transaction_feature_names
from fraud.config import REPO_ROOT, load_params, repo_path
from fraud.features.causal_graph import load
from fraud.features.encoders import Encoder
from fraud.features.graph import LABEL
from fraud.features.velocity import SECONDS_SINCE_LAST
from fraud.models.gnn import MERCHANT, TXN, USER, FraudGNN
from fraud.models.metrics import auc_pr


def short_velocity_columns(params: dict) -> list[str]:
    """The shortest window's four aggregates plus seconds-since-last."""
    w = min(params["velocity"]["windows_hours"])
    return [f"velocity_count_{w}h", f"velocity_amount_{w}h", f"velocity_merchants_{w}h",
            f"velocity_states_{w}h", SECONDS_SINCE_LAST]


class NodeTable:
    """The node features as one float32 tensor, rows sorted by txn_id."""

    def __init__(self, nodes: pl.DataFrame):
        self.ids = nodes["txn_id"].to_numpy()
        if not np.all(self.ids[1:] > self.ids[:-1]):
            raise ValueError("node table must be sorted by txn_id")
        self.x = torch.from_numpy(np.ascontiguousarray(
            nodes.drop("txn_id").to_numpy().astype(np.float32)))


class CausalBatches:
    """One split's examples, ready to slice into batched request graphs."""

    def __init__(self, examples: pl.DataFrame, nodes: pl.DataFrame | NodeTable,
                 encoder: Encoder, card_mapping: dict[str, int], n_card_values: int,
                 cardholder_history: bool = True, masked: list[int] | None = None):
        # One NodeTable can back every split and arm: the full-2018 val table is
        # ~2 GB, and a copy per CausalBatches would not fit five of them.
        table = nodes if isinstance(nodes, NodeTable) else NodeTable(nodes)
        node_ids, self.x = table.ids, table.x
        self.masked = masked or []

        users = examples["user_nbrs"].to_list() if cardholder_history else None
        merchants = examples["merchant_nbrs"].to_list()
        self.n_user = np.array([len(u) for u in users] if users else
                               [0] * examples.height, dtype=np.int64)
        self.n_merchant = np.array([len(m) for m in merchants], dtype=np.int64)
        flat_ids = np.fromiter(
            (i for k, t in enumerate(examples["txn_id"].to_list())
             for i in (t, *(users[k] if users else ()), *merchants[k])),
            dtype=np.int64)
        rows = np.searchsorted(node_ids, flat_ids)
        if not np.array_equal(node_ids[np.minimum(rows, len(node_ids) - 1)], flat_ids):
            raise ValueError("an example references a txn_id missing from the node table")
        self.rows = torch.from_numpy(rows)
        sizes = 1 + self.n_user + self.n_merchant
        self.ptr = np.concatenate([[0], np.cumsum(sizes)])

        ids = examples.select(*("User", "Card", "Merchant", "MCC")).to_dicts()
        pairs = [id_node_features(r, encoder, card_mapping, n_card_values) for r in ids]
        self.user_x = torch.cat([u for u, _ in pairs])
        self.merchant_x = torch.cat([m for _, m in pairs])
        self.y = torch.tensor(examples[LABEL].to_numpy(), dtype=torch.float32)
        self.weight = examples["weight"].to_numpy()
        self.txn_id = examples["txn_id"].to_numpy()

    def __len__(self) -> int:
        return len(self.y)

    def batch(self, ex: np.ndarray) -> tuple[HeteroData, torch.Tensor]:
        """Examples `ex` as one batched graph, plus each example's centre node."""
        n_u, n_m = self.n_user[ex], self.n_merchant[ex]
        sizes = 1 + n_u + n_m
        base = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        pos = (np.repeat(self.ptr[ex], sizes)
               + np.arange(sizes.sum()) - np.repeat(base, sizes))
        x_txn = self.x[self.rows[torch.from_numpy(pos)]]
        if self.masked:
            x_txn[:, self.masked] = 0.0

        def run(starts, counts):   # node ids starts[k] .. starts[k]+counts[k]-1, flat
            return (np.repeat(starts, counts)
                    + np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts))

        k = np.arange(len(ex))
        data = HeteroData()
        data[TXN].x = x_txn
        data[USER].x = self.user_x[ex]
        data[MERCHANT].x = self.merchant_x[ex]
        t = torch.from_numpy
        data[USER, "transacts", TXN].edge_index = t(np.stack([k, base]))
        data[MERCHANT, "rev_at", TXN].edge_index = t(np.stack([k, base]))
        data[TXN, "rev_transacts", USER].edge_index = t(np.stack(
            [run(base + 1, n_u), np.repeat(k, n_u)]))
        data[TXN, "at", MERCHANT].edge_index = t(np.stack(
            [run(base + 1 + n_u, n_m), np.repeat(k, n_m)]))
        return data, t(base)


@torch.no_grad()
def predict(model: FraudGNN, data: CausalBatches, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(data), batch_size):
        graph, centre = data.batch(np.arange(start, min(start + batch_size, len(data))))
        out.append(torch.sigmoid(model(graph.x_dict, graph.edge_index_dict)[centre]))
    return torch.cat(out).numpy()


def new_model(params: dict, data: CausalBatches) -> FraudGNN:
    graph, _ = data.batch(np.arange(min(16, len(data))))
    model = FraudGNN(graph.metadata(), params["gnn"])
    with torch.no_grad():   # materialise the lazy (-1, -1) layers
        model(graph.x_dict, graph.edge_index_dict)
    return model


def train_arm(params: dict, train: CausalBatches, val: CausalBatches, pos_weight: float,
              epochs: int) -> tuple[FraudGNN, dict]:
    cfg = params["gnn"]
    torch.manual_seed(cfg["random_state"])
    order_rng = np.random.default_rng(cfg["random_state"])
    model = new_model(params, train)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    history, best = [], {"val_auc_pr": -1.0, "epoch": -1, "state": None}
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        t0, total, seen = time.perf_counter(), 0.0, 0
        order = order_rng.permutation(len(train))
        for start in range(0, len(order), cfg["batch_size"]):
            ex = order[start:start + cfg["batch_size"]]
            graph, centre = train.batch(ex)
            optimizer.zero_grad()
            loss = criterion(model(graph.x_dict, graph.edge_index_dict)[centre], train.y[ex])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ex)
            seen += len(ex)
        scores = predict(model, val, cfg["batch_size"])
        val_auc = auc_pr(val.y.numpy(), scores, sample_weight=val.weight)
        history.append({"epoch": epoch, "train_loss": total / seen, "val_auc_pr": val_auc,
                        "seconds": round(time.perf_counter() - t0, 1)})
        print(f"    epoch {epoch:>2} loss {total / seen:.4f} val AUC-PR {val_auc:.4f} "
              f"({history[-1]['seconds']}s)", flush=True)
        if val_auc > best["val_auc_pr"]:
            best = {"val_auc_pr": val_auc, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    return model, {"best_epoch": best["epoch"], "val_auc_pr": best["val_auc_pr"],
                   "pos_weight": pos_weight, "epochs_run": epochs,
                   "seconds": round(time.perf_counter() - started, 1), "history": history}


def _log_mlflow(name: str, info: dict) -> None:
    """Every arm is an MLflow run, so losing arms stay on record too."""
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        print("    (MLFLOW_TRACKING_URI unset: not logged)")
        return
    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("gnn-causal-retrain")
    with mlflow.start_run(run_name=name):
        mlflow.log_params({k: info[k] for k in ("arm", "pos_weight", "cardholder_history",
                                                "mask_short_velocity", "best_epoch")})
        mlflow.log_metric("val_2018_weighted_auc_pr", info["val_auc_pr"])
        for h in info["history"]:
            mlflow.log_metric("val_auc_pr_by_epoch", h["val_auc_pr"], step=h["epoch"])
        mlflow.log_dict(info, "train_info.json")


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    cfg = params["gnn_causal"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=list(cfg["arms"]))
    ap.add_argument("--epochs", type=int, default=params["gnn"]["epochs"])
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args(argv)
    torch.set_num_threads(cfg["threads"])

    encoder = Encoder.from_json(repo_path(params["paths"]["encoder"]))
    cards = json.loads((repo_path(params["paths"]["graph"]) / "card_mapping.json").read_text())
    nodes = NodeTable(pl.read_parquet(repo_path(cfg["out_dir"]) / "nodes.parquet"))
    names = transaction_feature_names(encoder)
    masked = [names.index(c) for c in short_velocity_columns(params)]
    train_ex, val_ex = load(params, "train"), load(params, "val")

    def split(examples, arm):
        return CausalBatches(examples, nodes, encoder, cards["mapping"], cards["n_card_values"],
                             cardholder_history=arm["cardholder_history"],
                             masked=masked if arm["mask_short_velocity"] else None)

    # --- v3 on the same val examples, and the estimator check ------------------
    full_val = split(val_ex, cfg["arms"]["full"])
    v3 = new_model(params, full_val)
    v3_state = torch.load(repo_path(params["paths"]["models"]) / "gnn" / "model.pt",
                          map_location="cpu")
    if set(v3_state) != set(v3.state_dict()):
        raise RuntimeError("request-graph model and v3 disagree on parameter names")
    v3.load_state_dict(v3_state)
    v3_scores = predict(v3, full_val, params["gnn"]["batch_size"])
    replay = np.load(repo_path("reports/metrics_replay_2018.npz"))
    report = {"val_examples": len(full_val), "val_weighted_rows": float(full_val.weight.sum()),
              "v3": {"val_weighted_auc_pr": auc_pr(full_val.y.numpy(), v3_scores,
                                                   sample_weight=full_val.weight),
                     "replay_2018_auc_pr": auc_pr(replay["label"], replay["score"])},
              "runs": {}}
    # The same transactions scored two ways -- the sampled examples through these
    # batches, and the full replay -- must agree row for row, or the batches are
    # not the served graph.
    order = np.argsort(replay["txn_id"])
    at = np.searchsorted(replay["txn_id"], full_val.txn_id, sorter=order)
    report["v3"]["max_abs_diff_vs_replay"] = float(
        np.abs(replay["score"][order][at] - v3_scores).max())
    print(f"v3 on val examples: weighted AUC-PR {report['v3']['val_weighted_auc_pr']:.4f}  "
          f"(its full 2018 replay {report['v3']['replay_2018_auc_pr']:.4f}; "
          f"max |score diff| vs replay {report['v3']['max_abs_diff_vs_replay']:.2e})", flush=True)

    out_root = repo_path(cfg["models_dir"])
    if out_root.exists():
        shutil.rmtree(out_root)
    # One arm's batches in memory at a time: each full-2018 val set is ~1 GB of
    # index and id-node tensors on top of the shared node table. The `full`
    # arm reuses the val batches v3 was just scored on.
    best_pw = None
    for arm_name in args.arms:
        arm = cfg["arms"][arm_name]
        train = val = None
        val = full_val if arm == cfg["arms"]["full"] else split(val_ex, arm)
        train = split(train_ex, arm)
        sweep = cfg["pos_weight_sweep"] if arm_name == "full" else [best_pw or 10.0]
        for pw in sweep:
            run = f"{arm_name}-pw{pw:g}"
            print(f"  {run}: {len(train):,} train / {len(val):,} val examples", flush=True)
            model, info = train_arm(params, train, val, float(pw), args.epochs)
            info |= {"arm": arm_name, **arm,
                     "masked_features": short_velocity_columns(params)
                     if arm["mask_short_velocity"] else [],
                     "model_name": f"gnn-causal-{run}-epoch{info['best_epoch']}"}
            target = out_root / run
            target.mkdir(parents=True)
            torch.save(model.state_dict(), target / "model.pt")
            (target / "train_info.json").write_text(json.dumps(info, indent=2))
            report["runs"][run] = {k: info[k] for k in (
                "arm", "pos_weight", "best_epoch", "val_auc_pr", "seconds", "model_name")}
            if not args.no_mlflow:
                _log_mlflow(run, info)
        if arm_name == "full":
            best_pw = max((r for r in report["runs"].values() if r["arm"] == "full"),
                          key=lambda r: r["val_auc_pr"])["pos_weight"]

    winner = max(report["runs"], key=lambda r: report["runs"][r]["val_auc_pr"])
    report["winner"] = winner
    report["winner_gain_vs_v3_val"] = (report["runs"][winner]["val_auc_pr"]
                                       - report["v3"]["val_weighted_auc_pr"])
    out = repo_path(cfg["metrics"])
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
