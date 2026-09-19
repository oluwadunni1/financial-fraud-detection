"""Scoring one transaction. Shared by the API and the replay.

Both callers go through `Predictor.score`, deliberately. If the replay had its
own scoring path, it would be measuring something other than what the API does,
and its verdict on the offline number would be worth nothing.

The order of operations is the leakage control, and it is fixed:

    velocity(history)  ->  subgraph(history)  ->  forward pass  ->  score

`history` in both cases is rows with `ts < now`, strictly. Nothing here inserts;
the caller inserts *after* it has a score. That separation is what makes the
replay causally honest rather than merely careful.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import time
from dataclasses import dataclass
from typing import Any

import torch

from fraud.api.store import Neighbourhood
from fraud.api.subgraph import build_request_graph
from fraud.features.encoders import Encoder
from fraud.features.velocity_online import velocity_for_transaction
from fraud.models.gnn import TXN


@dataclass
class Prediction:
    score: float
    velocity: dict[str, float]
    cold_start_user: bool
    cold_start_merchant: bool
    latency_ms: float
    model_version: str


class Predictor:
    """The served model plus everything it needs to interpret a request.

    Loaded once and reused. The encoder and the card mapping are as much a part
    of the model as its weights: re-fitting either would silently change what
    every feature means, so they are loaded from the artifacts training wrote,
    never recomputed.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        encoder: Encoder,
        card_mapping: dict[str, int],
        n_card_values: int,
        windows_hours: list[int],
        model_version: str,
        max_neighbours: int = 10,
    ):
        self.model = model.eval()
        self.encoder = encoder
        self.card_mapping = card_mapping
        self.n_card_values = n_card_values
        self.windows_hours = windows_hours
        self.model_version = model_version
        self.max_neighbours = max_neighbours

    # -- construction ------------------------------------------------------
    @classmethod
    def from_registry(cls, params: dict, alias: str | None = None) -> Predictor:
        """Load the model by ALIAS, never by path (decision 14).

        Swapping the champion is then an alias move with no redeploy, which is
        the whole reason the registry exists in this project.
        """
        import os

        import mlflow
        from dotenv import load_dotenv

        from fraud.config import REPO_ROOT

        load_dotenv(REPO_ROOT / ".env")
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        alias = alias or params["serving"]["model_alias"]
        name = params["mlflow"]["registered_model_name"]

        uri = f"models:/{name}@{alias}"
        model = mlflow.pytorch.load_model(uri, map_location="cpu")
        version = mlflow.tracking.MlflowClient().get_model_version_by_alias(
            name, alias
        )
        return cls._assemble(
            model, params, f"{name}@{alias}:v{version.version}"
        )

    @classmethod
    def from_disk(cls, params: dict, model_dir: pathlib.Path) -> Predictor:
        """Local load, for the replay and for tests.

        Same weights, no network. Used when the point is to measure the model
        rather than to exercise the registry.
        """
        from torch_geometric.loader import NeighborLoader

        from fraud.config import repo_path
        from fraud.models.gnn import FraudGNN, load_graph

        cfg = params["gnn"]
        graph_dir = repo_path(params["paths"]["graph"])
        train = load_graph(graph_dir / "train")
        model = FraudGNN(train.metadata(), cfg)
        with torch.no_grad():  # materialise the lazy layers before loading
            batch = next(
                iter(
                    NeighborLoader(
                        train,
                        num_neighbors=cfg["num_neighbors"],
                        input_nodes=TXN,
                        batch_size=16,
                        shuffle=False,
                    )
                )
            )
            model(batch.x_dict, batch.edge_index_dict)
        model.load_state_dict(
            torch.load(model_dir / "model.pt", map_location="cpu")
        )
        info = json.loads((model_dir / "train_info.json").read_text())
        return cls._assemble(model, params, f"gnn-local-epoch{info['best_epoch']}")

    @classmethod
    def _assemble(
        cls, model: torch.nn.Module, params: dict, version: str
    ) -> Predictor:
        from fraud.config import repo_path

        encoder = Encoder.from_json(repo_path(params["paths"]["encoder"]))
        cards = json.loads(
            (repo_path(params["paths"]["graph"]) / "card_mapping.json").read_text()
        )
        return cls(
            model=model,
            encoder=encoder,
            card_mapping=cards["mapping"],
            n_card_values=cards["n_card_values"],
            windows_hours=params["velocity"]["windows_hours"],
            model_version=version,
            max_neighbours=params["serving"]["neighbours"],
        )

    # -- scoring -----------------------------------------------------------
    @torch.no_grad()
    def score(
        self, transaction: dict[str, Any], neighbourhood: Neighbourhood
    ) -> Prediction:
        """Score one arriving transaction against its own past.

        `neighbourhood` must contain only rows with `ts < transaction["ts"]`.
        `velocity_for_transaction` re-checks that rather than trusting it, since
        a single future row would be invisible in the output and catastrophic in
        the metrics.
        """
        started = time.perf_counter()

        velocity = velocity_for_transaction(
            transaction, neighbourhood.user_history, self.windows_hours
        )
        graph = build_request_graph(
            transaction,
            velocity,
            neighbourhood,
            self.encoder,
            self.card_mapping,
            self.n_card_values,
            max_neighbours=self.max_neighbours,
        )
        # Node 0 of the transaction type is the arriving transaction.
        logit = self.model(graph.x_dict, graph.edge_index_dict)[0]
        score = float(torch.sigmoid(logit))

        return Prediction(
            score=score,
            velocity=velocity,
            cold_start_user=neighbourhood.cold_start_user,
            cold_start_merchant=neighbourhood.cold_start_merchant,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            model_version=self.model_version,
        )


def transaction_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalise a processed-data row into what the predictor expects.

    One place to do this, so the API and the replay cannot disagree about which
    column is which.
    """
    ts = row["ts"]
    if not isinstance(ts, dt.datetime):
        ts = dt.datetime.fromisoformat(str(ts))
    return {
        "txn_id": int(row["txn_id"]),
        "User": int(row["User"]),
        "Card": int(row["Card"]),
        "ts": ts,
        "Amount": float(row["Amount"]),
        "Merchant": str(row["Merchant"]),
        "MCC": int(row["MCC"]),
        "City": str(row["City"]),
        "State": str(row["State"]),
        "Zip": int(row["Zip"]),
        "Errors": str(row["Errors"]),
        "Chip": str(row["Chip"]),
        "Time": int(row["Time"]),
        "Month": int(row["Month"]),
        "Day": int(row["Day"]),
    }
