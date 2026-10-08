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
from fraud.api.subgraph import build_request_graph, transaction_feature_names
from fraud.features.encoders import Encoder
from fraud.features.velocity_online import velocity_for_transaction


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
        self.family = "graphsage"
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
        import mlflow

        from fraud import settings

        settings.configure_mlflow()
        alias = alias or params["serving"]["champion_alias"]
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
        import polars as pl

        from fraud.config import repo_path
        from fraud.features.velocity import velocity_columns
        from fraud.models.gnn import FraudGNN

        # Lazy (-1, -1) layers need one forward pass to learn their widths. One
        # request graph -- built by the serving builder itself -- is enough, and
        # carries the same node/edge types as the training graph (the retrain
        # asserts identical parameter names). It used to be a NeighborLoader batch
        # from the 277k-node training graph: seconds of loading, and pyg-lib,
        # which the API image does not ship -- so the dashboard image could not
        # load the model it explains.
        encoder = Encoder.from_json(repo_path(params["paths"]["encoder"]))
        cards = json.loads(
            (repo_path(params["paths"]["graph"]) / "card_mapping.json").read_text()
        )
        probe = {"txn_id": 0, "User": 0, "Card": 0, "ts": dt.datetime(2000, 1, 1),
                 "Amount": 0.0, "Merchant": "0", "MCC": 0, "City": "", "State": "",
                 "Zip": 0, "Errors": "", "Chip": "Swipe Transaction", "Time": 0,
                 "Month": 1, "Day": 1}
        empty = pl.DataFrame(schema={})
        graph = build_request_graph(
            probe, dict.fromkeys(velocity_columns(params["velocity"]["windows_hours"]), 0.0),
            Neighbourhood(user_history=empty, merchant_history=empty), encoder,
            cards["mapping"], cards["n_card_values"],
        )
        info = json.loads((model_dir / "train_info.json").read_text())
        # A retrained arm may have been trained blind to some feature columns;
        # the mask travels inside the model (FraudGNN.txn_mask).
        names = transaction_feature_names(encoder)
        masked = [names.index(c) for c in info.get("masked_features", [])]
        model = FraudGNN(graph.metadata(), {**params["gnn"], "txn_features": len(names)},
                         masked_txn_features=masked)
        with torch.no_grad():
            model(graph.x_dict, graph.edge_index_dict)
        model.load_state_dict(
            torch.load(model_dir / "model.pt", map_location="cpu")
        )
        # The name is what the gate and the registry key on (decision 32): a
        # retrained arm carries its own, v3 keeps its historical one.
        return cls._assemble(model, params,
                             info.get("model_name", f"gnn-local-epoch{info['best_epoch']}"))

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

        history = (neighbourhood.user_history if neighbourhood.velocity_history is None
                   else neighbourhood.velocity_history)
        velocity = velocity_for_transaction(transaction, history, self.windows_hours)
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


class XGBoostPredictor:
    """The tabular champion behind the same contract as the graph model.

    `score(transaction, neighbourhood)` takes the same inputs the GraphSAGE
    predictor does, so the API, shadow mode and the replay never branch on
    model type. The 86 features are rebuilt exactly as `features/build.py`
    builds them offline: the train-fitted encoder over the transaction plus
    its velocity, in the booster's own column order (asserted at load).
    Velocity comes from `velocity_for_transaction` -- the one module, two
    callers rule -- and `tests/test_scorers.py` holds online == offline.
    """

    def __init__(self, booster: Any, encoder: Encoder, windows_hours: list[int],
                 model_version: str, max_neighbours: int = 10):
        names = list(booster.feature_names or [])
        if names != encoder.feature_names():
            raise RuntimeError(
                "booster and encoder disagree on feature order -- the encoder must "
                "ship with the model it was fitted for")
        self.booster = booster
        self.family = "xgboost"
        self.encoder = encoder
        self.windows_hours = windows_hours
        self.model_version = model_version
        self.max_neighbours = max_neighbours
        best = booster.attr("best_iteration")
        self._iterations = (0, int(best) + 1) if best is not None else (0, 0)

    def features(self, transaction: dict[str, Any], velocity: dict[str, float]):
        return self.encoder.transform_rows([{**transaction, **velocity}])

    def score(self, transaction: dict[str, Any], neighbourhood: Neighbourhood) -> Prediction:
        import xgboost as xgb

        started = time.perf_counter()
        history = (neighbourhood.user_history if neighbourhood.velocity_history is None
                   else neighbourhood.velocity_history)
        velocity = velocity_for_transaction(transaction, history, self.windows_hours)
        x = xgb.DMatrix(self.features(transaction, velocity),
                        feature_names=self.encoder.feature_names())
        score = float(self.booster.predict(x, iteration_range=self._iterations)[0])
        return Prediction(
            score=score,
            velocity=velocity,
            cold_start_user=neighbourhood.cold_start_user,
            cold_start_merchant=neighbourhood.cold_start_merchant,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            model_version=self.model_version,
        )


def resolve_alias(params: dict, alias: str) -> str:
    """The registry version an alias points at right now, as a string."""
    import mlflow

    name = params["mlflow"]["registered_model_name"]
    return str(mlflow.tracking.MlflowClient().get_model_version_by_alias(name, alias).version)


def load_scorer(params: dict, alias: str):
    """Whatever model `alias` points at, behind the one scoring contract.

    The flavour is read from the logged model itself (xgboost or pytorch), so
    promoting a different kind of model is still just an alias move.
    """
    import mlflow

    from fraud import settings
    from fraud.config import repo_path

    settings.configure_mlflow()
    name = params["mlflow"]["registered_model_name"]
    uri = f"models:/{name}@{alias}"
    flavours = mlflow.models.get_model_info(uri).flavors
    version = f"{name}@{alias}:v{resolve_alias(params, alias)}"
    if "xgboost" in flavours:
        booster = mlflow.xgboost.load_model(uri)
        booster = booster.get_booster() if hasattr(booster, "get_booster") else booster
        return XGBoostPredictor(
            booster, Encoder.from_json(repo_path(params["paths"]["encoder"])),
            params["velocity"]["windows_hours"], version,
            max_neighbours=params["serving"]["neighbours"])
    if "pytorch" in flavours:
        model = mlflow.pytorch.load_model(uri, map_location="cpu")
        return Predictor._assemble(model, params, version)
    raise RuntimeError(f"{uri} has no flavour this API can serve: {sorted(flavours)}")
