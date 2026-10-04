"""The serving API.

Two models, both loaded **by alias** from the registry, never by path
(decision 14): `@champion` decides, `@challenger` is scored in shadow after the
response is sent. A background thread re-resolves both aliases every
`serving.alias_refresh_seconds`, so moving an alias in the registry changes
live scoring with no redeploy -- and `/health` reports exactly which versions
are live, because a model/encoder mismatch is otherwise invisible until the
metrics quietly sag.

Request ordering is the leakage control and is identical to the replay's:

    read history (ts < now)  ->  score  ->  log  ->  INSERT  ->  (shadow score)

The insert happens before the response; a transaction becomes visible to the
next request and not to its own. The shadow score reuses the same history and
only ever updates `predictions.challenger_*`.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from fraud.api import store
from fraud.api.predictor import load_scorer, resolve_alias
from fraud.api.timing import Spans
from fraud.config import load_params

log = logging.getLogger("fraud.api")
_state: dict[str, Any] = {}


class Models:
    """The live champion and challenger, swappable without a restart.

    A swap loads the new model fully BEFORE replacing the reference, so a
    request always sees one complete model -- never a half-loaded one -- and a
    failed load leaves the old model serving.
    """

    def __init__(self, params: dict):
        self.params = params
        self.lock = threading.Lock()
        self.champion = None
        self.challenger = None
        self.versions: dict[str, str] = {}
        self.last_check: float | None = None
        self.swaps = 0
        self._stop = threading.Event()

    def _aliases(self) -> dict[str, str]:
        serving = self.params["serving"]
        out = {"champion": serving["champion_alias"]}
        if serving["shadow"]:
            out["challenger"] = serving["challenger_alias"]
        return out

    def refresh(self) -> bool:
        """Re-resolve every alias; load and swap any that moved. True if swapped."""
        swapped = False
        for role, alias in self._aliases().items():
            version = resolve_alias(self.params, alias)
            if self.versions.get(role) == version:
                continue
            scorer = load_scorer(self.params, alias)
            with self.lock:
                setattr(self, role, scorer)
                previous = self.versions.get(role)
                self.versions[role] = version
            if previous is not None:
                self.swaps += 1
                log.warning("%s swapped: v%s -> v%s (%s)", role, previous, version,
                            scorer.model_version)
            swapped = True
        self.last_check = time.time()
        return swapped

    def watch(self) -> None:
        period = self.params["serving"]["alias_refresh_seconds"]
        while not self._stop.wait(period):
            try:
                self.refresh()
            except Exception:              # a registry blip must not stop serving
                log.exception("alias refresh failed; keeping the live models")

    def stop(self) -> None:
        self._stop.set()


class Transaction(BaseModel):
    """One arriving transaction, as the network sees it."""

    txn_id: int
    user: int = Field(alias="User")
    card: int = Field(alias="Card")
    ts: dt.datetime
    amount: float = Field(alias="Amount")
    merchant: str = Field(alias="Merchant")
    mcc: int = Field(alias="MCC")
    city: str = Field(alias="City")
    state: str = Field(alias="State")
    zip_code: int = Field(alias="Zip")
    errors: str = Field(alias="Errors")
    chip: str = Field(alias="Chip")
    time_min: int = Field(alias="Time")
    month: int = Field(alias="Month")
    day: int = Field(alias="Day")

    model_config = {"populate_by_name": True}

    def to_features(self) -> dict[str, Any]:
        return {
            "txn_id": self.txn_id,
            "User": self.user,
            "Card": self.card,
            "ts": self.ts,
            "Amount": self.amount,
            "Merchant": self.merchant,
            "MCC": self.mcc,
            "City": self.city,
            "State": self.state,
            "Zip": self.zip_code,
            "Errors": self.errors,
            "Chip": self.chip,
            "Time": self.time_min,
            "Month": self.month,
            "Day": self.day,
        }


class PredictionResponse(BaseModel):
    txn_id: int
    score: float
    decision: bool
    model_version: str
    latency_ms: float
    cold_start_user: bool
    cold_start_merchant: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both models and open the pool once, at startup -- not per request."""
    params = load_params()
    _state["params"] = params
    # One request is a ~21-node graph: parallelising it inside torch thrashes
    # rather than helps (decision 20). Set before any model is loaded.
    import torch

    torch.set_num_threads(params["serving"]["torch_threads"])
    models = Models(params)
    models.refresh()
    _state["models"] = models
    # Merged round trips read and write in autocommit: no BEGIN before the read,
    # and the combined write is one atomic statement (decision 22).
    _state["merged"] = bool(params["serving"]["merged_round_trips"])
    _state["pool"] = store.pool(params, autocommit=_state["merged"])
    watcher = threading.Thread(target=models.watch, name="alias-watch", daemon=True)
    watcher.start()
    yield
    models.stop()
    _state["pool"].close()


app = FastAPI(
    title="Fraud detection",
    description="GraphSAGE scored end to end over a per-request subgraph.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    """What is actually loaded, so a mismatch is visible rather than inferred."""
    models: Models | None = _state.get("models")
    if models is None or models.champion is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    params = _state["params"]
    champion = models.champion
    return {
        "status": "ok",
        # `model_version` / `alias` describe the model that DECIDES -- the
        # replay and any client read these two keys.
        "model_version": champion.model_version,
        "alias": params["serving"]["champion_alias"],
        "challenger_version": models.challenger.model_version if models.challenger else None,
        "shadow": bool(params["serving"]["shadow"]),
        "alias_swaps": models.swaps,
        "aliases_checked_s_ago": (round(time.time() - models.last_check, 1)
                                  if models.last_check else None),
        "encoder_features": len(champion.encoder.feature_names()),
        "velocity_windows": champion.windows_hours,
        "neighbours": champion.max_neighbours,
    }


def _shadow(challenger, features: dict[str, Any], neighbourhood) -> None:
    """Score the challenger on the same input, after the response has gone.

    It never affects the decision, and a failure here is logged, not raised:
    shadow mode must not be able to take serving down.
    """
    try:
        prediction = challenger.score(features, neighbourhood)
        with _state["pool"].connection() as conn:
            store.record_shadow(conn, features["txn_id"], prediction.score,
                                prediction.model_version, prediction.latency_ms)
            if not conn.autocommit:
                conn.commit()
    except Exception:
        log.exception("shadow scoring failed for txn %s", features["txn_id"])


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction, response: Response,
            background: BackgroundTasks) -> PredictionResponse:
    params = _state["params"]
    models: Models = _state["models"]
    with models.lock:                       # one consistent pair per request
        champion, challenger = models.champion, models.challenger
    features = transaction.to_features()
    # Training timestamps are naive UTC. An aware one from a client is folded
    # to that, so `ts < now` compares like with like.
    if features["ts"].tzinfo is not None:
        features["ts"] = features["ts"].astimezone(dt.UTC).replace(tzinfo=None)

    # Each span is timed so the response can say where the time went; see
    # fraud.api.timing for why a total alone is not enough.
    spans = Spans()
    history_args = {
        "user_id": features["User"],
        "merchant_id": int(features["Merchant"]),
        "now": features["ts"],
        "history_hours": params["serving"]["history_hours"],
        "graph_rows": params["serving"]["neighbours"],
    }
    merged = _state["merged"]

    with _state["pool"].connection() as conn:
        # 1-2. Everything strictly before this transaction. Bounded by time,
        #      not row count -- velocity counts a window and truncation skews it.
        if merged:
            with spans.span("fetch", round_trips=1):
                neighbourhood = store.fetch_neighbourhood_once(conn, **history_args)
        else:
            # Three round trips, not two: outside autocommit psycopg sends
            # BEGIN on its own before the first statement. Measured, not
            # assumed -- the first latency replay found `fetch` at 3x the RTT.
            with spans.span("fetch", round_trips=3):
                neighbourhood = store.fetch_neighbourhood(conn, **history_args)

        # 3. Score -- the champion decides.
        with spans.span("score"):
            prediction = champion.score(features, neighbourhood)
            # Each model's own validation-chosen threshold (params.yaml).
            threshold = params["serving"]["decision_thresholds"][champion.family]
            decision = prediction.score >= threshold

        # 4. Log, then 5. make visible. Order matters: a row inserted before
        #    scoring would be visible to its own prediction.
        logged = {
            "txn_id": features["txn_id"],
            "score": prediction.score,
            "decision": decision,
            "model_version": prediction.model_version,
            "latency_ms": prediction.latency_ms,
            "embedding_age_seconds": None,
            "cold_start_user": prediction.cold_start_user,
            "cold_start_merchant": prediction.cold_start_merchant,
        }
        row = store.stored_row(features, {})
        if merged:
            with spans.span("write", round_trips=1):
                store.log_and_insert(conn, logged, row, prediction.velocity)
        else:
            with spans.span("log", round_trips=1):
                store.log_prediction(conn, logged)
            with spans.span("insert", round_trips=1):
                store.insert_transaction(conn, row, prediction.velocity)
            with spans.span("commit", round_trips=1):
                conn.commit()

    # 6. Shadow: after the response, on the same history, never acted on.
    if challenger is not None:
        background.add_task(_shadow, challenger, features, neighbourhood)

    response.headers["Server-Timing"] = spans.header()
    return PredictionResponse(
        txn_id=features["txn_id"],
        score=prediction.score,
        decision=decision,
        model_version=prediction.model_version,
        latency_ms=prediction.latency_ms,
        cold_start_user=prediction.cold_start_user,
        cold_start_merchant=prediction.cold_start_merchant,
    )


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """Operational counters, read back from what was actually logged -- the
    champion's, and the shadow challenger's beside them."""
    with _state["pool"].connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select count(*) as predictions,
                   avg(latency_ms) as avg_latency_ms,
                   percentile_disc(0.95) within group (order by latency_ms)
                       as p95_latency_ms,
                   sum(case when decision then 1 else 0 end) as alerts,
                   sum(case when cold_start_user then 1 else 0 end) as cold_users,
                   sum(case when cold_start_merchant then 1 else 0 end)
                       as cold_merchants,
                   count(challenger_score) as shadow_scored,
                   avg(challenger_latency_ms) as shadow_avg_latency_ms,
                   corr(score, challenger_score) as champion_challenger_corr
              from predictions
            """
        )
        row = cur.fetchone()
    return dict(row) if row else {}
