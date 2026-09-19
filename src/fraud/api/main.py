"""The serving API.

Loads the model **by alias** from the registry, never by path (decision 14).
That is what makes a champion swap an alias move with no redeploy, and it is
why `/health` reports the alias, the resolved version and the encoder it is
paired with -- a model/encoder mismatch is otherwise invisible until the
metrics quietly sag.

Request ordering is the leakage control and is identical to the replay's:

    read history (ts < now)  ->  score  ->  log  ->  INSERT

The insert happens last. A transaction becomes visible to the next request and
not to its own.
"""

from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraud.api import store
from fraud.api.predictor import Predictor
from fraud.config import load_params

_state: dict[str, Any] = {}


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
    """Load the model once, at startup -- not per request."""
    params = load_params()
    _state["params"] = params
    _state["predictor"] = Predictor.from_registry(params)
    _state["threshold"] = params["evaluate"]["target_false_positive_rate"]
    _state["conn"] = store.connect()
    yield
    conn = _state.get("conn")
    if conn is not None:
        conn.close()


app = FastAPI(
    title="Fraud detection",
    description="GraphSAGE scored end to end over a per-request subgraph.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    """What is actually loaded, so a mismatch is visible rather than inferred."""
    predictor: Predictor | None = _state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    params = _state["params"]
    return {
        "status": "ok",
        "model_version": predictor.model_version,
        "alias": params["serving"]["model_alias"],
        "encoder_features": len(predictor.encoder.feature_names()),
        "velocity_windows": predictor.windows_hours,
        "neighbours": predictor.max_neighbours,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction) -> PredictionResponse:
    predictor: Predictor = _state["predictor"]
    params = _state["params"]
    conn = _state["conn"]
    features = transaction.to_features()

    # 1-2. Everything strictly before this transaction. Bounded by time, not
    #      row count -- velocity counts a window and truncation would skew it.
    neighbourhood = store.fetch_neighbourhood(
        conn,
        user_id=features["User"],
        merchant_id=int(features["Merchant"]),
        now=features["ts"],
        history_hours=params["serving"]["history_hours"],
    )

    # 3. Score.
    prediction = predictor.score(features, neighbourhood)
    decision = prediction.score >= _state["threshold"]

    # 4. Log, then 5. make visible. Order matters: a row inserted before
    #    scoring would be visible to its own prediction.
    store.log_prediction(
        conn,
        {
            "txn_id": features["txn_id"],
            "score": prediction.score,
            "decision": decision,
            "model_version": prediction.model_version,
            "latency_ms": prediction.latency_ms,
            "embedding_age_seconds": None,
            "cold_start_user": prediction.cold_start_user,
            "cold_start_merchant": prediction.cold_start_merchant,
        },
    )
    store.insert_transaction(
        conn,
        {
            "txn_id": features["txn_id"],
            "user_id": features["User"],
            "merchant_id": int(features["Merchant"]),
            "ts": features["ts"],
            "amount": features["Amount"],
            "mcc": features["MCC"],
            "merchant": features["Merchant"],
            "city": features["City"],
            "state": features["State"],
            "zip": features["Zip"],
            "errors": features["Errors"],
            "chip": features["Chip"],
            "time_min": features["Time"],
            "month": features["Month"],
            "day": features["Day"],
        },
        prediction.velocity,
    )
    conn.commit()

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
    """Operational counters, read back from what was actually logged."""
    conn = _state["conn"]
    with conn.cursor() as cur:
        cur.execute(
            """
            select count(*) as predictions,
                   avg(latency_ms) as avg_latency_ms,
                   percentile_disc(0.95) within group (order by latency_ms)
                       as p95_latency_ms,
                   sum(case when decision then 1 else 0 end) as alerts,
                   sum(case when cold_start_user then 1 else 0 end) as cold_users,
                   sum(case when cold_start_merchant then 1 else 0 end)
                       as cold_merchants
              from predictions
            """
        )
        row = cur.fetchone()
    return dict(row) if row else {}
