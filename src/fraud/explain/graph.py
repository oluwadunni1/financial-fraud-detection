"""Exact Shapley values for GraphSAGE, over the groups that make up a request.

GraphSAGE has no TreeSHAP. Its input is a ~21-node graph, and per-feature
attribution across that graph is neither cheap nor something an analyst can
use. The question an analyst actually asks is coarser: was it the transaction
itself, how fast the card was spending, the card's history, the merchant's
history, or who the card and merchant are? So the players are five groups:

    transaction       the row's own attributes (amount, time, chip, location...)
    velocity          the 13 velocity features computed from the card's past
    card_history      the card's recent transactions, as graph neighbours
    merchant_history  the merchant's recent transactions, as graph neighbours
    identity          the card / merchant / MCC codes on the id nodes

With five players there are 2^5 = 32 coalitions, so the Shapley values are
computed EXACTLY -- every coalition is scored -- in 32 forward passes of a
~30k-parameter model (well under a second). No sampling noise, and the values
sum to score(all) - score(none) by construction.

"Absent" must mean something the model has seen, or the attribution measures
an input it never learned to read. Every group is removed along a path the
model already handles:

    card_history / merchant_history  -> empty history (the cold-start path)
    velocity                         -> velocity computed from empty history
                                        (what a first-ever transaction gets)
    identity                         -> an unseen card, merchant and MCC, which
                                        encode to the reserved all-zero codes
    transaction                      -> a typical transaction: the reference
                                        period's median numbers and modal
                                        categories

Values are in log-odds, like TreeSHAP, so the two models' explanations are on
the same scale.
"""

from __future__ import annotations

import datetime as dt
import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import torch

from fraud.api.predictor import Predictor
from fraud.api.store import Neighbourhood
from fraud.api.subgraph import build_request_graph
from fraud.config import repo_path
from fraud.features.velocity_online import velocity_for_transaction

GROUPS = ("transaction", "velocity", "card_history", "merchant_history", "identity")

# The arriving row's own encoder inputs -- everything except the id columns
# (Merchant, MCC live on the merchant node; User/Card on the card node).
TRANSACTION_FIELDS = ("Amount", "City", "State", "Zip", "Errors", "Chip",
                      "Time", "Month", "Day")
_NUMERIC = {"Amount", "Zip", "Time", "Month", "Day"}

EMPTY = pl.DataFrame(schema={})


def typical_transaction(params: dict, year: int, sample: int = 200_000) -> dict[str, Any]:
    """The 'transaction absent' baseline: median numbers, modal categories."""
    frame = (
        pl.scan_parquet(repo_path(params["paths"]["processed"]) / "**/*.parquet",
                        hive_partitioning=True)
        .filter(pl.col("Year") == year)
        .select(TRANSACTION_FIELDS)
        .collect(engine="streaming")
    )
    if frame.height > sample:
        frame = frame.sample(sample, seed=0)
    baseline = {}
    for field in TRANSACTION_FIELDS:
        column = frame.get_column(field)
        if field in _NUMERIC:
            value = column.median()
            baseline[field] = float(value) if field == "Amount" else int(round(value))
        else:
            baseline[field] = str(column.drop_nulls().mode().sort()[0])
    return baseline


@dataclass
class GroupShapley:
    values: dict[str, float]        # log-odds contribution per group
    baseline_logit: float           # every group absent
    full_logit: float               # every group present -- the real score
    coalitions: dict[frozenset, float]

    @property
    def score(self) -> float:
        return 1.0 / (1.0 + math.exp(-self.full_logit))

    def efficiency_gap(self) -> float:
        """sum(values) - (full - baseline). Zero up to float error, always."""
        return sum(self.values.values()) - (self.full_logit - self.baseline_logit)


def exact_shapley(value: Callable[[frozenset], float],
                  players: tuple[str, ...]) -> tuple[dict[str, float], dict[frozenset, float]]:
    """Shapley values by full enumeration: every coalition evaluated once."""
    n = len(players)
    v = {frozenset(s): value(frozenset(s))
         for k in range(n + 1) for s in itertools.combinations(players, k)}
    phi = {}
    for p in players:
        others = [q for q in players if q != p]
        total = 0.0
        for k in range(n):
            weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for s in itertools.combinations(others, k):
                s = frozenset(s)
                total += weight * (v[s | {p}] - v[s])
        phi[p] = total
    return phi, v


def _logit(predictor: Predictor, transaction: dict[str, Any], velocity: dict[str, float],
           neighbourhood: Neighbourhood) -> float:
    graph = build_request_graph(
        transaction, velocity, neighbourhood, predictor.encoder,
        predictor.card_mapping, predictor.n_card_values,
        max_neighbours=predictor.max_neighbours,
    )
    with torch.no_grad():
        return float(predictor.model(graph.x_dict, graph.edge_index_dict)[0])


def explain(predictor: Predictor, transaction: dict[str, Any],
            neighbourhood: Neighbourhood, typical: dict[str, Any]) -> GroupShapley:
    """Exact group Shapley values for one scored transaction."""
    windows = predictor.windows_hours
    velocity_full = velocity_for_transaction(
        transaction, neighbourhood.user_history, windows)
    velocity_cold = velocity_for_transaction(transaction, EMPTY, windows)
    anonymous = {"User": -1, "Card": 0, "Merchant": "__unseen__", "MCC": -1}

    def value(present: frozenset) -> float:
        txn = dict(transaction)
        if "transaction" not in present:
            txn.update(typical)
        if "identity" not in present:
            txn.update(anonymous)
        nb = Neighbourhood(
            user_history=(neighbourhood.user_history
                          if "card_history" in present else EMPTY),
            merchant_history=(neighbourhood.merchant_history
                              if "merchant_history" in present else EMPTY),
        )
        velocity = velocity_full if "velocity" in present else velocity_cold
        return _logit(predictor, txn, velocity, nb)

    phi, v = exact_shapley(value, GROUPS)
    return GroupShapley(values=phi, baseline_logit=v[frozenset()],
                        full_logit=v[frozenset(GROUPS)], coalitions=v)


def neighbourhood_for(params: dict, transaction: dict[str, Any],
                      graph_since: dt.datetime) -> Neighbourhood:
    """The past the causal replay saw for this transaction, rebuilt offline.

    Seeded through `seed_rows`, the one selection the in-memory and the HTTP
    replay share, restricted to this card and merchant. In production the same
    input comes from `store.fetch_neighbourhood_once`.
    """
    from fraud.features.velocity import velocity_columns
    from fraud.jobs.replay import CausalHistory, row_velocity, seed_rows, transaction_from_row

    hours = params["serving"]["history_hours"]
    names = velocity_columns(params["velocity"]["windows_hours"])
    history = CausalHistory(hours, params["serving"]["neighbours"])
    rows = seed_rows(params, transaction["ts"], hours, graph_since,
                     users={transaction["User"]}, merchants={str(transaction["Merchant"])})
    for row in rows.iter_rows(named=True):
        history.add(transaction_from_row(row), row_velocity(row, names))
    return history.neighbourhood(transaction)


def as_frame(results: dict[int, GroupShapley]) -> pl.DataFrame:
    """One row per explained transaction: score, baseline and each group's value."""
    return pl.DataFrame([
        {"txn_id": t, "score": r.score, "baseline_logit": r.baseline_logit,
         "full_logit": r.full_logit, **r.values,
         "top_group": max(r.values, key=r.values.get)}
        for t, r in results.items()
    ])


def efficiency_errors(results: dict[int, GroupShapley]) -> np.ndarray:
    return np.array([abs(r.efficiency_gap()) for r in results.values()])
