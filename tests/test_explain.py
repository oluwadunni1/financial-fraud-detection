"""Explanations must be exact where they claim to be, and mean what they say.

Shapley values have properties that hold or do not -- efficiency, symmetry,
the dummy axiom -- so they are tested as properties, first on toy functions
where the answer is known, then on the real models.
"""

from __future__ import annotations

import datetime as dt
import pathlib

import numpy as np
import pytest

from fraud.explain.graph import GROUPS, exact_shapley
from fraud.explain.tree import fold, reason_codes, source_field

# --- exact Shapley on functions with a known answer -------------------------------

def test_additive_game_returns_each_players_own_weight():
    weights = {"a": 1.5, "b": -2.0, "c": 0.25}
    phi, _ = exact_shapley(lambda s: sum(weights[p] for p in s), tuple(weights))
    assert phi == pytest.approx(weights)


def test_interaction_is_split_equally_between_symmetric_players():
    phi, _ = exact_shapley(lambda s: 4.0 if {"a", "b"} <= s else 0.0, ("a", "b", "c"))
    assert phi["a"] == pytest.approx(2.0) and phi["b"] == pytest.approx(2.0)
    assert phi["c"] == pytest.approx(0.0)        # dummy: never changes the value


def test_values_sum_to_the_full_minus_the_empty_coalition():
    rng = np.random.default_rng(0)
    table = {}

    def v(s: frozenset) -> float:
        return table.setdefault(s, float(rng.normal()))

    phi, coalitions = exact_shapley(v, GROUPS)
    assert sum(phi.values()) == pytest.approx(
        coalitions[frozenset(GROUPS)] - coalitions[frozenset()])
    assert len(coalitions) == 2 ** len(GROUPS)   # every coalition, once


# --- reason codes -----------------------------------------------------------------

def test_encoded_columns_fold_back_to_their_field():
    assert source_field("Merchant_bin_7") == "Merchant"
    assert source_field("Chip_oh_Swipe Transaction") == "Chip"
    assert source_field("velocity_count_24h") == "velocity_count_24h"


def test_folding_preserves_every_rows_total():
    columns = ["Merchant_bin_0", "Merchant_bin_1", "Chip_oh_A", "Chip_oh_B", "Amount"]
    contribs = np.random.default_rng(1).normal(size=(6, len(columns) + 1))
    fields, folded = fold(contribs, columns)
    assert fields == ["Merchant", "Chip", "Amount"]
    np.testing.assert_allclose(folded.sum(axis=1), contribs.sum(axis=1))
    np.testing.assert_allclose(folded[:, -1], contribs[:, -1])     # bias kept


def test_reason_codes_are_the_largest_positive_pushes_only():
    reasons = reason_codes(["A", "B", "C", "D"], np.array([0.5, -3.0, 2.0, 0.1, 9.9]), 3)
    assert [r["field"] for r in reasons] == ["C", "A", "D"]       # bias ignored


# --- the real models ----------------------------------------------------------------

XGB = pathlib.Path("data/models/xgb/model.json")
GNN = pathlib.Path("data/models/gnn/model.pt")
MATRIX = pathlib.Path("data/features/matrix")


@pytest.mark.skipif(not (XGB.exists() and MATRIX.exists()), reason="champion absent")
def test_treeshap_sums_to_the_champions_margin():
    from fraud.config import load_params
    from fraud.explain.tree import contributions, load_champion, matrix_rows

    params = load_params()
    champion = load_champion(params)
    rows = matrix_rows(params, years=[2019]).head(200)
    x = rows.select(champion.feature_names).to_numpy().astype(np.float32)
    phi = contributions(champion, x)
    np.testing.assert_allclose(phi.sum(axis=1), champion.margin(x), atol=1e-4)


@pytest.mark.skipif(not GNN.exists(), reason="GraphSAGE absent")
def test_group_shapley_on_graphsage_is_exact_and_matches_serving():
    """The full coalition must be the score serving produces -- otherwise the
    explanation is of some other input -- and the values must be efficient."""
    import polars as pl
    import torch

    from fraud.api.predictor import Predictor, transaction_from_row
    from fraud.config import load_params, repo_path
    from fraud.explain.graph import explain, neighbourhood_for, typical_transaction

    torch.set_num_threads(1)
    params = load_params()
    predictor = Predictor.from_disk(params, repo_path(params["paths"]["models"]) / "gnn")
    row = (pl.scan_parquet(repo_path(params["paths"]["processed"]) / "**/*.parquet",
                           hive_partitioning=True)
           .filter(pl.col("Year") == 2019).head(500).collect().sort("ts").row(-1, named=True))
    txn = transaction_from_row(row)
    nb = neighbourhood_for(params, txn, dt.datetime(2018, 12, 25))
    result = explain(predictor, txn, nb, typical_transaction(params, 2018, sample=20_000))

    assert abs(result.efficiency_gap()) < 1e-6
    assert result.score == pytest.approx(predictor.score(txn, nb).score, abs=1e-6)
    assert set(result.values) == set(GROUPS)
