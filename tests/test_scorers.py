"""Both models behind one scoring contract -- and neither drifts from training.

The XGBoost champion is now served live, so it inherits the project's top
risk: features rebuilt at request time must equal the offline feature matrix,
row for row. And the registry must hold the weights that were evaluated --
v1 of the champion was the pre-velocity-fix model and stayed @champion for
two weeks after the honest retrain, the same failure as decision 23.
"""

from __future__ import annotations

import datetime as dt
import pathlib

import numpy as np
import polars as pl
import pytest

XGB = pathlib.Path("data/models/xgb/model.json")
PROCESSED = pathlib.Path("data/processed")
needs_data = pytest.mark.skipif(not (XGB.exists() and PROCESSED.exists()),
                                reason="champion or processed data absent")


def _local_predictor(params):
    import xgboost as xgb

    from fraud.api.predictor import XGBoostPredictor
    from fraud.config import repo_path
    from fraud.features.encoders import Encoder

    booster = xgb.Booster()
    booster.load_model(repo_path(params["paths"]["models"]) / "xgb" / "model.json")
    return XGBoostPredictor(booster, Encoder.from_json(repo_path(params["paths"]["encoder"])),
                            params["velocity"]["windows_hours"], "xgb-local")


def test_a_booster_and_encoder_that_disagree_are_refused():
    from fraud.api.predictor import XGBoostPredictor

    class Booster:
        feature_names = ["b", "a"]

        def attr(self, _):
            return None

    class Enc:
        def feature_names(self):
            return ["a", "b"]

    with pytest.raises(RuntimeError, match="feature order"):
        XGBoostPredictor(Booster(), Enc(), [1], "x")


@needs_data
def test_xgboost_online_features_and_scores_equal_offline():
    from fraud.api.predictor import transaction_from_row
    from fraud.config import load_params
    from fraud.explain.graph import neighbourhood_for
    from fraud.explain.tree import matrix_rows

    params = load_params()
    predictor = _local_predictor(params)
    reference = np.load("reports/scores_xgb_2019.npz")
    offline = dict(zip(reference["txn_id"].tolist(), reference["score"].tolist(), strict=True))

    rows = (pl.scan_parquet(PROCESSED / "**/*.parquet", hive_partitioning=True)
            .filter(pl.col("Year") == 2019).head(3000).collect().sort("ts").tail(4))
    for row in rows.iter_rows(named=True):
        txn = transaction_from_row(row)
        nb = neighbourhood_for(params, txn, dt.datetime(2018, 12, 25))
        prediction = predictor.score(txn, nb)
        online = predictor.features(txn, prediction.velocity)[0]
        matrix = (matrix_rows(params, txn_ids=[txn["txn_id"]], years=[2019])
                  .select(predictor.encoder.feature_names()).to_numpy()[0])
        np.testing.assert_allclose(online, matrix, atol=1e-6)
        assert prediction.score == pytest.approx(offline[txn["txn_id"]], abs=1e-6)
