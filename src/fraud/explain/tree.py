"""Exact TreeSHAP for the XGBoost champion, as reason codes.

XGBoost computes exact TreeSHAP itself (`pred_contribs=True`): one value per
feature plus a bias column, summing to the raw margin (log-odds) for that row.
No sampling, no approximation -- the `shap` library is used only for plots and
as a cross-check.

An analyst cannot act on `Merchant_bin_7`. The encoder splits one field into
many columns (a 17-bit binary code for Merchant, one-hot columns for Chip), and
SHAP values are additive, so folding a field's columns back together is exact:
the folded values still sum to the same margin.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
import polars as pl
import xgboost as xgb

from fraud.config import repo_path


@dataclass
class Champion:
    booster: xgb.Booster
    best_iteration: int
    feature_names: list[str]

    def dmatrix(self, x: np.ndarray) -> xgb.DMatrix:
        return xgb.DMatrix(x, feature_names=self.feature_names)

    def margin(self, x: np.ndarray) -> np.ndarray:
        return self.booster.predict(self.dmatrix(x), output_margin=True,
                                    iteration_range=(0, self.best_iteration + 1))

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.booster.predict(self.dmatrix(x),
                                    iteration_range=(0, self.best_iteration + 1))


def load_champion(params: dict) -> Champion:
    """The booster `evaluate` scored, at the same best iteration."""
    model_dir = repo_path(params["paths"]["models"]) / "xgb"
    booster = xgb.Booster()
    booster.load_model(model_dir / "model.json")
    info = json.loads((model_dir / "train_info.json").read_text())
    return Champion(booster, int(info["best_iteration"]), list(booster.feature_names))


def matrix_rows(params: dict, txn_ids: list[int] | None = None,
                years: list[int] | None = None) -> pl.DataFrame:
    """Feature-matrix rows (txn_id, Fraud, features...) by id or by year."""
    scan = pl.scan_parquet(
        repo_path(params["paths"]["feature_matrix"]) / "**/*.parquet",
        hive_partitioning=True,
    )
    if years is not None:
        scan = scan.filter(pl.col("Year").is_in(years))
    if txn_ids is not None:
        scan = scan.filter(pl.col("txn_id").is_in(txn_ids))
    return scan.collect(engine="streaming")


def contributions(champion: Champion, x: np.ndarray) -> np.ndarray:
    """[n, features + 1] exact SHAP values in log-odds; the last column is the bias."""
    return champion.booster.predict(
        champion.dmatrix(x), pred_contribs=True,
        iteration_range=(0, champion.best_iteration + 1),
    )


def source_field(column: str) -> str:
    """`Merchant_bin_7` -> `Merchant`, `Chip_oh_Swipe Transaction` -> `Chip`."""
    for marker in ("_bin_", "_oh_"):
        if marker in column:
            return column.split(marker, 1)[0]
    return column


def fold(contribs: np.ndarray, columns: list[str]) -> tuple[list[str], np.ndarray]:
    """Sum each field's encoded columns. Exact, because SHAP is additive.

    Returns (fields, [n, fields + 1]) with the bias kept as the last column.
    """
    fields = list(dict.fromkeys(source_field(c) for c in columns))
    index = {f: i for i, f in enumerate(fields)}
    folded = np.zeros((contribs.shape[0], len(fields) + 1), dtype=contribs.dtype)
    for j, column in enumerate(columns):
        folded[:, index[source_field(column)]] += contribs[:, j]
    folded[:, -1] = contribs[:, -1]
    return fields, folded


def reason_codes(fields: list[str], folded_row: np.ndarray, top_k: int) -> list[dict]:
    """The fields that pushed this score up the most, largest first.

    Only positive contributions are reasons for an alert; a field that pulled
    the score down is context, not a reason.
    """
    values = folded_row[:-1]
    order = np.argsort(-values)
    return [
        {"field": fields[i], "log_odds": float(values[i])}
        for i in order[:top_k]
        if values[i] > 0
    ]


def explain_rows(champion: Champion, rows: pl.DataFrame, top_k: int) -> pl.DataFrame:
    """Reason codes for each row of a feature-matrix frame."""
    x = rows.select(champion.feature_names).to_numpy().astype(np.float32)
    fields, folded = fold(contributions(champion, x), champion.feature_names)
    scores = champion.score(x)
    out = []
    for i, txn in enumerate(rows["txn_id"].to_list()):
        reasons = reason_codes(fields, folded[i], top_k)
        out.append({
            "txn_id": txn,
            "score": float(scores[i]),
            "fraud": int(rows["Fraud"][i]) if "Fraud" in rows.columns else None,
            "reasons": " · ".join(f"{r['field']} (+{r['log_odds']:.2f})" for r in reasons),
            "top_field": reasons[0]["field"] if reasons else None,
        })
    return pl.DataFrame(out)


def save_dir(params: dict) -> pathlib.Path:
    return repo_path(params["monitoring"]["output_dir"]).parent / "explain"
