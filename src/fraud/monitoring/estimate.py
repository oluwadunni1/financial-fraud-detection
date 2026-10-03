"""Performance without labels, performance with late labels, and input drift.

Three views of the same months, so they can be compared directly:

    realized (hindsight)   every label, as if chargebacks were instant -- the
                           truth, unavailable in production
    realized (known)       only the labels that exist by a given date -- what
                           a team can actually measure, and when
    estimated (CBPE)       NannyML's confidence-based estimate from the scores
                           alone, available the moment a month closes

CBPE calibrates the scores on the labelled reference period and then reads
expected performance off the analysis period's score distribution. It sees
covariate shift (the scores move). It is blind, by construction, to concept
drift: if the same scores start meaning something different, nothing in the
scores says so. Which of the two drove XGBoost's 2018 -> 2019 drop is part of
what the experiment answers.

The analysis frame handed to CBPE never contains labels. NannyML will use them
if present (it reports a `realized` column); passing them would make the
estimate look better informed than a production run could be.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud.config import repo_path
from fraud.models.metrics import threshold_at_fpr

METRICS = ("average_precision", "roc_auc")


# --- loading -------------------------------------------------------------------

def scored_frame(params: dict, model: str, period: str,
                 sample: int | None = None, seed: int = 0) -> pl.DataFrame:
    """Scores + labels for one model and period, joined to the drift inputs.

    `period` is "reference" or "analysis" (params.monitoring). Scores come
    from the npz files earlier stages wrote; nothing is re-scored here.
    """
    cfg = params["monitoring"]
    z = np.load(repo_path(cfg["scores"][model][period]))
    scores = pl.DataFrame({
        "txn_id": z["txn_id"].astype(np.int64),
        "score": z["score"].astype(np.float64),
        "label": z["label"].astype(np.int8),
    })
    years = ([cfg["reference_year"]] if period == "reference"
             else list(cfg["analysis_years"]))
    drift = cfg["drift"]
    raw = [c for c in (*drift["continuous"], *drift["categorical"])
           if not c.startswith("velocity_")]
    vel = [c for c in drift["continuous"] if c.startswith("velocity_")]

    processed = pl.scan_parquet(
        repo_path(params["paths"]["processed"]) / "**/*.parquet", hive_partitioning=True
    ).filter(pl.col("Year").is_in(years)).select("txn_id", "ts", *raw)
    velocity = pl.scan_parquet(
        repo_path(params["paths"]["velocity"]) / "**/*.parquet", hive_partitioning=True
    ).filter(pl.col("Year").is_in(years)).select("txn_id", *vel)

    frame = (
        scores.lazy()
        .join(processed, on="txn_id", how="inner")
        .join(velocity, on="txn_id", how="left")
        .collect(engine="streaming")
        .with_columns(pl.col(c).cast(pl.String) for c in drift["categorical"])
        .sort("ts", "txn_id")
    )
    if sample and frame.height > sample:
        frame = frame.sample(sample, seed=seed).sort("ts", "txn_id")
    return frame


def operating_threshold(reference: pl.DataFrame, target_fpr: float) -> float:
    """The score at `target_fpr` on the reference period, frozen thereafter."""
    return threshold_at_fpr(reference["label"].to_numpy(),
                            reference["score"].to_numpy(), target_fpr)


def _nannyml_frame(frame: pl.DataFrame, threshold: float, with_labels: bool) -> pd.DataFrame:
    df = frame.with_columns(
        (pl.col("score") >= threshold).cast(pl.Int8).alias("y_pred")
    )
    if not with_labels:
        df = df.drop("label", "labeled_at", strict=False)
    return df.to_pandas()


def _chunks(frame: pl.DataFrame, period: str) -> pl.DataFrame:
    """Calendar-month chunk bounds, matching NannyML's chunk_period='M'."""
    if period != "M":
        raise ValueError("only monthly chunks are implemented")
    return frame.with_columns(
        pl.col("ts").dt.truncate("1mo").alias("chunk_start"),
        (pl.col("ts").dt.truncate("1mo").dt.offset_by("1mo")
         - pl.duration(microseconds=1)).alias("chunk_end"),
    )


# --- estimated -----------------------------------------------------------------

def estimate_performance(reference: pl.DataFrame, analysis: pl.DataFrame,
                         params: dict) -> pl.DataFrame:
    """CBPE per chunk, tidy: one row per (chunk, metric)."""
    import nannyml as nml

    cfg = params["monitoring"]
    threshold = operating_threshold(reference, cfg["target_false_positive_rate"])
    estimator = nml.CBPE(
        y_pred_proba="score", y_pred="y_pred", y_true="label",
        timestamp_column_name="ts", metrics=list(METRICS),
        chunk_period=cfg["chunk_period"], problem_type="classification_binary",
    )
    estimator.fit(_nannyml_frame(reference, threshold, with_labels=True))
    result = estimator.estimate(_nannyml_frame(analysis, threshold, with_labels=False))
    df = result.filter(period="analysis").to_df()

    rows = []
    for metric in METRICS:
        part = df[metric]
        rows.append(pl.DataFrame({
            "chunk_start": pd.to_datetime(df[("chunk", "start_date")]).to_numpy(),
            "chunk_end": pd.to_datetime(df[("chunk", "end_date")]).to_numpy(),
            "metric": f"estimated_{metric}",
            "value": part["value"].astype(float).to_numpy(),
            "lower_bound": part["lower_confidence_boundary"].astype(float).to_numpy(),
            "upper_bound": part["upper_confidence_boundary"].astype(float).to_numpy(),
            "lower_threshold": part["lower_threshold"].astype(float).to_numpy(),
            "upper_threshold": part["upper_threshold"].astype(float).to_numpy(),
            "alert": part["alert"].astype(bool).to_numpy(),
        }))
    return pl.concat(rows).with_columns(pl.col("chunk_start", "chunk_end").cast(pl.Datetime("us")))


def flag_degradation(estimates: pl.DataFrame, reference_level: float,
                     tolerance: float) -> pl.DataFrame:
    """Mark chunks whose estimate sits more than `tolerance` below the reference.

    The complement to NannyML's own alert. Its band is +/- 3 sigma of the
    reference CHUNKS, and with ~200 frauds a month the chunk-to-chunk noise in
    AUC-PR is wide enough to swallow a real 20% drop. Comparing with the
    reference period's level asks the operational question directly: is this
    model meaningfully worse than when it was approved?
    """
    return estimates.with_columns(
        (pl.col("value") / reference_level - 1.0).alias("relative_change"),
        (pl.col("value") < (1.0 - tolerance) * reference_level).alias("degraded"),
    )


# --- realized ------------------------------------------------------------------

def _metric(name: str, y: np.ndarray, s: np.ndarray) -> float:
    if y.size == 0 or y.min() == y.max():
        return float("nan")  # no positives (2020) or nothing labelled yet
    return float(average_precision_score(y, s) if name == "average_precision"
                 else roc_auc_score(y, s))


def realized_performance(frame: pl.DataFrame, params: dict,
                         as_of: dt.datetime | None = None) -> pl.DataFrame:
    """Per-chunk metrics from labels. `as_of=None` is hindsight (every label);
    with `as_of`, only rows whose label exists by then (needs `labeled_at`).

    `label_coverage` says how much of the chunk that was -- the honest caveat
    on any metric computed from a partly labelled month.
    """
    chunked = _chunks(frame, params["monitoring"]["chunk_period"])
    rows = []
    for (start, end), part in chunked.group_by(["chunk_start", "chunk_end"], maintain_order=True):
        known = (part if as_of is None
                 else part.filter(pl.col("labeled_at") <= as_of))
        y = known["label"].to_numpy()
        s = known["score"].to_numpy()
        for metric in METRICS:
            rows.append({
                "chunk_start": start, "chunk_end": end,
                "metric": f"realized_{metric}",
                "value": _metric(metric, y, s),
                "label_coverage": known.height / part.height,
                "positives_known": int(y.sum()),
            })
    return pl.DataFrame(rows).sort("chunk_start", "metric")


def first_label_based_view(frame: pl.DataFrame, params: dict,
                           horizon_days: int = 180) -> pl.DataFrame:
    """For each month: the first day its labels are complete enough to trust.

    "Complete enough" is `monitoring.min_label_coverage` of the month's rows
    labelled. Until then a label-based metric is computed on a biased subset:
    frauds surface first and legitimate rows only when their dispute window
    closes, so an early partial AP is not a smaller version of the real one.
    """
    cfg = params["monitoring"]
    chunked = _chunks(frame, cfg["chunk_period"])
    out = []
    for (start, end), part in chunked.group_by(["chunk_start", "chunk_end"], maintain_order=True):
        arrivals = np.sort(part["labeled_at"].to_numpy())
        need = int(np.ceil(cfg["min_label_coverage"] * part.height))
        ready = arrivals[need - 1] if need <= arrivals.size else None
        ready_dt = pd.Timestamp(ready).to_pydatetime() if ready is not None else None
        out.append({
            "chunk_start": start, "chunk_end": end,
            "labels_ready_at": ready_dt,
            "days_after_month_end": (
                (ready_dt - end).total_seconds() / 86_400 if ready_dt else None
            ),
        })
    return pl.DataFrame(out)


# --- drift ---------------------------------------------------------------------

def univariate_drift(reference: pl.DataFrame, analysis: pl.DataFrame,
                     params: dict) -> pl.DataFrame:
    """Jensen-Shannon distance per feature per chunk (score included)."""
    import nannyml as nml

    cfg = params["monitoring"]
    continuous = [*cfg["drift"]["continuous"], "score"]
    categorical = list(cfg["drift"]["categorical"])
    calc = nml.UnivariateDriftCalculator(
        column_names=continuous + categorical,
        treat_as_categorical=categorical,
        timestamp_column_name="ts",
        continuous_methods=["jensen_shannon"],
        categorical_methods=["jensen_shannon"],
        chunk_period=cfg["chunk_period"],
    )
    calc.fit(reference.drop("label", "labeled_at", strict=False).to_pandas())
    df = calc.calculate(analysis.drop("label", "labeled_at", strict=False).to_pandas()).filter(
        period="analysis").to_df().sort_index(axis=1)

    rows = []
    for feature in continuous + categorical:
        part = df[(feature, "jensen_shannon")]
        rows.append(pl.DataFrame({
            "chunk_start": pd.to_datetime(df[("chunk", "chunk", "start_date")]).to_numpy(),
            "chunk_end": pd.to_datetime(df[("chunk", "chunk", "end_date")]).to_numpy(),
            "metric": "jensen_shannon",
            "feature": feature,
            "value": part["value"].astype(float).to_numpy(),
            "upper_threshold": part["upper_threshold"].astype(float).to_numpy(),
            "alert": part["alert"].astype(bool).to_numpy(),
        }))
    return pl.concat(rows).with_columns(pl.col("chunk_start", "chunk_end").cast(pl.Datetime("us")))


def multivariate_drift(reference: pl.DataFrame, analysis: pl.DataFrame,
                       params: dict) -> pl.DataFrame:
    """PCA reconstruction error over the continuous drift inputs, per chunk.

    Catches a change in how features move together that no single
    feature's distribution shows.
    """
    import nannyml as nml

    cfg = params["monitoring"]
    columns = list(cfg["drift"]["continuous"])
    calc = nml.DataReconstructionDriftCalculator(
        column_names=columns, timestamp_column_name="ts",
        chunk_period=cfg["chunk_period"],
    )
    calc.fit(reference.select("ts", *columns).to_pandas())
    df = calc.calculate(analysis.select("ts", *columns).to_pandas()).filter(
        period="analysis").to_df()
    part = df["reconstruction_error"]
    return pl.DataFrame({
        "chunk_start": pd.to_datetime(df[("chunk", "start_date")]).to_numpy(),
        "chunk_end": pd.to_datetime(df[("chunk", "end_date")]).to_numpy(),
        "metric": "reconstruction_error",
        "value": part["value"].astype(float).to_numpy(),
        "upper_threshold": part["upper_threshold"].astype(float).to_numpy(),
        "lower_threshold": part["lower_threshold"].astype(float).to_numpy(),
        "alert": part["alert"].astype(bool).to_numpy(),
    }).with_columns(pl.col("chunk_start", "chunk_end").cast(pl.Datetime("us")))
