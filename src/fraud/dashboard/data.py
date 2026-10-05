"""Everything the presentation app shows, as plain data -- no Streamlit here.

The app (`app/streamlit_app.py`) only lays this out, and `fraud.demo` prints the
same functions, so the shell demo and the dashboard cannot disagree. Nothing in
this module computes a result of its own: numbers come from committed reports,
the live API, the registry, or the explain modules that tests hold exact.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import polars as pl

from fraud.config import repo_path


def _report(path: str) -> dict:
    return json.loads(repo_path(path).read_text())


def api_url(params: dict) -> str:
    return os.getenv("FRAUD_API", params["dashboard"]["api_url"])


def db_url(params: dict) -> str:
    return os.getenv("FRAUD_DB_URL", params["dashboard"]["db_url"])


def connect(params: dict):
    """A store connection to the DASHBOARD's database (Compose by default).

    `store.connect` reads DATABASE_URL, which .env points at Supabase; setting it
    here first wins, because load_dotenv never overrides an existing variable.
    """
    from fraud.api import store

    os.environ["DATABASE_URL"] = db_url(params)
    return store.connect()


# --- the answer ------------------------------------------------------------------

def scoreboard(params: dict) -> list[dict[str, Any]]:
    """Test 2019 AUC-PR for every model and control, from the committed reports."""
    xgb = _report("reports/metrics_xgb.json")["splits"]["test_2019_only"]
    offline = _report("reports/metrics_gnn_direct.json")["test_2019_only"]
    served = _report("reports/metrics_replay.json")["metrics"]
    off = _report("reports/metrics_replay_no_neighbours.json")["metrics"]
    shuffled = {k: _report(f"reports/metrics_replay_shuffled_{k}.json")["metrics"]
                for k in ("merchant", "card")}
    fm = _report("reports/metrics_fm.json")["headline_test_2019_auc_pr"]
    return [
        {"model": "XGBoost", "structure": "tabular", "auc_pr": xgb["auc_pr"],
         "p100": xgb["precision_at_100"], "measured": "offline = served", "kind": "model"},
        {"model": "GraphSAGE, offline", "structure": "relational",
         "auc_pr": offline["auc_pr"], "p100": offline["precision_at_100"],
         "measured": "offline graph", "kind": "model"},
        {"model": "GraphSAGE, served (@champion)", "structure": "relational",
         "auc_pr": served["auc_pr"], "p100": served["precision_at_100"],
         "measured": "causal replay, 1.72M rows", "kind": "champion"},
        {"model": "Foundation model + base", "structure": "sequential",
         "auc_pr": fm["isolated_combined"], "p100": None,
         "measured": "offline, NVIDIA extraction", "kind": "model"},
        {"model": "GraphSAGE, no neighbourhood", "structure": "control",
         "auc_pr": off["auc_pr"], "p100": off["precision_at_100"],
         "measured": "causal replay, ablation", "kind": "control"},
        {"model": "GraphSAGE, another merchant's history", "structure": "control",
         "auc_pr": shuffled["merchant"]["auc_pr"],
         "p100": shuffled["merchant"]["precision_at_100"],
         "measured": "causal replay, shuffled", "kind": "control"},
        {"model": "GraphSAGE, another card's history", "structure": "control",
         "auc_pr": shuffled["card"]["auc_pr"], "p100": shuffled["card"]["precision_at_100"],
         "measured": "causal replay, shuffled", "kind": "control"},
    ]


def base_rate() -> float:
    return _report("reports/metrics_replay.json")["metrics"]["base_rate"]


def serving_summary() -> dict[str, Any]:
    lat = _report("reports/metrics_latency.json")["runs"]
    co = lat["compose_colocated"]
    return {"colocated": co["latency"]["wall_ms"],
            "spans": co["latency"].get("spans_ms", {}),
            "equivalence": co["equivalence_vs_in_process"],
            "load": _report("reports/loadtest.json")["levels"]}


# --- monitoring -------------------------------------------------------------------

def monitoring_monthly(params: dict) -> pl.DataFrame:
    """Per model and 2019 month: CBPE's label-free estimate beside the TRUE AUC-PR.

    Estimates are what the monitoring job wrote (`drift_metrics`); true values
    are computed here from the committed scores and labels -- what you would only
    know once every chargeback had arrived.
    """
    from fraud.models.metrics import auc_pr

    est = (pl.read_parquet(repo_path("reports/monitoring/drift_metrics.parquet"))
           .filter(pl.col("metric") == "estimated_average_precision")
           .select("model", pl.col("chunk_start").dt.date().alias("month"),
                   pl.col("value").alias("estimate"), "alert"))
    sources = {m: params["monitoring"]["scores"][m]["analysis"] for m in ("xgboost", "graphsage")}
    months = (pl.scan_parquet(repo_path(params["paths"]["processed"]) / "**/*.parquet",
                              hive_partitioning=True)
              .filter(pl.col("Year") == 2019)
              .select("txn_id", pl.col("ts").dt.truncate("1mo").dt.date().alias("month"))
              .collect())
    frames = []
    for model, path in sources.items():
        z = np.load(repo_path(path))
        scores = pl.DataFrame({"txn_id": z["txn_id"], "score": z["score"], "label": z["label"]})
        joined = scores.join(months, on="txn_id")
        for (month,), g in joined.group_by("month"):
            frames.append({"model": model, "month": month,
                           "true": auc_pr(g["label"].to_numpy(), g["score"].to_numpy())})
    true = pl.DataFrame(frames)
    summary = pl.read_csv(repo_path("reports/monitoring/detection_summary.csv"))
    ref = dict(zip(summary["model"], summary["reference_2018"], strict=True))
    tol = params["monitoring"]["degradation_tolerance"]
    return (est.join(true, on=["model", "month"], how="full", coalesce=True)
            .with_columns(reference=pl.col("model").replace_strict(ref),
                          flag_line=pl.col("model").replace_strict(ref) * (1 - tol))
            .sort("model", "month"))


def detection_summary() -> pl.DataFrame:
    return pl.read_csv(repo_path("reports/monitoring/detection_summary.csv"))


# --- the retrain --------------------------------------------------------------------

def retrain_summary(params: dict) -> dict[str, Any]:
    """Step 2 of the retrain: every arm against v3 on all of 2018, plus curves
    where the (DVC-tracked) train_info files are present."""
    report = _report(params["gnn_causal"]["metrics"])
    graphs = _report(params["gnn_causal"]["summary"])
    curves = {}
    root = repo_path(params["gnn_causal"]["models_dir"])
    for run in report["runs"]:
        info = root / run / "train_info.json"
        if info.exists():
            curves[run] = [h["val_auc_pr"] for h in json.loads(info.read_text())["history"]]
    # Step 3: the winner's served 2019 number, once its causal replay exists.
    replay = repo_path("reports/metrics_replay_gnn_causal.json")
    test = None
    if replay.exists():
        served = json.loads(replay.read_text())
        test = {"model": served["model_version"], "auc_pr": served["metrics"]["auc_pr"],
                "p100": served["metrics"]["precision_at_100"],
                "v3_auc_pr": _report("reports/metrics_replay.json")["metrics"]["auc_pr"],
                "gate": params["promotion"]["min_auc_pr_gain"]}
        cmp = repo_path("reports/retrain_2019_comparison.json")
        if cmp.exists():
            test["bootstrap"] = json.loads(cmp.read_text())["paired_bootstrap"]
    return {"report": report, "graphs": graphs, "curves": curves, "test_2019": test}


# --- registry and the live API ---------------------------------------------------------

def registry_status(params: dict) -> list[dict[str, Any]]:
    from fraud.models.promote import _client

    client = _client(params)
    name = params["mlflow"]["registered_model_name"]
    out = []
    for alias in ("champion", "challenger", "previous"):
        try:
            v = client.get_model_version_by_alias(name, alias)
        except Exception:
            out.append({"alias": alias, "version": None})
            continue
        run = client.get_run(v.run_id)
        out.append({"alias": alias, "version": v.version, "run": run.info.run_name,
                    "evaluated_as": run.data.tags.get("evaluated_model", "-")})
    return out


def api_health(params: dict) -> dict[str, Any] | None:
    import httpx

    try:
        return httpx.get(f"{api_url(params)}/health", timeout=3).json()
    except Exception:
        return None


def live_version(health: dict | None, key: str = "model_version") -> str | None:
    return None if not health else str(health.get(key, "")).rsplit(":", 1)[-1] or None


def watch_swap(params: dict, want: str, seconds: float | None = None
               ) -> Iterator[tuple[float, str | None, int]]:
    """Yield (elapsed, champion a worker served, consecutive checks on `want`)
    until every worker serves `want` (a streak > the worker count) or time runs out."""
    seconds = seconds or params["dashboard"]["watch_seconds"]
    started, streak = time.time(), 0
    while time.time() - started < seconds:
        live = live_version(api_health(params))
        streak = streak + 1 if live == want else 0
        yield time.time() - started, live, streak
        if streak >= 12:
            return
        time.sleep(0.25)


# --- one transaction ---------------------------------------------------------------------

def transaction(params: dict, txn_id: int) -> dict[str, Any]:
    from fraud.api.predictor import transaction_from_row

    frame = (pl.scan_parquet(repo_path(params["paths"]["processed"]) / "**/*.parquet",
                             hive_partitioning=True)
             .filter(pl.col("txn_id") == txn_id).collect())
    if frame.is_empty():
        raise KeyError(f"txn {txn_id} not found")
    row = frame.row(0, named=True)
    return {"txn": transaction_from_row(row), "fraud": int(row["Fraud"])}


def score_via_api(params: dict, txn: dict[str, Any]) -> dict[str, Any]:
    """POST one transaction to the live API; body plus the Server-Timing spans."""
    import httpx

    from fraud.api.timing import parse_server_timing

    t0 = time.perf_counter()
    r = httpx.post(f"{api_url(params)}/predict", json={**txn, "ts": txn["ts"].isoformat()},
                   timeout=30)
    wall = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    return {**r.json(), "wall_ms": wall,
            "spans": parse_server_timing(r.headers.get("server-timing", ""))}


def shadow_score(params: dict, txn_id: int, wait_s: float = 3.0) -> dict[str, Any] | None:
    """The challenger's score for the latest prediction of `txn_id` -- written by
    the API AFTER the response, so poll briefly."""
    deadline = time.time() + wait_s
    while True:
        with connect(params) as conn, conn.cursor() as cur:
            cur.execute("""select challenger_version, challenger_score from predictions
                            where txn_id = %s order by predicted_at desc limit 1""", (txn_id,))
            row = cur.fetchone()
        if (row and row["challenger_score"] is not None) or time.time() > deadline:
            return row
        time.sleep(0.2)


def history_count(params: dict, txn: dict[str, Any]) -> int:
    """Rows the store holds for this card strictly before the transaction --
    0 means the replay has not seeded its past, and the score would be a cold start."""
    with connect(params) as conn, conn.cursor() as cur:
        cur.execute("select count(*) as n from transaction_events where user_id = %s and ts < %s",
                    (txn["User"], txn["ts"]))
        return int(cur.fetchone()["n"])


def explain_both(params: dict, txn_id: int, models: dict | None = None) -> dict[str, Any]:
    """Exact reasons from both models (TreeSHAP; GraphSAGE group Shapley).

    `models` caches the loaded boosters/predictor and the typical transaction
    between calls (the app keeps it in st.cache_resource).
    """
    import torch

    from fraud.explain import graph as G
    from fraud.explain import tree as T

    torch.set_num_threads(1)
    models = models if models is not None else load_explainers(params)
    found = transaction(params, txn_id)
    txn = found["txn"]
    rows = T.matrix_rows(params, txn_ids=[txn_id], years=[txn["ts"].year])
    reasons = T.explain_rows(models["xgb"], rows, params["explain"]["top_k_reasons"]).row(
        0, named=True)
    since = dt.datetime(txn["ts"].year, 1, 1) - dt.timedelta(
        hours=params["serving"]["history_hours"])
    nb = G.neighbourhood_for(params, txn, since)
    result = G.explain(models["gnn"], txn, nb, models["typical"])
    thr = params["serving"]["decision_thresholds"]
    return {
        "txn": txn, "fraud": found["fraud"],
        "xgboost": {"score": reasons["score"], "threshold": thr["xgboost"],
                    "reasons": reasons["reasons"]},
        "graphsage": {"score": result.score, "threshold": thr["graphsage"],
                      "groups": dict(sorted(result.values.items(), key=lambda kv: -kv[1])),
                      "card_rows": nb.user_history.height,
                      "merchant_rows": nb.merchant_history.height,
                      "efficiency_gap": abs(result.efficiency_gap())},
    }


def load_explainers(params: dict) -> dict[str, Any]:
    from fraud.api.predictor import Predictor
    from fraud.explain import graph as G
    from fraud.explain import tree as T

    return {"xgb": T.load_champion(params),
            "gnn": Predictor.from_disk(params, repo_path(params["paths"]["models"]) / "gnn"),
            "typical": G.typical_transaction(params, 2018, sample=50_000)}


# --- predictions and the live replay ----------------------------------------------------

def recent_predictions(params: dict, last: int = 10) -> tuple[pl.DataFrame, dict[str, Any]]:
    with connect(params) as conn, conn.cursor() as cur:
        cur.execute("""
            select p.txn_id, p.model_version, p.score, p.decision, p.challenger_version,
                   p.challenger_score, t.amount, t.chip
              from predictions p join transaction_events t using (txn_id)
             order by p.predicted_at desc limit %s""", (last,))
        rows = cur.fetchall()
        cur.execute("""select count(*) as n, sum(decision::int) as alerts,
                              count(challenger_score) as shadowed,
                              corr(score, challenger_score) as corr from predictions""")
        agg = cur.fetchone()
    return pl.DataFrame(rows), agg


def _is_supabase(url: str) -> bool:
    return "supabase" in url.lower()


def prepare_replay(params: dict, start: str | None = None, limit: int | None = None
                   ) -> tuple[pl.DataFrame, int]:
    """Reset the dashboard's store and seed exactly the slice's past
    (`replay.seed_store`, the HTTP replay's own seeding). Returns (rows, seeded)."""
    from fraud.jobs.replay import http_slice, seed_store

    if _is_supabase(db_url(params)):
        raise RuntimeError("refusing to reset a Supabase store from the dashboard")
    cfg = params["dashboard"]
    rows, graph_since = http_slice(params, limit or cfg["replay_limit"],
                                   start or cfg["replay_start"])
    with connect(params) as conn:
        seeded = seed_store(params, conn, rows, graph_since, reset=True)
    return rows, seeded


def replay_stream(params: dict, rows: pl.DataFrame) -> Iterator[dict[str, Any]]:
    """Send `rows` to the live API in time order, one request at a time (each
    request inserts its row after scoring -- the order IS the causality control)."""
    import httpx

    from fraud.api.predictor import transaction_from_row
    from fraud.api.timing import parse_server_timing

    with httpx.Client(base_url=api_url(params), timeout=60) as client:
        for row in rows.iter_rows(named=True):
            txn = transaction_from_row(row)
            t0 = time.perf_counter()
            r = client.post("/predict", json={**txn, "ts": txn["ts"].isoformat()})
            wall = (time.perf_counter() - t0) * 1000.0
            r.raise_for_status()
            body = r.json()
            spans = parse_server_timing(r.headers.get("server-timing", ""))
            yield {"txn_id": txn["txn_id"], "ts": txn["ts"], "amount": txn["Amount"],
                   "chip": txn["Chip"], "fraud": int(row["Fraud"]), "score": body["score"],
                   "alert": body["decision"], "wall_ms": wall,
                   **{f"{k}_ms": v["dur"] for k, v in spans.items()}}
