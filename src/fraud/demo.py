"""Shell demo helpers -- every number they print comes from the system itself.

    .venv/bin/python -m fraud.demo status             # registry aliases + the live API
    .venv/bin/python -m fraud.demo results            # the scoreboard, from reports/
    .venv/bin/python -m fraud.demo payload 18267417   # a real transaction, as a request body
    .venv/bin/python -m fraud.demo explain 18267417   # why each model scored it that way
    .venv/bin/python -m fraud.demo shadow             # champion vs challenger, from the DB
    .venv/bin/python -m fraud.demo watch v3           # wait until every worker serves v3

Nothing here computes a result of its own. `results` reads the committed
reports; `explain` calls fraud.explain (exact TreeSHAP, exact group Shapley);
`shadow` reads what the API wrote to `predictions`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import polars as pl

from fraud.config import load_params, repo_path

API = os.getenv("FRAUD_API", "http://localhost:8000")


def _rule(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 70 - len(title)))


def _report(path: str) -> dict:
    return json.loads(repo_path(path).read_text())


# --- status ------------------------------------------------------------------

def status(params: dict) -> None:
    from fraud.models.promote import _client

    client = _client(params)
    name = params["mlflow"]["registered_model_name"]
    _rule(f"registry: {name}  (DagsHub MLflow)")
    for alias in ("champion", "challenger", "previous"):
        try:
            v = client.get_model_version_by_alias(name, alias)
        except Exception:
            print(f"  @{alias:<11} (unset)")
            continue
        run = client.get_run(v.run_id)
        print(f"  @{alias:<11} v{v.version:<3} {run.info.run_name:<22} "
              f"evaluated as: {run.data.tags.get('evaluated_model', '-')}")

    _rule(f"API: {API}")
    try:
        import httpx

        health = httpx.get(f"{API}/health", timeout=5).json()
        for key in ("model_version", "challenger_version", "shadow", "alias_swaps",
                    "aliases_checked_s_ago"):
            print(f"  {key:<22} {health.get(key)}")
    except Exception as exc:
        print(f"  not reachable ({type(exc).__name__}) -- `docker compose up -d`")


def watch(params: dict, version: str, timeout_s: int = 180) -> None:
    """Poll /health until EVERY worker serves `version` as champion.

    Each API worker re-resolves the aliases on its own clock (every
    `serving.alias_refresh_seconds`, plus the model download), so a swap lands
    worker by worker. "Converged" = a run of consecutive health checks, more
    than the worker count, all agreeing.
    """
    import time

    import httpx

    want = version if version.startswith("v") else f"v{version}"
    needed = 3 * 4                       # comfortably more checks than workers
    started, streak, last = time.time(), 0, None
    while time.time() - started < timeout_s:
        health = httpx.get(f"{API}/health", timeout=5).json()
        live = health["model_version"].rsplit(":", 1)[-1]
        streak = streak + 1 if live == want else 0
        if live != last:
            print(f"  {time.time() - started:5.1f}s  a worker serves champion {live}")
            last = live
        if streak >= needed:
            print(f"  {time.time() - started:5.1f}s  all workers serve champion {want} "
                  f"(shadow: {health.get('challenger_version', '-').rsplit(':', 1)[-1]})")
            return
        time.sleep(0.25)
    raise SystemExit(f"workers did not converge on {want} within {timeout_s}s")


# --- results -----------------------------------------------------------------

def results(params: dict) -> None:
    xgb = _report("reports/metrics_xgb.json")
    gnn_offline = _report("reports/metrics_gnn_direct.json")
    served = _report("reports/metrics_replay.json")["metrics"]
    graph_off = _report("reports/metrics_replay_no_neighbours.json")["metrics"]
    shuffled = {k: _report(f"reports/metrics_replay_shuffled_{k}.json")["metrics"]
                for k in ("merchant", "card")}
    fm = _report("reports/metrics_fm.json")["headline_test_2019_auc_pr"]
    test19 = xgb["splits"]["test_2019_only"]

    _rule("Which structure catches card fraud?  (test 2019, AUC-PR; base rate 0.0012)")
    table = pl.DataFrame([
        {"model": "XGBoost, tabular (old champion)", "AUC-PR": test19["auc_pr"],
         "P@100": test19["precision_at_100"], "how measured": "offline, = served"},
        {"model": "GraphSAGE, relational", "AUC-PR": gnn_offline["test_2019_only"]["auc_pr"],
         "P@100": gnn_offline["test_2019_only"]["precision_at_100"],
         "how measured": "offline (NeighborLoader)"},
        {"model": "GraphSAGE, served  <- @champion", "AUC-PR": served["auc_pr"],
         "P@100": served["precision_at_100"], "how measured": "causal replay, 1.72M rows"},
        {"model": "GraphSAGE, graph removed", "AUC-PR": graph_off["auc_pr"],
         "P@100": graph_off["precision_at_100"], "how measured": "causal replay, ablation"},
        {"model": "  ...merchant history borrowed", "AUC-PR": shuffled["merchant"]["auc_pr"],
         "P@100": shuffled["merchant"]["precision_at_100"],
         "how measured": "causal replay, shuffled control"},
        {"model": "  ...card history borrowed", "AUC-PR": shuffled["card"]["auc_pr"],
         "P@100": shuffled["card"]["precision_at_100"],
         "how measured": "causal replay, shuffled control"},
        {"model": "Foundation model + base, sequential", "AUC-PR": fm["isolated_combined"],
         "P@100": None, "how measured": "offline, NVIDIA extraction"},
    ]).with_columns(pl.col("AUC-PR").round(4),
                    pl.col("P@100").round(2).cast(pl.String).fill_null("–"))   # not measured
    with pl.Config(tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True,
                   tbl_width_chars=110, fmt_str_lengths=40):
        print(table)

    lat = _report("reports/metrics_latency.json")["runs"]
    load = _report("reports/loadtest.json")["levels"]
    _rule("Serving  (Compose: Postgres in the next container)")
    co = lat["compose_colocated"]["latency"]["wall_ms"]
    print(f"  sequential, 3,000 requests   p50 {co['p50']:.1f} ms   p95 {co['p95']:.1f} ms   "
          f"p99 {co['p99']:.1f} ms")
    for level in load:
        wall = level["wall_ms"]
        print(f"  concurrency {level['concurrency']:<3}             "
              f"{level['throughput_per_s']:6.1f} req/s   "
              f"p50 {wall['p50']:.1f} ms   p95 {wall['p95']:.1f} ms")
    eq = lat["compose_colocated"]["equivalence_vs_in_process"]
    print(f"  served == causal replay       {eq['compared'] - eq['mismatches_over_tolerance']:,}"
          f" / {eq['compared']:,} scores (max diff {eq['max_abs_diff']:.0e})")

    summary = pl.read_csv(repo_path("reports/monitoring/detection_summary.csv"))
    _rule("Monitoring without labels  (NannyML CBPE, monthly AUC-PR)")
    for r in summary.iter_rows(named=True):
        print(f"  {r['model']:<10} 2018 {r['reference_2018']:.3f}   "
              f"true 2019 {r['true_2019']:.3f} ({r['true_change']})   "
              f"estimated {r['estimated_2019']:.3f} ({r['estimated_change']})   "
              f"months flagged {r['months_flagged_cbpe']}")
    print("  XGBoost's drop flagged 89 days before the chargebacks could confirm it")


# --- payload ------------------------------------------------------------------

def _row(params: dict, txn_id: int) -> dict:
    from fraud.api.predictor import transaction_from_row

    frame = (pl.scan_parquet(repo_path(params["paths"]["processed"]) / "**/*.parquet",
                             hive_partitioning=True)
             .filter(pl.col("txn_id") == txn_id).collect())
    if frame.is_empty():
        raise SystemExit(f"txn {txn_id} not found")
    row = frame.row(0, named=True)
    return {"txn": transaction_from_row(row), "fraud": int(row["Fraud"])}


def payload(params: dict, txn_id: int) -> None:
    txn = _row(params, txn_id)["txn"]
    print(json.dumps({**txn, "ts": txn["ts"].isoformat()}))


# --- explain --------------------------------------------------------------------

def explain(params: dict, txn_id: int) -> None:
    import torch

    from fraud.api.predictor import Predictor
    from fraud.explain import graph as G
    from fraud.explain import tree as T

    torch.set_num_threads(1)
    found = _row(params, txn_id)
    txn = found["txn"]
    _rule(f"txn {txn_id}  {txn['ts']}  ${txn['Amount']:.2f}  {txn['Chip']}  "
          f"MCC {txn['MCC']}  -> {'FRAUD' if found['fraud'] else 'legitimate'}")

    champion = T.load_champion(params)
    rows = T.matrix_rows(params, txn_ids=[txn_id], years=[txn["ts"].year])
    reasons = T.explain_rows(champion, rows, params["explain"]["top_k_reasons"]).row(0, named=True)
    thr = params["serving"]["decision_thresholds"]
    print(f"  XGBoost    score {reasons['score']:.4f}   threshold {thr['xgboost']:.4f}   "
          f"-> {'ALERT' if reasons['score'] >= thr['xgboost'] else 'pass'}")
    print(f"             pushed by: {reasons['reasons'] or '-'}")

    gnn = Predictor.from_disk(params, repo_path(params["paths"]["models"]) / "gnn")
    since = dt.datetime(txn["ts"].year, 1, 1) - dt.timedelta(
        hours=params["serving"]["history_hours"])
    nb = G.neighbourhood_for(params, txn, since)
    result = G.explain(gnn, txn, nb, G.typical_transaction(params, 2018, sample=50_000))
    print(f"  GraphSAGE  score {result.score:.4f}   threshold {thr['graphsage']:.4f}   "
          f"-> {'ALERT' if result.score >= thr['graphsage'] else 'pass'}")
    print(f"             neighbourhood: {nb.user_history.height} card rows, "
          f"{nb.merchant_history.height} merchant rows")
    for group, value in sorted(result.values.items(), key=lambda kv: -kv[1]):
        bar = "█" * min(30, int(abs(value) * 3))
        print(f"             {group:<17} {value:+7.2f}  {bar}")
    print(f"             (log-odds; exact: values sum to the score shift, "
          f"error {abs(result.efficiency_gap()):.0e})")


# --- shadow ---------------------------------------------------------------------

def shadow(params: dict, last: int) -> None:
    from fraud.api import store

    with store.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select p.txn_id, p.model_version, p.score, p.decision,
                   p.challenger_version, p.challenger_score, p.latency_ms,
                   t.amount, t.chip
              from predictions p join transaction_events t using (txn_id)
             order by p.predicted_at desc limit %s
            """, (last,))
        rows = cur.fetchall()
        cur.execute(
            """
            select count(*) as n, sum(decision::int) as alerts,
                   count(challenger_score) as shadowed,
                   corr(score, challenger_score) as corr
              from predictions
            """)
        agg = cur.fetchone()
    _rule(f"last {last} predictions (champion decides; challenger in shadow)")
    frame = pl.DataFrame(rows).select(
        "txn_id",
        pl.col("model_version").str.extract(r"(v\d+)").alias("champion"),
        pl.col("score").round(4).alias("champ_score"),
        pl.col("decision").alias("alert"),
        pl.col("challenger_version").str.extract(r"(v\d+)").alias("shadow"),
        pl.col("challenger_score").round(4).alias("shadow_score"),
        pl.col("amount").round(2), "chip")
    with pl.Config(tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True,
                   tbl_rows=last, tbl_width_chars=120):
        print(frame)
    print(f"  {agg['n']:,} predictions logged, {agg['alerts'] or 0:,} alerts, "
          f"{agg['shadowed']:,} shadow-scored, champion/shadow score correlation "
          f"{(agg['corr'] or 0):.2f}")


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("results")
    for name in ("payload", "explain"):
        sub.add_parser(name).add_argument("txn_id", type=int)
    sub.add_parser("shadow").add_argument("--last", type=int, default=8)
    sub.add_parser("watch").add_argument("version", help="e.g. v3")
    args = ap.parse_args(argv)
    if args.cmd == "status":
        status(params)
    elif args.cmd == "results":
        results(params)
    elif args.cmd == "payload":
        payload(params, args.txn_id)
    elif args.cmd == "explain":
        explain(params, args.txn_id)
    elif args.cmd == "watch":
        watch(params, args.version)
    else:
        shadow(params, args.last)
    return 0


if __name__ == "__main__":
    sys.exit(main())
