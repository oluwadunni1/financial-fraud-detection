"""Champion promotion and rollback: alias moves behind a gate.

    python -m fraud.models.promote                      # dry run: would it pass?
    python -m fraud.models.promote --apply              # move the aliases
    python -m fraud.models.promote --rollback --apply   # @champion <- @previous

The gate, in order -- any failure stops it, and nothing moves:

1. **The registry holds the evaluated weights.** Both the candidate and the
   current champion are loaded THROUGH the registry and re-score a sample of
   2019 transactions; each must reproduce its evaluated scores. Checked by
   scores, not by name, because names lied twice: @challenger v2 was an
   unevaluated epoch, and @champion v1 was the pre-velocity-fix model
   (decision 23).
2. **The evidence exists** -- e.g. GraphSAGE needs its graph-off ablation.
3. **The candidate is better where it counts:** 2019 AUC-PR of its evaluated
   scores beats the champion's by `promotion.min_auc_pr_gain`.

On success: @previous <- old champion, @champion <- candidate, @challenger <-
old champion (so it keeps scoring in shadow and is one command from a
rollback). The API picks the change up within `alias_refresh_seconds` -- no
redeploy. Every attempt, passed or refused, is logged as an MLflow run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import polars as pl

from fraud.config import REPO_ROOT, load_params, repo_path
from fraud.models.metrics import auc_pr


def _client(params: dict):
    import mlflow
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    return mlflow.tracking.MlflowClient()


def family(params: dict, alias: str) -> str:
    """'xgboost' or 'graphsage', read from how the version was logged."""
    import mlflow

    # Point MLflow at DagsHub FIRST: unset, it silently falls back to a local
    # ./mlflow.db and reports that the registered model does not exist.
    _client(params)
    name = params["mlflow"]["registered_model_name"]
    flavours = mlflow.models.get_model_info(f"models:/{name}@{alias}").flavors
    if "xgboost" in flavours:
        return "xgboost"
    if "pytorch" in flavours:
        return "graphsage"
    raise RuntimeError(f"unknown model family for @{alias}: {sorted(flavours)}")


def reference_scores(params: dict, model_family: str) -> pl.DataFrame:
    z = np.load(repo_path(params["promotion"]["references"][model_family]))
    return pl.DataFrame({"txn_id": z["txn_id"], "score": z["score"], "label": z["label"]})


def verify_registry(params: dict, alias: str, model_family: str) -> dict:
    """Re-score sampled 2019 transactions through the registry; compare."""
    import torch

    from fraud.api.predictor import load_scorer, transaction_from_row
    from fraud.explain.graph import neighbourhood_for

    torch.set_num_threads(1)
    cfg = params["promotion"]
    scorer = load_scorer(params, alias)
    reference = reference_scores(params, model_family)
    sample = reference.sample(cfg["verify_rows"], seed=7)["txn_id"].to_list()
    rows = (pl.scan_parquet(repo_path(params["paths"]["processed"]) / "**/*.parquet",
                            hive_partitioning=True)
            .filter(pl.col("txn_id").is_in(sample)).collect().sort("ts"))
    expected = dict(zip(reference["txn_id"].to_list(), reference["score"].to_list(),
                        strict=True))
    since = dt.datetime(2019, 1, 1) - dt.timedelta(hours=params["serving"]["history_hours"])
    diffs = []
    for row in rows.iter_rows(named=True):
        txn = transaction_from_row(row)
        got = scorer.score(txn, neighbourhood_for(params, txn, since)).score
        diffs.append(abs(got - expected[txn["txn_id"]]))
    worst = float(max(diffs))
    return {"alias": alias, "version": scorer.model_version, "family": model_family,
            "rows": len(diffs), "max_abs_diff": worst,
            "passed": worst <= cfg["verify_tolerance"]}


def gate(params: dict, candidate_alias: str) -> dict:
    cfg = params["promotion"]
    champ_family = family(params, params["serving"]["champion_alias"])
    cand_family = family(params, candidate_alias)
    report: dict = {"candidate_alias": candidate_alias, "checks": []}

    for alias, fam in ((candidate_alias, cand_family),
                       (params["serving"]["champion_alias"], champ_family)):
        check = verify_registry(params, alias, fam)
        report["checks"].append({"check": f"registry @{alias} reproduces its evaluation", **check})
        if not check["passed"]:
            report["passed"] = False
            report["reason"] = (f"@{alias} ({check['version']}) does not reproduce its "
                                f"evaluated scores: max diff {check['max_abs_diff']:.2e}")
            return report

    missing = [p for p in cfg["evidence"].get(cand_family, []) if not repo_path(p).exists()]
    report["checks"].append({"check": "evidence present", "missing": missing,
                             "passed": not missing})
    if missing:
        report.update(passed=False, reason=f"missing evidence: {missing}")
        return report

    def year_auc(fam: str) -> float:
        ref = reference_scores(params, fam)
        return auc_pr(ref["label"].to_numpy(), ref["score"].to_numpy())

    cand, champ = year_auc(cand_family), year_auc(champ_family)
    gain = cand - champ
    report["checks"].append({"check": "beats the champion", "candidate_auc_pr": cand,
                             "champion_auc_pr": champ, "gain": gain,
                             "required": cfg["min_auc_pr_gain"],
                             "passed": gain >= cfg["min_auc_pr_gain"]})
    report["passed"] = gain >= cfg["min_auc_pr_gain"]
    if not report["passed"]:
        report["reason"] = f"gain {gain:+.4f} below the required {cfg['min_auc_pr_gain']}"
    return report


def _move(client, name: str, moves: dict[str, str]) -> None:
    for alias, version in moves.items():
        client.set_registered_model_alias(name, alias, version)


def promote(params: dict, candidate_alias: str, apply: bool) -> dict:
    client = _client(params)
    name = params["mlflow"]["registered_model_name"]
    champion_alias = params["serving"]["champion_alias"]
    old = client.get_model_version_by_alias(name, champion_alias).version
    new = client.get_model_version_by_alias(name, candidate_alias).version
    if old == new:
        return {"passed": False, "reason": f"@{candidate_alias} is already @champion (v{new})"}

    report = gate(params, candidate_alias)
    report.update(champion_before=old, candidate=new, applied=False)
    if report["passed"] and apply:
        _move(client, name, {"previous": old, champion_alias: new,
                             params["serving"]["challenger_alias"]: old})
        report["applied"] = True
    _audit(params, "promotion", report)
    return report


def champion_alerting(params: dict) -> dict:
    """Is the live champion in a PERSISTENT degradation alert right now?

    Reads what the monitoring job wrote (decision 25): the newest monthly
    estimate for the champion's model family, from the newest run. The alert
    flag there already requires `persistence_months` consecutive flagged months.
    """
    from fraud.api import store

    fam = family(params, params["serving"]["champion_alias"])
    with store.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select chunk_start, value, alert, run_at
              from drift_metrics
             where model = %(model)s and metric = 'estimated_average_precision'
             order by run_at desc, chunk_start desc
             limit 1
            """,
            {"model": fam},
        )
        row = cur.fetchone()
    if row is None:
        return {"family": fam, "alerting": False, "reason": "no monitoring rows yet"}
    return {"family": fam, "alerting": bool(row["alert"]), "month": str(row["chunk_start"]),
            "estimate": row["value"], "run_at": str(row["run_at"])}


def rollback(params: dict, apply: bool, if_alerting: bool = False) -> dict:
    """@champion <- @previous; the demoted model keeps scoring in shadow.

    With `if_alerting`, only when the monitoring job has the champion in a
    persistent alert -- the automatic path the scheduled workflow takes.
    """
    if if_alerting:
        status = champion_alerting(params)
        if not status["alerting"]:
            return {"applied": False, "skipped": "champion not alerting", **status}
    client = _client(params)
    name = params["mlflow"]["registered_model_name"]
    champion_alias = params["serving"]["champion_alias"]
    current = client.get_model_version_by_alias(name, champion_alias).version
    previous = client.get_model_version_by_alias(name, "previous").version
    report = {"champion_before": current, "champion_after": previous, "applied": False}
    if apply:
        _move(client, name, {champion_alias: previous, "previous": current,
                             params["serving"]["challenger_alias"]: current})
        report["applied"] = True
    _audit(params, "rollback", report)
    return report


def _audit(params: dict, kind: str, report: dict) -> None:
    """Every attempt is a run: who was champion, what was checked, what moved."""
    import mlflow

    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"{kind}-{'applied' if report.get('applied') else 'dry'}"):
        mlflow.set_tags({"kind": kind, "applied": str(report.get("applied", False)),
                         "passed": str(report.get("passed", True))})
        mlflow.log_dict(report, f"{kind}.json")


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", default=params["serving"]["challenger_alias"])
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--if-alerting", action="store_true",
                    help="with --rollback: only if monitoring has the champion alerting")
    ap.add_argument("--apply", action="store_true", help="move aliases (default: dry run)")
    args = ap.parse_args(argv)

    report = (rollback(params, args.apply, args.if_alerting) if args.rollback
              else promote(params, args.candidate, args.apply))
    out = repo_path("reports/rollback.json" if args.rollback else "reports/promotion.json")
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("passed", True) else 1


if __name__ == "__main__":
    sys.exit(main())
