"""How far can the hot store's history lag before GraphSAGE gets worse?

ARCHITECTURE 6.2 planned this as "AUC-PR vs embedding age". Phase 4 changed the
question: GraphSAGE no longer serves precomputed embeddings, it reads the card's
and merchant's recent rows from Postgres on every request. What can go stale is
the STORE -- if ingest runs Δ behind, a request sees history only up to now − Δ,
and both its velocity features and its graph neighbourhood lose the most recent
transactions. That is the number an ingest SLA should come from.

The replay is the ordinary causal replay (`jobs/replay.py`) with one change: a
row becomes visible only once it is `lag` old (`CausalHistory(visibility_lag_hours)`).
Lag 0 is serving as built, and is checked against `reports/metrics_replay.npz`
row for row before any other lag is trusted.

Differences between lags are judged with a PAIRED bootstrap -- the same
transactions resampled for both lags -- because with ~700 frauds in the slice
the noise on AUC-PR alone is larger than the effects being measured.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time

import numpy as np
import polars as pl

from fraud.config import load_params, repo_path
from fraud.models.metrics import auc_pr, precision_at_k


def load_slice(params: dict, start: str, end: str, limit: int | None = None) -> pl.DataFrame:
    """Processed rows in [start, end), chronological -- the replay's own ordering."""
    from fraud.jobs.replay import load_split_rows

    lo, hi = dt.datetime.fromisoformat(start), dt.datetime.fromisoformat(end)
    rows = load_split_rows(params, sorted({lo.year, (hi - dt.timedelta(seconds=1)).year}), None)
    rows = rows.filter((pl.col("ts") >= lo) & (pl.col("ts") < hi))
    return rows.head(limit) if limit else rows


def graph_bound(params: dict, year: int) -> dt.datetime:
    """Where the full-year replay's history began: the year's first row minus one
    velocity window. Using the same bound is what makes lag 0 reproducible."""
    first = (
        pl.scan_parquet(repo_path(params["paths"]["processed"]) / "**/*.parquet",
                        hive_partitioning=True)
        .filter(pl.col("Year") == year).select(pl.col("ts").min()).collect().item()
    )
    return first - dt.timedelta(hours=params["serving"]["history_hours"])


def paired_bootstrap(labels: np.ndarray, base: np.ndarray, other: np.ndarray,
                     n: int, seed: int = 0) -> dict:
    """AUC-PR(other) − AUC-PR(base) on the same resampled transactions."""
    rng = np.random.default_rng(seed)
    deltas = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, labels.size, labels.size)
        deltas[i] = auc_pr(labels[idx], other[idx]) - auc_pr(labels[idx], base[idx])
    return {
        "delta": float(auc_pr(labels, other) - auc_pr(labels, base)),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_worse": float((deltas < 0).mean()),
    }


def run(params: dict, lags: list[float], rows: pl.DataFrame, shards: int,
        bootstrap: int, check_reference: bool = True, lag_applies_to: str = "both",
        baseline: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    from fraud.jobs.replay import replay_sharded

    model_dir = repo_path(params["paths"]["models"]) / "gnn"
    since = graph_bound(params, rows["ts"].min().year)
    scores: dict[float, np.ndarray] = {}
    labels = txn_ids = None
    timings = {}
    for lag in lags:
        started = time.perf_counter()
        print(f"lag {lag:g} h: replaying {rows.height:,} rows on {shards} shards", flush=True)
        result = replay_sharded(rows, params, [], model_dir, shards,
                                lag_hours=lag, graph_since=since,
                                lag_applies_to=lag_applies_to)
        scores[lag] = result["scores"]
        labels, txn_ids = result["labels"], result["txn_ids"]
        timings[lag] = round(time.perf_counter() - started, 1)

    report: dict = {
        "slice": {"start": str(rows["ts"].min()), "end": str(rows["ts"].max()),
                  "rows": rows.height, "frauds": int(labels.sum())},
        "graph_since": str(since),
        "lag_applies_to": lag_applies_to,
        "seconds": timings,
    }
    if check_reference and baseline is None:
        ref = np.load(repo_path("reports/metrics_replay.npz"))
        lookup = dict(zip(ref["txn_id"].tolist(), ref["score"].tolist(), strict=True))
        expected = np.array([lookup[t] for t in txn_ids.tolist()])
        diff = np.abs(scores[0.0 if 0.0 in scores else lags[0]] - expected)
        report["lag0_vs_full_replay"] = {"max_abs_diff": float(diff.max()),
                                         "mismatches_over_1e-6": int((diff > 1e-6).sum())}

    if baseline is not None:
        # lag 0 from an earlier run, aligned by txn_id -- it does not depend on
        # which consumer a lag is applied to, so it need not be replayed again.
        lookup = dict(zip(baseline[0].tolist(), baseline[1].tolist(), strict=True))
        base = np.array([lookup[t] for t in txn_ids.tolist()])
        scores = {0.0: base, **scores}
        lags = [0.0, *lags]
    base = scores[lags[0]]
    report["lags"] = [
        {
            "lag_hours": lag,
            "auc_pr": auc_pr(labels, scores[lag]),
            "precision_at_100": precision_at_k(labels, scores[lag], 100),
            "precision_at_1000": precision_at_k(labels, scores[lag], 1000),
            **({"vs_lag0": paired_bootstrap(labels, base, scores[lag], bootstrap)}
               if lag != lags[0] else {}),
        }
        for lag in lags
    ]
    report["scores"] = {"txn_id": txn_ids, "label": labels,
                        **{f"lag_{lag:g}h": s for lag, s in scores.items()}}
    return report


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    cfg = params["staleness"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (smoke)")
    ap.add_argument("--lags", type=float, nargs="+", default=None)
    ap.add_argument("--shards", type=int, default=cfg["shards"])
    ap.add_argument("--output", default=cfg["output_dir"])
    ap.add_argument("--lag-applies-to", choices=["both", "graph", "velocity"], default="both")
    ap.add_argument("--baseline", default=None,
                    help="scores.npz of an earlier run; reuse its lag_0h instead of replaying it")
    args = ap.parse_args(argv)

    rows = load_slice(params, cfg["start"], cfg["end"], args.limit)
    lags = args.lags or [float(x) for x in cfg["lags_hours"]]
    baseline = None
    if args.baseline:
        z = np.load(repo_path(args.baseline))
        baseline = (z["txn_id"], z["lag_0h"])
    report = run(params, lags, rows, args.shards, cfg["bootstrap"], check_reference=True,
                 lag_applies_to=args.lag_applies_to, baseline=baseline)
    out = repo_path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "scores.npz", **report.pop("scores"))
    (out / "summary.json").write_text(json.dumps(report, indent=2, default=float))

    print("\n=== AUC-PR vs ingest lag ===")
    for row in report["lags"]:
        extra = ""
        if "vs_lag0" in row:
            b = row["vs_lag0"]
            extra = f"  Δ {b['delta']:+.4f} [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]"
        print(f"  {row['lag_hours']:>5g} h   AUC-PR {row['auc_pr']:.4f}   "
              f"P@100 {row['precision_at_100']:.2f}{extra}")
    if "lag0_vs_full_replay" in report:
        print("lag 0 vs the full replay:", report["lag0_vs_full_replay"])
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
