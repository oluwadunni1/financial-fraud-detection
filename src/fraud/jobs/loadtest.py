"""Load test: throughput and tail latency under concurrency, per serving setup.

    DATABASE_URL=postgresql://fraud:fraud@localhost:5433/fraud \\
      .venv/bin/python -m fraud.jobs.loadtest --api-url http://localhost:8000 --reset-store

Phase 4 measured ONE request at a time, from a Studio 70 ms from its database,
and projected the co-located number. Against the Compose stack the database is
in the next container, so this measures what Phase 4 could only project -- and
adds what it could not measure at all: behaviour under concurrency (the pool,
torch pinned to one thread, the shadow model sharing the CPU).

Each concurrency level sends its own consecutive slice of real 2019
transactions, so no transaction is scored twice. Within a level requests
overlap, so this is a load test, not a causal replay -- scores are not
compared here; `replay.py --http` is the correctness check.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import sys
import time

from fraud.config import load_params, repo_path
from fraud.jobs.latency import percentiles


def run_level(base: str, payloads: list[dict], concurrency: int) -> dict:
    import httpx

    from fraud.api.timing import parse_server_timing

    wall, server, errors = [], [], 0
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    with httpx.Client(base_url=base, timeout=60.0, limits=limits) as client:

        def send(payload: dict) -> tuple[float, float | None]:
            t0 = time.perf_counter()
            response = client.post("/predict", json=payload)
            elapsed = (time.perf_counter() - t0) * 1000.0
            if response.status_code != 200:
                return elapsed, None
            spans = parse_server_timing(response.headers.get("server-timing", ""))
            return elapsed, sum(v["dur"] for v in spans.values())

        started = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
            for elapsed, server_ms in pool.map(send, payloads):
                if server_ms is None:
                    errors += 1
                else:
                    wall.append(elapsed)
                    server.append(server_ms)
        seconds = time.perf_counter() - started
    return {
        "concurrency": concurrency,
        "requests": len(payloads),
        "errors": errors,
        "throughput_per_s": round(len(wall) / seconds, 1),
        "wall_ms": percentiles(wall),
        "server_ms": percentiles(server),
    }


def main(argv: list[str] | None = None) -> int:
    from fraud.api import store
    from fraud.api.predictor import transaction_from_row
    from fraud.jobs.replay import http_slice, seed_store

    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--per-level", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--reset-store", action="store_true")
    ap.add_argument("--output", default="reports/loadtest.json")
    args = ap.parse_args(argv)

    total = args.warmup + args.per_level * len(args.levels)
    rows, graph_since = http_slice(params, total)
    conn = store.connect()
    seeded = seed_store(params, conn, rows, graph_since, args.reset_store)
    conn.close()
    print(f"seeded {seeded:,} rows; sending {total:,} requests to {args.api_url}")

    payloads = []
    for row in rows.iter_rows(named=True):
        txn = transaction_from_row(row)
        payloads.append({**txn, "ts": txn["ts"].isoformat()})

    import httpx

    health = httpx.get(f"{args.api_url}/health", timeout=10).json()
    run_level(args.api_url, payloads[: args.warmup], 1)          # model + pool warm
    results, cursor = [], args.warmup
    for level in args.levels:
        chunk = payloads[cursor: cursor + args.per_level]
        cursor += args.per_level
        result = run_level(args.api_url, chunk, level)
        results.append(result)
        print(f"  c={level:<3} {result['throughput_per_s']:>7.1f} req/s   "
              f"p50 {result['wall_ms']['p50']:6.1f}  p95 {result['wall_ms']['p95']:6.1f}  "
              f"p99 {result['wall_ms']['p99']:6.1f} ms   errors {result['errors']}", flush=True)

    metrics = httpx.get(f"{args.api_url}/metrics", timeout=10).json()
    report = {
        "api": args.api_url,
        "champion": health["model_version"],
        "challenger": health.get("challenger_version"),
        "shadow": health.get("shadow"),
        "at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "seeded_rows": seeded,
        "levels": results,
        "shadow_scored": metrics.get("shadow_scored"),
        "predictions_logged": metrics.get("predictions"),
    }
    out = repo_path(args.output)
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"shadow-scored {report['shadow_scored']} of {report['predictions_logged']} "
          f"logged predictions -> {out}")
    return 0 if all(r["errors"] == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
