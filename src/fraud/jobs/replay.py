"""Causal replay: does the offline number survive honest serving?

Phase 3 measured the GNN on a graph holding the whole test period at once, so
scoring a January transaction could aggregate over December. Serving never can.
This replays the same transactions the way production would see them and reports
what the model is actually worth.

The protocol is the entire leakage control:

    for each transaction, in STRICT CHRONOLOGICAL ORDER:
        1. read  history   (ts < this transaction's ts)
        2. score
        3. record
        4. ADD to history  <- only now is it visible to anything

Nothing can see the future because nothing exists until after it has been
scored. Compare with the alternative -- load the test period, then replay over
it -- which produces the same code paths, the same queries, plausible latencies
and a flattering metric, while every prediction is clairvoyant.

Two modes, because there are two questions:

    --in-process   all of 2019, no HTTP, no database. Answers "is the offline
                   number honest?" using every one of the 2,087 frauds.
    --http         a few thousand requests against the running API. Answers
                   "what does this cost in production?" -- p50/p95/p99.

Both call `Predictor.score`, the same path the API uses. A replay with its own
scoring logic would be measuring something else and its verdict would be worth
nothing.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
import time
from typing import Any

import numpy as np
import polars as pl

from fraud.api.predictor import Predictor, transaction_from_row
from fraud.api.store import Neighbourhood
from fraud.config import load_params, repo_path
from fraud.features.velocity import velocity_columns
from fraud.models.metrics import evaluate_scores, threshold_at_fpr

EMPTY = pl.DataFrame(schema={})


class CausalHistory:
    """Per-entity history that only ever contains the past.

    Rows are added *after* the transaction that produced them has been scored,
    so a lookup can never return something the caller should not have seen.

    User history serves two consumers with different needs, and keeping only
    one of them was worth 0.5339 -> 0.5804 AUC-PR on a June 2019 slice:

      velocity  every row in the window. `velocity_count_168h` counts a window,
                so dropping a row inside it under-reports and feeds the model a
                number training never produced.
      the graph the `graph_rows` most recent rows, HOWEVER OLD. The offline
                user node aggregates transactions sampled across the whole
                split, so pruning the graph to the velocity window shows it one
                week of a card that training saw a year of.

    So rows leave only when they are outside the window AND no longer among the
    most recent `graph_rows`. Merchant history feeds the graph alone, so it is a
    pure row cap with no age bound.
    """

    def __init__(self, window_hours: int, merchant_cap: int):
        self.window = dt.timedelta(hours=window_hours)
        self.merchant_cap = merchant_cap
        self._users: dict[int, collections.deque] = collections.defaultdict(
            collections.deque
        )
        self._merchants: dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=merchant_cap)
        )

    def neighbourhood(self, transaction: dict[str, Any]) -> Neighbourhood:
        now = transaction["ts"]
        rows = self._users[transaction["User"]]
        # Out of the velocity window AND no longer needed as a graph neighbour.
        # The second clause is what keeps a quiet card's older transactions
        # reachable; without it the graph sees nothing but the last week.
        while len(rows) > self.merchant_cap and rows[0]["ts"] < now - self.window:
            rows.popleft()

        # `ts < now`, STRICTLY -- the same rule store.py applies in SQL.
        # Same-minute transactions are mutually invisible offline
        # (`closed="left"`), so they must be here too. Without this filter the
        # in-memory replay and the database path would disagree with each other
        # *and* with training, on the 142,010 rows that share a user and a
        # minute. Caught by the guard in velocity_for_transaction.
        visible = [r for r in reversed(rows) if r["ts"] < now]
        user = pl.DataFrame(visible) if visible else EMPTY

        merchant_rows = self._merchants.get(str(transaction["Merchant"]))
        merchant_visible = (
            [r for r in reversed(merchant_rows) if r["ts"] < now]
            if merchant_rows
            else []
        )
        merchant = pl.DataFrame(merchant_visible) if merchant_visible else EMPTY
        return Neighbourhood(user_history=user, merchant_history=merchant)

    def add(self, transaction: dict[str, Any], velocity: dict[str, float]) -> None:
        """Make a transaction visible -- called only after it was scored."""
        row = {**transaction, **velocity}
        self._users[transaction["User"]].append(row)
        self._merchants[str(transaction["Merchant"])].append(row)


def replay_in_process(
    matrix_rows: pl.DataFrame,
    predictor: Predictor,
    params: dict,
    progress_every: int = 50_000,
    history: CausalHistory | None = None,
) -> dict:
    """Score every row in chronological order, seeing only its own past."""
    if history is None:
        history = CausalHistory(
            window_hours=params["serving"]["history_hours"],
            merchant_cap=params["serving"]["neighbours"],
        )
    scores = np.zeros(matrix_rows.height, dtype=np.float64)
    labels = np.zeros(matrix_rows.height, dtype=np.int8)
    latencies = np.zeros(matrix_rows.height, dtype=np.float64)
    cold_user = cold_merchant = 0

    started = time.perf_counter()
    previous_ts: dt.datetime | None = None

    for index, row in enumerate(matrix_rows.iter_rows(named=True)):
        transaction = transaction_from_row(row)

        # The ordering invariant, checked rather than assumed: a replay that
        # wandered out of chronological order would leak silently.
        if previous_ts is not None and transaction["ts"] < previous_ts:
            raise RuntimeError(
                f"replay went backwards at txn {transaction['txn_id']}: "
                f"{transaction['ts']} after {previous_ts}. Rows must be sorted."
            )
        previous_ts = transaction["ts"]

        neighbourhood = history.neighbourhood(transaction)
        prediction = predictor.score(transaction, neighbourhood)

        scores[index] = prediction.score
        labels[index] = int(row["Fraud"])
        latencies[index] = prediction.latency_ms
        cold_user += prediction.cold_start_user
        cold_merchant += prediction.cold_start_merchant

        # Visible only now.
        history.add(transaction, prediction.velocity)

        if progress_every and index and index % progress_every == 0:
            rate = index / (time.perf_counter() - started)
            print(
                f"  {index:>9,} / {matrix_rows.height:,}  "
                f"({rate:,.0f}/s, {(matrix_rows.height - index) / rate / 60:.0f} min left)",
                flush=True,
            )

    return {
        "rows": int(matrix_rows.height),
        "positives": int(labels.sum()),
        "scores": scores,
        "labels": labels,
        "latency_ms": latencies,
        "cold_start_user": int(cold_user),
        "cold_start_merchant": int(cold_merchant),
        "seconds": round(time.perf_counter() - started, 1),
    }


def shard_boundaries(rows: pl.DataFrame, shards: int) -> list[tuple[int, int]]:
    """Split into contiguous row ranges that never cut a timestamp in two.

    Rows sharing a timestamp are mutually invisible (`ts < now`), but only if
    they are on the same side of a boundary: seeding uses `ts < start`, so a
    tied row left behind in the previous shard would be invisible to the whole
    of this one rather than just to its twin. Each boundary is therefore
    advanced to the next change in `ts`.
    """
    if shards <= 1:
        return [(0, rows.height)]
    ts = rows.get_column("ts").to_numpy()
    cuts = [0]
    for k in range(1, shards):
        i = min(k * rows.height // shards, rows.height - 1)
        while i < rows.height and ts[i] == ts[i - 1]:
            i += 1
        if i < rows.height and i > cuts[-1]:
            cuts.append(i)
    cuts.append(rows.height)
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:], strict=True) if b > a]


def seed_history(
    history: CausalHistory,
    params: dict,
    start_ts: dt.datetime,
    hours: int,
    graph_since: dt.datetime | None = None,
) -> int:
    """Fill the store with the window immediately BEFORE the replay begins.

    In production the store is never empty: by the time a 2019 transaction
    arrives, the previous week is already there. Starting cold would make every
    early transaction a cold start and understate the model badly -- the 5,000
    row smoke test showed 27.6% cold-start users for exactly that reason.

    Everything seeded is strictly before `start_ts`, so this adds context
    without adding foresight. Their velocity values come from the offline
    stage, which computed them over each row's own past -- the same numbers
    scoring them live would have produced and written.
    """
    processed = repo_path(params["paths"]["processed"])
    velocity_dir = repo_path(params["paths"]["velocity"])
    since = start_ts - dt.timedelta(hours=hours)

    # Two bounded reads, exactly as store.py issues them: every row in the
    # velocity window, plus the N most recent per card and per merchant however
    # old, so the graph is not silently capped at the velocity window.
    #
    # The graph tail is bounded below by `graph_since` -- the start of the
    # replayed period. The offline test graph holds test-split rows only, so
    # seeding neighbours from before it would hand serving history the offline
    # measurement never had, and the comparison would flatter us.
    graph_rows = history.merchant_cap
    prior = pl.scan_parquet(
        processed / "**/*.parquet", hive_partitioning=True
    ).filter(pl.col("ts") < start_ts)
    window = prior.filter(pl.col("ts") >= since)

    frames = [window]
    if graph_since is not None and graph_since < start_ts:
        in_period = prior.filter(pl.col("ts") >= graph_since).collect(
            engine="streaming"
        )
        if in_period.height:
            ordered = in_period.sort("ts", "txn_id")
            # group_by(...).tail() promotes the key to the first column, and
            # pl.concat matches schemas by position -- so reselect.
            columns = in_period.columns
            frames = [
                window.collect(engine="streaming").lazy(),
                ordered.group_by("User").tail(graph_rows).select(columns).lazy(),
                ordered.group_by("Merchant").tail(graph_rows).select(columns).lazy(),
            ]

    txns = pl.concat(frames).unique(subset=["txn_id"])
    vel = pl.scan_parquet(
        velocity_dir / "**/*.parquet", hive_partitioning=True
    ).drop("Year")
    rows = (
        txns.join(vel, on="txn_id", how="left")
        .collect(engine="streaming")
        .sort("ts", "txn_id")
    )

    names = velocity_columns(params["velocity"]["windows_hours"])
    for row in rows.iter_rows(named=True):
        history.add(
            transaction_from_row(row),
            {n: float(row[n] if row[n] is not None else 0.0) for n in names},
        )
    return rows.height


def load_split_rows(params: dict, years: list[int], limit: int | None) -> pl.DataFrame:
    """Transactions plus their velocity inputs, in chronological order."""
    processed = repo_path(params["paths"]["processed"])
    frame = (
        pl.scan_parquet(processed / "**/*.parquet", hive_partitioning=True)
        .filter(pl.col("Year").is_in(years))
        .collect(engine="streaming")
        .sort("ts", "txn_id")
    )
    return frame.head(limit) if limit else frame


def _run_shard(job: tuple) -> dict:
    """One shard, in its own process: seed from its own past, then replay.

    A shard is exact, not an approximation. Everything before `start` is seeded
    through the same two bounded reads the store performs, and within the shard
    the loop is strictly sequential -- so the concatenation equals the
    single-process replay row for row.
    """
    import torch

    # The graphs here are ~21 nodes; torch's intra-op threads cost more than
    # they save and would fight the other shards for cores.
    torch.set_num_threads(1)

    index, (lo, hi), years, params, model_dir, graph_since = job
    rows = load_split_rows(params, years, None)[lo:hi]
    predictor = Predictor.from_disk(params, model_dir)
    history = CausalHistory(
        window_hours=params["serving"]["history_hours"],
        merchant_cap=params["serving"]["neighbours"],
    )
    seeded = seed_history(
        history, params, rows["ts"].min(), params["serving"]["history_hours"],
        graph_since=graph_since,
    )
    print(f"  shard {index}: {rows.height:,} rows, seeded {seeded:,}", flush=True)
    result = replay_in_process(rows, predictor, params, progress_every=0,
                               history=history)
    result["index"] = index
    return result


def replay_sharded(
    rows: pl.DataFrame, params: dict, years: list[int], model_dir, shards: int
) -> dict:
    """Replay contiguous time ranges in parallel and stitch them back together."""
    import multiprocessing as mp

    ranges = shard_boundaries(rows, shards)
    print(f"{len(ranges)} shards: {[b - a for a, b in ranges]}", flush=True)

    # Where the sequential run's history begins: its own seed reaches one
    # velocity window back, so for a sparse card the 10 most recent rows can
    # sit in that window. A shard whose tail stopped later would hold a
    # different history and score those rows differently -- 2,659 of 20,000 on
    # the first attempt. Bounding both at the same instant is what makes the
    # sharded replay equal to the sequential one rather than merely close.
    graph_since = rows["ts"].min() - dt.timedelta(
        hours=params["serving"]["history_hours"]
    )
    jobs = [
        (i, r, years, params, model_dir, graph_since)
        for i, r in enumerate(ranges)
    ]

    started = time.perf_counter()
    with mp.get_context("spawn").Pool(len(ranges)) as pool:
        parts = sorted(pool.map(_run_shard, jobs), key=lambda r: r["index"])

    return {
        "rows": sum(p["rows"] for p in parts),
        "positives": sum(p["positives"] for p in parts),
        "scores": np.concatenate([p["scores"] for p in parts]),
        "labels": np.concatenate([p["labels"] for p in parts]),
        "latency_ms": np.concatenate([p["latency_ms"] for p in parts]),
        "cold_start_user": sum(p["cold_start_user"] for p in parts),
        "cold_start_merchant": sum(p["cold_start_merchant"] for p in parts),
        "seconds": round(time.perf_counter() - started, 1),
        "shards": len(ranges),
    }


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-process", action="store_true", default=True)
    ap.add_argument("--years", type=int, nargs="+", default=[2019])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", default="reports/metrics_replay.json")
    ap.add_argument(
        "--shards", type=int, default=1,
        help="replay contiguous time ranges in parallel; results are identical",
    )
    args = ap.parse_args(argv)

    rows = load_split_rows(params, args.years, args.limit)
    print(
        f"replaying {rows.height:,} transactions from {args.years} "
        f"({int(rows['Fraud'].sum()):,} frauds), chronologically"
    )

    model_dir = repo_path(params["paths"]["models"]) / "gnn"
    predictor = Predictor.from_disk(params, model_dir)
    print(f"model: {predictor.model_version}")

    if args.shards > 1:
        result = replay_sharded(rows, params, args.years, model_dir, args.shards)
        seeded = -1
    else:
        history = CausalHistory(
            window_hours=params["serving"]["history_hours"],
            merchant_cap=params["serving"]["neighbours"],
        )
        seeded = seed_history(
            history, params, rows["ts"].min(), params["serving"]["history_hours"]
        )
        print(f"seeded {seeded:,} rows of prior history (all strictly before "
              f"the first replayed transaction)\n")
        result = replay_in_process(rows, predictor, params, history=history)

    ev = params["evaluate"]
    fpr = ev["target_false_positive_rate"]
    ks = list(ev["precision_at_k"])
    # Threshold from the replay's own scores at the target FPR. The offline
    # threshold came from val under a different regime, so it does not transfer.
    threshold = threshold_at_fpr(result["labels"], result["scores"], fpr)
    metrics = evaluate_scores(
        result["labels"], result["scores"], ks, fpr, threshold=threshold
    )

    latency = result["latency_ms"]
    report = {
        "mode": "in_process_causal",
        "years": args.years,
        "model_version": predictor.model_version,
        "seeded_rows": seeded,
        "rows": result["rows"],
        "seconds": result["seconds"],
        "throughput_per_s": round(result["rows"] / result["seconds"], 1),
        "cold_start_user": result["cold_start_user"],
        "cold_start_merchant": result["cold_start_merchant"],
        "latency_ms": {
            "p50": float(np.percentile(latency, 50)),
            "p95": float(np.percentile(latency, 95)),
            "p99": float(np.percentile(latency, 99)),
        },
        "metrics": metrics,
    }

    out = repo_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))

    print(f"\n=== causal replay, {args.years} ===")
    print(f"  AUC-PR          {metrics['auc_pr']:.4f}   (base {metrics['base_rate']:.5f})")
    print(f"  P@100           {metrics['precision_at_100']:.3f}")
    print(f"  recall @ {fpr:.0%} FPR {metrics[f'recall_at_fpr_{fpr}']:.3f}")
    print(f"  cold start      user {result['cold_start_user']:,} / "
          f"merchant {result['cold_start_merchant']:,}")
    print(f"  latency p50/p95 {report['latency_ms']['p50']:.1f} / "
          f"{report['latency_ms']['p95']:.1f} ms")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
