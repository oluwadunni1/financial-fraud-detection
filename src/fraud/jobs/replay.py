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

    def __init__(self, window_hours: int, merchant_cap: int,
                 visibility_lag_hours: float = 0.0, lag_applies_to: str = "both"):
        self.window = dt.timedelta(hours=window_hours)
        self.merchant_cap = merchant_cap
        # The staleness experiment: a row becomes visible only once it is
        # `visibility_lag_hours` old, as if the hot store's ingest ran that far
        # behind. 0 is serving as built -- and must reproduce the replay exactly.
        self.lag = dt.timedelta(hours=visibility_lag_hours)
        # "both" is a lagging store. "graph" / "velocity" lag one consumer and
        # leave the other fresh -- the decomposition that says which one moves.
        if lag_applies_to not in ("both", "graph", "velocity"):
            raise ValueError(f"lag_applies_to must be both/graph/velocity, not {lag_applies_to!r}")
        self.lag_applies_to = lag_applies_to
        self._users: dict[int, collections.deque] = collections.defaultdict(
            collections.deque
        )
        # Not a `deque(maxlen=cap)`: that evicts on append, BEFORE the
        # `ts < now` filter, so a same-minute row already added pushed out the
        # 10th strictly-earlier neighbour and the model saw 9. The HTTP latency
        # replay found it -- Postgres's `ts < now ... limit 10` returned 10 on
        # 81 of 3,000 rows where this returned 9. Rows tied with the newest
        # timestamp now never count toward the cap; see `_prune`.
        self._merchants: dict[str, collections.deque] = collections.defaultdict(
            collections.deque
        )

    @staticmethod
    def _tied_tail(rows: collections.deque, ts: dt.datetime) -> int:
        """How many rows at the right end have `ts >= ts` -- invisible to a
        request at `ts`, so they must not use up its neighbour budget."""
        n = 0
        for row in reversed(rows):
            if row["ts"] < ts:
                break
            n += 1
        return n

    def neighbourhood(self, transaction: dict[str, Any]) -> Neighbourhood:
        now = transaction["ts"]
        cutoff = now - self.lag   # == now unless the staleness experiment lags ingest
        rows = self._users[transaction["User"]]
        # Out of the velocity window AND no longer needed as a graph neighbour.
        # The second clause is what keeps a quiet card's older transactions
        # reachable; without it the graph sees nothing but the last week.
        # Rows at or after `now` are invisible here, so they do not count
        # toward the `merchant_cap` strictly-earlier rows the graph needs.
        keep = self.merchant_cap + self._tied_tail(rows, cutoff)
        while len(rows) > keep and rows[0]["ts"] < now - self.window:
            rows.popleft()

        # `ts < now`, STRICTLY -- the same rule store.py applies in SQL.
        # Same-minute transactions are mutually invisible offline
        # (`closed="left"`), so they must be here too. Without this filter the
        # in-memory replay and the database path would disagree with each other
        # *and* with training, on the 142,010 rows that share a user and a
        # minute. Caught by the guard in velocity_for_transaction.
        lagged = [r for r in reversed(rows) if r["ts"] < cutoff]
        fresh = lagged if self.lag == dt.timedelta(0) else [
            r for r in reversed(rows) if r["ts"] < now]
        graph_rows = fresh if self.lag_applies_to == "velocity" else lagged
        user = pl.DataFrame(graph_rows) if graph_rows else EMPTY
        velocity_rows = {"both": None, "graph": fresh, "velocity": lagged}[self.lag_applies_to]

        merchant_rows = self._merchants.get(str(transaction["Merchant"]))
        merchant_cutoff = now if self.lag_applies_to == "velocity" else cutoff
        merchant_visible = (
            [r for r in reversed(merchant_rows) if r["ts"] < merchant_cutoff][: self.merchant_cap]
            if merchant_rows
            else []
        )
        merchant = pl.DataFrame(merchant_visible) if merchant_visible else EMPTY
        return Neighbourhood(
            user_history=user, merchant_history=merchant,
            velocity_history=(None if velocity_rows is None
                              else pl.DataFrame(velocity_rows) if velocity_rows else EMPTY),
        )

    def add(self, transaction: dict[str, Any], velocity: dict[str, float]) -> None:
        """Make a transaction visible -- called only after it was scored."""
        row = {**transaction, **velocity}
        self._users[transaction["User"]].append(row)
        merchant = self._merchants[str(transaction["Merchant"])]
        merchant.append(row)
        # Bounded memory: no later request can see past the newest timestamp's
        # ties, so `cap` rows before them are all a merchant ever needs.
        # With a visibility lag, everything inside the lag is still invisible to
        # the next request, so it must not use up the cap either.
        keep = self.merchant_cap + self._tied_tail(merchant, row["ts"] - self.lag)
        while len(merchant) > keep:
            merchant.popleft()


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
    # Keyed so a replay can be joined against any other model's scores later.
    # Without this, answering "what would an ensemble do?" costs a whole re-run.
    txn_ids = np.zeros(matrix_rows.height, dtype=np.int64)
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
        txn_ids[index] = int(transaction["txn_id"])
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
        "txn_ids": txn_ids,
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


def seed_rows(
    params: dict,
    start_ts: dt.datetime,
    hours: int,
    graph_since: dt.datetime | None = None,
    users: set[int] | None = None,
    merchants: set[str] | None = None,
) -> pl.DataFrame:
    """The prior history a replay starting at `start_ts` should see, with velocity.

    In production the store is never empty: by the time a 2019 transaction
    arrives, the previous week is already there. Starting cold would make every
    early transaction a cold start and understate the model badly -- the 5,000
    row smoke test showed 27.6% cold-start users for exactly that reason.

    Everything selected is strictly before `start_ts`, so this adds context
    without adding foresight. Velocity values come from the offline stage,
    which computed them over each row's own past -- the same numbers scoring
    them live would have produced and written.

    One selection, two consumers: `seed_history` feeds it to the in-memory
    replay and the HTTP replay COPYs it into Postgres, so both start from the
    same past by construction.

    `users` / `merchants` restrict the selection to the keys a replay slice
    will actually look up. A request reads only its own card's and merchant's
    rows, so the rest of the history can never reach a score -- and seeding it
    all would not fit Supabase's free tier.
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
    graph_rows = params["serving"]["neighbours"]
    prior = pl.scan_parquet(
        processed / "**/*.parquet", hive_partitioning=True
    ).filter(pl.col("ts") < start_ts)
    # Velocity reads the card's window only; merchant history feeds the graph.
    window = prior.filter(pl.col("ts") >= since)
    if users is not None:
        window = window.filter(pl.col("User").is_in(list(users)))

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
            user_tail = ordered.group_by("User").tail(graph_rows).select(columns)
            merchant_tail = (
                ordered.group_by("Merchant").tail(graph_rows).select(columns)
            )
            if users is not None:
                user_tail = user_tail.filter(pl.col("User").is_in(list(users)))
            if merchants is not None:
                merchant_tail = merchant_tail.filter(
                    pl.col("Merchant").cast(pl.String).is_in(list(merchants))
                )
            frames = [
                window.collect(engine="streaming").lazy(),
                user_tail.lazy(),
                merchant_tail.lazy(),
            ]

    txns = pl.concat(frames).unique(subset=["txn_id"])
    vel = pl.scan_parquet(
        velocity_dir / "**/*.parquet", hive_partitioning=True
    ).drop("Year")
    return (
        txns.join(vel, on="txn_id", how="left")
        .collect(engine="streaming")
        .sort("ts", "txn_id")
    )


def row_velocity(row: dict[str, Any], names: list[str]) -> dict[str, float]:
    return {n: float(row[n] if row[n] is not None else 0.0) for n in names}


def seed_history(
    history: CausalHistory,
    params: dict,
    start_ts: dt.datetime,
    hours: int,
    graph_since: dt.datetime | None = None,
) -> int:
    """Fill the in-memory history with `seed_rows` -- see there for the rules."""
    rows = seed_rows(params, start_ts, hours, graph_since=graph_since)
    names = velocity_columns(params["velocity"]["windows_hours"])
    for row in rows.iter_rows(named=True):
        history.add(transaction_from_row(row), row_velocity(row, names))
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

    # The shard's own rows travel with the job (a few MB pickled) rather than
    # being re-read by index, so a slice replays as correctly as a whole year.
    index, rows, params, model_dir, graph_since, lag_hours, lag_applies_to = job
    predictor = Predictor.from_disk(params, model_dir)
    history = CausalHistory(
        window_hours=params["serving"]["history_hours"],
        merchant_cap=params["serving"]["neighbours"],
        visibility_lag_hours=lag_hours,
        lag_applies_to=lag_applies_to,
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
    rows: pl.DataFrame, params: dict, years: list[int], model_dir, shards: int,
    lag_hours: float = 0.0, graph_since: dt.datetime | None = None,
    lag_applies_to: str = "both",
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
    if graph_since is None:
        graph_since = rows["ts"].min() - dt.timedelta(
            hours=params["serving"]["history_hours"]
        )
    jobs = [
        (i, rows[lo:hi], params, model_dir, graph_since, lag_hours, lag_applies_to)
        for i, (lo, hi) in enumerate(ranges)
    ]

    started = time.perf_counter()
    with mp.get_context("spawn").Pool(len(ranges)) as pool:
        parts = sorted(pool.map(_run_shard, jobs), key=lambda r: r["index"])

    return {
        "rows": sum(p["rows"] for p in parts),
        "positives": sum(p["positives"] for p in parts),
        "scores": np.concatenate([p["scores"] for p in parts]),
        "labels": np.concatenate([p["labels"] for p in parts]),
        "txn_ids": np.concatenate([p["txn_ids"] for p in parts]),
        "latency_ms": np.concatenate([p["latency_ms"] for p in parts]),
        "cold_start_user": sum(p["cold_start_user"] for p in parts),
        "cold_start_merchant": sum(p["cold_start_merchant"] for p in parts),
        "seconds": round(time.perf_counter() - started, 1),
        "shards": len(ranges),
    }


def http_slice(params: dict, limit: int | None) -> tuple[pl.DataFrame, dt.datetime]:
    """The latency slice, and the graph bound the in-process replay used for it.

    The bound is the first row of the slice's year minus one velocity window --
    exactly where the (sharded or sequential) in-process 2019 replay's history
    began. Seeding to the same bound is what lets the HTTP scores be checked
    against `metrics_replay.npz` row for row.
    """
    cfg = params["serving"]["latency"]
    start = dt.datetime.fromisoformat(cfg["start"])
    processed = repo_path(params["paths"]["processed"])
    scan = pl.scan_parquet(processed / "**/*.parquet", hive_partitioning=True)
    rows = (
        scan.filter(pl.col("ts") >= start)
        .sort("ts", "txn_id")
        .head(limit or cfg["n"])
        .collect(engine="streaming")
    )
    year_start = (
        scan.filter(pl.col("Year") == start.year)
        .select(pl.col("ts").min())
        .collect()
        .item()
    )
    return rows, year_start - dt.timedelta(hours=params["serving"]["history_hours"])


def _wait_for_health(client, timeout_s: float = 300.0) -> dict:
    """The API downloads the model from DagsHub at startup; wait for it."""
    import httpx

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            response = client.get("/health")
            if response.status_code == 200:
                return response.json()
        except httpx.TransportError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"API not healthy after {timeout_s:.0f}s")


def replay_http(params: dict, limit: int | None, reset: bool) -> dict:
    """Send a slice through the running API, one request at a time.

    Answers "what does this cost in production?" with the database in the loop,
    and checks on the way that the database path scores exactly what the
    in-memory replay did. Sequential on purpose: each request inserts its row
    after scoring, so the order IS the causality control. Concurrency is the
    load test's question, not this one's.
    """
    import subprocess

    import httpx

    from fraud.api import store
    from fraud.api.timing import parse_server_timing
    from fraud.config import REPO_ROOT
    from fraud.jobs.latency import analyse, equivalence

    cfg = params["serving"]["latency"]
    hours = params["serving"]["history_hours"]
    rows, graph_since = http_slice(params, limit)
    cursor = rows["ts"].min()
    print(f"slice: {rows.height:,} transactions from {cursor} "
          f"({int(rows['Fraud'].sum())} frauds)")

    # --- seed -------------------------------------------------------------
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("select count(*) as n from transaction_events")
        existing = cur.fetchone()["n"]
    if existing and not reset:
        raise RuntimeError(
            f"transaction_events holds {existing:,} rows from a previous run. "
            f"Pass --reset-store: the store must contain only this slice's past."
        )
    if reset:
        store.reset_store(conn)

    users = set(rows["User"].to_list())
    merchants = {str(m) for m in rows["Merchant"].to_list()}
    seed = seed_rows(params, cursor, hours, graph_since, users, merchants)
    names = velocity_columns(params["velocity"]["windows_hours"])
    started = time.perf_counter()
    seeded = store.bulk_load_history(
        conn,
        [store.stored_row(transaction_from_row(r), row_velocity(r, names))
         for r in seed.iter_rows(named=True)],
    )
    print(f"seeded {seeded:,} rows for {len(users):,} cards / "
          f"{len(merchants):,} merchants in {time.perf_counter() - started:.1f}s")
    store.assert_causal(conn, cursor)

    # --- network baseline -------------------------------------------------
    probes = []
    with conn.cursor() as cur:
        for _ in range(cfg["rtt_probes"]):
            t0 = time.perf_counter()
            cur.execute("select 1")
            cur.fetchone()
            probes.append((time.perf_counter() - t0) * 1000.0)
        cur.execute("select pg_database_size(current_database()) as b")
        db_bytes = int(cur.fetchone()["b"])
    conn.rollback()
    rtt = float(np.median(probes))
    print(f"network: select 1 median {rtt:.1f} ms; database {db_bytes / 1e6:.0f} MB")

    # --- the API ----------------------------------------------------------
    base = f"http://127.0.0.1:{cfg['port']}"
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "fraud.api.main:app",
         "--port", str(cfg["port"]), "--log-level", "warning"],
        cwd=REPO_ROOT,
    )
    wall, scores, txn_ids, cold_user, cold_merchant = [], [], [], 0, 0
    span_ms: dict[str, list[float]] = collections.defaultdict(list)
    span_rt: dict[str, list[float]] = collections.defaultdict(list)
    try:
        with httpx.Client(base_url=base, timeout=60.0) as client:
            health = _wait_for_health(client)
            print(f"API up: {health['model_version']} (@{health['alias']})")
            started = time.perf_counter()
            for index, row in enumerate(rows.iter_rows(named=True)):
                txn = transaction_from_row(row)
                payload = {**txn, "ts": txn["ts"].isoformat()}
                t0 = time.perf_counter()
                response = client.post("/predict", json=payload)
                wall.append((time.perf_counter() - t0) * 1000.0)
                response.raise_for_status()
                body = response.json()
                for name, v in parse_server_timing(
                    response.headers["server-timing"]
                ).items():
                    span_ms[name].append(v["dur"])
                    span_rt[name].append(v["rt"])
                scores.append(body["score"])
                txn_ids.append(body["txn_id"])
                cold_user += body["cold_start_user"]
                cold_merchant += body["cold_start_merchant"]
                if index and index % 500 == 0:
                    rate = index / (time.perf_counter() - started)
                    print(f"  {index:>6,} / {rows.height:,} ({rate:.1f}/s, "
                          f"p50 so far {np.median(wall):.0f} ms)", flush=True)
    finally:
        api.terminate()
        api.wait(timeout=30)
    seconds = time.perf_counter() - started

    # --- server-side query time, on Postgres's own clock -------------------
    rng = np.random.default_rng(params["xgboost"]["random_state"])
    sample = rng.choice(rows.height, size=min(cfg["explain_samples"], rows.height),
                        replace=False)
    explain = [
        store.explain_fetch_ms(
            conn, int(rows["User"][i]), int(rows["Merchant"][i]), rows["ts"][i],
            hours, params["serving"]["neighbours"],
        )
        for i in sample.tolist()
    ]
    conn.close()

    # --- does the database path score what the in-memory replay did? -----
    reference = np.load(repo_path("reports/metrics_replay.npz"))
    got = np.asarray(scores)
    same = equivalence(txn_ids, got, reference["txn_id"], reference["score"])

    analysis = analyse(
        np.asarray(wall),
        {k: np.asarray(v) for k, v in span_ms.items()},
        {k: np.asarray(v) for k, v in span_rt.items()},
        rtt,
        cfg["target_rtt_ms"],
    )
    return {
        "per_request": {"txn_id": np.asarray(txn_ids), "score": got,
                        "wall_ms": np.asarray(wall)},
        "mode": "http_sequential",
        "slice": {"start": str(cursor), "rows": rows.height,
                  "frauds": int(rows["Fraud"].sum())},
        "model_version": health["model_version"],
        "alias": health["alias"],
        "seeded_rows": seeded,
        "database_mb": round(db_bytes / 1e6, 1),
        "rtt_probe_ms": percentiles_dict(probes),
        "seconds": round(seconds, 1),
        "throughput_per_s": round(rows.height / seconds, 2),
        "cold_start_user": int(cold_user),
        "cold_start_merchant": int(cold_merchant),
        "latency": analysis,
        "explain_fetch_ms": percentiles_dict(explain),
        "equivalence_vs_in_process": same,
    }


def percentiles_dict(values) -> dict[str, float]:
    from fraud.jobs.latency import percentiles

    return percentiles(values)


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
    ap.add_argument(
        "--http", action="store_true",
        help="send serving.latency's slice through the API against Supabase",
    )
    ap.add_argument(
        "--reset-store", action="store_true",
        help="truncate transaction_events and predictions before seeding",
    )
    ap.add_argument("--label", default="separate",
                    help="name for this --http run inside the latency report")
    ap.add_argument("--latency-output", default="reports/metrics_latency.json")
    args = ap.parse_args(argv)

    if args.http:
        report = replay_http(params, args.limit, args.reset_store)
        out = repo_path(args.latency_output)
        # Per-request scores beside the summary, so equivalence can be
        # re-checked against a later in-process replay without re-sending.
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out.with_name(f"{out.stem}_{args.label}.npz"), **report.pop("per_request")
        )
        existing = json.loads(out.read_text()) if out.exists() else {"runs": {}}
        existing["runs"][args.label] = report
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(existing, indent=2, default=float))
        lat = report["latency"]
        eq = report["equivalence_vs_in_process"]
        print(f"\n=== HTTP replay [{args.label}], {report['slice']['rows']:,} requests ===")
        print(f"  wall p50/p95/p99   {lat['wall_ms']['p50']:.1f} / "
              f"{lat['wall_ms']['p95']:.1f} / {lat['wall_ms']['p99']:.1f} ms")
        for name, stats in lat["spans_ms"].items():
            print(f"  {name:<8} p50 {stats['p50']:8.2f} ms")
        for name, stats in lat["projected_same_region_ms"].items():
            print(f"  projected {name}: p50 {stats['p50']:.1f} / p95 {stats['p95']:.1f} ms")
        print(f"  equivalence: {eq['mismatches_over_tolerance']} of {eq['compared']} "
              f"over {eq['tolerance']} (max diff {eq['max_abs_diff']:.2e})")
        print(f"\n-> {out}")
        return 0

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

    # Per-row scores, keyed by txn_id. A 53-minute replay should be answerable
    # more than once: ensembles, PR curves and threshold choices all need the
    # raw scores, and none of them justify re-running the whole thing.
    scores_path = out.with_suffix(".npz")
    np.savez_compressed(
        scores_path,
        txn_id=result["txn_ids"],
        score=result["scores"],
        label=result["labels"],
        latency_ms=result["latency_ms"],
    )
    print(f"-> {scores_path}")

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
