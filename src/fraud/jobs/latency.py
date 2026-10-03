"""What a request costs, split so geography cannot masquerade as design.

The HTTP replay records, per request, the client's wall time and the server's
`Server-Timing` spans, each carrying its database round-trip count. From those
and a measured network round trip this module derives:

    http_overhead   wall - sum(spans): JSON, uvicorn, the loopback hop
    db_server       sum over spans of (duration - round_trips x RTT): the time
                    Postgres and the driver spend, with the wire subtracted
    projected       wall - round_trips x (RTT_measured - RTT_target): the same
                    request with the database next door

The projection is arithmetic on measured parts, nothing more. It assumes the
server-side time does not change with distance, which holds for a database that
is not saturated -- one request at a time, as the replay sends them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

PERCENTILES = (50, 95, 99)


def percentiles(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {f"p{p}": float("nan") for p in PERCENTILES} | {"mean": float("nan")}
    return {f"p{p}": float(np.percentile(arr, p)) for p in PERCENTILES} | {
        "mean": float(arr.mean())
    }


def analyse(
    wall_ms: np.ndarray,
    spans: dict[str, np.ndarray],
    round_trips: dict[str, np.ndarray],
    rtt_ms: float,
    target_rtt_ms: Sequence[float],
    budget_ms: float = 100.0,
) -> dict:
    """Decompose per-request latency and project it to other round-trip times.

    `spans[name]` and `round_trips[name]` are per-request arrays, aligned with
    `wall_ms`.
    """
    wall = np.asarray(wall_ms, dtype=np.float64)
    names = list(spans)
    server = np.sum([spans[n] for n in names], axis=0)
    trips = np.sum([round_trips[n] for n in names], axis=0)
    db_server = np.sum(
        [np.asarray(spans[n]) - np.asarray(round_trips[n]) * rtt_ms
         for n in names if np.any(round_trips[n])],
        axis=0,
    )

    projections = {}
    for target in target_rtt_ms:
        projected = wall - trips * (rtt_ms - target)
        stats = percentiles(projected)
        projections[f"rtt_{target:g}ms"] = stats | {
            "within_budget": bool(stats["p95"] <= budget_ms)
        }

    return {
        "requests": int(wall.size),
        "rtt_ms_measured": float(rtt_ms),
        "round_trips_per_request": percentiles(trips),
        "wall_ms": percentiles(wall),
        "spans_ms": {n: percentiles(spans[n]) for n in names},
        "decomposition_ms": {
            "network": percentiles(trips * rtt_ms),
            "db_server": percentiles(db_server),
            "score": percentiles(spans["score"]) if "score" in spans else None,
            "http_overhead": percentiles(wall - server),
        },
        "projected_same_region_ms": projections,
        "budget_ms": budget_ms,
    }


def equivalence(
    txn_ids: Sequence[int], scores: Sequence[float],
    reference_txn_ids: Sequence[int], reference_scores: Sequence[float],
    tolerance: float = 1e-4,
) -> dict:
    """Did the served path score what the in-memory causal replay scored?

    Scores are stored as float32 in Postgres, so agreement is to a tolerance,
    not to the bit; a real divergence (a different neighbour, a stale model)
    moves a score by orders of magnitude more.
    """
    ref = dict(zip(np.asarray(reference_txn_ids).tolist(),
                   np.asarray(reference_scores).tolist(), strict=True))
    got = np.asarray(scores, dtype=np.float64)
    expected = np.asarray([ref.get(t, np.nan) for t in np.asarray(txn_ids).tolist()])
    diff = np.abs(got - expected)
    return {
        "compared": int(np.isfinite(expected).sum()),
        "max_abs_diff": float(np.nanmax(diff)) if diff.size else float("nan"),
        "mismatches_over_tolerance": int(np.nansum(diff > tolerance)),
        "tolerance": tolerance,
    }
