"""Postgres access for the serving path.

Every read here is `ts < :now`, strictly. That is not a stylistic choice: it is
the online half of the offline `closed="left"` contract, and the two must agree
or the model is fed numbers it was never trained on. 142,010 rows in this
dataset share a user and a minute, so `<=` would diverge on ordinary traffic
rather than on an edge case. `tests/test_skew.py` holds the two halves together.

Writes happen only *after* a prediction is made. A row becomes visible to future
predictions at insert time and not one moment sooner, which is what makes the
replay causally honest rather than merely well-intentioned.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Any

import polars as pl
import psycopg
from psycopg.rows import dict_row

from fraud.config import REPO_ROOT
from fraud.features.velocity import velocity_columns

# What velocity needs back from the store, named as compute_velocity expects.
HISTORY_COLUMNS = ("txn_id", "User", "ts", "Amount", "Merchant", "State")

# Raw encoder inputs carried per row so a neighbour's feature vector can be
# rebuilt without re-deriving anything.
RAW_COLUMNS = (
    "merchant", "city", "state", "zip", "errors", "chip",
    "time_min", "month", "day",
)

# The database spells these differently from the feature pipeline. Mapping them
# once, here, keeps the schema's naming from leaking into the model code.
_DB_TO_CANONICAL = {
    "user_id": "User",
    "amount": "Amount",
    "mcc": "MCC",
    "merchant": "Merchant",
    "state": "State",
    "city": "City",
    "zip": "Zip",
    "errors": "Errors",
    "chip": "Chip",
    "time_min": "Time",
    "month": "Month",
    "day": "Day",
}


@dataclass
class Neighbourhood:
    """One request's view of the past. Everything here has `ts < now`."""

    user_history: pl.DataFrame      # this card's recent rows
    merchant_history: pl.DataFrame  # this merchant's recent rows
    # Velocity normally reads `user_history`. Only the staleness experiment sets
    # this, to give velocity a different view of the past than the graph and
    # find out which of the two a lagging store actually affects.
    velocity_history: pl.DataFrame | None = None

    @property
    def cold_start_user(self) -> bool:
        return self.user_history.is_empty()

    @property
    def cold_start_merchant(self) -> bool:
        return self.merchant_history.is_empty()


def connection_string() -> str:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    url = os.getenv("DATABASE_URL", "")
    if not url or "<" in url:
        raise RuntimeError("DATABASE_URL missing or placeholder in .env")
    return url


def connect() -> psycopg.Connection:
    # Session pinned to UTC: training timestamps are naive UTC, and a naive
    # `now` parameter is interpreted in the session's zone. Leaving that to the
    # server default would make `ts < now` depend on where the database lives.
    return psycopg.connect(
        connection_string(), row_factory=dict_row, options="-c TimeZone=UTC"
    )


def _to_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Rows as velocity expects them, with the column names it requires."""
    if not rows:
        return pl.DataFrame(
            schema={
                "txn_id": pl.Int64,
                "User": pl.Int64,
                "ts": pl.Datetime("us"),
                "Amount": pl.Float64,
                "Merchant": pl.String,
                "State": pl.String,
                "merchant_id": pl.Int64,
                **{c: pl.Float64 for c in velocity_columns([1, 24, 168])},
                **{c: pl.String for c in ("City", "Errors", "Chip")},
                **{c: pl.Int64 for c in ("Zip", "Time", "Month", "Day", "MCC")},
            }
        )
    # Canonical names here, not in the caller: subgraph.py and the predictor
    # should not need to know what the database columns are called.
    frame = pl.DataFrame(rows).rename(_DB_TO_CANONICAL)
    # `timestamptz` comes back zone-aware; everything offline is naive UTC.
    # Normalised once, here, so a comparison against the arriving transaction
    # can never mix the two.
    if getattr(frame.schema["ts"], "time_zone", None):
        frame = frame.with_columns(
            pl.col("ts").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        )
    return frame


_HISTORY_COLUMNS = """
    txn_id, user_id, ts, amount, mcc, merchant, state, merchant_id,
    city, zip, errors, chip, time_min, month, day,
    velocity_count_1h, velocity_amount_1h, velocity_merchants_1h,
    velocity_states_1h, velocity_count_24h, velocity_amount_24h,
    velocity_merchants_24h, velocity_states_24h, velocity_count_168h,
    velocity_amount_168h, velocity_merchants_168h, velocity_states_168h,
    velocity_seconds_since_last
"""

# The card's history serves two consumers with different needs, so it is the
# union of two bounded reads rather than one compromise:
#
#   velocity  every row in the window. A count over a window truncated by row
#             count under-reports, and the model sees a number training never
#             produced.
#   the graph the N most recent rows, HOWEVER OLD. The offline graph's user
#             node aggregates transactions sampled across the whole split, so
#             capping the graph at the velocity window shows it one week of a
#             card that training saw a year of. Measured at 0.5339 -> 0.5804
#             AUC-PR on a June 2019 slice -- and the 168h bound never had a
#             justification for the graph, only for velocity.
#
# Both arms ride idx_txe_user_ts, and both are bounded.
#
# `txn_id desc` breaks timestamp ties. subgraph.py keeps the first N rows, and
# 142,010 rows share a user and a minute, so an unordered tie would let the
# database pick different neighbours from the in-memory replay -- a different
# score from the same history, with nothing to show for it.
_USER_HISTORY_SQL = f"""
    (select {_HISTORY_COLUMNS}
       from transaction_events
      where user_id = %(key)s
        and ts < %(now)s          -- STRICTLY before. See the module docstring.
        and ts >= %(since)s       -- the velocity window
      order by ts desc, txn_id desc
      limit %(limit)s)
    union
    (select {_HISTORY_COLUMNS}
       from transaction_events
      where user_id = %(key)s
        and ts < %(now)s
      order by ts desc, txn_id desc
      limit %(graph_rows)s)
    order by ts desc, txn_id desc
"""

# The merchant side feeds the graph only -- velocity reads user history alone --
# so it is a pure row-count lookup with no age bound, matching the offline
# merchant node.
_MERCHANT_HISTORY_SQL = f"""
    select {_HISTORY_COLUMNS}
      from transaction_events
     where merchant_id = %(key)s
       and ts < %(now)s
     order by ts desc, txn_id desc
     limit %(graph_rows)s
"""


# Velocity counts every row in its window, so the fetch must not truncate one.
# A user averaging 1.4 transactions a day has ~10 rows in 168h; this cap is
# generous enough to be unreachable in practice while still bounding a
# pathological card.
_HISTORY_ROW_CAP = 2000


def fetch_neighbourhood(
    conn: psycopg.Connection,
    user_id: int,
    merchant_id: int,
    now: dt.datetime,
    history_hours: int,
    graph_rows: int,
) -> Neighbourhood:
    """The card's and the merchant's recent past, two indexed reads.

    `history_hours` bounds VELOCITY; `graph_rows` bounds the GRAPH. They are
    separate because they measure different things -- see _USER_HISTORY_SQL.

    Both queries ride `idx_txe_user_ts` / `idx_txe_merchant_ts`, which already
    existed for velocity -- the neighbourhood fetch added no new index.
    """
    since = now - dt.timedelta(hours=history_hours)
    with conn.cursor() as cur:
        cur.execute(
            _USER_HISTORY_SQL,
            {
                "key": user_id,
                "now": now,
                "since": since,
                "limit": _HISTORY_ROW_CAP,
                "graph_rows": graph_rows,
            },
        )
        user_rows = cur.fetchall()
        cur.execute(
            _MERCHANT_HISTORY_SQL,
            {"key": merchant_id, "now": now, "graph_rows": graph_rows},
        )
        merchant_rows = cur.fetchall()

    return Neighbourhood(
        user_history=_to_frame(user_rows),
        merchant_history=_to_frame(merchant_rows),
    )


# The same two reads as one statement, for the one-round-trip serving path.
# Each arm keeps its own placeholders (the user and merchant keys differ) and
# its own order; `side` says which neighbourhood a row belongs to, and the
# outer ORDER BY restores each side's order, because a subquery's ORDER BY does
# not survive a UNION.
_NEIGHBOURHOOD_SQL = f"""
    select 'user' as side, u.* from (
        {_USER_HISTORY_SQL.replace("%(key)s", "%(user)s")}
    ) u
    union all
    select 'merchant' as side, m.* from (
        {_MERCHANT_HISTORY_SQL.replace("%(key)s", "%(merchant)s")}
    ) m
    order by side desc, ts desc, txn_id desc
"""


def fetch_neighbourhood_once(
    conn: psycopg.Connection,
    user_id: int,
    merchant_id: int,
    now: dt.datetime,
    history_hours: int,
    graph_rows: int,
) -> Neighbourhood:
    """`fetch_neighbourhood` in one round trip instead of two.

    Same predicates, same bounds, same order -- tests/test_latency.py holds
    the two together on a fixture. Run on an autocommit connection, it also
    avoids the BEGIN that a transactional connection sends first.
    """
    since = now - dt.timedelta(hours=history_hours)
    with conn.cursor() as cur:
        cur.execute(
            _NEIGHBOURHOOD_SQL,
            {
                "user": user_id,
                "merchant": merchant_id,
                "now": now,
                "since": since,
                "limit": _HISTORY_ROW_CAP,
                "graph_rows": graph_rows,
            },
        )
        rows = cur.fetchall()
    user_rows = [{k: v for k, v in r.items() if k != "side"}
                 for r in rows if r["side"] == "user"]
    merchant_rows = [{k: v for k, v in r.items() if k != "side"}
                     for r in rows if r["side"] == "merchant"]
    return Neighbourhood(
        user_history=_to_frame(user_rows),
        merchant_history=_to_frame(merchant_rows),
    )


def explain_fetch_ms(
    conn: psycopg.Connection,
    user_id: int,
    merchant_id: int,
    now: dt.datetime,
    history_hours: int,
    graph_rows: int,
) -> float:
    """Server-side execution time of one neighbourhood fetch, in ms.

    The same statements `fetch_neighbourhood` issues, under EXPLAIN ANALYZE:
    Postgres's own clock, with no network in it. The latency replay uses this
    to cross-check what it derives by subtracting round trips from wall time.
    """
    since = now - dt.timedelta(hours=history_hours)
    total = 0.0
    with conn.cursor() as cur:
        for sql, args in (
            (_USER_HISTORY_SQL, {"key": user_id, "now": now, "since": since,
                                 "limit": _HISTORY_ROW_CAP,
                                 "graph_rows": graph_rows}),
            (_MERCHANT_HISTORY_SQL, {"key": merchant_id, "now": now,
                                     "graph_rows": graph_rows}),
        ):
            cur.execute(f"explain (analyze, format json) {sql}", args)
            plan = cur.fetchone()["QUERY PLAN"][0]
            total += float(plan["Execution Time"]) + float(plan["Planning Time"])
    conn.rollback()
    return total


_INSERT_SQL = """
    insert into transaction_events (
        txn_id, user_id, merchant_id, ts, amount, mcc,
        merchant, city, state, zip, errors, chip, time_min, month, day,
        velocity_count_1h, velocity_amount_1h, velocity_merchants_1h,
        velocity_states_1h, velocity_count_24h, velocity_amount_24h,
        velocity_merchants_24h, velocity_states_24h, velocity_count_168h,
        velocity_amount_168h, velocity_merchants_168h, velocity_states_168h,
        velocity_seconds_since_last
    ) values (
        %(txn_id)s, %(user_id)s, %(merchant_id)s, %(ts)s, %(amount)s, %(mcc)s,
        %(merchant)s, %(city)s, %(state)s, %(zip)s, %(errors)s, %(chip)s,
        %(time_min)s, %(month)s, %(day)s,
        %(velocity_count_1h)s, %(velocity_amount_1h)s, %(velocity_merchants_1h)s,
        %(velocity_states_1h)s, %(velocity_count_24h)s, %(velocity_amount_24h)s,
        %(velocity_merchants_24h)s, %(velocity_states_24h)s,
        %(velocity_count_168h)s, %(velocity_amount_168h)s,
        %(velocity_merchants_168h)s, %(velocity_states_168h)s,
        %(velocity_seconds_since_last)s
    )
    on conflict (txn_id) do nothing
"""


def insert_transaction(
    conn: psycopg.Connection, row: dict[str, Any], velocity: dict[str, float]
) -> None:
    """Make a transaction visible to *future* predictions.

    Called after scoring, never before. Storing the velocity computed at scoring
    time is what lets a later request reuse this row as a neighbour: those values
    were derived from `ts < this row's ts`, so they cannot contain anything this
    row could not legitimately have seen.
    """
    with conn.cursor() as cur:
        cur.execute(_INSERT_SQL, {**row, **velocity})


# Columns a stored row carries, in the order _INSERT_SQL writes them. The bulk
# loader reuses this list so a seeded row and a live insert cannot drift apart.
STORED_COLUMNS = (
    "txn_id", "user_id", "merchant_id", "ts", "amount", "mcc",
    "merchant", "city", "state", "zip", "errors", "chip", "time_min", "month",
    "day", *velocity_columns([1, 24, 168]),
)


def stored_row(row: dict[str, Any], velocity: dict[str, float]) -> dict[str, Any]:
    """A transaction (canonical names) plus its velocity, as the table stores it."""
    return {
        "txn_id": row["txn_id"],
        "user_id": row["User"],
        "merchant_id": int(row["Merchant"]),
        "ts": row["ts"],
        "amount": row["Amount"],
        "mcc": row["MCC"],
        "merchant": row["Merchant"],
        "city": row["City"],
        "state": row["State"],
        "zip": row["Zip"],
        "errors": row["Errors"],
        "chip": row["Chip"],
        "time_min": row["Time"],
        "month": row["Month"],
        "day": row["Day"],
        **velocity,
    }


def reset_store(conn: psycopg.Connection) -> None:
    """Empty the hot store. Only the latency replay calls this, behind a flag.

    The store must hold nothing at or after the replay cursor, and a previous
    run leaves exactly that behind.
    """
    with conn.cursor() as cur:
        cur.execute("truncate transaction_events, predictions")
    conn.commit()


def bulk_load_history(
    conn: psycopg.Connection, rows: list[dict[str, Any]]
) -> int:
    """Seed the store with prior history in one COPY stream.

    `rows` are `stored_row()` dicts. One COPY rather than one INSERT each: from
    a Studio ~70 ms away, 40k inserts would take most of an hour.
    """
    columns = ", ".join(STORED_COLUMNS)
    sql = f"copy transaction_events ({columns}) from stdin"
    with conn.cursor() as cur, cur.copy(sql) as copy:
        for row in rows:
            copy.write_row(tuple(row[c] for c in STORED_COLUMNS))
    conn.commit()
    return len(rows)


def assert_causal(conn: psycopg.Connection, cursor_ts: dt.datetime) -> int:
    """Fail loudly if the store holds anything at or after the replay cursor.

    The one way a live simulation leaks: seed the store with the test period,
    replay over it, and every prediction is clairvoyant while looking entirely
    normal -- same code, same queries, plausible latencies, flattering metrics.
    """
    with conn.cursor() as cur:
        cur.execute("select assert_store_is_causal(%s)", (cursor_ts,))
        row = cur.fetchone()
    return int(row["assert_store_is_causal"]) if row else 0


def log_prediction(conn: psycopg.Connection, prediction: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(_LOG_PREDICTION_SQL, prediction)


_LOG_PREDICTION_SQL = """
    insert into predictions (
        txn_id, score, decision, model_version, latency_ms,
        embedding_age_seconds, cold_start_user, cold_start_merchant
    ) values (
        %(txn_id)s, %(score)s, %(decision)s, %(model_version)s,
        %(latency_ms)s, %(embedding_age_seconds)s,
        %(cold_start_user)s, %(cold_start_merchant)s
    )
    on conflict (txn_id) do nothing
"""

# Log and insert as ONE statement: a data-modifying CTE runs whether or not its
# result is used, and a single statement in autocommit is its own transaction,
# so the pair is still all-or-nothing -- in one round trip instead of three
# (log, insert, commit).
_LOG_AND_INSERT_SQL = f"""
    with logged as ({_LOG_PREDICTION_SQL})
    {_INSERT_SQL}
"""


def log_and_insert(
    conn: psycopg.Connection,
    prediction: dict[str, Any],
    row: dict[str, Any],
    velocity: dict[str, float],
) -> None:
    """`log_prediction` + `insert_transaction` + commit, in one round trip.

    Still strictly after scoring: the row becomes visible to the next request,
    never to its own. `conn` must be in autocommit.
    """
    if not conn.autocommit:
        raise RuntimeError("log_and_insert needs an autocommit connection")
    with conn.cursor() as cur:
        cur.execute(_LOG_AND_INSERT_SQL, {**row, **velocity, **prediction})
