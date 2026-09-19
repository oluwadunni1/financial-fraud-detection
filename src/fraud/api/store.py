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
    return psycopg.connect(connection_string(), row_factory=dict_row)


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
                **{c: pl.Int64 for c in ("Zip", "Time", "Month", "Day")},
            }
        )
    # Canonical names here, not in the caller: subgraph.py and the predictor
    # should not need to know what the database columns are called.
    return pl.DataFrame(rows).rename(_DB_TO_CANONICAL)


_HISTORY_SQL = """
    select txn_id, user_id, ts, amount, merchant, state, merchant_id,
           city, zip, errors, chip, time_min, month, day,
           velocity_count_1h, velocity_amount_1h, velocity_merchants_1h,
           velocity_states_1h, velocity_count_24h, velocity_amount_24h,
           velocity_merchants_24h, velocity_states_24h, velocity_count_168h,
           velocity_amount_168h, velocity_merchants_168h, velocity_states_168h,
           velocity_seconds_since_last
      from transaction_events
     where {key} = %(key)s
       and ts < %(now)s          -- STRICTLY before. See the module docstring.
       and ts >= %(since)s       -- the velocity window; older rows affect nothing
     order by ts desc
     limit %(limit)s
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
) -> Neighbourhood:
    """The card's and the merchant's recent past, one indexed read each.

    Bounded by **time**, not by row count. That distinction is the difference
    between correct velocity and silent skew: `velocity_count_168h` is a count
    over a window, so fetching "the last 10 rows" would under-report it for any
    card busier than that, and the model would see a number training never
    produced. The graph takes the most recent few of these rows; velocity needs
    all of them.

    Both queries ride `idx_txe_user_ts` / `idx_txe_merchant_ts`, which already
    existed for velocity -- the neighbourhood fetch added no new index.
    """
    since = now - dt.timedelta(hours=history_hours)
    with conn.cursor() as cur:
        cur.execute(
            _HISTORY_SQL.format(key="user_id"),
            {"key": user_id, "now": now, "since": since, "limit": _HISTORY_ROW_CAP},
        )
        user_rows = cur.fetchall()
        cur.execute(
            _HISTORY_SQL.format(key="merchant_id"),
            {
                "key": merchant_id,
                "now": now,
                "since": since,
                "limit": _HISTORY_ROW_CAP,
            },
        )
        merchant_rows = cur.fetchall()

    return Neighbourhood(
        user_history=_to_frame(user_rows),
        merchant_history=_to_frame(merchant_rows),
    )


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
        cur.execute(
            """
            insert into predictions (
                txn_id, score, decision, model_version, latency_ms,
                embedding_age_seconds, cold_start_user, cold_start_merchant
            ) values (
                %(txn_id)s, %(score)s, %(decision)s, %(model_version)s,
                %(latency_ms)s, %(embedding_age_seconds)s,
                %(cold_start_user)s, %(cold_start_merchant)s
            )
            on conflict (txn_id) do nothing
            """,
            prediction,
        )
