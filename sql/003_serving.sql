-- Widen the hot store so it can actually serve the GraphSAGE champion.
-- Apply after 002_challenger.sql.
--
-- Phase 3 reversed the serving design (CLAUDE.md decision 3). Scoring the graph
-- end to end means a request needs its neighbours' FULL feature vectors, not
-- just their ids: a GNN transaction node carries 62 features, and 13 of those
-- are the neighbour's own velocity aggregates as of the neighbour's timestamp.
--
-- The original table carried txn_id / user_id / merchant_id / ts / amount / mcc,
-- which cannot reconstruct any of that.
--
-- Two things are added:
--
--   1. The raw columns the encoder consumes. Raw, not the encoded 62-wide
--      vector: ~80 B/row instead of ~248 B/row, and the encoder stays the single
--      source of truth for what a column means. Re-encoding on read costs
--      microseconds.
--
--   2. The 13 velocity aggregates, written at insert time. This is the causal
--      part: velocity is already computed in order to score the transaction, and
--      it was computed over `ts < that row's ts`. Storing it means a later
--      request reusing this row as a neighbour gets a value that could only ever
--      have seen that row's own past. Recomputing a neighbour's velocity at read
--      time would be both slower and easier to get wrong.

alter table transaction_events
    -- Encoder inputs. `merchant` is the hashed id as text: it spans nearly the
    -- full int64 range and must never round-trip through a float.
    add column if not exists merchant   text,
    add column if not exists city       text,
    add column if not exists state      text,
    add column if not exists zip        integer,
    add column if not exists errors     text,
    add column if not exists chip       text,
    add column if not exists time_min   smallint,   -- minutes since midnight
    add column if not exists month      smallint,
    add column if not exists day        smallint,

    -- Velocity, computed when this row was scored. Named to match
    -- fraud.features.velocity.velocity_columns() so the mapping is mechanical.
    add column if not exists velocity_count_1h            real,
    add column if not exists velocity_amount_1h           real,
    add column if not exists velocity_merchants_1h        real,
    add column if not exists velocity_states_1h           real,
    add column if not exists velocity_count_24h           real,
    add column if not exists velocity_amount_24h          real,
    add column if not exists velocity_merchants_24h       real,
    add column if not exists velocity_states_24h          real,
    add column if not exists velocity_count_168h          real,
    add column if not exists velocity_amount_168h         real,
    add column if not exists velocity_merchants_168h      real,
    add column if not exists velocity_states_168h         real,
    add column if not exists velocity_seconds_since_last  real;


-- The two reads a /predict does are both "this card's / this merchant's recent
-- rows, strictly before now". idx_txe_user_ts and idx_txe_merchant_ts from
-- 001_init.sql already cover them; this is the covering index for the payload so
-- the neighbourhood fetch does not need a heap lookup per row.
create index if not exists idx_txe_user_ts_covering
    on transaction_events (user_id, ts desc)
    include (merchant_id, amount, mcc, state);


-- The replay's safety net, in the database rather than in Python.
--
-- Leakage in a live simulation has exactly one cause: a row being visible to a
-- prediction made before it existed. Bulk-loading the test period and then
-- replaying would do that invisibly -- same code, same queries, plausible
-- latencies, flattering metrics. This function makes the invariant checkable:
-- nothing in the store may be at or after the replay cursor.
create or replace function assert_store_is_causal(cursor_ts timestamptz)
returns bigint
language plpgsql
as $$
declare
    violations bigint;
begin
    select count(*) into violations
    from transaction_events
    where ts >= cursor_ts;

    if violations > 0 then
        raise exception
            'causality violated: % rows in transaction_events at or after the '
            'replay cursor %. The store must only ever contain the past.',
            violations, cursor_ts;
    end if;
    return violations;
end;
$$;
