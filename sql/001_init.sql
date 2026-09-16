-- Supabase schema for the fraud detection platform.
-- Apply via: Supabase SQL editor, or `psql $DATABASE_URL -f sql/001_init.sql`

create extension if not exists vector;

-- ---------------------------------------------------------------------------
-- Precomputed GNN node embeddings.
--
-- model_version is NOT optional: embeddings produced by model v1 are
-- meaningless to a v2 serving head. They version and deploy together.
-- ---------------------------------------------------------------------------
create table if not exists node_embeddings (
    node_type     text        not null check (node_type in ('user', 'merchant')),
    node_id       bigint      not null,
    embedding     vector(64)  not null,
    computed_at   timestamptz not null default now(),
    model_version text        not null,
    primary key (node_type, node_id)
);

-- Freshness SLI queries scan by age.
create index if not exists idx_embeddings_computed_at
    on node_embeddings (computed_at);


-- ---------------------------------------------------------------------------
-- HOT transaction store. Deliberately narrow and rolling.
--
-- This is NOT the transaction history -- the full 24M rows live in Parquet on
-- the DVC remote. Supabase free tier is 500 MB; the full table plus indexes is
-- ~5.6 GB, so it would not fit and does not need to.
--
-- Velocity windows top out at 7 days, so serving never reads anything older
-- than that. Retention drops the tail (see prune_transaction_events below).
-- Only the columns the velocity aggregates actually consume are stored.
-- ---------------------------------------------------------------------------
create table if not exists transaction_events (
    txn_id      bigint      primary key,
    user_id     bigint      not null,
    merchant_id bigint      not null,
    ts          timestamptz not null,
    amount      real        not null,
    mcc         integer     not null
);

-- These two indexes are what keep the live velocity aggregates inside the
-- request budget. Without them /predict degrades to a sequential scan.
create index if not exists idx_txe_user_ts     on transaction_events (user_id, ts desc);
create index if not exists idx_txe_merchant_ts on transaction_events (merchant_id, ts desc);


-- Retention: keep a rolling window, drop the rest. Called by the scheduled
-- jobs container. Default 30d gives the 7d velocity window generous slack.
create or replace function prune_transaction_events(retain interval default '30 days')
returns bigint
language plpgsql
as $$
declare
    removed bigint;
begin
    delete from transaction_events where ts < now() - retain;
    get diagnostics removed = row_count;
    return removed;
end;
$$;


-- ---------------------------------------------------------------------------
-- Every prediction the API makes. This is the monitoring substrate: NannyML
-- reads it, the latency dashboard reads it, the staleness experiment reads it.
-- ---------------------------------------------------------------------------
create table if not exists predictions (
    txn_id                bigint      primary key,
    score                 real        not null,
    decision              boolean     not null,
    model_version         text        not null,
    latency_ms            real        not null,
    embedding_age_seconds integer,              -- null when cold start
    cold_start_user       boolean     not null default false,
    cold_start_merchant   boolean     not null default false,
    predicted_at          timestamptz not null default now()
);

create index if not exists idx_pred_predicted_at on predictions (predicted_at);
create index if not exists idx_pred_model        on predictions (model_version);


-- ---------------------------------------------------------------------------
-- Late-arriving ground truth. Simulates chargebacks: labeled_at is materially
-- later than the transaction, which is the whole reason NannyML's label-free
-- performance estimation is needed.
-- ---------------------------------------------------------------------------
create table if not exists labels (
    txn_id     bigint      primary key,
    is_fraud   boolean     not null,
    labeled_at timestamptz not null default now()
);

create index if not exists idx_labels_labeled_at on labels (labeled_at);


-- ---------------------------------------------------------------------------
-- NannyML output: estimated performance and drift, per chunk.
-- ---------------------------------------------------------------------------
create table if not exists drift_metrics (
    id          bigserial   primary key,
    run_at      timestamptz not null default now(),
    metric      text        not null,   -- e.g. 'estimated_roc_auc', 'jensen_shannon'
    feature     text,                   -- null for model-level metrics
    value       real        not null,
    lower_bound real,
    upper_bound real,
    alert       boolean     not null default false,
    chunk_start timestamptz,
    chunk_end   timestamptz
);

create index if not exists idx_drift_run_at on drift_metrics (run_at);
create index if not exists idx_drift_metric on drift_metrics (metric, chunk_start);


-- ---------------------------------------------------------------------------
-- Watermark for incremental jobs (embedding refresh, label simulation).
-- ---------------------------------------------------------------------------
create table if not exists job_watermarks (
    job_name     text        primary key,
    last_run_at  timestamptz not null,
    last_cursor  timestamptz,
    rows_processed bigint    not null default 0
);
