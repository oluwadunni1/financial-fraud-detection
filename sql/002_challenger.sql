-- Adds the foundation-model challenger alongside the GNN champion.
-- Apply after 001_init.sql.

-- ---------------------------------------------------------------------------
-- Foundation-model embeddings.
--
-- Separate table rather than a column on node_embeddings because pgvector
-- dimensions are fixed per column: the GNN emits 64-d, this emits 512-d.
--
-- Users only. The sequence model has no notion of a merchant node, so the
-- assembled feature vector differs in shape from the GNN path -- store.py
-- handles both.
--
-- ~2k users x 512 float32 = ~4 MB. Storage is a non-issue here.
-- ---------------------------------------------------------------------------
create table if not exists fm_embeddings (
    node_id       bigint      primary key,
    embedding     vector(512) not null,
    computed_at   timestamptz not null default now(),
    model_version text        not null
);

create index if not exists idx_fm_computed_at on fm_embeddings (computed_at);


-- ---------------------------------------------------------------------------
-- Shadow mode: the challenger scores every request alongside the champion,
-- but only the champion's decision is acted on. Both are recorded so the two
-- can be compared on real traffic at zero risk.
--
-- Nullable throughout -- when no challenger is deployed these stay empty.
-- ---------------------------------------------------------------------------
alter table predictions
    add column if not exists challenger_score         real,
    add column if not exists challenger_version       text,
    add column if not exists challenger_latency_ms    real;

-- Comparison queries scan champion-vs-challenger over a time window.
create index if not exists idx_pred_challenger
    on predictions (challenger_version, predicted_at)
    where challenger_version is not null;
