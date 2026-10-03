-- Phase 5: monitoring writes for more than one model.
--
-- drift_metrics was written for a single champion, so a row did not say which
-- model it describes. Two models are now monitored (XGBoost @champion,
-- GraphSAGE @challenger) and their estimates must not mix. Additive and
-- nullable, so rows written before this migration stay valid.

alter table drift_metrics
    add column if not exists model text;

create index if not exists idx_drift_model_chunk
    on drift_metrics (model, metric, chunk_start);
