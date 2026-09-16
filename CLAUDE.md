# CLAUDE.md

## What this repo is

A fork of the **NVIDIA AI Blueprint: Financial Fraud Detection**, being extended on the
`mlops-platform` branch into an end-to-end MLOps pipeline (DVC, MLflow, Pandera, NannyML,
FastAPI, Supabase, Docker).

Three models compete through one serving contract and one registry:
1. **XGBoost** on tabular features -- baseline, ships first
2. **GraphSAGE** -- relational structure, ported from the blueprint
3. **Foundation model** -- sequential structure, from
   https://github.com/oluwadunni1/transaction-foundation-model (decoder-only Llama,
   ~29M params, frozen feature extractor)

The comparison is the point: does relational or sequential structure matter more for
card fraud? Champion/challenger swapping is what gives the registry a real job.

- `main` tracks NVIDIA upstream. **Do not commit work there.**
- `mlops-platform` is the working branch.

## Read this first

**`docs/ARCHITECTURE.md`** is the reference doc: system design, storage budget, phased
build plan, verification strategy. Read it before proposing changes. Section 6 has the
phases; section 10 is the Lightning AI setup.

## Current state (update as phases complete)

**Phase 1 complete (2026-09-16).** `dvc repro` now runs the whole pipeline from the raw
CSV. Next up is Phase 2 (feature engineering + the XGBoost baseline).

| | Status |
|---|---|
| Scaffold (`pyproject.toml`, `params.yaml`, `src/fraud/`, `sql/`) | Done |
| TabFormer data downloaded | **Yes** -- 24,386,900 rows, 2.35 GB, DVC-tracked |
| Python env installed | **Yes** -- `.venv`, CPython 3.12.13, `[dev]` extras |
| Schemas validated against real data | **Yes** -- `Amount` regex bug found and fixed |
| DVC initialised | **Yes** -- local cache, **no remote** (deliberate, see below) |
| DagsHub repo + token | **Verified** -- tracking + Model Registry both work |
| Supabase project + `sql/*.sql` applied | **Done** -- eu-west-1, pgvector + 7 tables verified |
| Pipeline (`dvc.yaml`) | **Done** -- `ingest` -> `validate` / `split` / `sample` |
| Processed data | **Done** -- 30 year partitions, 538 MB Parquet (from 2.35 GB CSV) |
| Tests | **50 passing** (`test_schemas`, `test_ingest`, `test_split`) |

Measured dataset facts now live in `docs/ARCHITECTURE.md` section 2.1. Read that before
writing any feature code -- several of them contradict what the scaffolding assumed.

## Decisions already made — do not relitigate

These were settled deliberately. Reopen only if new evidence appears.

1. **The NVIDIA training container is not used.** It is NGC-gated and opaque; you cannot
   version or instrument what you cannot see. The GNN is rebuilt in open PyTorch Geometric.
2. **Triton is not in the serving path.** CPU-servable champion behind FastAPI instead.
3. **Serving uses precomputed embeddings + live velocity features**, not live neighbourhood
   lookup. Frozen user/merchant embeddings from Postgres, plus velocity aggregates computed
   live in SQL. See ARCHITECTURE.md section 3.1 for why.
4. **Supabase is a rolling hot store, not an archive.** The full 24.4M-row history stays
   in DVC-tracked Parquet (`data/processed/`, 538 MB). Free tier is 500 MB; the full table
   plus indexes measures ~6.2 GB, 12.4x over.
5. **MLflow is hosted on DagsHub.** DVC is not -- see decision 10, which supersedes the
   earlier "DVC remote on DagsHub" plan.
6. **XGBoost baseline ships before the GNN**, so serving/monitoring/deploy phases are not
   blocked behind GPU work.
7. **Pandera over Great Expectations** — same value, far less config.
8. **Do not pretrain the foundation model.** The repo ships a 56 MB checkpoint; pretraining
   needs 8x A100 and would burn the whole Lightning budget. Forward pass only.
9. **No full orchestrator** (Airflow/Dagster/Prefect-server). The DVC DAG plus GitHub
   Actions scheduled workflows plus a cron container cover it. Adding a scheduler +
   webserver + metadata DB is the classic overengineering trap. Prefect Cloud free tier
   if an orchestration UI is genuinely wanted.
10. **DagsHub hosts MLflow only -- it is not the DVC remote.** DVC runs with a local cache
   and **no remote at all** for now: the raw CSV is re-downloadable, the big graph
   artifacts are regenerated rather than pushed, and model artifacts go to MLflow. A
   remote earns its place when a second machine needs `dvc pull` (the Phase 3 GPU Studio,
   or CI), and at that point it is **Cloudflare R2** -- S3-compatible so `dvc[s3]` already
   covers it, no egress fees. Commands are in ARCHITECTURE.md section 4.3.
11. **The 2016+ GNN subsample stays, but the storage justification is dead.** The feature
   matrix measures 7.32 GB fp32, not the estimated 9.6 GB, and there is no DagsHub ceiling
   to hit any more. Justify the subsample on modelling grounds only.
12. **2020 stays in the test set, but metrics are reported two ways.** Its 336,500 rows
   carry zero positives, so including them drops test prevalence from 0.121% to 0.101% and
   lowers precision at any threshold -- a movement unrelated to model quality. Headline
   **AUC-PR is 2019-only**; precision@k, alert volume and recall-at-fixed-FPR use
   2019+2020. Both variants are recorded in `reports/splits.json` under `test_variants`.
13. **The split is a manifest, not three copies of the data.** Year-partitioned Parquet
   makes a split a predicate resolved by partition pruning. Always load a split via
   `fraud.data.split.load_split(name)`; never hand-write a year filter.

## Conventions

- **All config lives in `params.yaml`.** Nothing hardcoded in `src/`.
- **`src/preprocess_TabFormer*.py` are read-only reference.** They are the NVIDIA originals
  (cuDF, GPU-only, ~80% duplicated). Port their *logic* into `src/fraud/features/`; do not
  edit or import them.
- src-layout package: `from fraud.features import velocity`.
- Pipeline stages are DVC stages in `dvc.yaml`. If it is not a stage, it is not the pipeline.

## Gotchas that will silently corrupt results

- **Never use a random train/test split.** Transactions are time-ordered; a random split
  leaks the future. Temporal only: train <=2017, val 2018, test >=2019.
- **Train/serve skew on velocity features is the top project risk.** They must be computed
  by the same code offline and online — one module in `src/fraud/features/velocity.py`,
  two callers, with a test asserting they agree on fixed input.
- **`model_version` must match the serving head.** v1 embeddings are meaningless to a v2
  model. They version and deploy together — this is also what makes model swapping safe.
- **The two embedding sources have different shapes.** GNN gives 64-d user *and* merchant
  embeddings; the foundation model gives 512-d user embeddings only. Separate tables
  (pgvector dims are fixed per column) and different assembled feature vectors.
- **Fraud rate is ~0.1%.** Accuracy is a useless metric here. Use AUC-PR, precision@k, and
  recall at a fixed FPR.
- **Write Parquet, not CSV.** The blueprint's `to_csv` calls would turn a ~3 GB feature
  matrix into ~30 GB.
- **Refunds are `$-292.00`, not `-$292.00`.** The minus is *inside* the dollar sign, on
  1,244,689 rows (5.1%). The original schema regex assumed the US convention and rejected
  every refund. Parse with `^\$-?\d+\.\d{2}$`.
- **The CSV is ordered by user, not by time.** The first 200k rows span 1999-2020.
  Partitioning by `Year` is a full shuffle; never assume file order is time order.
- **`Merchant Name` is a hashed int64 near the full range and often negative.** It must
  never round-trip through a float -- it exceeds float64's exact-integer range. `MCC` is
  the unrelated 4-digit category code.
- **2020 has zero fraud** in all 336,500 rows and the file stops at 2020-02-28, so a
  `Year >= 2019` test set draws every positive from 2019 alone.
- **Reading the 2.35 GB CSV with pandas' default inference yields mixed dtypes.** Pass
  explicit `dtype=`; a silent `object` column will break validation in confusing ways.
- **Do not load the full graph onto the GPU.** Use PyG `NeighborLoader`; features stay in
  CPU RAM and are gathered per batch.
- **`txn_id` is synthetic and order-dependent.** It is the row's index in the original CSV,
  assigned in one sequential pass. There is no natural key: `(User, Card, Year, Month, Day,
  Time)` collides on 142,010 rows and 66 rows are exact duplicates. Anything that reorders
  or filters rows before ingest silently renumbers every transaction -- and `predictions`,
  `labels` and `transaction_events` all key on it.
- **`source .venv/bin/activate` does NOT change which `python` runs.** The shell profile
  activates conda and zsh caches the lookup, so a bare `python` silently stays on
  `/home/zeus/miniconda3/envs/cloudspace/bin/python` even though `which python` says
  otherwise. Always write `.venv/bin/python`, `.venv/bin/dvc`, and
  `uv pip install --python .venv/bin/python`. This already split the dependencies across two
  environments once. See ARCHITECTURE.md 10.1.

## Unresolved

- Deploy target (Cloud Run recommended). Build Compose-first until Phase 6.
- Is the foundation-model checkpoint loadable as standard Llama in plain `transformers`?
  If yes, skip the NeMo container entirely — pip install vs multi-GB image. Check early,
  it changes the Phase 3.5 environment.
- ~~All storage estimates in ARCHITECTURE.md section 4.4 are unmeasured.~~ **Resolved
  2026-09-16** -- section 4.4 now carries measured figures.

## Commands

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"   # add api,gnn,monitoring per phase

# Always invoke the venv explicitly -- activate is not enough here (see gotchas).
.venv/bin/python -m pytest                         # tests
.venv/bin/python -m ruff check src tests scripts   # lint
.venv/bin/dvc data status                          # tracked data clean?
.venv/bin/dvc repro                                # the pipeline: ingest/validate/split/sample
.venv/bin/dvc dag                                  # show the DAG
.venv/bin/python scripts/verify_services.py        # DagsHub + Supabase, 11 checks
```

GPU is needed only for Phase 3. Run other phases on a CPU machine to conserve credits.
