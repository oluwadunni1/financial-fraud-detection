# CLAUDE.md

## What this repo is

A fork of the **NVIDIA AI Blueprint: Financial Fraud Detection**, being extended on the
`mlops-platform` branch into an end-to-end MLOps pipeline (DVC, MLflow, Pandera, NannyML,
FastAPI, Supabase, Docker).

- `main` tracks NVIDIA upstream. **Do not commit work there.**
- `mlops-platform` is the working branch.

## Read this first

**`docs/ARCHITECTURE.md`** is the reference doc: system design, storage budget, phased
build plan, verification strategy. Read it before proposing changes. Section 6 has the
phases; section 10 is the Lightning AI setup.

## Current state (update as phases complete)

Phase 0, partially done. Scaffolding committed; nothing runs end to end yet.

| | Status |
|---|---|
| Scaffold (`pyproject.toml`, `params.yaml`, `src/fraud/`, `sql/`) | Done |
| TabFormer data downloaded | **No** |
| Python env installed | **No** |
| DagsHub repo + token | **No** |
| Supabase project + `sql/001_init.sql` applied | **No** |
| DVC initialised | **No** |

## Decisions already made — do not relitigate

These were settled deliberately. Reopen only if new evidence appears.

1. **The NVIDIA training container is not used.** It is NGC-gated and opaque; you cannot
   version or instrument what you cannot see. The GNN is rebuilt in open PyTorch Geometric.
2. **Triton is not in the serving path.** CPU-servable champion behind FastAPI instead.
3. **Serving uses precomputed embeddings + live velocity features**, not live neighbourhood
   lookup. Frozen user/merchant embeddings from Postgres, plus velocity aggregates computed
   live in SQL. See ARCHITECTURE.md section 3.1 for why.
4. **Supabase is a rolling hot store, not an archive.** Full 24M-row history stays in
   Parquet on the DVC remote. Free tier is 500 MB; the full table is ~5.6 GB.
5. **MLflow and DVC are hosted on DagsHub.**
6. **XGBoost baseline ships before the GNN**, so serving/monitoring/deploy phases are not
   blocked behind GPU work.
7. **Pandera over Great Expectations** — same value, far less config.

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
- **`node_embeddings.model_version` must match the serving head.** v1 embeddings are
  meaningless to a v2 model. They version and deploy together.
- **Fraud rate is ~0.1%.** Accuracy is a useless metric here. Use AUC-PR, precision@k, and
  recall at a fixed FPR.
- **Write Parquet, not CSV.** The blueprint's `to_csv` calls would turn a ~3 GB feature
  matrix into ~30 GB.
- **Do not load the full graph onto the GPU.** Use PyG `NeighborLoader`; features stay in
  CPU RAM and are gathered per batch.

## Unresolved

- Does DagsHub's hosted MLflow support the **Model Registry** API? If not, fall back to
  tag-based champion resolution in `src/fraud/models/registry.py`. Phase 2 depends on this.
- Deploy target (Cloud Run recommended). Build Compose-first until Phase 6.
- Second TabFormer model for comparison — repo link not yet provided.
- All storage estimates in ARCHITECTURE.md section 4.4 are **unmeasured**. Correct them once
  the CSV lands.

## Commands

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"              # add api,gnn,monitoring as phases need them
pytest                                   # tests
ruff check src tests                     # lint
dvc repro                                # run the pipeline
```

GPU is needed only for Phase 3. Run other phases on a CPU machine to conserve credits.
