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

### Two remotes, on purpose

| Remote | Repo | Role |
|---|---|---|
| `standalone` | `oluwadunni1/fraud-detection-mlops` | **The real home.** Not a fork, so commits count toward the contribution graph. Work lands on its `main`. |
| `origin` | `oluwadunni1/financial-fraud-detection` | The fork. Keeps `main` as a clean NVIDIA upstream mirror; work mirrors to `mlops-platform`. |

Push both: `git push standalone mlops-platform:main && git push origin mlops-platform`.

GitHub excludes **all** commits in forked repositories from the contribution
graph, regardless of branch -- which is why the standalone repo exists. Counting
needs all three of: not a fork, on the default branch, author email verified on
the account. All three are satisfied; verified via the API that commits are
attributed to `oluwadunni1`.

- In the fork, `main` tracks NVIDIA upstream. **Do not commit work there.**
- `mlops-platform` is the working branch locally.

## Read this first

**`docs/ARCHITECTURE.md`** is the reference doc: system design, storage budget, phased
build plan, verification strategy. Read it before proposing changes. Section 6 has the
phases; section 10 is the Lightning AI setup.

## Current state (update as phases complete)

**Phase 3 modelling complete (2026-09-19).** GraphSAGE beats the XGBoost champion
**0.5614 vs 0.2501** test-2019 AUC-PR -- relational structure matters, and by 2.2x. But
flattening it to embeddings-as-features destroys the gain (0.1351), so the section 3.1
serving design cannot deliver it. Promotion deferred; see ARCHITECTURE section 6 Phase 3.

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
| Features | **Done** -- 86 columns (69 encoded + 4 numeric + 13 velocity) |
| Champion | **Registered** -- `models:/fraud-champion@champion`, AUC-PR 0.3190 |
| Tests | **117 passing** across 7 files |

Measured dataset facts now live in `docs/ARCHITECTURE.md` section 2.1. Read that before
writing any feature code -- several of them contradict what the scaffolding assumed.

## Decisions already made — do not relitigate

These were settled deliberately. Reopen only if new evidence appears.

1. **The NVIDIA training container is not used.** It is NGC-gated and opaque; you cannot
   version or instrument what you cannot see. The GNN is rebuilt in open PyTorch Geometric.
2. **Triton is not in the serving path.** CPU-servable champion behind FastAPI instead.
3. ~~**Serving uses precomputed embeddings + live velocity features**, not live
   neighbourhood lookup.~~ **REVERSED 2026-09-19.** Serving fetches the card's and
   merchant's recent transactions and scores the GNN **end to end** over a ~25-node
   subgraph, alongside live velocity aggregates. The original decision was made on latency
   grounds before we had evidence: frozen embeddings score **0.1351** against a 0.2113
   base, while the same GNN end to end scores **0.6474**. Flattening does not merely fail
   to help, it costs the entire 2x. The subgraph is tiny, the model is ~30k parameters, and
   `transaction_events` already carries `idx_txe_user_ts` / `idx_txe_merchant_ts` -- the
   exact two lookups needed, already there for velocity. Cost: torch in the API image
   (~200 MB CPU wheel) and a latency budget that must be re-measured against the <100ms
   target.
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
14. **Model promotion uses MLflow aliases, not stages.** ARCHITECTURE said "promote to
   Production"; MLflow 3 deprecates stages. The API loads
   `models:/fraud-champion@champion`, so swapping the champion is an alias move with no
   redeploy. Verified working on DagsHub.
15. **Sweep `scale_pos_weight` per model; never inherit it.** Measured on the baseline:
   10 beats the guessed 50 by 29% headline AUC-PR and cuts missed frauds 63%. Full inverse
   prevalence (819) is the *worst* value on the grid -- "balance the classes" is a trap
   here. Re-sweep for the GNN and the foundation model.
16. **Do not port a preprocessing step just because the blueprint has one.** Their
   non-fraud dedupe cost **19x** on test once we evaluated on the real distribution -- it
   strips the repeated legitimate behaviour that is most of real traffic. Harmless for them
   because they evaluate on the deduped distribution too. Check every inherited step
   against *our* evaluation, not theirs.
17. **The GNN's value does not survive flattening into embeddings.** End to end it scores
   0.5614 on test 2019; its embeddings bolted onto XGBoost score 0.1351, *worse than no
   embeddings at all* (0.2113). Section 3.1's precomputed-embedding serving design cannot
   deliver the 2x, so Phase 4 must resolve how to serve the GNN before it can be promoted.

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
- **`to_hetero` FX-traces the model, so `F.dropout(training=...)` bakes the flag in as a
  constant** -- dropout stays ON during eval and silently corrupts every validation score.
  Use `nn.Dropout`. Likewise `torch.manual_seed` does not reach `NeighborLoader`'s workers:
  pass an explicit generator or the same config scores differently run to run.
- **The blueprint's graph edges are one-directional.** `user->txn` and `txn->merchant` mean
  a transaction can aggregate from its user but never its merchant -- half the graph
  silently unreachable. Apply `ToUndirected`.
- **"Do not load the full graph onto the GPU"** was sized for a 24.4M-node graph and 9.6 GB
  of features. After under-sampling the training graph is 277k nodes and peak VRAM is
  **154 MB**, so the constraint no longer binds -- but keep `NeighborLoader` for the 2M-node
  val/test graphs.
- **Every velocity feature must use ONE notion of "previous".** `seconds_since_last`
  originally used `ts.diff()` (which includes same-timestamp rows) while the window
  aggregates used `closed="left"` (which excludes them). Serving queries `ts < :now` and
  would have disagreed with training on 142,010 rows, silently. It is now bounded by the
  largest window and strictly-earlier, matching everything else. Cost: champion test AUC-PR
  0.3190 -> 0.2501, because the old number depended on a burst signal serving cannot see.
- **Velocity is `closed="left"` offline and `ts < :now` online.** Both mean *strictly
  earlier*, so two transactions in the same minute are mutually invisible. 142,010 rows
  share a user and a minute, so this is not a corner case. A `<=` in the online SQL would
  let the later of a pair see the earlier while the earlier saw nothing -- asymmetric, and
  invisible without the skew test. See `src/fraud/features/velocity.py`.
- **Velocity windows cross year partitions.** A 7-day window on 2018-01-02 needs December
  2017, so velocity is computed in *user* chunks, never per year partition. Computing it
  per partition silently zeroes every window at each year boundary.
- **The encoder is fitted on train only and ships with the model.** Re-fitting it on a
  different split changes what every column means. Train sees 93,298 of 100,343 merchants,
  so ~0.46% of test rows carry an unseen merchant and encode to the reserved all-zero
  code -- that is the cold-start path, exercised for real.
- **`Year` is deliberately not a feature.** The split is temporal, so no training row
  carries 2019 or 2020; a tree cannot extrapolate and every test row would land in one
  terminal bucket. `Month` and `Day` are fine -- they recur.
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
