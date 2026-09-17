# Fraud Detection MLOps Platform — Architecture & Build Plan

> The reference doc for this repo: system design, storage budget, phased build plan,
> verification strategy. Section 6 has the phases; section 10 is the Lightning AI setup.
> Last updated: 2026-09-16 (Phase 0 complete -- estimates replaced by measurements)

---

## 1. Context

### What we're building
An end-to-end, industry-standard MLOps pipeline for credit-card fraud detection, using
the NVIDIA Financial Fraud Detection blueprint as the **modelling starting point** rather
than the runtime.

The deliverable is a live demo: transactions replay through a deployed API, get scored by
a champion model pulled from a registry, and every prediction is logged, monitored for
drift, and measured for latency — with the whole thing reproducible from raw data by a
single command.

### Why this dataset / problem
Fraud is a genuinely good MLOps showcase because it has three properties most portfolio
projects lack:

- **Extreme class imbalance** (~0.1% positive) → accuracy is meaningless, forces AUC-PR,
  proper thresholding, and cost-sensitive evaluation.
- **Delayed labels** — chargebacks arrive weeks after the transaction → you cannot compute
  live accuracy. This is what makes NannyML's CBPE *load-bearing* rather than decorative.
- **Real concept drift** — fraud tactics actively adapt, so monitoring and retraining
  triggers have a real justification.

### What we inherit from the NVIDIA blueprint
| Keep | Drop |
|---|---|
| TabFormer dataset + the fraud framing | `financial-fraud-training` container (NGC-gated black box) |
| Graph construction logic (`src/preprocess_TabFormer_lp.py`) | Triton serving (GPU-gated, overkill for demo) |
| Tri-partite User→Transaction→Merchant modelling | cuDF hard dependency (no Windows, forces GPU for *preprocessing*) |
| Temporal split strategy (train <=2017 / val 2018 / test >=2019) | The notebook-driven workflow |

**Why drop the training container:** you cannot version, instrument, or MLflow-log what you
can't see inside. Rebuilding the GNN in open PyG is the entire point of the exercise.

---

## 2. Where we are right now

**Phases 0 and 1 complete (2026-09-16).** Every number in this doc that was previously an
estimate has been replaced by a measurement against the real file. Where a figure is
still derived rather than observed, it says so.

| | Status |
|---|---|
| Repo | Fork at `github.com/oluwadunni1/financial-fraud-detection`, branch `mlops-platform` |
| Scaffolded | `pyproject.toml`, `params.yaml`, `src/fraud/`, `sql/*.sql`, `tests/`, this doc |
| Data | **Downloaded, DVC-tracked, and processed.** 24,386,900 rows; 2.35 GB CSV -> 538 MB Parquet |
| Pipeline | **`dvc repro` runs end to end**: `ingest` -> `validate` / `split` / `sample` |
| Dev machine | Lightning AI Studio -- Linux, 14 GB RAM, 4 vCPU, **no GPU** (correct for Phases 0-2) |
| Python env | `.venv` on CPython 3.12.13. **Invoke as `.venv/bin/python`** -- see 10.1 |
| GPU | Lightning credits available; needed only for Phase 3 |
| MLflow | **DagsHub hosted and verified** -- tracking + Model Registry both confirmed working |
| DVC | Initialised, local cache only. **No remote, deliberately** -- see 4.3 |
| Supabase | **Live** (eu-west-1). Both migrations applied; pgvector + all 7 tables verified |
| Deploy target | Undecided -- Compose-first, cloud-agnostic (see section 7) |

### 2.1 The dataset, measured

`card_transaction.v1.csv`, 2,354,626,737 bytes, 24,386,900 rows, 15 columns.

| | Measured |
|---|---|
| Rows | 24,386,900 |
| Fraud | 29,757 (**0.122%**, i.e. 1 in 819) |
| Users | 2,000 |
| (User, Card) pairs | 6,139 |
| Merchants | 100,343 |
| MCC codes | 109 (range 1711-9402) |
| Cities / States | 13,429 / 223 |
| Distinct Zip | 27,322 |
| `Use Chip` | 3 -- Swipe / Chip / Online |
| `Errors?` | 23 distinct non-null, **98.4% null** (23,998,469 rows) |
| Null `Zip` / `Merchant State` | 2,878,135 (11.8%) / 2,720,821 (11.2%) |
| Year range | 1991-2020 |

**Graph node count is therefore 102,343** (2,000 users + 100,343 merchants), which sizes
`node_embeddings` exactly.

#### Four things the file does that the scaffolding assumed otherwise

1. **Refunds are `$-292.00`, not `-$292.00`** -- the minus sits *inside* the dollar sign.
   1,244,689 rows (5.1%) are negative. The original `RawTransactionSchema` regex assumed
   the US convention and would have failed validation on every refund in the dataset.
   Corrected regex: `^\$-?\d+\.\d{2}$`, verified to match all 24,386,900 rows.
2. **The file is ordered by user, not by time.** The first 200k rows span 1999-2020.
   Partitioning by `Year` in Phase 1 is a full shuffle, not a sequential cut -- do not
   assume file order is time order anywhere.
3. **`Merchant Name` is a hashed int64 spanning nearly the full range**
   (-9,222,899,435,637,403,521 .. 9,223,291,803,303,717,674), so it is frequently negative
   and **must never round-trip through a float** -- it exceeds float64's exact-integer
   range. `MCC` is the unrelated 4-digit *category* code.
4. **Reading with pandas' default type inference produces mixed dtypes** on a file this
   size. Phase 1's `ingest.py` must pass explicit `dtype=` rather than rely on inference.

#### The fraud rate is wildly non-stationary -- this matters for the split

| Split | Years | Rows | Fraud | Rate |
|---|---|---|---|---|
| train | 1991-2017 | 20,604,847 | 25,179 | 0.122% |
| val | 2018 | 1,721,615 | 2,491 | 0.145% |
| test | 2019-2020 | 2,060,438 | 2,087 | 0.101% |

Per-year rates swing by two orders of magnitude: 0.0035% (2011) to 0.303% (2008). Two
specific hazards:

> **2020 contains zero fraud** across all 336,500 rows, and the file ends 2020-02-28. It is
> a partial year that contributes only negatives to the test set. Consider testing on 2019
> alone, or state plainly that test positives come entirely from 2019.

> **2017 is a 14x outlier low** (255 fraud in 1,723,360 rows = 0.015%) sitting immediately
> before the train/val boundary, where 2018 jumps back to 0.145%. Training ends on an
> anomalously clean year and validates on a ~10x dirtier one. This is real, exploitable
> concept drift for the monitoring story -- but it means a val-set metric drop is not
> automatically a modelling bug, and `scale_pos_weight` tuned on 2017-heavy data will be
> miscalibrated for 2018.

Both are dataset properties, not pipeline bugs. Name them in the README.

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  OFFLINE  (DVC pipeline — reproducible, versioned)                   │
│                                                                      │
│   raw CSV ──▶ validate ──▶ features ──▶ build_graph                  │
│              (Pandera)       │              │                        │
│                              │              ├─▶ edges.parquet        │
│                              │              ├─▶ node_features.parquet│
│                              │              └─▶ labels.parquet       │
│                              │                                       │
│                    ┌─────────┴─────────┐                             │
│                    ▼                   ▼                             │
│              train_xgboost        train_gnn  ◀── Lightning AI (GPU)  │
│                    │                   │                             │
│                    └────────┬──────────┘                             │
│                             ▼                                        │
│                     MLflow Tracking ──▶ Model Registry               │
│                             │              (champion / challenger)   │
│                             ▼                                        │
│                     precompute_embeddings                            │
│                             │                                        │
└─────────────────────────────┼────────────────────────────────────────┘
                              │ upsert
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SUPABASE POSTGRES (+ pgvector)                                      │
│                                                                      │
│   node_embeddings │ transactions │ predictions │ labels │ drift      │
└──────┬──────────────────┬───────────────────────────┬────────────────┘
       │ read             │ read/write                │ read/write
       ▼                  ▼                           ▼
┌──────────────┐   ┌──────────────┐          ┌─────────────────┐
│  ONLINE      │   │  Refresh Job │          │  NannyML Job    │
│  FastAPI     │   │  (scheduled) │          │  (scheduled)    │
│              │   │              │          │                 │
│ /predict     │   │ forward-pass │          │ CBPE estimated  │
│ /explain     │   │ new embeds,  │          │ perf + drift,   │
│ /health      │   │ upsert w/    │          │ no labels req'd │
│ /metrics     │   │ computed_at  │          │                 │
└──────┬───────┘   └──────────────┘          └─────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  Demo dashboard (Streamlit)  │
│  replay · scores · drift ·   │
│  latency · staleness curve   │
└──────────────────────────────┘
```

### 3.1 The serving path (the core design decision)

Settled in discussion: **precomputed embeddings + live velocity features.**

```
txn arrives
  ├─▶ user_emb      ← Supabase lookup (pgvector, indexed)   [cold-start → zero vector + flag]
  ├─▶ merchant_emb  ← Supabase lookup                        [cold-start → zero vector + flag]
  ├─▶ velocity aggs ← live SQL window over recent txns
  │     (count 1h/24h, amount sum, distinct merchants, distinct states, time since last)
  └─▶ txn features  ← from request payload
         │
         ▼
   concat ──▶ champion model ──▶ score
         │
         └──▶ log to predictions(score, latency_ms, embedding_age_seconds, model_version,
                                 cold_start_user, cold_start_merchant)
```

**Why this split:** freezing user/merchant embeddings loses the fastest-moving fraud signal
(velocity bursts, freshly-breached merchants). The live SQL aggregates recover exactly that,
so staleness degrades gracefully instead of blinding the model. Structural signal moves
slowly and is safe to batch; velocity signal moves in seconds and is computed on demand.

**Cold start** is a first-class path, not an error: unseen user/merchant → no embedding →
fall back to tabular-only scoring, flagged in the response and logged. The accuracy gap
between warm and cold predictions is a demo metric.

### 3.2 Database schema (Supabase)

Canonical definition lives in `sql/001_init.sql`. Shape:

| Table | Role |
|---|---|
| `node_embeddings` | Precomputed GNN embeddings, `vector(64)`, users + merchants |
| `fm_embeddings` | Foundation-model embeddings, `vector(512)`, users only |
| `transaction_events` | **Narrow, rolling hot store** for velocity aggregates -- not the history |
| `predictions` | Every score, with `latency_ms`, `embedding_age_seconds`, cold-start flags |
| `labels` | Late-arriving ground truth (simulated chargebacks) |
| `drift_metrics` | NannyML output |
| `job_watermarks` | Incremental-job cursors |

Two things that are not optional:

> **`model_version` on every embedding table.** Embeddings from model v1 are meaningless
> to a v2 serving head. They version and deploy together. This is also the mechanism that
> makes champion/challenger swapping safe.

> **Two embedding tables, not one.** pgvector columns are fixed-dimension, so 64-d GNN
> and 512-d foundation-model embeddings cannot share a column.

> **`transaction_events` is a hot store, not an archive.** It holds a rolling window
> (`prune_transaction_events()`); the full 24M-row history lives in Parquet on the
> DVC remote. See section 4.4 for why.

### 3.3 Two cadences, don't conflate them

| | Recompute embeddings | Retrain model |
|---|---|---|
| What changes | Node embeddings | Model weights |
| Cost | Cheap, CPU-fine | Expensive, GPU |
| Where | Scheduled job | Lightning AI |
| Cadence | Hourly/daily (measure it, §6.2) | On drift trigger or scheduled |
| Trigger | Freshness SLO breach | NannyML estimated-perf drop |

---

## 4. Tech Stack & Rationale

| Layer | Choice | Why |
|---|---|---|
| Data versioning | **DVC** | `dvc.yaml` gives a reproducible DAG — this *is* the pipeline, not just storage |
| Validation | **Pandera** (GE optional) | 90% of GE's value, 10% of the config. Revisit if GE on the résumé matters |
| Experiments | **MLflow** | Tracking + Model Registry + champion promotion |
| Graph ML | **PyTorch Geometric** | Open, readable, `NeighborLoader` for sampling |
| Baseline | **XGBoost** | CPU-servable, strong tabular baseline, honest comparison point |
| GPU training | **Lightning AI** | Credits available |
| Database | **Supabase** (Postgres + pgvector) | Embedding store + prediction log + hot serving store, one system. MLflow lives on DagsHub, not here |
| API | **FastAPI** + Pydantic | Async, auto OpenAPI docs, typed validation |
| Monitoring | **NannyML** | CBPE estimates performance *without labels* — the delayed-label problem |
| Explainability | **SHAP** | `/explain` endpoint; mirrors blueprint's Shapley story |
| Container | **Docker** + Compose | Local parity with cloud |
| CI | **GitHub Actions** | Lint, test, `dvc repro` on sample, build image |

### 4.1 Storage sizing
Superseded by section 4.4, which covers all three tiers against measured data.

### 4.2 GPU memory note
Do not load the full graph onto the GPU. `NeighborLoader` samples subgraphs; the feature
matrix stays in CPU RAM and is gathered per batch. That makes 9.6 GB of features trainable
on a 24 GB card.

### 4.3 MLflow hosting -- DagsHub. DVC remote -- none yet.

**MLflow lives on DagsHub.** Hosted tracking plus Model Registry, free, nothing to self-host.

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<user>
export MLFLOW_TRACKING_PASSWORD=<dagshub-token>
```

> **Confirmed (2026-09-16):** DagsHub's hosted MLflow supports the Model Registry API,
> so champion/challenger promotion uses the real registry. No tag-based fallback needed.

**DVC has no remote, and that is deliberate.** DVC is initialised with a local cache only.
DagsHub is *not* used as the DVC remote.

The reasoning: a remote exists to move data between machines. Right now nothing needs
moving. The raw CSV is re-downloadable from IBM Box in about a minute; the large graph
artifacts are explicitly regenerated on the GPU Studio rather than pushed (see 4.4); model
artifacts and embeddings go to MLflow, not the DVC remote. That leaves a handful of small
Parquet files that do not exist yet. Standing up remote storage now would be versioning
infrastructure with nothing to version.

A remote earns its place the moment a **second machine** needs `dvc pull` -- i.e. the
Phase 3 GPU Studio, or CI. At that point it is **Cloudflare R2**: S3-compatible, so the
already-installed `dvc[s3]` covers it with no new dependency, 10 GB free, and no egress
fees (the trap with S3 proper when a GPU Studio pulls the same features repeatedly).

```bash
dvc remote add -d r2 s3://<bucket>
dvc remote modify r2 endpointurl https://<account-id>.r2.cloudflarestorage.com
dvc remote modify --local r2 access_key_id     <key>
dvc remote modify --local r2 secret_access_key <secret>
```

> Until then, **the DVC cache is the only copy of any derived artifact**, and a Studio's
> local disk is not a backup. Nothing irreplaceable should live only there -- which is
> true today, since everything is either re-downloadable or regenerable.

> With no pipeline stages yet, `dvc status` reports "no data or pipelines tracked". The
> data-level check is **`dvc data status`**, which reports "No changes." Once `dvc.yaml`
> exists in Phase 1, `dvc status` becomes the right command.

### 4.4 Storage budget -- measured

All figures below are computed from the real file (section 2.1) unless marked *derived*.
Three tiers, because no single free tier holds everything.

```
DagsHub (MLflow only)      Lightning Studio drive       Supabase (500 MB DB)
-----------------------    ----------------------       --------------------------
MLflow tracking+registry   raw CSV (2.35 GB)            transaction_events (rolling)
model artifacts            .dvc/cache (2.2 GB)          node_embeddings   (64-d, GNN)
embeddings (~38 MB)        node_features.parquet <-BIG  fm_embeddings     (512-d, FM)
                           edges.parquet                predictions / labels
                           FM checkpoint (56 MB, LFS)   drift_metrics
```

The Studio drive has 348 GB free, so local capacity is not a constraint. The binding
constraint is Supabase's 500 MB.

#### Why the full history cannot live in Postgres

The `transactions` table at all 24,386,900 rows (*derived* from measured row count):

| | Size |
|---|---|
| Table (~134 B/row incl. tuple overhead) | ~3.27 GB |
| 3 btree indexes (~40 B/entry effective) | ~2.93 GB |
| **Total** | **~6.2 GB vs a 500 MB limit -- 12.4x over** |

The argument survives measurement, and got slightly stronger (the earlier estimate was
5.6 GB / 11x).

#### What actually belongs in Postgres

Velocity windows top out at 7 days, so **serving never reads a transaction older than
the longest window**. Older rows belong in Parquet. Supabase becomes a rolling hot
store with a retention job -- which is what production does anyway.

| Table | Sizing basis | Est. |
|---|---|---|
| `transaction_events` (narrow, rolling ~30d, cap 500k) | 40 B payload + 2 indexes | ~74 MB |
| `node_embeddings` (**102,343** x vector(64)) | ~330 B/row | **~34 MB** |
| `fm_embeddings` (**2,000** x vector(512)) | ~2.1 KB/row | **~4 MB** |
| `predictions` (cap 200k) | ~100 B/row | ~20 MB |
| `labels` | ~50 B/row | ~10 MB |
| `drift_metrics`, `job_watermarks` | negligible | ~2 MB |
| **Total hot store** | | **~144 MB** (3.5x headroom) |

The node counts are now exact rather than assumed, and the earlier ~102k guess was
almost exactly right. Levers if it gets tight: `halfvec(64)` halves the GNN embeddings;
drop `city`/`errors` from the hot table (velocity does not use them).

> **Free-tier gotcha:** Supabase pauses projects after 7 days idle, which would
> silently break a demo link. The scheduled refresh/NannyML jobs hit the DB, so a
> daily cron keeps it awake as a side-effect. Make that explicit, do not rely on luck.

#### The GNN feature matrix is smaller than estimated

Encoded width, applying `params.yaml: features.one_hot_max_cardinality: 8` to the measured
cardinalities (one-hot below 8, binary at or above):

| Column | Cardinality | Encoding | Columns |
|---|---|---|---|
| Merchant | 100,343 | binary | 17 |
| Zip | 27,322 | binary | 15 |
| City | 13,429 | binary | 14 |
| State | 224 | binary | 8 |
| MCC | 109 | binary | 7 |
| Errors | 24 | binary | 5 |
| Chip | 3 | one-hot | 3 |
| Amount, Hour, Minute, Year, Month, Day | -- | scaled | 6 |
| **Total** | | | **75** |

| | Rows | fp32 | fp16 |
|---|---|---|---|
| Full history | 24,386,900 | **7.32 GB** | 3.66 GB |
| 2016+ subsample | 7,214,337 | **2.16 GB** | 1.08 GB |

*Derived: final width lands in Phase 1 once the encoder is actually fitted.*

> **This weakens the stated reason for subsampling.** Section 4.4 previously justified
> the 2016+ cut by a 9.6 GB matrix not fitting in DagsHub's ~10 GB. The real matrix is
> **7.32 GB**, DagsHub is no longer the DVC remote, and the Studio has 348 GB free -- so
> the *storage* argument no longer binds. The *modelling* argument still does: recent data
> is more representative and 2016+ still contains ~8,400 fraud cases, which is ample.
> Keep the subsample, but justify it on modelling grounds in the README. Do not repeat the
> storage rationale, which measurement has retired.

Big graph artifacts are still **regenerated on the Studio** via `dvc repro build_graph`
rather than stored anywhere central -- with no DVC remote that is now the only option,
and it was already the plan.

## 5. Repo Structure (target)

```
financial-fraud-detection/
├── data/                     # DVC-tracked, gitignored contents
├── dvc.yaml                  # the pipeline DAG
├── params.yaml               # all hyperparameters / config
├── src/
│   ├── data/
│   │   ├── ingest.py         # CSV → partitioned Parquet
│   │   ├── schemas.py        # Pandera schemas
│   │   └── split.py          # temporal split (no random split!)
│   ├── features/
│   │   ├── tabular.py        # ported from preprocess_TabFormer_lp.py, cuDF → pandas/polars
│   │   ├── graph.py          # edge list + node feature construction
│   │   └── velocity.py       # live SQL aggregates (shared offline/online)
│   ├── models/
│   │   ├── xgb.py
│   │   ├── gnn.py            # GraphSAGE (PyG)
│   │   └── registry.py       # MLflow promotion helpers
│   ├── jobs/
│   │   ├── refresh_embeddings.py
│   │   └── run_nannyml.py
│   └── api/
│       ├── main.py           # FastAPI app
│       ├── schemas.py        # Pydantic request/response
│       └── store.py          # Supabase client
├── dashboard/                # Streamlit demo
├── tests/
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.jobs
│   └── docker-compose.yml
├── .github/workflows/ci.yml
└── docs/ARCHITECTURE.md      # this file
```

> **Note on `src/preprocess_TabFormer*.py`:** the three existing scripts are ~80% duplicated
> with no shared module. We port the *logic* into `src/features/`, we don't extend them.
> Keep the originals untouched as reference.

---

## 6. Build Phases

Each phase ends with something that runs. Don't start the next until the current one does.

### Phase 0 — Foundations — **COMPLETE (2026-09-16)**
- [x] Linux dev loop (Lightning AI Studio, not WSL2 -- the migration made WSL2 moot)
- [x] `uv` project on Python 3.12.11, `uv pip install -e ".[dev]"`
- [x] Download TabFormer, verify row count and fraud rate (24,386,900 rows / 0.122%)
- [x] DVC init; raw CSV tracked. **No remote** -- see 4.3 for why
- [x] Schemas validated against the real file; `Amount` regex bug found and fixed
- [x] `tests/test_schemas.py` -- 26 tests, corruption cases rejected
- [x] Storage estimates in 4.4 replaced with measurements
- [x] Supabase project created, schema from 3.2 applied and verified
- [x] DagsHub MLflow connectivity + **Model Registry** smoke test (register, alias, read back)
- **Done when:** `dvc data status` is clean and raw data is tracked. **Met.**

Re-check any time with `.venv/bin/python scripts/verify_services.py` (11 checks).

### Phase 1 — Data & validation — **COMPLETE (2026-09-16)**
- [x] `ingest.py`: CSV -> Parquet partitioned by year (one streaming pass, 33.6s, 48 batches)
- [x] Pandera schemas in the build path: raw validated per batch, clean per partition
- [x] Temporal split (train <=2017 / val 2018 / test >=2019) with no-leakage assertions
- [x] DVC stages: `ingest` -> `validate` / `split` / `sample`
- [x] `sample` stage: 200,004-row stratified subset, 4.1 MB, committed to git for CI
- **Done when:** `dvc repro` runs clean, and a deliberately corrupted row fails the build.
  **Met** -- both verified, including the negative test.

Outputs: `data/processed/Year=1991..2020` (538 MB Parquet, down from 2.35 GB CSV),
`reports/validation_report.json`, `reports/splits.json`,
`data/sample/transactions.parquet`.

| Split | Years | Rows | Fraud | Rate |
|---|---|---|---|---|
| train | 1991-2017 | 20,604,847 | 25,179 | 0.122% |
| val | 2018 | 1,721,615 | 2,491 | 0.145% |
| test | 2019-2020 | 2,060,438 | 2,087 | 0.101% |

> **Evaluation note.** 2020 is kept (336,500 rows, zero positives) because the extra
> negatives make the false-positive burden realistic and give Phase 4/5 a long stretch of
> replay traffic. It is not metric-neutral: it drops test prevalence from 0.121% to 0.101%.
> So headline **AUC-PR is 2019-only**, while precision@k, alert volume and recall at fixed
> FPR use 2019+2020. `reports/splits.json` records both under `test_variants`.

> **Load splits via `fraud.data.split.load_split(name)`.** The split is a manifest plus a
> predicate over the year-partitioned Parquet, not three copies of the data. Hand-writing a
> year filter is how an accidental random split gets in.

### Phase 2 — Baseline model — **COMPLETE (2026-09-17)**
- [x] Ported tabular feature engineering (cuDF -> polars), logic not library
- [x] Velocity features, brought forward from Phase 4 (see below)
- [x] `scale_pos_weight` chosen on val by sweep, not guessed
- [x] XGBoost + MLflow logging: AUC-PR, precision@k, recall at fixed FPR, confusion matrix
- [x] Registered as `fraud-champion` v1 under the `champion` alias
- **Done when:** a champion exists in the registry with real metrics. **Met.**

Load it with `mlflow.pyfunc.load_model("models:/fraud-champion@champion")`.

#### Results

86 features, `scale_pos_weight=10`, best iteration 228, threshold 0.0360 (val @ 1% FPR).

| Split | Rows | Positives | Base rate | AUC-PR | AUC-ROC | P@100 | P@1000 |
|---|---|---|---|---|---|---|---|
| val (2018) | 1,721,615 | 2,491 | 0.00145 | 0.5181 | 0.9974 | 1.000 | 0.709 |
| **test 2019 (headline)** | 1,723,938 | 2,087 | 0.00121 | **0.3190** | 0.9973 | 0.450 | 0.449 |
| test 2019+2020 | 2,060,438 | 2,087 | 0.00101 | 0.2855 | 0.9974 | 0.400 | 0.413 |

**Headline AUC-PR is 263x the base rate.** At the operating point the model catches 1,990
of 2,087 frauds (95.4% recall) for 17,025 false positives.

#### `scale_pos_weight` was guessed, and the guess was costly

`params.yaml` shipped `scale_pos_weight: 50` under a comment claiming it was "tuned on val,
not guessed". It was guessed. Sweeping it (`src/fraud/models/sweep.py`,
`reports/sweep_scale_pos_weight.json`) on a prevalence-preserving subsample:

| `scale_pos_weight` | val AUC-PR (sweep) |
|---|---|
| 1 | 0.3175 |
| **10** | **0.3215** |
| 50 (the guess) | 0.2714 |
| 200 | 0.1789 |
| 819 (full inverse prevalence) | 0.1169 |

Retraining at 10 on full data:

| | spw=50 | spw=10 | change |
|---|---|---|---|
| headline AUC-PR | 0.2481 | **0.3190** | **+29%** |
| P@100 | 0.410 | 0.450 | +10% |
| recall at 1% FPR | 0.873 | **0.954** | +9pp |
| missed frauds | 265 | **97** | **-63%** |
| training time | 40 min | 11.5 min | early stop at 228 |

> **Reweighting to full inverse prevalence (819) is the worst option on the grid.** That is
> the standard trap with extreme imbalance: "balance the classes" sounds principled and
> destroys precision. The sweep is cheap and should be repeated for the GNN and the
> foundation model rather than inheriting this value.

> **Two traps in sweeping it.** First, a *stratified* subsample that keeps all positives
> raises training prevalence, and `scale_pos_weight` is defined relative to that ratio -- so
> the winner does not transfer. The sweep uses a uniform subsample for this reason.
> Second, a low round cap flatters high weights: they converge faster on the minority class.
> Watch a low weight look worse early and overtake -- spw=10 trailed spw=50 at round 50, and
> had beaten its 446-round peak by round 100.

#### Why ROC-AUC is never the headline

The same model scores **0.997 AUC-ROC** and **0.319 AUC-PR**. The first reads as essentially
solved; the second says under a third of the ranked alerts are real. At a 1-in-819 base rate
ROC-AUC is dominated by the true-negative pool and flatters everything. Report it for
comparability with published numbers, never as the headline.

#### The 2020 dilution, as predicted

Adding 336,500 fraud-free rows adds 3,118 false positives and zero true positives, moving
P@100 from 0.450 to 0.400 and precision at the operating point from 0.105 to 0.090. The
model did not change. This is why the headline is 2019-only (decision 12).

> **A 1% FPR budget is loose at this volume:** ~17,000 false positives for ~2,000 catches,
> about 9 false alarms per fraud. Precision@k is the more honest operational lens. Phase 5
> should revisit the target rather than inherit 1% unexamined.

#### Cost, measured

| Stage | Wall clock | Peak RSS |
|---|---|---|
| `velocity` | 29s | 1.7 GB |
| `features` | 185s | 2.8 GB |
| `train_xgb` | 688s | 5.9 GB |

Trained on CPU deliberately: the Lightning GPU budget is reserved for Phase 3, and a booster
trained on GPU still serves on CPU, so this is a credits trade rather than a correctness
one. `QuantileDMatrix` walks the data iterator **4 times** (sketching, then filling), which
dominates both the wall clock and the memory.

> Deliberately before the GNN: it gives a servable model early, so Phases 4–6 can proceed
> without waiting on GPU work, and it's the honest comparison baseline.

### Phase 3 — Graph & GNN (Lightning AI)
- [ ] Port graph construction → `edges/node_features/labels` Parquet
- [ ] GraphSAGE in PyG with `NeighborLoader`
- [ ] Lightning Studio: `dvc pull` → train → log to remote MLflow → push artifacts
- [ ] Compare vs XGBoost; promote champion on merit, not on novelty
- [ ] `precompute_embeddings.py` → upsert user/merchant embeddings to Supabase
- **Done when:** embeddings are in Postgres with `computed_at` + `model_version`.

### Phase 3.5 — Foundation model challenger (Lightning AI)

Source: https://github.com/oluwadunni1/transaction-foundation-model -- a decoder-only
Llama (~29M params) pretrained on TabFormer sequences via causal LM, used as a frozen
feature extractor (last-token pooling -> 512-d embeddings -> XGBoost).

- [ ] **Do NOT pretrain.** The repo ships a 56 MB checkpoint via Git LFS; pretraining
      needs 8x A100 and would burn the entire Lightning budget. Forward pass only.
- [ ] Check the checkpoint format first. If it loads as standard Llama in plain
      `transformers`, skip the NeMo container entirely -- that is a pip install versus
      a multi-GB image, and it keeps the CPU-servable story intact.
- [ ] Extract user-level embeddings -> `fm_embeddings` table (vector(512))
- [ ] Train the downstream head; log to MLflow as a **challenger**, not a champion
- [ ] Record serving profile alongside accuracy: this model can run precomputed
      (stale, cheap) *or* live (fresh, slower). Measure both -- it is a real tradeoff.
- **Done when:** three models sit in the registry with comparable metrics, and the
  challenger is servable through the same `/predict` contract as the champion.

#### Why this is cheap to add
Both approaches have an **identical serving contract**: precompute embeddings offline,
store, look up per request, feed a downstream head. That is the seam the architecture
already has, so nothing needs redesigning. What differs is only the inductive bias.

| | GNN | Foundation model |
|---|---|---|
| Learns from | relational structure (who transacts with whom) | sequential structure (order of a user's transactions) |
| Embedding | 64-d, users + merchants | 512-d, **users only** |
| Storage | ~34 MB | ~4 MB (2k users x 512 floats) |

> **Asymmetry to plan for, not discover in Phase 4:** the GNN yields user *and* merchant
> embeddings; the sequence model yields user embeddings only. The assembled feature
> vectors differ. `store.py` must handle both shapes.

### Phase 4 — Serving API
- [ ] FastAPI: `/predict`, `/explain`, `/health`, `/metrics`
- [ ] Embedding lookup + **cold-start fallback path**
- [ ] Live velocity aggregates (`velocity.py`, shared with offline to avoid train/serve skew)
- [ ] Prediction logging incl. `latency_ms` + `embedding_age_seconds`
- [ ] Load model from MLflow registry by stage, not by path
- **Done when:** `curl` returns a score in <100ms and the row lands in Postgres.

> **Train/serve skew is the top risk here.** Velocity features must be computed by the same
> code offline and online. One module, two callers.

### Phase 5 — Monitoring
- [ ] `refresh_embeddings.py` — watermarked incremental job
- [ ] Label simulator: reveal ground truth on a delay (simulated chargebacks)
- [ ] NannyML CBPE + multivariate drift → `drift_metrics`
- [ ] Freshness SLI: `max(now - computed_at)`
- [ ] **Staleness experiment** (see §6.2)
- **Done when:** artificially drifted traffic shows up as a NannyML estimated-perf drop.

### Phase 6 — CI/CD, model swapping & deploy

With two genuinely competing models the registry stops being ceremony and becomes the
centre of the project. This is where that pays off.

- [ ] `Dockerfile.api` (slim, multi-stage), `Dockerfile.jobs`
- [ ] Compose: api + dashboard + scheduler
- [ ] **Shadow mode** -- challenger scores every request alongside champion; only the
      champion's decision is used, both are logged. Highest value-per-unit-work item in
      the project: real-traffic comparison at zero risk.
- [ ] **Promotion gate in CI** -- challenger must beat champion on held-out AUC-PR
      before it can be promoted. Automated and auditable.
- [ ] **Registry-driven swap** -- flip the MLflow stage, the API picks up the new
      champion with no redeploy. This is the demo moment.
- [ ] **Rollback path** -- NannyML degradation alert -> revert to previous champion
- [ ] k6/locust load test -> throughput + p50/p95/p99, per model
- [ ] GitHub Actions: lint, test, `dvc repro` on a sample, eval, build/push image
- [ ] Deploy API to cloud; schedule jobs
- **Done when:** swapping the champion in the registry changes live scoring behaviour
  without a redeploy, and CI blocks a promotion that fails the gate.

> **Deliberately NOT adding a full orchestrator.** Airflow/Dagster/Prefect-server means
> running a scheduler, a webserver and a metadata DB -- substantial infra for a demo and
> the classic overengineering trap. The DVC DAG already expresses the pipeline; GitHub
> Actions scheduled workflows plus a cron container cover the recurring jobs. If an
> orchestration UI is wanted, **Prefect Cloud free tier** gives it with zero infra.

### Phase 7 — Demo & docs
- [ ] Streamlit: replay stream, live scores, drift charts, latency, staleness curve
- [ ] README with architecture diagram and honest limitations section
- **Done when:** a stranger can follow the README end to end.

### Phase 8 — Stretch
- [ ] Neo4j/Memgraph purely as a visual graph explorer (presentation asset, not compute)
- [ ] Tiered refresh (hot entities hourly, long tail weekly)

### 6.2 The staleness experiment (do not skip)
Hold the model fixed, artificially age the embeddings, plot **AUC-PR vs embedding age**
(0h / 6h / 24h / 7d / 30d), with and without velocity features.

Delivers: an empirically justified refresh SLA, a quantified defence of the hybrid design,
and a direct NannyML hook — stale embeddings fail *silently* (no error, just worse scores),
which is precisely what label-free performance estimation is for.

---

## 7. Deployment target — open decision

Building Compose-first and cloud-agnostic, so this can stay open until Phase 6.

| Option | Case for it |
|---|---|
| **Cloud Run** (recommended) | Scale-to-zero, generous free tier, Docker-native, ~$0 idle |
| Fly.io / Railway | Simplest DX; Railway can also host MLflow |
| AWS ECS/App Runner | Best signal if targeting AWS shops; most config overhead |

---

## 8. Honest limitations (state these in the README)

Naming these reads stronger than letting a reviewer find them:

1. **TabFormer has only ~2,000 users.** Recomputing 2k embeddings is nearly free, which
   hides the hard part of real-scale refresh (tens of millions of entities → tiered
   cadence). We demo the pattern, not the scale.
2. **Synthetic data** — drift is simulated, not organic.
3. **Labels are simulated as delayed**, not genuinely delayed.
4. **Single-node throughput** — no Kafka, no horizontal scale story.
5. We **do not reproduce NVIDIA's numbers** — different framework, CPU serving, different
   feature pipeline. This is inspired by the blueprint, not a reimplementation of it.
6. **The models are reference implementations, not original work.** The graph approach is
   ported from the blueprint and the foundation model is a pretrained NVIDIA checkpoint
   used frozen. The original contribution is the platform: the pipeline, serving
   architecture, monitoring, and the champion/challenger comparison between them. Say this
   plainly rather than letting a reviewer infer it.

---

## 9. Verification

| Level | How |
|---|---|
| Unit | `pytest tests/` — feature logic, schema validation, cold-start path |
| **Train/serve skew** | Assert offline and online velocity features match on fixed input |
| Pipeline | `dvc repro --force` from raw → champion, on a sampled subset in CI |
| Data quality | Corrupt a row → Pandera fails the DVC stage |
| Model | AUC-PR floor; deliberately drifted holdout → NannyML flags it |
| API | `curl /predict` → score + Postgres row; `/health` shows version + freshness |
| Load | k6: throughput, p50/p95/p99, sustained-rate error budget |
| End-to-end | Replay a day of transactions through the deployed URL, watch the dashboard |
```

---

## 10. Lightning AI handoff

The dev loop moves to a Lightning Studio (Linux + GPU), which also sidesteps the
Windows/`cudf` problem entirely.

### One-time setup on the Studio
```bash
git clone https://github.com/oluwadunni1/financial-fraud-detection.git
cd financial-fraud-detection && git checkout mlops-platform

curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,api,gnn,monitoring]"

# Secrets -- never commit these
cp .env.example .env    # then fill in DagsHub + Supabase credentials
```

### Where each artifact lives on the Studio
| Artifact | Location | Pushed anywhere? |
|---|---|---|
| Raw CSV (2.35 GB) | Studio drive (`data/TabFormer/raw/`) | n/a -- no DVC remote; re-downloadable |
| Processed tabular Parquet | Studio drive + DVC cache | n/a -- see 4.3 |
| `node_features.parquet` (~2.16 GB at 2016+) | Studio drive | **No** -- regenerate via `dvc repro` |
| `edges.parquet` | Studio drive | n/a (small) |
| Model artifacts / embeddings | MLflow on DagsHub | Yes |

### 10.1 Always invoke `.venv/bin/python` explicitly

**`source .venv/bin/activate` is not sufficient on this Studio.** The shell starts from a
profile that activates the conda `cloudspace` env, and zsh caches command lookups in a hash
table. Activating the venv prepends to `PATH`, but a bare `python` still resolves from the
stale hash to `/home/zeus/miniconda3/envs/cloudspace/bin/python`. `which python` reports the
venv while `sys.executable` reports conda -- the two genuinely disagree.

This bit once already: `uv pip install -e ".[dev]"` run before any activate saw
`CONDA_PREFIX` with no `VIRTUAL_ENV` and installed the whole project into the **conda** env,
while a later `uv pip install psycopg` run after an activate went to `.venv`. Two
environments, each holding half the dependencies, and every check appearing to pass.

Rules that avoid it entirely:

```bash
.venv/bin/python -m pytest              # not: python -m pytest
.venv/bin/python -m ruff check src tests scripts
.venv/bin/dvc repro                     # not: dvc repro
uv pip install --python .venv/bin/python <pkg>    # always pin the target
```

`hash -r` after activating also works, but explicit paths cannot be forgotten.

### Studio hygiene
- Studios **sleep when idle** -- long training runs need the job to keep the session
  alive, or use a Lightning Job rather than an interactive Studio.
- The Teamspace Drive persists across Studio restarts; `/tmp` does not.
- Keep the GPU machine for Phase 3 only. Phases 0-2 and 4-7 are CPU work and should
  run on a cheaper CPU Studio to conserve credits.

### First tasks on arrival
1. ~~Download TabFormer and measure the real numbers~~ **Done** -- see section 2.1.
2. Verify DagsHub MLflow Model Registry support against your own repo and token.
3. Apply `sql/001_init.sql` then `sql/002_challenger.sql` to Supabase, confirm `pgvector`.
4. Resume at Phase 1.

Items 2 and 3 need credentials in `.env` and are the only things standing between here
and Phase 1.
