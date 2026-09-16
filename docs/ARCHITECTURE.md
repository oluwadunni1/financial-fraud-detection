# Fraud Detection MLOps Platform — Architecture & Build Plan

> Working reference doc. On implementation, copy to `docs/ARCHITECTURE.md` in the repo.
> Last updated: 2026-09-16

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
| Temporal split strategy (train <2018 / val 2018 / test >2018) | The notebook-driven workflow |

**Why drop the training container:** you cannot version, instrument, or MLflow-log what you
can't see inside. Rebuilding the GNN in open PyG is the entire point of the exercise.

---

## 2. Where we are right now

| | Status |
|---|---|
| Repo | Fork at `github.com/oluwadunni1/financial-fraud-detection`, branch `mlops-platform` |
| Scaffolded | `pyproject.toml`, `params.yaml`, `src/fraud/` package, `sql/001_init.sql`, this doc |
| Data | **Not downloaded.** `data/TabFormer/raw/` contains only a readme |
| Dev machine | Windows 11 + Python 3.14 system, `uv` available. **Migrating to Lightning AI.** |
| GPU | Lightning AI credits available |
| MLflow | **DagsHub hosted** (decided) |
| DVC remote | **DagsHub S3-compatible** (decided) |
| Supabase | Not yet created. Free tier -- see storage budget in section 4.4 |
| Deploy target | Undecided -- Compose-first, cloud-agnostic (see section 7) |

### Immediate unblocks needed
1. Download TabFormer `transactions.tgz` (~2.3 GB) -- IBM Box link in `docs/setup.md`
2. Create DagsHub repo, connect to the GitHub fork, grab access token
3. Create Supabase project, apply `sql/001_init.sql`
4. **Verify DagsHub MLflow supports the Model Registry API** (see section 4.3 caveat)

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
| `node_embeddings` | Precomputed GNN embeddings, `vector(64)` + `computed_at` + `model_version` |
| `transaction_events` | **Narrow, rolling hot store** for velocity aggregates -- not the history |
| `predictions` | Every score, with `latency_ms`, `embedding_age_seconds`, cold-start flags |
| `labels` | Late-arriving ground truth (simulated chargebacks) |
| `drift_metrics` | NannyML output |
| `job_watermarks` | Incremental-job cursors |

Two things that are not optional:

> **`model_version` on `node_embeddings`.** Embeddings from model v1 are meaningless
> to a v2 serving head. They version and deploy together.

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
| Database | **Supabase** (Postgres + pgvector) | Embedding store + prediction log + MLflow backend, one system |
| API | **FastAPI** + Pydantic | Async, auto OpenAPI docs, typed validation |
| Monitoring | **NannyML** | CBPE estimates performance *without labels* — the delayed-label problem |
| Explainability | **SHAP** | `/explain` endpoint; mirrors blueprint's Shapley story |
| Container | **Docker** + Compose | Local parity with cloud |
| CI | **GitHub Actions** | Lint, test, `dvc repro` on sample, build image |

### 4.1 Storage sizing
Superseded by section 4.4, which covers all three tiers and the free-tier limits.

### 4.2 GPU memory note
Do not load the full graph onto the GPU. `NeighborLoader` samples subgraphs; the feature
matrix stays in CPU RAM and is gathered per batch. That makes 9.6 GB of features trainable
on a 24 GB card.

### 4.3 MLflow + DVC hosting -- DagsHub
DagsHub provides hosted MLflow tracking, an S3-compatible DVC remote, and a git
remote in one free account. This replaces the earlier "Supabase as MLflow backend"
idea -- fewer moving parts, nothing to self-host.

```bash
# MLflow
export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<user>
export MLFLOW_TRACKING_PASSWORD=<dagshub-token>

# DVC remote
dvc remote add -d origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<user>/<repo>.s3
dvc remote modify --local origin access_key_id  <dagshub-token>
dvc remote modify --local origin secret_access_key <dagshub-token>
```

> **Caveat to verify early:** DagsHub's hosted MLflow definitely supports *tracking*.
> Model *Registry* API support is less certain. If the registry is unavailable, fall
> back to tagging runs (`champion=true`) and resolving by tag in
> `src/fraud/models/registry.py`. Same outcome, slightly more code. Phase 2 depends
> on knowing which.

### 4.4 Storage budget -- the binding constraint

Three tiers, because no single free tier holds everything.

```
DagsHub (~10 GB)          Lightning Studio drive        Supabase (500 MB DB)
--------------------      ------------------------      --------------------
raw CSV (or hash only)    node_features.parquet  <-BIG  transaction_events (rolling)
processed tabular pq      edges.parquet                 node_embeddings
model artifacts           training checkpoints          predictions / labels
embeddings (~26 MB)       (regenerated, never pushed)   drift_metrics
MLflow tracking/registry
```

#### Why the full history cannot live in Postgres
The `transactions` table at all 24M rows:

| | Size |
|---|---|
| Table (~134 B/row incl. tuple overhead) | ~3.2 GB |
| 3 btree indexes (~40 B/entry effective) | ~2.4 GB |
| **Total** | **~5.6 GB vs a 500 MB limit -- 11x over** |

#### What actually belongs in Postgres
Velocity windows top out at 7 days, so **serving never reads a transaction older than
the longest window**. Older rows belong in Parquet. Supabase becomes a rolling hot
store with a retention job -- which is what production does anyway (hot store + cold
archive).

| Table | Sizing assumption | Est. |
|---|---|---|
| `transaction_events` (narrow, rolling ~30d, cap 500k) | 40 B payload + 2 indexes | ~74 MB |
| `node_embeddings` (~102k x vector(64)) | ~330 B/row | ~34 MB |
| `predictions` (cap 200k) | ~100 B/row | ~20 MB |
| `labels` | ~50 B/row | ~10 MB |
| `drift_metrics`, `job_watermarks` | negligible | ~2 MB |
| **Total hot store** | | **~140 MB** (3.5x headroom) |

Levers if it gets tight: `halfvec(64)` instead of `vector(64)` halves embeddings;
drop `city`/`errors` from the hot table (velocity does not use them).

> **Free-tier gotcha:** Supabase pauses projects after 7 days idle, which would
> silently break a demo link. The scheduled refresh/NannyML jobs hit the DB, so a
> daily cron keeps it awake as a side-effect. Make that explicit, do not rely on luck.

#### DagsHub ceiling and the subsampling decision
~10 GB free. The ~9.6 GB fp32 node-feature matrix does not fit alongside raw data
and artifacts, so:

- Big graph artifacts are **regenerated on the Lightning Studio** via
  `dvc repro build_graph` and never pushed to the remote.
- **Temporally subsample GNN training to 2016+ (~8M transactions)** -> ~3.2 GB fp32,
  ~1.6 GB fp16. Defensible on modelling grounds too: recent data is more
  representative and ~8k fraud cases is ample. State this in the README as a
  deliberate choice, not a hidden shortcut.

All sizes above are estimates -- **measure once the CSV lands** and correct this table.

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

### Phase 0 — Foundations
- [ ] WSL2 + Ubuntu (needed for a Linux-parity dev loop)
- [ ] `uv` or Poetry project, pin Python 3.11/3.12
- [ ] Download TabFormer, verify row count and fraud rate
- [ ] `git init` the new structure on a branch; DVC init; remote → Supabase Storage
- [ ] Supabase project created, schema from §3.2 applied
- **Done when:** `dvc status` is clean and raw data is tracked.

### Phase 1 — Data & validation
- [ ] `ingest.py`: CSV → Parquet partitioned by year
- [ ] Pandera schemas: types, ranges, nullability, **fraud rate sanity bound**
- [ ] Temporal split (train <2018 / val 2018 / test >2018) — assert no leakage
- [ ] DVC stages: `ingest` → `validate` → `split`
- **Done when:** `dvc repro` runs clean, and a deliberately corrupted row fails the build.

### Phase 2 — Baseline model
- [ ] Port tabular feature engineering (cuDF → pandas/polars)
- [ ] XGBoost + MLflow logging: AUC-PR, precision@k, recall at fixed FPR, confusion matrix
- [ ] Register to Model Registry, promote to `Production`
- **Done when:** a champion exists in the registry with real metrics.

> Deliberately before the GNN: it gives a servable model early, so Phases 4–6 can proceed
> without waiting on GPU work, and it's the honest comparison baseline.

### Phase 3 — Graph & GNN (Lightning AI)
- [ ] Port graph construction → `edges/node_features/labels` Parquet
- [ ] GraphSAGE in PyG with `NeighborLoader`
- [ ] Lightning Studio: `dvc pull` → train → log to remote MLflow → push artifacts
- [ ] Compare vs XGBoost; promote champion on merit, not on novelty
- [ ] `precompute_embeddings.py` → upsert user/merchant embeddings to Supabase
- **Done when:** embeddings are in Postgres with `computed_at` + `model_version`.

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

### Phase 6 — Containerize & deploy
- [ ] `Dockerfile.api` (slim, multi-stage), `Dockerfile.jobs`
- [ ] Compose: api + mlflow + dashboard + scheduler
- [ ] k6/locust load test → throughput + p50/p95/p99
- [ ] Deploy API to cloud; schedule jobs
- [ ] GitHub Actions CI
- **Done when:** there's a public URL scoring live transactions.

### Phase 7 — Demo & docs
- [ ] Streamlit: replay stream, live scores, drift charts, latency, staleness curve
- [ ] README with architecture diagram and honest limitations section
- **Done when:** a stranger can follow the README end to end.

### Phase 8 — Stretch
- [ ] **Second NVIDIA/TabFormer model** — ⚠️ *repo link pending, paste here:* `________`
      Slots in as another `train_*` DVC stage + MLflow run; no restructuring needed.
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
| Artifact | Location | Pushed to DagsHub? |
|---|---|---|
| Raw CSV | Studio drive (`data/TabFormer/raw/`) | No -- re-downloadable |
| Processed tabular Parquet | Studio drive | Yes |
| `node_features.parquet` (~3.2 GB) | Studio drive | **No** -- regenerate via `dvc repro` |
| `edges.parquet` | Studio drive | Yes (small) |
| Model artifacts / embeddings | MLflow | Yes |

### Studio hygiene
- Studios **sleep when idle** -- long training runs need the job to keep the session
  alive, or use a Lightning Job rather than an interactive Studio.
- The Teamspace Drive persists across Studio restarts; `/tmp` does not.
- Keep the GPU machine for Phase 3 only. Phases 0-2 and 4-7 are CPU work and should
  run on a cheaper CPU Studio to conserve credits.

### First tasks on arrival
1. Download TabFormer, **measure the real numbers** (row count, unique users, unique
   merchants, fraud rate, feature width) and correct the estimates in section 4.4.
2. Verify DagsHub MLflow Model Registry support (section 4.3 caveat).
3. Apply `sql/001_init.sql` to Supabase, confirm `pgvector` is available.
4. Resume at Phase 1.
