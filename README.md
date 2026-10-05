# Fraud Detection MLOps Platform

An end-to-end MLOps platform for card-fraud detection on IBM's TabFormer dataset
(24,386,900 transactions, 1991–2020), built to answer one question:

> **Does relational or sequential structure matter more for card fraud?**

Three models compete through **one serving contract and one registry**, so the answer is
like-for-like rather than three separate experiments — and the champion/challenger
machinery gives the registry a real job.

## The answer

Test year 2019: 1,723,938 transactions, 2,087 frauds (a 1-in-826 base rate, so a random
ranker scores AUC-PR ≈ 0.0012).

| Model | Structure it uses | AUC-PR | P@100 | How it was measured |
|---|---|---|---|---|
| XGBoost | tabular (86 features incl. velocity) | 0.2501 | 0.41 | offline = served (features verified equal) |
| **GraphSAGE** | **relational** — card ↔ transaction ↔ merchant | **0.4665** | **0.71** | **causal replay, served path** — `@champion` |
| GraphSAGE, neighbourhood removed | — (ablation) | 0.0418 | 0.05 | causal replay |
| GraphSAGE, *another* merchant's history | — (shuffled control) | 0.0379 | 0.13 | causal replay |
| GraphSAGE, *another* card's history | — (shuffled control) | 0.5046 | 0.71 | causal replay |
| Foundation model + base features | sequential (NVIDIA's 29M decoder, frozen) | 0.2215 | — | offline |

**Relational structure wins, and the ablation shows the graph is doing the work:** remove
the neighbourhood at scoring time and GraphSAGE falls from 0.4665 to 0.0418. A shuffled
control rules out the obvious objection (an empty neighbourhood is simply unfamiliar): give
each merchant another, equally busy merchant's real history and it falls just as far, to
0.0379. The signal is *this merchant's* recent activity. The card's own history, by
contrast, is not helping -- borrowing another card's scores 0.5046 (+0.038 [+0.020, +0.053]),
an open finding for the retrain. The sequential
foundation model adds +23% over a same-rows tabular base, but stays below XGBoost trained on
all history and at half of GraphSAGE.

Every headline is the **served** number: a causal replay scores 2019 in time order through
the same code path as the API, inserting each transaction only after it is scored, so
nothing can see its own future. Offline GraphSAGE scored 0.5614; the honest served number
is 17% lower, and the gap was measured, not hidden (leakage is worth ~1%).

## The platform

```mermaid
flowchart LR
    subgraph Data["Data & pipeline (DVC, Cloudflare R2)"]
        CSV[TabFormer CSV<br/>24.4M rows] --> ING[ingest → validate<br/>Pandera] --> SPL[temporal split<br/>≤2017 / 2018 / ≥2019]
        SPL --> FEAT[features + velocity<br/>one module, two callers]
        FEAT --> XGB[XGBoost]
        FEAT --> GNN[GraphSAGE<br/>PyG, T4 GPU]
        SPL --> FM[foundation model<br/>embeddings]
    end
    subgraph Registry["MLflow on DagsHub"]
        REG[(fraud-champion<br/>@champion · @challenger · @previous)]
    end
    XGB & GNN --> GATE{promotion gate<br/>re-scores through registry}
    GATE --> REG
    subgraph Serving["Docker Compose"]
        API[FastAPI /predict<br/>champion decides,<br/>challenger in shadow] <--> PG[(Postgres + pgvector<br/>hot store)]
        MON[monitoring job<br/>NannyML CBPE + drift]
    end
    REG -- alias, re-resolved every 30 s --> API
    MON --> PG
    MON -- persistent alert --> RB[rollback] --> REG
```

| Concern | How |
|---|---|
| Reproducibility | DVC DAG (19 stages), data + evidence on Cloudflare R2, all config in `params.yaml` |
| Data quality | Pandera schemas as a pipeline gate |
| Leakage | temporal split; velocity strictly-earlier offline and online; causal replay score-then-insert |
| Serving | FastAPI, per-request ~21-node subgraph, 2 DB round trips, **10.4 ms p50 / 15.4 p95** with the DB co-located |
| Model swaps | MLflow aliases re-resolved at runtime — a promotion reached the running API in ~11 s, no redeploy |
| Safety | promotion gate re-scores both models **through the registry**; shadow challenger on every request; rollback on persistent alerts |
| Monitoring | NannyML CBPE estimates performance **without labels** — flagged XGBoost's 2019 decline 89 days before chargebacks could |
| Explainability | exact TreeSHAP reason codes (XGBoost); exact group Shapley over the subgraph (GraphSAGE) |
| CI/CD | GitHub Actions: lint + tests in two environments; images built once and published to ghcr.io; scheduled monitoring, rollback, keep-alive |

## Quickstart

Requirements: Docker, [uv](https://docs.astral.sh/uv/), and a `.env` copied from
`.env.example` with DagsHub (MLflow) credentials — the API loads its models from the
registry by alias. The R2 keys are needed for `dvc pull`.

```bash
# 1. Environment and data
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev,api,gnn,notebook]"
.venv/bin/dvc pull                       # processed data, features, models, evidence

# 2. The platform: Postgres + API + monitoring + the presentation app, from the
#    CI-built images on ghcr.io
docker compose pull && docker compose up -d    # or `up -d --build` to build locally
# -> the app: http://localhost:8501   (the API: http://localhost:8000)
curl -s localhost:8000/health | jq

# 3. Score a real 2019 fraud that GraphSAGE catches and XGBoost misses
export DATABASE_URL=postgresql://fraud:fraud@localhost:5433/fraud
.venv/bin/python -m fraud.jobs.replay --http --api-url http://localhost:8000 \
    --reset-store --start 2019-02-22T17:00:00 --limit 377 --label demo
.venv/bin/python -m fraud.demo payload 18267417 > /tmp/fraud.json
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d @/tmp/fraud.json | jq
.venv/bin/python -m fraud.demo explain 18267417

# 4. The scoreboard and the live system
.venv/bin/python -m fraud.demo results
.venv/bin/python -m fraud.demo status
```

### The presentation app

`app/streamlit_app.py` walks the project in talk order — the answer, a live replay of
a 2019 evening through the API, one transaction scored and explained by both models,
the registry with **real promotion and rollback behind a confirmation**, monitoring,
serving and the retrain. It computes nothing of its own: every number comes from
the committed reports, the live API, the registry or the exact explain modules
(`fraud.dashboard.data`). Run it outside Compose with
`.venv/bin/streamlit run app/streamlit_app.py` (it talks to `localhost:8000` and the
Compose Postgres on `localhost:5433`; `FRAUD_API` / `FRAUD_DB_URL` override).

Monitoring runs in its own environment (NannyML conflicts with the main one — see
`pyproject.toml`); `docker compose run --rm monitor python -m fraud.monitoring.run --as-of 2019-11-01`
runs it in the jobs image.

## What makes the numbers trustworthy

Most of this project's findings were bugs that returned a confident number rather than an
error. The checks that caught them are now tests or gates:

- **Served == offline, row for row.** The HTTP replay compares every served score with the
  in-memory causal replay. It caught a stale registered model (decision 23), a missing
  encoder column, and a neighbour-eviction bug.
- **The registry holds the evaluated weights.** The promotion gate re-scores sampled 2019
  transactions through the registry. It found `@champion` v1 was the pre-velocity-fix model.
- **Edge attribution, not shapes.** Three subgraph defects (4.4×, 8%, 7.5%) passed every
  shape test; the tests now assert which rows feed which node.
- **One notion of "previous".** Velocity is computed by one module for training and serving,
  strictly-earlier on both sides, with a skew test.
- **Ablations over attribution.** Group Shapley suggested the graph contributed ~15%; the
  graph-off ablation showed the model collapses without it. The ablation is what decides.

## Honest limitations

- **Synthetic data, ~2,000 users.** TabFormer is generated; drift is the generator's, and a
  2,000-card graph may not transfer to real issuer scale.
- **AUC-PR, not dollars.** Thresholds are 1% FPR on validation; a cost-sensitive operating
  point (fraud losses vs. false-decline costs) is not modelled.
- **Thin features.** No device, IP, geolocation velocity or dispute history.
- **Served GraphSAGE is 17% below offline** because serving takes the 10 most recent
  neighbours, not a sample across the card's history — a recorded latency trade-off.
- **Short-window velocity has drifted** since training (a 1 h ingest lag *raises* 2019
  AUC-PR by +0.07, almost all through velocity). The fix is a retrain, not a lag.
- **The card's own history lowers 2019 AUC-PR** (shuffled control: +0.038 without it), for
  reasons not yet established -- it is not fraud bursts. Candidate for the retrain.
- **Compose, not cloud.** Deployment target deferred; latency is measured co-located.

## Repository map

| Path | What |
|---|---|
| `src/fraud/data/` | fetch, ingest, Pandera schemas, validation, temporal split |
| `src/fraud/features/` | velocity (shared offline/online), encoders, graph, FM tokenizer |
| `src/fraud/models/` | XGBoost, GraphSAGE, foundation-model head, registry, **promotion gate** |
| `src/fraud/api/` | FastAPI app, scorers, Postgres store, subgraph builder, Server-Timing |
| `src/fraud/jobs/` | causal replay (`--http`, `--no-neighbours`), latency, staleness, load test |
| `src/fraud/monitoring/` | delayed labels, NannyML CBPE + drift, `drift_metrics` sink, the monthly job |
| `src/fraud/explain/` | exact TreeSHAP reason codes, GraphSAGE group Shapley |
| `src/fraud/demo.py` | shell demo helpers: `status`, `results`, `payload`, `explain`, `shadow`, `watch` |
| `app/`, `src/fraud/dashboard/` | the Streamlit presentation app: pages, and the data layer they read |
| `notebooks/phase5/` | monitoring, explainability, staleness & operations experiments (executed) |
| `sql/` | Postgres schema (Supabase and the Compose database run the same migrations) |
| `docs/ARCHITECTURE.md` | the full design record, phase by phase |
| `CLAUDE.md` | 30 settled decisions and the gotchas that silently corrupt results |

Built on the [NVIDIA AI Blueprint: Financial Fraud Detection](docs/NVIDIA_BLUEPRINT_README.md)
and NVIDIA's [transaction foundation model](https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model);
the GNN is rebuilt in open PyTorch Geometric.
