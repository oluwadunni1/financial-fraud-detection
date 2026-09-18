# Fraud Detection MLOps Platform

An end-to-end MLOps platform for credit-card fraud detection on IBM's TabFormer
dataset (24,386,900 transactions), built to answer one question:

> **Does relational or sequential structure matter more for card fraud?**

Three models compete through **one serving contract and one registry**, so the
comparison is like-for-like rather than three separate experiments:

| Model | Inductive bias | Embedding | Status |
|---|---|---|---|
| XGBoost | tabular only | — | **champion, registered** |
| GraphSAGE | relational — who transacts with whom | 64-d, users *and* merchants | next |
| Foundation model | sequential — the order of a user's transactions | 512-d, users only | planned |

The models are reference implementations. **The platform is the contribution:**
the reproducible pipeline, the serving architecture, the monitoring, and the
champion/challenger machinery that makes swapping between them safe.

## Current results

The XGBoost champion, evaluated on 2019 (1,723,938 transactions, 2,087 frauds —
a 1-in-826 base rate):

| Metric | Value | Meaning |
|---|---|---|
| **AUC-PR** | **0.3190** | **263x better than random** |
| Recall @ 1% FPR | 95.4% | catches 1,990 of 2,087 frauds |
| Precision @ 100 | 0.450 | 45 of the top 100 alerts are real fraud |
| AUC-ROC | 0.9973 | reported for comparability — see below |

> **Accuracy and ROC-AUC are deliberately not headlined.** At a 1-in-826 base
> rate, a model predicting "never fraud" scores 99.88% accuracy. The same
> champion scores 0.997 ROC-AUC while raising 17,025 false alarms. AUC-PR's floor
> is the base rate itself, which makes "better than chance" unambiguous.

## Why this dataset

Fraud has three properties most portfolio datasets lack:

- **Extreme imbalance** (0.122% positive) forces AUC-PR, precision@k and
  cost-sensitive thresholds instead of accuracy.
- **Delayed labels** — chargebacks arrive weeks later, so live accuracy cannot be
  computed. This is what makes label-free performance estimation load-bearing.
- **Real concept drift** — the fraud rate swings two orders of magnitude across
  years in this data, so monitoring has something genuine to detect.

## Architecture

```
OFFLINE (DVC pipeline)                    ONLINE
raw CSV                                   FastAPI /predict
  -> validate      (Pandera)                 <- embeddings   (Supabase/pgvector)
  -> split         (temporal, asserted)      <- velocity     (live SQL)
  -> velocity      (shared with serving)     <- champion     (MLflow registry)
  -> features      (encoder fit on train)
  -> train         (XGBoost / GraphSAGE)   MONITORING
  -> evaluate                               NannyML drift + estimated performance
  -> register      (MLflow alias)
```

Full design, measured storage budget and phase plan: **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**.

## Quickstart

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"

cp .env.example .env          # DagsHub + Supabase credentials
.venv/bin/dvc repro           # raw CSV -> registered champion
.venv/bin/python -m pytest    # 117 tests
```

The champion loads from the registry by alias, so swapping it needs no redeploy:

```python
mlflow.pyfunc.load_model("models:/fraud-champion@champion")
```

## Engineering notes

A few things measurement contradicted, all documented in `CLAUDE.md`:

- **Refunds are `$-292.00`, not `-$292.00`** — the minus sits inside the dollar
  sign on 1,244,689 rows. The obvious regex rejects every refund in the dataset.
- **Velocity features must be strictly causal.** `closed="left"` offline and
  `ts < :now` online, so two transactions in the same minute are mutually
  invisible. 142,010 rows share a user and a minute, so this is not academic.
- **`scale_pos_weight` was swept, not guessed.** The value 10 beat the initial
  guess of 50 by 29% AUC-PR; full inverse prevalence (819) was the *worst* value
  on the grid. "Balance the classes" is a trap at this imbalance.
- **`Year` is not a feature.** The split is temporal, so no training row carries
  a test year, and a tree cannot extrapolate.

## Honest limitations

1. TabFormer has only ~2,000 users, so embedding refresh is nearly free here —
   the pattern is demonstrated, not the scale.
2. The data is synthetic; drift is simulated rather than organic.
3. Labels are artificially delayed, not genuinely delayed.
4. Single-node throughput — no streaming ingest, no horizontal scaling story.
5. NVIDIA's published numbers are not reproduced: different framework, CPU
   serving, different feature pipeline.

## Credits

Derived from the [NVIDIA AI Blueprint: Financial Fraud Detection](https://github.com/NVIDIA-AI-Blueprints/financial-fraud-detection),
licensed under Apache 2.0. The original blueprint README is preserved at
[`docs/NVIDIA_BLUEPRINT_README.md`](docs/NVIDIA_BLUEPRINT_README.md); the graph
construction logic and the fraud framing come from there. The NGC-gated training
container and Triton serving path are deliberately not used — see
`docs/ARCHITECTURE.md` §1.

Dataset: [IBM TabFormer](https://github.com/IBM/TabFormer), Apache 2.0.

This project remains Apache 2.0 — see [`LICENSE`](LICENSE).
