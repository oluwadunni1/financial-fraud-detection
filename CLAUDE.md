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

**Phase 6 built (2026-10-04).** GraphSAGE is **@champion** (v3), promoted through the new
gate; XGBoost v4 scores every request in shadow and is one command from a rollback. The
graph-off ablation settled the project's question: without its neighbourhood GraphSAGE
falls from **0.4665 to 0.0418** (decision 28). Compose runs Postgres + API + monitoring;
co-located latency **measured** at **10.4 ms p50 / 15.4 p95**; 3 workers serve ~180 req/s
with shadow on. A live alias move swapped the running API's champion in 11 s, no redeploy.
Our CI replaces NVIDIA's; the GitHub push of it waits on the `workflow` token scope.

**Phase 5 done (2026-10-03).** Three notebooks on production code
(`notebooks/phase5/`). Without a single label, NannyML CBPE reads XGBoost's 2019 drop
(true −24%, estimated −18%) and GraphSAGE's stability (−2% / −2%), flagging XGBoost **89
days** before delayed chargebacks could. Exact explanations for both models: XGBoost's
false positives are almost all `Chip = Online`; GraphSAGE's group Shapley gives the graph
~15% of the push on frauds (read at first as "not the graph" -- disproved, decision 28).
A 1 h ingest lag *raises* GraphSAGE's AUC-PR (+0.074), almost all through velocity: the
short-window velocity signal has drifted since training (decision 27). Ingest SLA 24 h. The
monthly job (`fraud.monitoring.run`) alerts on XGBoost from May 2019, never on GraphSAGE.
Monitoring runs in `.venv-monitoring` (decision 24).

**Phase 4 closed (2026-10-03).** Latency measured with Supabase in the loop: 427 ms from
this Studio, 96% of it the 68 ms transatlantic round trip. Cutting 6 database round trips to
2 brings it to 151 ms here and a projected **14 ms p50 / 17 ms p95** beside the database. The
run also found the API serving a model nobody evaluated (decision 23) and a neighbour bug in
the in-memory replay; the restated headline is **0.4665**.

**Phase 3.5 complete (2026-10-01).** The foundation model is measured and logged to MLflow
but **not registered**: NVIDIA's own extraction (each transaction alone) adds +23% over a
same-rows base (0.1797 -> **0.2215**), still below the champion and half of GraphSAGE. Our
in-sequence extraction scores at chance -- see decision 21.

**Phase 4 complete (2026-09-19).** The causal replay settled the question Phase 3 could
not: GraphSAGE served honestly scores **0.4665** AUC-PR on 2019 against the offline
**0.5614** and the XGBoost champion's **0.2501**. The offline number was ~17% optimistic,
not fake -- measured leakage is worth about 1%. Serving is FastAPI + a per-request
subgraph, p50 5.4ms / p95 13.0ms in-process (the database is measured in Phase 4's close-out above).

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
| Champion | **GraphSAGE v3** -- `@champion`, served AUC-PR 0.4665; XGBoost v4 `@challenger` in shadow and `@previous` |
| Serving | **Done** -- FastAPI `/predict`, per-request subgraph, 2 DB round trips; projected 14 ms p50 / 17 ms p95 same-AZ |
| Causal replay | **Done** -- 1,723,938 rows of 2019, AUC-PR **0.4665**, P@100 0.710 |
| Latency replay | **Done** -- `replay.py --http`, 3,000 requests, scores equal to the in-memory replay on all 3,000 |
| Foundation model | **Measured, not registered** -- best head 0.2215, MLflow run `fm-phase3.5-experiment` |
| Monitoring | **Experiment done** -- `01_monitoring.ipynb`; CBPE + drift → `drift_metrics` rows (Parquet; Supabase write is opt-in) |
| Explainability | **Experiment done** -- `02_explainability.ipynb`; exact TreeSHAP reason codes + GraphSAGE group Shapley |
| Staleness | **Measured** -- `03_staleness_and_operations.ipynb`; ingest SLA 24 h; velocity drift found |
| Monitoring job | **Running** -- `python -m fraud.monitoring.run`; rows + watermarks in Supabase (as of 2019-11-01) |
| Containers | **Done** -- `Dockerfile.api` (1.9 GB), `Dockerfile.jobs`, `docker-compose.yml` (postgres + api + monitor) |
| Load | **Measured** -- co-located p50 10.4 / p95 15.4 ms; ~180 req/s at 3 workers, shadow on |
| CI/CD | **Built** -- `ci.yml` (lint+tests, 2 envs), `operate.yml` (monitor, promote, rollback, keep-alive); needs secrets |
| Tests | **312 passing** in `.venv` across 21 files, **14** under `.venv-monitoring` |

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
   target. **Measured 2026-10-03:** 14 ms p50 / 17 ms p95 projected same-AZ (decision 22).
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
10. **DagsHub hosts MLflow only. The DVC remote is Cloudflare R2 -- and it now exists.**
   DVC ran with a local cache and no remote through Phase 4, deliberately: a remote moves
   data between machines and nothing needed moving. Both triggers named at the time have
   since fired -- Phase 3 ran on a separate GPU Studio, and Phase 6 puts `dvc repro` in
   GitHub Actions -- so R2 was wired on 2026-09-21 and 321 files pushed.
   R2 over S3 because CI is the main consumer: S3 egress to a GitHub runner is charged per
   GB, R2's is free. S3-compatible, so the already-installed `dvc[s3]` covers it.
   **The whole dataset is 4.3 GB on disk against R2's 10 GB free tier**, so everything is
   pushed and nothing needs `push: false`. (The 7.32 GB in ARCHITECTURE 4.4 is the matrix
   in fp32 memory; as compressed Parquet it is 767 MB.)
   Bucket and endpoint live in `.dvc/config`, committed. The key pair lives in
   `.dvc/config.local`, gitignored by `.dvc/.gitignore` and never committed. CI has no
   config.local, so the same secrets go in GitHub Actions secrets exported as
   `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, which s3fs picks up for any
   S3-compatible endpoint.
   **Do not `source .env` in zsh.** Placeholder values containing `<...>` parse as input
   redirections; that silently emptied `$R2_BUCKET` and produced a malformed remote whose
   error message pointed at `endpointurl` instead. Read it with `dotenv` instead.
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
18. **The causal replay is the number we report, not the offline one.** Offline 0.5614
   vs served 0.4665 on 2019 (0.4667 before the same-minute neighbour fix, see gotchas).
   The gap is NOT leakage -- letting the graph see the future is worth ~1%. It is that offline samples 10 neighbours spread across the card's whole
   period while serving takes the 10 most recent, which is what an indexed lookup can
   cheaply return. A defensible tradeoff, recorded rather than closed.
19. **A sharded replay must be proven equal to the sequential one, not assumed.** The
   first attempt differed on 2,659 of 20,000 rows because shard seeding stopped at
   2019-01-01 while the sequential run's history reaches one velocity window further
   back. Both bounds now match and the two agree to one float32 ULP. `--shards 4` cut
   2019 from ~2.2h to 53 min.
20. **Pin torch to one thread for per-request scoring.** On ~21-node graphs
   `threads=4` measured **10 rows/s** against `threads=1` at **178** -- torch thrashes on
   work far too small to parallelise, and it starves the shards of cores.
21. **The foundation model's downstream embedding follows NVIDIA: each transaction ALONE.**
   The checkpoint and tokenizer are NVIDIA's exactly (fork is 0 ahead / 0 behind, merchant
   hash verified against cuDF on all 100,343 merchants). Their notebook 04 embeds a
   transaction as `<bos> T <eos>` with no history; we also tried embedding it inside its
   card's 315-transaction window. That **contextual** arm fails: within a card ~0% of the
   embedding variance changes per transaction (vs 66% isolated) -- it is a card identifier,
   scores 0.0016 alone and drags the base from 0.1797 to 0.1368. Isolated + base is 0.2215.
   Both arms are kept (`fm_embeddings`, `fm_embeddings_isolated`) so the comparison is
   reproducible. Never move the contextual read point to `<eos>`/`<sep>` after the
   transaction: which one appears depends on whether a later transaction exists.
22. **Serving makes 2 database round trips, not 6.** One statement reads both histories;
   one data-modifying CTE logs the prediction and inserts the row, in autocommit, so the
   pair stays atomic (`serving.merged_round_trips`). Measured on the same 3,000 requests:
   427 -> 151 ms from us-east-1, 19.9 -> 14.0 ms p50 projected same-AZ. Latency is reported
   decomposed (network / Postgres / score / HTTP) from the `Server-Timing` header, because
   from this Studio a single total measures the Atlantic, not the design.
23. **The registered model must be the evaluated model -- check it by scores, not by name.**
   `@challenger` v2 was GraphSAGE epoch 17; the model was retrained to epoch 19 36 minutes
   later and every reported number came from that. The API served unevaluated weights for
   two weeks. Caught only because the HTTP replay compares served scores with the causal
   replay row for row. v3 (epoch 19) is now `@challenger`; v2 is kept. `registry_gnn.py`
   tags each version with the `evaluated_model` it was logged from.
24. **Monitoring runs in its own environment, `.venv-monitoring`.** NannyML 0.13 cannot
   share `.venv`: its pins pull XGBoost 3.4.1 -> 2.1.4 or numpy/pandas/pyarrow down. Our
   package is installed there `--no-deps`; `fraud.monitoring` never imports xgboost or
   torch (tested) -- it reads scores, never models. Explainability stays in `.venv`. This
   is also the production shape: monitoring is a job container, not part of the API image.
25. **Label-free degradation is judged against the reference LEVEL, and must persist.**
   NannyML's ±3σ band over reference months never fires here: ~200 frauds a month makes
   monthly AUC-PR too noisy (XGBoost's lower threshold is 0). `flag_degradation` flags an
   estimate more than `degradation_tolerance` (15%) below the 2018 level. Single-month flags
   happen for the stable model too, so alerting needs persistence (consecutive months or
   quarterly chunks). CBPE flagged XGBoost on 7/10 months, GraphSAGE on 1/10.
26. **Explanations are exact or they are not shipped.** TreeSHAP via `pred_contribs`
   (verified equal to the `shap` library), folded to source fields; GraphSAGE gets exact
   Shapley over five groups (32 forward passes), each group removed along a path the model
   already knows. Tests assert efficiency and that the full coalition equals the served
   score -- an explanation of a different input than the one scored is worse than none.
27. **GraphSAGE's short-window velocity features have drifted; fix the model, not ingest.**
   Lagging only velocity by 1 h raises 2019 AUC-PR by +0.072 [+0.052, +0.095]; lagging only
   the graph, +0.014. Velocity is computed identically offline and online (skew test), so the
   learned meaning of the 1 h / seconds-since-last signal is what moved -- monitoring flagged
   `velocity_seconds_since_last` independently. Do NOT ship a deliberate ingest lag as a
   "fix": it is a symptom. Retrain or review those features. Ingest SLA stays 24 h.
28. **GraphSAGE's edge depends on its neighbourhood -- the ablation, not Shapley, decides.**
   Replaying all of 2019 with no card/merchant neighbours (velocity kept) drops AUC-PR
   0.4665 -> **0.0418** (P@100 0.71 -> 0.05; Δ −0.425 [−0.445, −0.404]), below XGBoost.
   Phase 5 had read group Shapley (graph ~15% of the push on frauds) as "not the graph";
   wrong -- Shapley on positives measures push from a baseline, not ranking among 1.7M
   legitimate rows. Caveat: an empty neighbourhood is out of training distribution too, so
   this proves dependence; a shuffled-neighbour control would isolate the relationships.
29. **Promotion goes through `fraud.models.promote`, never a hand-set alias.** The gate
   re-scores 2019 transactions THROUGH the registry for both candidate and champion (each
   must reproduce its evaluated scores), requires the family's evidence, and demands
   `min_auc_pr_gain` on the served 2019 number. On pass: @previous <- old champion,
   @champion <- candidate, @challenger <- old champion (shadow + rollback). Every attempt
   is an MLflow run. Needed twice over: @champion v1 was the pre-velocity-fix XGBoost (0.3190)
   and stayed registered after the honest 0.2501 retrain -- now v4, verified bit-equal.
30. **The API is two models behind one contract, swapped by alias at runtime.** Scorers are
   chosen from the logged MLflow flavour; the champion decides, the challenger scores the
   same history after the response (`predictions.challenger_*`); a watcher thread re-resolves
   aliases every `alias_refresh_seconds` and loads a new model fully before swapping it in.
   Workers are processes: one serialises everything behind the GIL (~85 req/s ceiling).

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
  embeddings; the foundation model gives a 512-d embedding *per transaction* (PCA'd to 64
  by the head, PCA fitted on train and shipped with it). Different assembled feature vectors.
- **A per-request subgraph can be the right shape and still be wrong.** Three defects
  got through shape tests because every one of them returns a confident number rather
  than an exception. Each was worth more than the model:
  1. Wiring every neighbour to BOTH id nodes. The card's recent transactions happened at
     other merchants and the merchant's belong to other cards, so each identity node
     averaged ~50% rows that were never connected to it. Cost **4.4x** -- and it showed
     up as inflated scores on legitimate rows (0.1091 vs 0.0536), not as missed fraud.
  2. Letting the 168h velocity window bound the GRAPH. Nothing older can affect a
     velocity feature, but the offline user node aggregates across the whole split, so
     serving saw one week of a card that training saw a year of. Cost **8%**.
  3. `ToUndirected` putting the arriving transaction inside its own id nodes' aggregates.
     It must RECEIVE from them; offline it is one candidate among ~280 and
     `NeighborLoader` usually does not draw it. Cost **7.5%**.
  The tests that catch these assert *edge attribution*, not node counts. Write the four
  edge types explicitly -- direction is the thing being controlled, and `ToUndirected`
  controls it wrongly here.
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
- **A `deque(maxlen=N)` evicts BEFORE the `ts < now` filter.** The in-memory replay's
  merchant history did exactly that, so a same-minute row already added cost the model its
  10th neighbour (Postgres's `ts < now ... limit 10` returned 10). 2.5% of 2019 rows moved;
  AUC-PR 0.4667 -> 0.4665. Tied rows no longer count toward the cap on either side.
- **psycopg outside autocommit sends `BEGIN` as its own round trip.** Invisible locally,
  70 ms from this Studio. Count round trips from measurements, never from the code.
- **The API holds one shared connection.** Fine for the sequential replay; the Phase 6 load
  test needs `psycopg_pool` first, or concurrent requests interleave on one session.
- **Seed the latency replay only through `seed_rows()`.** It is the one selection both the
  in-memory and the HTTP replay start from; any other seeding breaks the score check.
- **Never install `.[monitoring]` (NannyML) into `.venv`.** It silently downgrades
  XGBoost or numpy/pandas/pyarrow under the whole pipeline. Use `.venv-monitoring`, and
  note NannyML imports `statsmodels` without declaring it.
- **`.venv-monitoring` is a symlink on Lightning**, so `.gitignore` must name it without a
  trailing slash -- `dir/` never matches a symlink and git would offer to commit it.
- **TabFormer has zero fraud after October 2019.** Nov 2019 - Feb 2020 carry no positives,
  so realized metrics are undefined there while label-free estimates are not -- CBPE keeps
  estimating 0.27-0.38. A change in the label distribution is invisible in the scores.
- **Monthly AUC-PR at ~200 frauds swings ±0.1 on noise alone.** Never alert on one month.
- **Saving a notebook in the Studio's JupyterLab re-binds it to the default `cloudspace`
  kernel** (`python3`, no NannyML). It happened to 01 twice; the only diff is metadata.
  Pick `fraud` / `fraud-monitoring` from the kernel menu before saving, or restore with
  `git checkout`. `tests/test_notebooks.py` asserts each notebook names its kernel.
- **MLflow with no tracking URI silently uses a local `./mlflow.db`** and reports that the
  registered model "does not exist". Set the URI before ANY registry call (`promote.family`
  once did it in the wrong order and created the file).
- **XGBoost pulls `nvidia-nccl-cu13` (345 MB) on Linux** for multi-GPU training; the API
  image uninstalls it and proves at build time that the booster still predicts.
- **The jobs image needs `libgomp1`**: NannyML imports LightGBM, which links system OpenMP.
- **Pushing workflow files needs the `workflow` token scope**: `gh auth refresh -s workflow`.
- **Never run a sharded replay from a stdin script.** `spawn` workers re-import `__main__`
  from a file that does not exist and the pool hangs. Use the module entry points.
- **`source .venv/bin/activate` does NOT change which `python` runs.** The shell profile
  activates conda and zsh caches the lookup, so a bare `python` silently stays on
  `/home/zeus/miniconda3/envs/cloudspace/bin/python` even though `which python` says
  otherwise. Always write `.venv/bin/python`, `.venv/bin/dvc`, and
  `uv pip install --python .venv/bin/python`. This already split the dependencies across two
  environments once. See ARCHITECTURE.md 10.1.

## Unresolved

- Deploy target (Cloud Run recommended). Build Compose-first until Phase 6.
- ~~Is the foundation-model checkpoint loadable as standard Llama?~~ **Resolved** -- yes,
  `transformers` 5.17, no NeMo.
- ~~All storage estimates in ARCHITECTURE.md section 4.4 are unmeasured.~~ **Resolved
  2026-09-16** -- section 4.4 now carries measured figures.

## Commands

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"   # add api,gnn,notebook per phase
# NannyML lives apart -- see pyproject.toml's `monitoring` extra for the recipe.

# Always invoke the venv explicitly -- activate is not enough here (see gotchas).
.venv/bin/python -m pytest                         # tests
.venv/bin/python -m ruff check src tests scripts   # lint
.venv/bin/dvc data status                          # tracked data clean?
.venv/bin/dvc repro                                # the pipeline: ingest/validate/split/sample
.venv/bin/dvc dag                                  # show the DAG
.venv/bin/python scripts/verify_services.py        # DagsHub + Supabase, 11 checks
.venv-monitoring/bin/python -m pytest tests/test_monitoring.py   # NannyML-backed tests
# Notebooks: kernels `fraud` (.venv) and `fraud-monitoring` (.venv-monitoring)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/phase5/01_monitoring.ipynb
```

GPU is needed only for Phase 3. Run other phases on a CPU machine to conserve credits.
