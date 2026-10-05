"""The presentation app's pages -- layout only; every number comes from
`fraud.dashboard.data`. `app/streamlit_app.py` wires them into navigation, and
tests render each one alone through `AppTest.from_function`.

The Operations page moves registry aliases for real, always behind a
confirmation dialog; everything else is read-only. The replay page resets the
DASHBOARD's store (Compose Postgres by default; a Supabase URL is refused).
"""

from __future__ import annotations

import re

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from fraud.config import load_params
from fraud.dashboard import data as D

PARAMS = load_params()
ACCENT, ALERT, PASS, MUTED, CONTROL = "#0E6A8C", "#B4441C", "#2C7A4B", "#7F93A3", "#B9C4CE"
STRUCTURE_COLOURS = {"tabular": "#8A6FB0", "relational": ACCENT, "sequential": "#C08A2B",
                     "control": CONTROL}


def _layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="IBM Plex Sans, sans-serif"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(gridcolor="rgba(127,147,163,0.2)")
    fig.update_yaxes(gridcolor="rgba(127,147,163,0.2)")
    return fig


# --- cached reads ----------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def scoreboard():
    return D.scoreboard(PARAMS), D.base_rate()


@st.cache_data(show_spinner="Computing the true monthly AUC-PR from committed scores...")
def monitoring():
    return D.monitoring_monthly(PARAMS), D.detection_summary()


@st.cache_data(show_spinner=False)
def serving():
    return D.serving_summary()


@st.cache_data(show_spinner=False)
def retrain():
    return D.retrain_summary(PARAMS)


@st.cache_data(ttl=10, show_spinner=False)
def registry():
    try:
        return D.registry_status(PARAMS)
    except Exception as exc:
        return [{"alias": "error", "version": None, "run": str(exc)}]


@st.cache_resource(show_spinner="Loading both models and the 2018 baseline transaction...")
def explainers():
    return D.load_explainers(PARAMS)


@st.cache_data(show_spinner="Looking the transaction up in 24.4M rows...")
def transaction(txn_id: int):
    return D.transaction(PARAMS, txn_id)


def family_of(evaluated_as: str | None) -> str:
    return "xgboost" if (evaluated_as or "").startswith("xgb") else "graphsage"


# --- sidebar --------------------------------------------------------------------------

def sidebar() -> None:
    with st.sidebar:
        st.markdown("### Live system")
        health = D.api_health(PARAMS)
        if health:
            st.success(f"API up · champion **{D.live_version(health)}** · shadow "
                       f"**{D.live_version(health, 'challenger_version')}**")
        else:
            st.error(f"API not reachable at {D.api_url(PARAMS)} — `docker compose up -d`")
        for r in registry():
            if r["version"]:
                st.caption(f"@{r['alias']} → v{r['version']} · {r.get('evaluated_as', '')}")
        host = D.db_url(PARAMS).rsplit("@", 1)[-1]
        st.caption(f"store: {host}")
        if st.button("Refresh", width="stretch"):
            st.cache_data.clear()
            st.rerun()


# --- 1. the answer --------------------------------------------------------------------

def page_answer() -> None:
    st.title("Does relational or sequential structure catch card fraud?")
    st.markdown("Three models bet on three kinds of structure and compete through **one API "
                "and one model registry** on IBM's TabFormer ledger: 24.4M transactions, "
                "1 in 826 fraudulent in 2019.")
    rows, base = scoreboard()
    by = {r["model"]: r for r in rows}
    served, xgb = by["GraphSAGE, served (@champion)"], by["XGBoost"]
    lat = serving()["colocated"]
    c = st.columns(4)
    c[0].metric("Champion, served 2019 AUC-PR", f"{served['auc_pr']:.4f}",
                f"{served['auc_pr'] / xgb['auc_pr']:.2f}× XGBoost")
    c[1].metric("Top-100 alerts that are fraud", f"{served['p100']:.0%}",
                f"XGBoost {xgb['p100']:.0%}", delta_color="off", delta_arrow="off")
    c[2].metric("Without its neighbourhood", f"{by['GraphSAGE, no neighbourhood']['auc_pr']:.4f}",
                "the graph is doing the work", delta_color="off", delta_arrow="off")
    c[3].metric("Latency, DB co-located", f"{lat['p50']:.1f} ms",
                f"p95 {lat['p95']:.1f} ms", delta_color="off", delta_arrow="off")

    frame = pl.DataFrame(rows).reverse()
    fig = go.Figure(go.Bar(
        x=frame["auc_pr"], y=frame["model"], orientation="h",
        marker_color=[STRUCTURE_COLOURS[s] for s in frame["structure"]],
        text=[f"{v:.4f}" for v in frame["auc_pr"]], textposition="outside",
        customdata=frame.select("measured", "structure").rows(),
        hovertemplate="%{y}<br>AUC-PR %{x:.4f}<br>%{customdata[0]} · %{customdata[1]}"
                      "<extra></extra>"))
    fig.add_vline(x=base, line_dash="dot", line_color=MUTED,
                  annotation_text=f"chance ≈ {base:.4f}", annotation_position="top right")
    fig.update_xaxes(title="test 2019 AUC-PR", range=[0, 0.68])
    st.plotly_chart(_layout(fig, 380), width="stretch")
    st.caption("Colour = structure: tabular · **relational** · sequential · grey = controls "
               "(the same GraphSAGE with its neighbourhood removed or swapped).")

    st.markdown("""
**Relational structure wins, and the merchant relationship carries it.**
- GraphSAGE served honestly scores **0.4665**, almost twice XGBoost's 0.2501.
- Remove its neighbourhood: **0.0418**. Give each merchant *another, equally busy*
  merchant's real history instead: **0.0379** — the model reads *this* merchant's activity,
  not just any plausible history.
- The sequential foundation model helps a tabular model by 23% but stays below both.
- Every number is the **served** number: a causal replay scores 2019 in time order through
  the API's own code. Offline GraphSAGE said 0.5614; the honest figure is 17% lower.
""")


# --- 2. replay ---------------------------------------------------------------------------

def page_replay() -> None:
    cfg = PARAMS["dashboard"]
    st.title("Replay an evening of 2019 through the live API")
    st.markdown(f"From **{cfg['replay_start'].replace('T', ' ')}**, {cfg['replay_limit']} real "
                "transactions in time order. The store is reset and seeded with exactly the "
                "past these cards and merchants had; each request inserts its own row after "
                "it is scored, so nothing can see its own future.")
    health = D.api_health(PARAMS)
    if not health:
        st.error("The API is not reachable — start the stack first.")
        return
    if st.button("▶ Reset the store and replay", type="primary"):
        # The status box covers seeding only: it collapses when complete, and the
        # results drawn inside it would collapse with it.
        with st.status("Seeding the store with the slice's past...") as status:
            rows, seeded = D.prepare_replay(PARAMS)
            status.update(label=f"Seeded {seeded:,} rows for {rows['User'].n_unique()} cards "
                                f"/ {rows['Merchant'].n_unique()} merchants", state="complete")
        kpis, chart, table = [c.empty() for c in st.columns(4)], st.empty(), st.empty()
        done = []
        for i, r in enumerate(D.replay_stream(PARAMS, rows)):
            done.append(r)
            if i % 10 == 0 or i == rows.height - 1:
                _replay_view(pl.DataFrame(done), rows.height, kpis, chart, table)
        st.success(f"Replayed {len(done)} transactions in time order through the live API.")
        st.session_state["replay"] = pl.DataFrame(done)
    elif "replay" in st.session_state:
        _replay_view(st.session_state["replay"], st.session_state["replay"].height,
                     [c.empty() for c in st.columns(4)], st.empty(), st.empty())


def _replay_view(frame: pl.DataFrame, total: int, kpis, chart, table) -> None:
    alerts, frauds = frame.filter(pl.col("alert")), frame.filter(pl.col("fraud") == 1)
    caught = frauds.filter(pl.col("alert")).height
    kpis[0].metric("Sent", f"{frame.height} / {total}")
    kpis[1].metric("Alerts", alerts.height)
    kpis[2].metric("Frauds caught", f"{caught} / {frauds.height}")
    kpis[3].metric("Latency p50", f"{frame['wall_ms'].median():.1f} ms",
                   f"score {frame['score_ms'].median():.1f} ms" if "score_ms" in frame.columns
                   else None, delta_color="off", delta_arrow="off")
    fig = go.Figure()
    for label, colour, f in (("legitimate", MUTED, frame.filter(pl.col("fraud") == 0)),
                             ("fraud", ALERT, frauds)):
        fig.add_trace(go.Scatter(
            x=f["ts"], y=f["score"], mode="markers", name=label,
            marker=dict(color=colour, size=10 if label == "fraud" else 6,
                        symbol="diamond" if label == "fraud" else "circle"),
            customdata=f.select("txn_id", "amount", "chip").rows(),
            hovertemplate="txn %{customdata[0]} · $%{customdata[1]:.2f} · %{customdata[2]}"
                          "<br>score %{y:.4f}<extra></extra>"))
    fig.add_hline(y=PARAMS["serving"]["decision_thresholds"]["graphsage"], line_dash="dash",
                  line_color=ACCENT, annotation_text="GraphSAGE alert threshold")
    fig.update_yaxes(title="champion score", type="log", range=[-4, 0.05])
    chart.plotly_chart(_layout(fig, 360), width="stretch", key=f"replay{frame.height}")
    table.dataframe(alerts.select("txn_id", "ts", "amount", "chip", "score", "fraud"),
                    width="stretch", hide_index=True)


# --- 3. one transaction --------------------------------------------------------------------

def page_transaction() -> None:
    st.title("One transaction, two models")
    featured = PARAMS["dashboard"]["featured"]
    choice = st.selectbox("Transaction", list(featured),
                          format_func=lambda t: f"{t} — {featured[t]}")
    custom = st.text_input("…or any txn_id", "")
    txn_id = int(custom) if custom.strip().isdigit() else int(choice)
    try:
        found = transaction(txn_id)
    except KeyError as exc:
        st.error(str(exc))
        return
    txn = found["txn"]
    truth = ":red[**FRAUD**]" if found["fraud"] else ":green[**legitimate**]"
    st.markdown(f"#### {txn['ts']:%d %b %Y, %H:%M} · ${txn['Amount']:.2f} · "
                f"{txn['City']}, {txn['State']} · {txn['Chip'].replace(' Transaction', '')} · "
                f"MCC {txn['MCC']} · cardholder {txn['User']} · truth: {truth}")

    try:
        seeded = D.history_count(PARAMS, txn)
    except Exception:
        seeded = None
    if seeded == 0:
        st.warning("The store holds none of this card's past, so the API would score it as a "
                   "cold start. Run the **Replay** page first (it seeds this evening's past).")

    left, right = st.columns(2)
    with left:
        st.subheader("Live: through the API")
        if st.button("Score it", type="primary"):
            try:
                st.session_state["scored"] = (txn_id, D.score_via_api(PARAMS, txn),
                                              D.shadow_score(PARAMS, txn_id))
            except Exception as exc:
                st.error(f"API call failed: {exc}")
        if st.session_state.get("scored", (None,))[0] == txn_id:
            _, body, shadow = st.session_state["scored"]
            _score_card(body, shadow)
    with right:
        st.subheader("Why: exact explanations")
        if st.button("Explain both models"):
            with st.spinner("TreeSHAP + 32-coalition group Shapley..."):
                st.session_state["explained"] = (
                    txn_id, D.explain_both(PARAMS, txn_id, explainers()))
        if st.session_state.get("explained", (None,))[0] == txn_id:
            _explanation(st.session_state["explained"][1])


def _score_card(body: dict, shadow: dict | None) -> None:
    reg = {r["alias"]: r for r in registry()}
    thr = PARAMS["serving"]["decision_thresholds"]
    champ_fam = family_of(reg.get("champion", {}).get("evaluated_as"))
    shadow_fam = family_of(reg.get("challenger", {}).get("evaluated_as"))
    a, b = st.columns(2)
    a.metric(f"Champion {body['model_version'].rsplit(':', 1)[-1]} ({champ_fam})",
             f"{body['score']:.4f}", "ALERT" if body["decision"] else "pass",
             delta_color="inverse" if body["decision"] else "off", delta_arrow="off")
    if shadow and shadow.get("challenger_score") is not None:
        s = shadow["challenger_score"]
        b.metric(f"Shadow {shadow['challenger_version'].rsplit(':', 1)[-1]} ({shadow_fam})",
                 f"{s:.4f}", "would alert" if s >= thr[shadow_fam] else "would pass",
                 delta_color="inverse" if s >= thr[shadow_fam] else "off", delta_arrow="off")
    spans = body.get("spans", {})
    if spans:
        fig = go.Figure(go.Bar(
            x=[v["dur"] for v in spans.values()], y=list(spans), orientation="h",
            marker_color=ACCENT, text=[f"{v['dur']:.2f} ms · {int(v['rt'])} DB round trip"
                                       f"{'s' if v['rt'] != 1 else ''}" for v in spans.values()],
            textposition="auto"))
        fig.update_xaxes(title="ms (Server-Timing header)")
        st.plotly_chart(_layout(fig, 200), width="stretch")
    st.caption(f"Wall time from here: {body['wall_ms']:.1f} ms. The champion decides; the "
               "shadow model scored the same history after the response was sent.")


def _explanation(e: dict) -> None:
    g, x = e["graphsage"], e["xgboost"]
    groups = g["groups"]
    fig = go.Figure(go.Bar(
        x=list(groups.values()), y=list(groups), orientation="h",
        marker_color=[ALERT if v > 0 else PASS for v in groups.values()],
        text=[f"{v:+.2f}" for v in groups.values()], textposition="outside",
        cliponaxis=False))
    span = max(abs(v) for v in groups.values())
    fig.update_xaxes(range=[min(0, min(groups.values())) - 0.25 * span, 1.2 * span])
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="push on the log-odds (exact Shapley)")
    st.markdown(f"**GraphSAGE {g['score']:.4f}** "
                f"({'ALERT' if g['score'] >= g['threshold'] else 'pass'} at {g['threshold']:.4f}) "
                f"— {g['card_rows']} cardholder rows, {g['merchant_rows']} merchant rows")
    st.plotly_chart(_layout(fig, 240), width="stretch")
    parsed = re.findall(r"([\w\s]+?) \(([+-][\d.]+)\)", x["reasons"] or "")
    st.markdown(f"**XGBoost {x['score']:.4f}** "
                f"({'ALERT' if x['score'] >= x['threshold'] else 'pass'} at {x['threshold']:.4f})")
    if parsed:
        fig = go.Figure(go.Bar(
            x=[float(v) for _, v in parsed], y=[n.strip() for n, _ in parsed], orientation="h",
            marker_color=[ALERT if float(v) > 0 else PASS for _, v in parsed]))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="TreeSHAP, folded to source fields")
        st.plotly_chart(_layout(fig, 220), width="stretch")
    st.caption(f"Both explanations are exact: GraphSAGE's groups sum to its score shift "
               f"(error {g['efficiency_gap']:.0e}).")


# --- 4. registry and operations ------------------------------------------------------------

def page_ops() -> None:
    st.title("Registry and operations")
    st.markdown("The API serves `models:/fraud-champion@champion` and re-reads the aliases "
                "every 30 s, so promotion and rollback are alias moves — **no redeploy**. "
                "Moves below are real and go through `fraud.models.promote`, the same gate "
                "GitHub Actions runs.")
    reg = registry()
    st.dataframe(pl.DataFrame([{k: r.get(k) for k in ("alias", "version", "run", "evaluated_as")}
                               for r in reg]), width="stretch", hide_index=True)
    health = D.api_health(PARAMS)
    if health:
        st.caption(f"API now: champion {D.live_version(health)} · shadow "
                   f"{D.live_version(health, 'challenger_version')} · "
                   f"{health.get('alias_swaps', 0)} swaps since start · aliases checked "
                   f"{health.get('aliases_checked_s_ago', 0):.0f}s ago")

    a, b, c = st.columns(3)
    if a.button("Dry-run the gate", width="stretch"):
        _run_gate(apply=False)
    if b.button("Roll back…", width="stretch"):
        _confirm_rollback()
    if c.button("Promote the challenger…", type="primary", width="stretch"):
        _confirm_promote()

    last = st.session_state.get("ops")
    if last:
        kind, report = last
        st.subheader(f"Last action: {kind}")
        _show_report(report)
        moved_to = report.get("champion_after") or (report.get("candidate")
                                                    if report.get("applied") else None)
        if report.get("applied") and moved_to:
            # Watch once per move, then rerun so the sidebar and alias table
            # show the new state; later reruns show the stored result instead.
            if st.session_state.get("watched") != id(report):
                st.session_state["watch_result"] = _watch(f"v{moved_to}")
                st.session_state["watched"] = id(report)
                st.cache_data.clear()
                st.rerun()
            st.success(st.session_state.get("watch_result", ""))


def _run_gate(apply: bool) -> None:
    from fraud.models import promote as P

    with st.status("Running the gate: both models re-score 2019 through the registry "
                   "(~75 s)...", expanded=False) as status:
        report = P.promote(PARAMS, PARAMS["serving"]["challenger_alias"], apply)
        status.update(label="Gate finished", state="complete")
    st.session_state["ops"] = ("promotion" if apply else "dry run", report)
    st.cache_data.clear()


@st.dialog("Roll back the champion?")
def _confirm_rollback() -> None:
    reg = {r["alias"]: r for r in registry()}
    st.markdown(f"This moves **@champion** from v{reg['champion']['version']} to "
                f"**v{reg['previous']['version']}** (@previous). The running API swaps within "
                "~30 s; the demoted model keeps scoring in shadow.")
    ok = st.checkbox("I understand this changes which model makes live decisions")
    if st.button("Roll back now", type="primary", disabled=not ok):
        from fraud.models import promote as P

        report = P.rollback(PARAMS, apply=True)
        st.session_state["ops"] = ("rollback", report)
        st.cache_data.clear()
        st.rerun()


@st.dialog("Promote the challenger?")
def _confirm_promote() -> None:
    reg = {r["alias"]: r for r in registry()}
    st.markdown(f"Runs the full gate on **@challenger v{reg['challenger']['version']}** "
                f"({reg['challenger'].get('evaluated_as')}) against the champion. Only if every "
                "check passes do the aliases move: @previous ← old champion, @champion ← "
                "candidate, @challenger ← old champion.")
    ok = st.checkbox("I understand a passing gate changes which model makes live decisions")
    if st.button("Run the gate and promote", type="primary", disabled=not ok):
        _run_gate(apply=True)
        st.rerun()


def _show_report(report: dict) -> None:
    if "checks" in report:
        st.dataframe(pl.DataFrame([
            {"check": c["check"], "passed": "✓" if c["passed"] else "✗",
             "detail": ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in c.items() if k not in ("check", "passed", "alias"))}
            for c in report["checks"]]), width="stretch", hide_index=True)
    before = report.get("champion_before")
    after = report.get("champion_after") or report.get("candidate")
    if report.get("applied"):
        st.success(f"Aliases moved: @champion v{before} → v{after}.")
    elif report.get("passed") is False:
        st.info(f"Refused, nothing moved — {report.get('reason', '')}")
    else:
        st.info("Dry run: nothing moved." + (f" {report['reason']}" if report.get("reason")
                                             else ""))


def _watch(want: str) -> str:
    """Follow the swap worker by worker; return a one-line summary."""
    st.markdown(f"**Watching the API swap to {want}** — each worker re-reads the aliases on "
                "its own 30-second clock, so health checks land on old and new workers "
                "until the last one switches.")
    line, bar = st.empty(), st.progress(0.0)
    first = last = None
    streak = 0
    for elapsed, live, streak in D.watch_swap(PARAMS, want):
        if live == want and first is None:
            first = elapsed
        last = elapsed
        line.caption(f"{elapsed:4.1f}s · this check hit a worker serving `{live}` · "
                     f"{streak} consecutive checks on {want}")
        bar.progress(min(1.0, streak / 12))
    if first is not None and streak >= 12:
        return (f"Every worker serves {want}: the first switched {first:.1f}s after the move, "
                f"the last by {last:.1f}s — no redeploy.")
    return f"Not every worker serves {want} yet — they refresh every 30 s; press Refresh."


# --- 5. monitoring ---------------------------------------------------------------------------

def page_monitoring() -> None:
    st.title("Monitoring without labels")
    st.markdown("Chargebacks arrive 30-90+ days late, so the newest months have almost no "
                "labels. NannyML's **CBPE** estimates each month's AUC-PR from the scores "
                "alone; the true value is drawn beside it for hindsight.")
    monthly, summary = monitoring()
    cols = st.columns(len(summary))
    for col, r in zip(cols, summary.iter_rows(named=True), strict=True):
        col.metric(f"{r['model']}: 2018 → 2019 (true / estimated)",
                   f"{r['true_change']} / {r['estimated_change']}",
                   f"flagged {r['months_flagged_cbpe']} months",
                   delta_color="off", delta_arrow="off")
    xgb = summary.filter(pl.col("model") == "xgboost").row(0, named=True)
    st.info(f"XGBoost's decline was flagged on **{xgb['cbpe_flags_on']}**; the labels could "
            f"confirm it on {xgb['labels_flag_on']} — **{xgb['lead_days']} days earlier**.")
    for model in ("xgboost", "graphsage"):
        m = monthly.filter(pl.col("model") == model)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m["month"], y=m["true"], name="true (hindsight)",
                                 mode="lines+markers", line=dict(color=MUTED)))
        fig.add_trace(go.Scatter(x=m["month"], y=m["estimate"], name="CBPE estimate, no labels",
                                 mode="lines+markers", line=dict(color=ACCENT, width=3)))
        # Flags only where labels exist: no fraud after October 2019 in this
        # data, so later months have estimates but nothing to confirm them.
        labelled = m.filter(pl.col("true").is_not_nan() & pl.col("true").is_not_null())
        flagged = labelled.filter(pl.col("estimate") < pl.col("flag_line"))
        fig.add_trace(go.Scatter(x=flagged["month"], y=flagged["estimate"], mode="markers",
                                 name="flagged month", marker=dict(color=ALERT, size=13,
                                                                   symbol="x")))
        if labelled.height < m.height:
            # Annotation added separately: plotly cannot centre a vrect's own
            # annotation on date edges (TypeError: int + datetime.date).
            fig.add_vrect(x0=labelled["month"].max(), x1=m["month"].max(),
                          fillcolor=MUTED, opacity=0.12, line_width=0)
            fig.add_annotation(x=m["month"].max(), y=0.78, text="no fraud in the data",
                               showarrow=False, xanchor="right", font=dict(color=MUTED))
        fig.add_hline(y=m["reference"][0], line_dash="dash", line_color=MUTED,
                      annotation_text="2018 level")
        fig.add_hline(y=m["flag_line"][0], line_dash="dot", line_color=ALERT,
                      annotation_text="−15%: flag")
        fig.update_yaxes(title="monthly AUC-PR", range=[0, 0.8])
        st.subheader(model)
        st.plotly_chart(_layout(fig, 300), width="stretch")
    st.caption("No fraud exists after October 2019 in this dataset, so true values stop there "
               "while estimates continue — the blind spot: a change in the labels is invisible "
               "in the scores. Alerts need two consecutive flagged months.")


# --- 6. serving ---------------------------------------------------------------------------------

def page_serving() -> None:
    st.title("Serving")
    s = serving()
    co, eq = s["colocated"], s["equivalence"]
    c = st.columns(4)
    c[0].metric("p50", f"{co['p50']:.1f} ms")
    c[1].metric("p95", f"{co['p95']:.1f} ms")
    c[2].metric("p99", f"{co['p99']:.1f} ms")
    c[3].metric("Served == causal replay",
                f"{eq['compared'] - eq['mismatches_over_tolerance']:,} / {eq['compared']:,}",
                f"max diff {eq['max_abs_diff']:.0e}", delta_color="off", delta_arrow="off")
    st.markdown("Each request builds a ~21-node graph from Postgres — the cardholder's and the "
                "merchant's 10 most recent earlier transactions — in **2 database round trips**, "
                "and scores it on CPU. 3,000 sequential requests, Postgres in the next container.")
    load = pl.DataFrame([{"concurrency": lv["concurrency"], "req/s": lv["throughput_per_s"],
                          "p50": lv["wall_ms"]["p50"], "p95": lv["wall_ms"]["p95"]}
                         for lv in s["load"]])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=load["concurrency"].cast(pl.String), y=load["req/s"],
                         name="throughput (req/s)", marker_color=ACCENT))
    fig.add_trace(go.Scatter(x=load["concurrency"].cast(pl.String), y=load["p95"],
                             name="p95 latency (ms)", yaxis="y2", mode="lines+markers",
                             line=dict(color=ALERT)))
    fig.update_layout(yaxis=dict(title="req/s"),
                      yaxis2=dict(title="p95 ms", overlaying="y", side="right"))
    fig.update_xaxes(title="concurrent clients (3 workers, shadow model on)")
    st.plotly_chart(_layout(fig, 320), width="stretch")


# --- 7. the retrain ---------------------------------------------------------------------

def page_retrain() -> None:
    st.title("The retrain (in progress)")
    r = retrain()
    rep, graphs = r["report"], r["graphs"]["splits"]
    st.markdown(f"""
v3 was trained on a graph that **serving never builds**: its neighbours came from the
under-sampled rows (~9% fraud, against 0.12% in reality), were sampled across 1991–2017
including *later* transactions, and its "card" node was one card while serving reads the
whole cardholder. The retrain builds every training example with **the serving code
itself** — neighbour fraud rate **{graphs['train']['neighbour_fraud_rate']:.2%}** — and
selects on **all of 2018** ({rep['val_examples']:,} transactions), where v3 scores
**{rep['v3']['val_weighted_auc_pr']:.4f}**.""")
    runs = rep["runs"]
    names = list(runs)
    vals = [runs[n]["val_auc_pr"] for n in names]
    fig = go.Figure(go.Bar(
        x=["v3 (champion)", *names], y=[rep["v3"]["val_weighted_auc_pr"], *vals],
        marker_color=[MUTED] + [ACCENT if n == rep["winner"] else CONTROL for n in names],
        text=[f"{v:.4f}" for v in [rep["v3"]["val_weighted_auc_pr"], *vals]],
        textposition="outside"))
    fig.update_yaxes(title="2018 AUC-PR (served)", range=[0, 0.7])
    st.plotly_chart(_layout(fig, 340), width="stretch")
    if r["curves"]:
        fig = go.Figure()
        for name, curve in r["curves"].items():
            fig.add_trace(go.Scatter(y=curve, name=name, mode="lines",
                                     line=dict(width=3 if name == rep["winner"] else 1.5)))
        fig.add_hline(y=rep["v3"]["val_weighted_auc_pr"], line_dash="dash", line_color=MUTED,
                      annotation_text="v3")
        fig.update_xaxes(title="epoch")
        fig.update_yaxes(title="2018 AUC-PR")
        st.plotly_chart(_layout(fig, 300), width="stretch")
    st.warning(f"**{rep['winner']}** leads by +{rep['winner_gain_vs_v3_val']:.3f} on 2018 by "
               "dropping the drifted short-window velocity features. It has **not** been "
               "tested on 2019, registered, or promoted: 2019 decides, through the gate.")


