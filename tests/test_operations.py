"""Phase 6 operations: live alias swaps, the promotion gate, guarded rollback.

The registry is faked: these tests pin down the LOGIC -- what moves, when, and
what never moves -- without credentials, so they run in CI. The real thing was
exercised against DagsHub: the gate verified both models through the registry,
promoted GraphSAGE, and the running Compose API swapped in 11 s.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

PARAMS = {
    "serving": {"champion_alias": "champion", "challenger_alias": "challenger",
                "shadow": True, "alias_refresh_seconds": 30},
    "mlflow": {"registered_model_name": "fraud-champion"},
    "promotion": {"min_auc_pr_gain": 0.02, "verify_rows": 5, "verify_tolerance": 1e-5,
                  "evaluated": {
                      "xgb-a": {"scores": "x.npz", "evidence": [], "monitoring": "xgboost"},
                      "gnn-a": {"scores": "g.npz", "evidence": ["evidence.json"],
                                "monitoring": "graphsage"},
                      "gnn-b": {"scores": "g2.npz", "evidence": ["evidence.json"],
                                "monitoring": "graphsage-b"},
                  }},
}


# --- the live swap -------------------------------------------------------------------

class FakeScorer:
    def __init__(self, alias, version):
        self.model_version = f"fraud-champion@{alias}:v{version}"


def _models(monkeypatch, registry: dict, fail: set | None = None):
    import fraud.api.main as main

    def load(params, alias):
        if fail and alias in fail:
            raise RuntimeError("registry unavailable")
        return FakeScorer(alias, registry[alias])

    monkeypatch.setattr(main, "resolve_alias", lambda params, alias: registry[alias])
    monkeypatch.setattr(main, "load_scorer", load)
    return main.Models(PARAMS)


def test_moving_an_alias_swaps_the_live_model(monkeypatch):
    registry = {"champion": "4", "challenger": "3"}
    models = _models(monkeypatch, registry)
    models.refresh()
    assert models.champion.model_version.endswith("v4") and models.swaps == 0

    registry.update(champion="3", challenger="4")       # a promotion
    assert models.refresh() is True
    assert models.champion.model_version.endswith("v3")
    assert models.challenger.model_version.endswith("v4")
    assert models.swaps == 2


def test_nothing_reloads_when_no_alias_moved(monkeypatch):
    registry = {"champion": "4", "challenger": "3"}
    models = _models(monkeypatch, registry)
    models.refresh()
    first = models.champion
    assert models.refresh() is False and models.champion is first


def test_a_failed_load_leaves_the_old_model_serving(monkeypatch):
    """The new model is loaded fully BEFORE the reference is swapped, so a
    registry blip mid-swap cannot leave the API without a champion."""
    registry = {"champion": "4", "challenger": "3"}
    models = _models(monkeypatch, registry)
    models.refresh()
    registry["champion"] = "3"
    broken = _models(monkeypatch, registry, fail={"champion"})
    broken.champion, broken.versions = models.champion, dict(models.versions)
    with pytest.raises(RuntimeError):
        broken.refresh()
    assert broken.champion.model_version.endswith("v4")


# --- the gate --------------------------------------------------------------------

def _gate(monkeypatch, tmp_path, *, verified=True, gain=0.2, evidence=True,
          names=("gnn-a", "xgb-a")):
    import fraud.models.promote as P

    monkeypatch.chdir(tmp_path)
    if evidence:
        (tmp_path / "evidence.json").write_text("{}")
    tags = {"challenger": names[0], "champion": names[1]}
    monkeypatch.setattr(P, "repo_path", lambda p: tmp_path / p)
    monkeypatch.setattr(P, "evaluated_model", lambda params, alias: tags[alias])
    monkeypatch.setattr(P, "family", lambda params, alias:
                        "xgboost" if tags[alias].startswith("xgb") else "graphsage")
    monkeypatch.setattr(P, "verify_registry", lambda params, alias, fam, name: {
        "alias": alias, "version": "v", "family": fam, "evaluated_model": name, "rows": 5,
        "max_abs_diff": 0.0 if verified else 0.2, "passed": verified})
    rng = np.random.default_rng(0)
    labels = (rng.uniform(size=4000) < 0.05).astype(int)

    def ref(params, name):
        # Overlapping classes, so AUC-PR moves with `quality` instead of
        # saturating at 1.0 for both models. The candidate is better by `gain`.
        quality = 0.3 + (gain if name == names[0] else 0.0)
        score = labels * quality + rng.uniform(size=labels.size)
        return pl.DataFrame({"txn_id": np.arange(labels.size), "score": score, "label": labels})

    monkeypatch.setattr(P, "reference_scores", ref)
    return P.gate(PARAMS, "challenger")


def test_gate_passes_a_verified_better_model_with_evidence(monkeypatch, tmp_path):
    report = _gate(monkeypatch, tmp_path)
    assert report["passed"]
    assert [c["check"] for c in report["checks"]][-1] == "beats the champion"


def test_gate_refuses_weights_that_do_not_reproduce_their_evaluation(monkeypatch, tmp_path):
    report = _gate(monkeypatch, tmp_path, verified=False)
    assert not report["passed"] and "does not reproduce" in report["reason"]


def test_gate_refuses_without_the_evidence(monkeypatch, tmp_path):
    report = _gate(monkeypatch, tmp_path, evidence=False)
    assert not report["passed"] and "missing evidence" in report["reason"]


def test_gate_refuses_a_model_that_is_not_better_enough(monkeypatch, tmp_path):
    report = _gate(monkeypatch, tmp_path, gain=0.0)
    assert not report["passed"] and "below the required" in report["reason"]


def test_two_versions_of_one_family_are_judged_on_their_own_scores(monkeypatch, tmp_path):
    # GraphSAGE v5 vs GraphSAGE v3: a family-keyed lookup would hand both the
    # same file and compare the model with itself (gain exactly 0).
    better = _gate(monkeypatch, tmp_path, names=("gnn-b", "gnn-a"))
    assert better["passed"]
    beats = [c for c in better["checks"] if c["check"] == "beats the champion"][0]
    assert (beats["candidate"], beats["champion"]) == ("gnn-b", "gnn-a")
    assert beats["gain"] > 0.02


def test_a_version_with_no_recorded_evaluation_cannot_be_promoted(monkeypatch, tmp_path):
    report = _gate(monkeypatch, tmp_path, names=("gnn-unknown", "gnn-a"))
    assert not report["passed"] and "no entry in promotion.evaluated" in report["reason"]


def test_every_evaluated_model_points_at_a_monitored_key():
    from fraud.config import load_params

    params = load_params()
    for name, entry in params["promotion"]["evaluated"].items():
        assert set(entry) == {"scores", "evidence", "monitoring"}, name
        assert entry["monitoring"] in params["monitoring"]["scores"], name


@pytest.mark.parametrize("passed, code", [(True, 0), (False, 3)])
def test_a_refusal_exits_3_so_it_is_not_mistaken_for_a_crash(monkeypatch, tmp_path, passed, code):
    # operate.yml lets a refused DRY RUN finish green but must still fail on a
    # traceback, which exits 1 -- so a refusal must never exit 1.
    import fraud.models.promote as P

    monkeypatch.setattr(P, "repo_path", lambda p: tmp_path / p.split("/")[-1])
    monkeypatch.setattr(P, "promote", lambda params, cand, apply: {"passed": passed})
    assert P.main([]) == code
    assert P.REFUSED == 3


# --- guarded rollback --------------------------------------------------------------

def test_rollback_if_alerting_does_nothing_for_a_healthy_champion(monkeypatch):
    import fraud.models.promote as P

    monkeypatch.setattr(P, "champion_alerting", lambda params: {
        "family": "graphsage", "alerting": False})
    monkeypatch.setattr(P, "_client", lambda params: pytest.fail("must not touch the registry"))
    report = P.rollback(PARAMS, apply=True, if_alerting=True)
    assert report["applied"] is False and report["skipped"] == "champion not alerting"


def test_decision_thresholds_are_the_validation_chosen_ones():
    """Each model alerts at the score where its VALIDATION FPR is 1% -- never at
    the FPR target itself (the API compared every score with 0.01 until Phase 7)."""
    import json
    import pathlib

    from fraud.config import load_params

    thresholds = load_params()["serving"]["decision_thresholds"]
    xgb = json.loads(pathlib.Path("reports/metrics_xgb.json").read_text())
    gnn = json.loads(pathlib.Path("reports/metrics_replay_2018.json").read_text())
    assert xgb["threshold_chosen_on"] == "val" and gnn["years"] == [2018]
    assert thresholds["xgboost"] == pytest.approx(xgb["threshold"], abs=1e-6)
    assert thresholds["graphsage"] == pytest.approx(gnn["metrics"]["threshold"], abs=1e-6)
