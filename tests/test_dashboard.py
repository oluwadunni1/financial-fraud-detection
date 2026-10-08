"""The presentation app renders every page, and shows the reports' numbers.

Pages are driven headless through Streamlit's AppTest. Pages that read
DVC-tracked data (processed rows, score files) skip without it; pages that call
the live API or the registry must still render -- with an error banner -- when
neither is reachable, which is exactly the CI situation.
"""

from __future__ import annotations

import pytest

from fraud.config import load_params, repo_path

PARAMS = load_params()
APP = str(repo_path("app/streamlit_app.py"))
HAS_DATA = (repo_path(PARAMS["paths"]["processed"]).exists()
            and repo_path("reports/metrics_replay.npz").exists())

streamlit = pytest.importorskip("streamlit")


@pytest.fixture(autouse=True)
def _restore_database_url(monkeypatch):
    """Pages call `dashboard.data.connect`, which re-points DATABASE_URL at the Compose
    Postgres on purpose. Inside one pytest process that would leak into every later
    test that reads DATABASE_URL; monkeypatch restores it after each test."""
    import os

    if "DATABASE_URL" in os.environ:
        monkeypatch.setenv("DATABASE_URL", os.environ["DATABASE_URL"])
    else:
        monkeypatch.delenv("DATABASE_URL", raising=False)


PAGES = {"answer": "page_answer", "replay": "page_replay", "transaction": "page_transaction",
         "operations": "page_ops", "monitoring": "page_monitoring", "serving": "page_serving",
         "retrain": "page_retrain"}


def _page(name: str):
    """Render one page alone (AppTest cannot switch to function-defined pages)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(
        "from fraud.dashboard import pages\n"
        "pages.sidebar()\n"
        f"pages.{PAGES[name]}()\n", default_timeout=120)
    return at.run()


def test_the_app_entry_point_renders():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, [e.message for e in at.exception]


def test_scoreboard_is_the_reports():
    from fraud.dashboard import data as D

    rows = {r["model"]: r["auc_pr"] for r in D.scoreboard(PARAMS)}
    assert round(rows["GraphSAGE, served (@champion)"], 4) == 0.4665
    assert round(rows["XGBoost"], 4) == 0.2501
    assert round(rows["GraphSAGE, another merchant's history"], 4) == 0.0379


def test_the_dashboard_refuses_to_reset_supabase(monkeypatch):
    from fraud.dashboard import data as D

    monkeypatch.setenv("FRAUD_DB_URL", "postgresql://u:p@aws-1-eu-west-1.pooler.supabase.com/db")
    with pytest.raises(RuntimeError, match="Supabase"):
        D.prepare_replay(PARAMS)


@pytest.mark.parametrize("page", ["answer", "replay", "operations", "serving", "retrain"])
def test_page_renders(page):
    at = _page(page)
    assert not at.exception, [e.message for e in at.exception]
    assert at.title, "every page has a title"


@pytest.mark.skipif(not HAS_DATA, reason="needs processed data and score files (dvc pull)")
@pytest.mark.parametrize("page", ["transaction", "monitoring"])
def test_data_pages_render(page):
    at = _page(page)
    assert not at.exception, [e.message for e in at.exception]


def test_the_answer_page_shows_the_served_number():
    at = _page("answer")
    assert any("0.4665" in m.value for m in at.metric)
