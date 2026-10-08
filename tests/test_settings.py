"""fraud.settings: one lazy, secret-safe door to the environment.

The laziness test is the one that matters most: the dashboard re-points
DATABASE_URL at the Compose Postgres immediately before resetting the store, so
a value cached at import would aim that reset at Supabase.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from fraud import settings

SECRET = "s3cr3t-value-that-must-never-be-printed"


@pytest.fixture
def no_dotenv(monkeypatch, tmp_path):
    """Point settings at an empty repo root, so the real .env cannot interfere."""
    monkeypatch.setattr(settings, "REPO_ROOT", tmp_path)
    settings.load_env.cache_clear()
    yield tmp_path
    settings.load_env.cache_clear()


def test_require_raises_on_an_unset_variable_naming_it(no_dotenv, monkeypatch):
    monkeypatch.delenv("FRAUD_TEST_VAR", raising=False)
    with pytest.raises(settings.MissingSetting, match="FRAUD_TEST_VAR") as exc:
        settings.require("FRAUD_TEST_VAR")
    assert exc.value.names == ["FRAUD_TEST_VAR"]


def test_a_placeholder_counts_as_missing_and_its_value_is_never_printed(no_dotenv, monkeypatch):
    monkeypatch.setenv("FRAUD_TEST_VAR", f"<{SECRET}>")
    with pytest.raises(settings.MissingSetting) as exc:
        settings.require("FRAUD_TEST_VAR")
    assert "FRAUD_TEST_VAR" in str(exc.value)
    assert SECRET not in str(exc.value) and SECRET not in repr(exc.value)


def test_require_all_reports_every_missing_name_in_one_error(no_dotenv, monkeypatch):
    monkeypatch.setenv("FRAUD_A", "ok")
    monkeypatch.delenv("FRAUD_B", raising=False)
    monkeypatch.setenv("FRAUD_C", "<placeholder>")
    with pytest.raises(settings.MissingSetting) as exc:
        settings.require_all("FRAUD_A", "FRAUD_B", "FRAUD_C", hint="extra advice")
    assert exc.value.names == ["FRAUD_B", "FRAUD_C"]
    assert str(exc.value).endswith("extra advice")
    monkeypatch.setenv("FRAUD_B", "b")
    monkeypatch.setenv("FRAUD_C", "c")
    assert settings.require_all("FRAUD_A", "FRAUD_B", "FRAUD_C") == {
        "FRAUD_A": "ok", "FRAUD_B": "b", "FRAUD_C": "c"}


def test_load_env_reads_dotenv_but_never_overrides_the_environment(no_dotenv, monkeypatch):
    (no_dotenv / ".env").write_text("FRAUD_FROM_FILE=file\nFRAUD_BOTH=file\n")
    monkeypatch.delenv("FRAUD_FROM_FILE", raising=False)
    monkeypatch.setenv("FRAUD_BOTH", "process")
    assert settings.require("FRAUD_FROM_FILE") == "file"
    assert settings.require("FRAUD_BOTH") == "process"
    monkeypatch.delenv("FRAUD_FROM_FILE")      # tidy: load_dotenv wrote os.environ


def test_a_missing_dotenv_is_a_normal_case(no_dotenv, monkeypatch):
    monkeypatch.setenv("FRAUD_TEST_VAR", "from-the-container")
    assert not (no_dotenv / ".env").exists()
    assert settings.require("FRAUD_TEST_VAR") == "from-the-container"


def test_reads_are_lazy_a_changed_value_is_seen(no_dotenv, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@supabase.example/db")
    assert settings.database_url().endswith("supabase.example/db")
    # What fraud.dashboard.data.connect does just before resetting the store.
    monkeypatch.setenv("DATABASE_URL", "postgresql://fraud:fraud@localhost:5433/fraud")
    assert settings.database_url() == "postgresql://fraud:fraud@localhost:5433/fraud"


def test_override_prefers_the_environment_then_the_params_default(no_dotenv, monkeypatch):
    monkeypatch.delenv("FRAUD_API", raising=False)
    assert settings.override("FRAUD_API", "http://localhost:8000") == "http://localhost:8000"
    monkeypatch.setenv("FRAUD_API", "http://api:8000")
    assert settings.override("FRAUD_API", "http://localhost:8000") == "http://api:8000"


def test_optional_returns_none_for_unset_or_placeholder(no_dotenv, monkeypatch):
    monkeypatch.delenv("FRAUD_TEST_VAR", raising=False)
    assert settings.optional("FRAUD_TEST_VAR") is None
    monkeypatch.setenv("FRAUD_TEST_VAR", "<placeholder>")
    assert settings.optional("FRAUD_TEST_VAR") is None
    monkeypatch.setenv("FRAUD_TEST_VAR", "set")
    assert settings.optional("FRAUD_TEST_VAR") == "set"


def test_configure_mlflow_needs_all_three_and_sets_the_uri(no_dotenv, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.org/x.mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", SECRET)
    with pytest.raises(settings.MissingSetting) as exc:
        settings.configure_mlflow()
    assert exc.value.names == ["MLFLOW_TRACKING_USERNAME"] and SECRET not in str(exc.value)

    import mlflow

    before = mlflow.get_tracking_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "someone")
    try:
        assert settings.configure_mlflow() == "https://example.org/x.mlflow"
        assert mlflow.get_tracking_uri() == "https://example.org/x.mlflow"
    finally:
        mlflow.set_tracking_uri(before)


def test_importing_settings_loads_no_heavy_dependency():
    # A fresh interpreter: this test process has long since imported mlflow.
    code = ("import sys, fraud.settings; "
            "print(sorted(m for m in ('mlflow', 'torch', 'xgboost', 'psycopg') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_nothing_else_loads_dotenv():
    """Every environment read goes through fraud.settings."""
    import pathlib

    root = pathlib.Path(settings.__file__).resolve().parents[2]
    offenders = [str(p.relative_to(root)) for d in ("src/fraud", "scripts")
                 for p in (root / d).rglob("*.py")
                 if "load_dotenv" in p.read_text() and p.name != "settings.py"]
    assert offenders == []
