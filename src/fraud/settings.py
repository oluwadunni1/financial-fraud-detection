"""The environment: secrets and per-machine overrides, read in one place.

Two kinds of configuration live in this project, and they never mix:

  params.yaml + fraud.config   versioned, non-secret configuration -- thresholds,
                               paths, model settings. Committed, tracked by DVC,
                               the same on every machine.
  fraud.settings (this file)   the environment -- credentials and per-machine
                               overrides. Comes from `.env` locally, from the
                               Compose `environment:`/`env_file:` in containers,
                               and from GitHub Actions secrets in CI. Never
                               committed.

Every read of a credential goes through a function here; nothing else in the
package calls `load_dotenv` or reads a secret from `os.environ` directly.

Reads are LAZY, on purpose: there are no module-level constants holding an
environment value, and every function looks at `os.environ` when it is called.
`fraud.dashboard.data.connect` relies on this -- it points DATABASE_URL at the
Compose Postgres immediately before `store.connect()`, and the replay page then
RESETS that store. A value captured at import time would still be `.env`'s
Supabase URL.

`.env` itself is loaded at most once, without overriding anything already set,
and a missing file is a normal case (containers and CI have none).

Values never appear in an error message or a log line -- only their names. A
value containing "<" is treated as missing: `.env.example` placeholders look
like `<dagshub-token>` (decision 10).

Imports at module level are the standard library and python-dotenv only, so
this module loads in `.venv-monitoring` (fraud installed --no-deps) and in
every image. MLflow is imported inside `configure_mlflow`.
"""

from __future__ import annotations

import functools
import os

from dotenv import load_dotenv

from fraud.config import REPO_ROOT

MLFLOW_VARS = ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")


class MissingSetting(RuntimeError):
    """One or more required variables are unset or still a placeholder.

    `names` lists them; the message never contains a value.
    """

    def __init__(self, names: list[str], hint: str = ""):
        self.names = names
        message = f"missing or placeholder in the environment (.env): {', '.join(names)}"
        super().__init__(f"{message}. {hint}" if hint else message)


@functools.cache
def load_env() -> None:
    """Load `.env` once. Never overrides a variable that is already set."""
    load_dotenv(REPO_ROOT / ".env", override=False)


def _usable(value: str | None) -> bool:
    return bool(value) and "<" not in value


def optional(name: str) -> str | None:
    """The variable's value, or None if it is unset or a placeholder."""
    load_env()
    value = os.environ.get(name)
    return value if _usable(value) else None


def require(name: str, hint: str = "") -> str:
    """The variable's value; MissingSetting (naming it) if unset or a placeholder."""
    return require_all(name, hint=hint)[name]


def require_all(*names: str, hint: str = "") -> dict[str, str]:
    """All of `names`, or one MissingSetting listing EVERY one that is missing."""
    load_env()
    missing = [n for n in names if not _usable(os.environ.get(n))]
    if missing:
        raise MissingSetting(missing, hint)
    return {n: os.environ[n] for n in names}


def override(name: str, default: str) -> str:
    """A per-machine override that falls back to a params.yaml value.

    For non-secret knobs such as FRAUD_API / FRAUD_DB_URL: the environment wins,
    otherwise `default` (which the caller takes from params.yaml).
    """
    load_env()
    return os.environ.get(name, default)


def database_url() -> str:
    """The hot store's connection string (Supabase in `.env`; Compose sets its own)."""
    return require("DATABASE_URL")


def mlflow_tracking_uri() -> str:
    return require("MLFLOW_TRACKING_URI")


def configure_mlflow(hint: str = "") -> str:
    """Point MLflow at DagsHub. Call before ANY tracking or registry call.

    With no tracking URI MLflow silently falls back to a local ./mlflow.db and
    then reports that the registered model "does not exist" (CLAUDE.md gotcha).
    All three credentials are required, and checked together. Returns the URI.
    """
    env = require_all(*MLFLOW_VARS, hint=hint)
    import mlflow

    mlflow.set_tracking_uri(env["MLFLOW_TRACKING_URI"])
    return env["MLFLOW_TRACKING_URI"]
