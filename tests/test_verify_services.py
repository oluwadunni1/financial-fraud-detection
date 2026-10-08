"""scripts/verify_services.py must check the retention function WITHOUT running it.

`prune_transaction_events('30 days')` deletes rows older than 30 days before NOW,
which with this project's 2019 data is every row. The check once committed it and
emptied the Supabase hot store (37,031 rows). It must roll back.
"""

from __future__ import annotations

import importlib.util
import sys
import types

from fraud.config import repo_path


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "verify_services", repo_path("scripts/verify_services.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCursor:
    def __init__(self, log):
        self.log, self.last = log, ""

    def execute(self, sql, params=None):
        self.last = " ".join(str(sql).split())
        self.log.append(("execute", self.last))

    def fetchone(self):
        if "pg_extension" in self.last:
            return ("vector",)
        if "vector_dims" in self.last:
            return (64,)
        if "prune_transaction_events" in self.last:
            return (37031,)
        return None

    def fetchall(self):
        if "pg_tables" in self.last:
            return [(t,) for t in ("node_embeddings", "fm_embeddings", "transaction_events",
                                   "predictions", "labels", "drift_metrics", "job_watermarks")]
        if "information_schema.columns" in self.last:
            return [("challenger_score",), ("challenger_version",), ("challenger_latency_ms",)]
        return []


class FakeConn:
    def __init__(self, log):
        self.log = log

    def cursor(self):
        return FakeCursor(self.log)

    def commit(self):
        self.log.append(("commit", ""))

    def rollback(self):
        self.log.append(("rollback", ""))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.log.append(("exit", ""))   # psycopg commits here: must find nothing pending


def test_the_prune_check_rolls_back_and_never_commits_the_prune(monkeypatch, capsys):
    script = _load_script()
    log: list[tuple[str, str]] = []
    fake_psycopg = types.SimpleNamespace(connect=lambda *a, **k: FakeConn(log))
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setattr(script.settings, "require_all", lambda *names, **kw: {})
    monkeypatch.setattr(script.settings, "database_url", lambda: "postgresql://fake")

    script.check_supabase(apply_sql=False)

    prune = next(i for i, (kind, sql) in enumerate(log)
                 if kind == "execute" and "prune_transaction_events" in sql)
    after = [kind for kind, _ in log[prune + 1:]]
    assert after[0] == "rollback", f"the prune must be rolled back first, got {after}"
    assert "commit" not in after, "nothing may be committed after the prune ran"
    # Earlier work was committed BEFORE the prune, so the rollback undoes nothing else.
    assert log[prune - 1][0] == "commit"
    out = capsys.readouterr().out
    assert "would remove 37031 rows; rolled back, nothing deleted" in out
    assert script.fail_count == 0
