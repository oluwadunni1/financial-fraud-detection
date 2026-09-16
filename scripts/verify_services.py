"""Phase 0 step 7: prove DagsHub MLflow and Supabase actually work.

Neither service is exercised anywhere else until Phase 2 (registry) and Phase 4
(serving store), so a silent misconfiguration here would surface a long way from
its cause. This script fails loudly instead.

    python scripts/verify_services.py                 # check both, change nothing
    python scripts/verify_services.py --apply-sql     # also apply sql/*.sql
    python scripts/verify_services.py --only mlflow   # or --only supabase

Needs credentials in .env. Supabase checks additionally need psycopg:
    uv pip install "psycopg[binary]"
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import uuid

from dotenv import load_dotenv

REPO = pathlib.Path(__file__).resolve().parent.parent
SQL_FILES = ["sql/001_init.sql", "sql/002_challenger.sql"]

EXPECTED_TABLES = {
    "node_embeddings", "fm_embeddings", "transaction_events",
    "predictions", "labels", "drift_metrics", "job_watermarks",
}

ok_count = 0
fail_count = 0


def ok(msg: str) -> None:
    global ok_count
    ok_count += 1
    print(f"  \033[32mPASS\033[0m {msg}")


def fail(msg: str) -> None:
    global fail_count
    fail_count += 1
    print(f"  \033[31mFAIL\033[0m {msg}")


def require_env(*keys: str) -> bool:
    missing = [k for k in keys if not os.getenv(k) or "<" in os.getenv(k, "")]
    if missing:
        fail(f"missing/placeholder in .env: {', '.join(missing)}")
        return False
    return True


# --------------------------------------------------------------------------
def check_mlflow() -> None:
    print("\nDagsHub MLflow")
    if not require_env("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME",
                       "MLFLOW_TRACKING_PASSWORD"):
        return

    import mlflow
    from mlflow.tracking import MlflowClient
    from sklearn.linear_model import LogisticRegression

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    try:
        mlflow.set_experiment("phase0-smoke")
        ok("tracking server reachable, experiment created")
    except Exception as e:
        fail(f"cannot reach tracking server: {type(e).__name__}: {e}")
        return

    # A throwaway name so a failed run never collides with the real registry.
    name = f"phase0-smoke-{uuid.uuid4().hex[:8]}"
    try:
        model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
        with mlflow.start_run(run_name="phase0-smoke") as run:
            mlflow.log_param("purpose", "phase-0 connectivity check")
            mlflow.log_metric("aucpr", 0.5)
            mlflow.sklearn.log_model(model, name="model", registered_model_name=name)
        ok(f"run logged (run_id={run.info.run_id[:8]}), params + metrics accepted")
    except Exception as e:
        fail(f"cannot log a run or model: {type(e).__name__}: {e}")
        return

    # THE check: ARCHITECTURE.md 4.3 asserts the Model Registry API works here.
    # Champion/challenger promotion depends on it entirely.
    try:
        versions = client.search_model_versions(f"name='{name}'")
        assert versions, "registered model has no versions"
        v = versions[0].version
        client.set_registered_model_alias(name, "champion", v)
        alias = client.get_model_version_by_alias(name, "champion")
        assert alias.version == v
        ok(f"Model Registry works: registered v{v}, alias 'champion' set and read back")
    except Exception as e:
        fail(f"Model Registry unusable -- champion/challenger swap will not work: "
             f"{type(e).__name__}: {e}")
    finally:
        try:
            client.delete_registered_model(name)
            ok("smoke model deleted (registry left clean)")
        except Exception as e:
            fail(f"could not clean up '{name}', delete it by hand: {e}")


# --------------------------------------------------------------------------
def check_supabase(apply_sql: bool) -> None:
    print("\nSupabase Postgres")
    if not require_env("DATABASE_URL"):
        return
    try:
        import psycopg
    except ImportError:
        fail('psycopg not installed -- run: uv pip install "psycopg[binary]"')
        return

    try:
        conn = psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=15)
    except Exception as e:
        fail(f"cannot connect: {type(e).__name__}: {e}")
        return

    with conn:
        cur = conn.cursor()

        if apply_sql:
            for f in SQL_FILES:
                try:
                    cur.execute((REPO / f).read_text())
                    conn.commit()
                    ok(f"applied {f}")
                except Exception as e:
                    conn.rollback()
                    fail(f"{f} failed: {type(e).__name__}: {e}")

        cur.execute("select extname from pg_extension where extname = 'vector'")
        if cur.fetchone():
            ok("pgvector extension installed")
        else:
            fail("pgvector NOT installed -- embeddings cannot be stored. "
                 "Run: create extension vector;")

        cur.execute("select tablename from pg_tables where schemaname = 'public'")
        present = {r[0] for r in cur.fetchall()}
        missing = EXPECTED_TABLES - present
        if missing:
            fail(f"missing tables: {', '.join(sorted(missing))} "
                 f"(run with --apply-sql)")
        else:
            ok(f"all {len(EXPECTED_TABLES)} tables present")

        # Shadow-mode columns from 002_challenger.sql.
        cur.execute("""select column_name from information_schema.columns
                       where table_name = 'predictions'""")
        cols = {r[0] for r in cur.fetchall()}
        need = {"challenger_score", "challenger_version", "challenger_latency_ms"}
        if need <= cols:
            ok("shadow-mode challenger columns present on predictions")
        elif cols:
            fail(f"predictions missing {', '.join(sorted(need - cols))} "
                 f"-- apply sql/002_challenger.sql")

        # A vector(64) must survive a round trip, or the whole serving path is moot.
        if "node_embeddings" in present:
            try:
                vec = "[" + ",".join(["0.5"] * 64) + "]"
                cur.execute(
                    """insert into node_embeddings
                         (node_type, node_id, embedding, model_version)
                       values ('user', -999, %s::vector, 'phase0-smoke')
                       on conflict (node_type, node_id) do update
                         set embedding = excluded.embedding""", (vec,))
                cur.execute("""select vector_dims(embedding) from node_embeddings
                               where node_type='user' and node_id=-999""")
                dims = cur.fetchone()[0]
                assert dims == 64, f"expected 64 dims, got {dims}"
                cur.execute("delete from node_embeddings where node_id = -999")
                conn.commit()
                ok("vector(64) insert/read/delete round-trips")
            except Exception as e:
                conn.rollback()
                fail(f"vector round-trip failed: {type(e).__name__}: {e}")

        try:
            cur.execute("select prune_transaction_events('30 days'::interval)")
            conn.commit()
            ok(f"prune_transaction_events() callable (removed {cur.fetchone()[0]} rows)")
        except Exception as e:
            conn.rollback()
            fail(f"prune_transaction_events() not callable: {type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply-sql", action="store_true",
                    help="apply sql/*.sql before checking (idempotent)")
    ap.add_argument("--only", choices=["mlflow", "supabase"])
    args = ap.parse_args()

    load_dotenv(REPO / ".env")
    if args.only != "supabase":
        check_mlflow()
    if args.only != "mlflow":
        check_supabase(args.apply_sql)

    print(f"\n{ok_count} passed, {fail_count} failed")
    return 1 if fail_count else 0


if __name__ == "__main__":
    sys.exit(main())
