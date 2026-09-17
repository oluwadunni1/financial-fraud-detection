"""Stage `register`: log the run to MLflow and set the `champion` alias.

Aliases, not stages. ARCHITECTURE §6 says "promote to Production", but MLflow 3
deprecates stage transitions in favour of named aliases, and Phase 0's smoke
test already proved `set_registered_model_alias` works against DagsHub. Aliases
also match the champion/challenger vocabulary the rest of the project uses: the
API loads `models:/fraud-champion@champion`, and swapping the champion is a
one-line alias move with no redeploy.

This stage needs network access and DagsHub credentials in `.env`, unlike every
other stage in the pipeline. CI without credentials should run
`dvc repro evaluate`, which covers the whole DAG up to but not including this.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import mlflow
import xgboost as xgb
from dotenv import load_dotenv

from fraud.config import REPO_ROOT, load_params, repo_path

ALIAS = "champion"


def _flatten(prefix: str, payload: dict) -> dict[str, float]:
    """MLflow metrics are flat floats; the report is nested."""
    flat: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            flat |= _flatten(f"{prefix}{key}_", value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[f"{prefix}{key}"] = float(value)
    return flat


def register(
    model_dir: pathlib.Path,
    encoder_path: pathlib.Path,
    report_path: pathlib.Path,
    params: dict,
) -> dict:
    load_dotenv(REPO_ROOT / ".env")
    for required in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME",
                     "MLFLOW_TRACKING_PASSWORD"):
        if not os.getenv(required) or "<" in os.getenv(required, ""):
            raise RuntimeError(
                f"{required} is missing or a placeholder in .env. "
                "This stage needs DagsHub credentials; the rest of the pipeline "
                "does not (run `dvc repro evaluate` to skip it)."
            )

    info = json.loads((model_dir / "train_info.json").read_text())
    report = json.loads(report_path.read_text())
    booster = xgb.Booster()
    booster.load_model(model_dir / "model.json")

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    name = params["mlflow"]["registered_model_name"]

    with mlflow.start_run(run_name="xgboost-baseline") as run:
        mlflow.log_params(
            {
                "model": "xgboost",
                **{k: v for k, v in params["xgboost"].items()},
                "scale_pos_weight": info["scale_pos_weight"],
                "best_iteration": info["best_iteration"],
                "n_features": info["n_features"],
                "train_rows": info["train_rows"],
                "velocity_windows": params["velocity"]["windows_hours"],
                "one_hot_max_cardinality": params["features"][
                    "one_hot_max_cardinality"
                ],
            }
        )
        # Headline metrics get unprefixed names so they sort to the top of a
        # registry comparison against the GNN and the foundation model.
        headline = report["splits"]["test_2019_only"]
        mlflow.log_metrics(
            {
                "auc_pr": headline["auc_pr"],
                "auc_pr_lift": report["auc_pr_lift_over_base_rate"],
                "val_auc_pr": info["val_auc_pr"],
            }
        )
        for split_name, split in report["splits"].items():
            mlflow.log_metrics(_flatten(f"{split_name}_", split))

        # The encoder ships with the model: serving must apply the identical
        # mapping, and a re-fitted one silently changes every column's meaning.
        mlflow.log_artifact(str(encoder_path), artifact_path="encoder")
        mlflow.log_dict(report, "metrics_xgb.json")
        mlflow.log_dict(
            {"feature_names": info["feature_names"]}, "feature_names.json"
        )

        mlflow.xgboost.log_model(
            booster, name="model", registered_model_name=name
        )
        run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{name}'")
    mine = [v for v in versions if v.run_id == run_id]
    if not mine:
        raise RuntimeError(f"registered model {name} has no version for {run_id}")
    version = mine[0].version
    client.set_registered_model_alias(name, ALIAS, version)

    resolved = client.get_model_version_by_alias(name, ALIAS)
    return {
        "run_id": run_id,
        "registered_model": name,
        "version": str(version),
        "alias": ALIAS,
        "alias_resolves_to": str(resolved.version),
        "model_uri": f"models:/{name}@{ALIAS}",
        "headline_auc_pr": report["splits"]["test_2019_only"]["auc_pr"],
    }


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=params["paths"]["models"])
    ap.add_argument("--encoder", default=params["paths"]["encoder"])
    ap.add_argument("--report", default=params["paths"]["metrics_report"])
    ap.add_argument("--output", default=params["paths"]["registry_report"])
    args = ap.parse_args(argv)

    result = register(
        repo_path(args.model) / "xgb",
        repo_path(args.encoder),
        repo_path(args.report),
        params,
    )

    out = repo_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(
        f"registered {result['registered_model']} v{result['version']} "
        f"as @{result['alias']} (run {result['run_id'][:8]}) "
        f"| headline AUC-PR {result['headline_auc_pr']:.4f}"
    )
    print(f"load with: mlflow.pyfunc.load_model('{result['model_uri']}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
