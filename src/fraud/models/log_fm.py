"""Log the Phase 3.5 foundation-model experiment to MLflow -- as a record, not a model.

Nothing here is registered or aliased. The best FM head (NVIDIA's isolated
extraction + base features, 0.2215 test-2019 AUC-PR) sits below both the
champion (0.2501) and GraphSAGE served (0.4667), so it has no claim on
`@champion` or `@challenger`. What it does have is a result worth keeping
findable next to the runs it was compared against:

    NVIDIA's extraction helps (+23% over the same-rows base); our in-sequence
    extraction does not (0.1368 combined, chance alone), because deep in a
    4096-token context the final layer encodes the CARD, not the transaction.

One parent run carries the comparison and the embedding reports; one nested
run per head variant carries its params, metrics and booster.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import mlflow

from fraud import settings
from fraud.config import load_params, repo_path
from fraud.models.registry_gnn import _flatten

VERDICT = (
    "NVIDIA's isolated extraction adds +23% over a same-rows base; the "
    "in-sequence (contextual) extraction carries ~0% per-transaction variance "
    "within a card and scores at chance alone. Not registered: best head "
    "0.2215 is below champion 0.2501 and GraphSAGE served 0.4667."
)


def log(metrics_path: pathlib.Path, head_dir: pathlib.Path,
        reports: list[pathlib.Path], params: dict) -> dict:
    settings.require_all(*settings.MLFLOW_VARS)

    report = json.loads(metrics_path.read_text())
    fm = params["fm"]
    settings.configure_mlflow()
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    children = {}
    with mlflow.start_run(run_name="fm-phase3.5-experiment") as parent:
        mlflow.set_tags({
            "phase": "3.5",
            "model_family": "transaction-foundation-model",
            "registered": "false",
            "verdict": VERDICT,
        })
        mlflow.log_params({
            "checkpoint": f"{fm['checkpoint']['repo']}@{fm['checkpoint']['ref']}",
            "upstream": "NVIDIA-AI-Blueprints/transaction-foundation-model",
            "min_year": fm["min_year"],
            "seq_length": fm["seq_length"],
            "pca_dim": fm["head"]["pca_dim"],
            "pca_fit_rows": fm["head"]["pca_fit_rows"],
            "train_years": json.dumps(report["train_years"]),
        })
        mlflow.log_metrics(
            {f"test2019_auc_pr_{k}": v
             for k, v in report["headline_test_2019_auc_pr"].items()}
            | {f"lift_{k}": v for k, v in report["lift_over_base"].items()}
            | {f"context_effect_{k}": v for k, v in report["context_effect"].items()}
        )
        mlflow.log_artifact(str(metrics_path))
        for path in reports:
            mlflow.log_artifact(str(path), artifact_path="embedding_reports")

        for name, variant in report["variants"].items():
            arm, _, config = name.partition("_")
            with mlflow.start_run(run_name=f"fm-head-{name}", nested=True) as child:
                mlflow.set_tags({"phase": "3.5", "arm": arm if config else "none",
                                 "config": config or "base", "registered": "false"})
                mlflow.log_params({
                    "variant": name,
                    "n_features": variant["n_features"],
                    "scale_pos_weight": variant["scale_pos_weight"],
                    "best_iteration": variant["best_iteration"],
                    "threshold": variant["threshold"],
                    **({"pca_explained_variance":
                        round(variant["pca_explained_variance"], 4)}
                       if "pca_explained_variance" in variant else {}),
                })
                mlflow.log_metrics(
                    {"auc_pr": variant["splits"]["test_2019_only"]["auc_pr"],
                     "val_auc_pr": variant["splits"]["val"]["auc_pr"]}
                )
                for split, payload in variant["splits"].items():
                    mlflow.log_metrics(_flatten(f"{split}_", payload))
                mlflow.log_artifact(str(head_dir / f"{name}.json"), artifact_path="model")
                if config:
                    mlflow.log_artifact(str(head_dir / f"pca_{arm}.npz"),
                                        artifact_path="model")
                children[name] = child.info.run_id

    return {"parent_run_id": parent.info.run_id, "child_runs": children,
            "registered": False, "verdict": VERDICT}


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    paths = params["paths"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", default=paths["fm_metrics"])
    ap.add_argument("--models", default=paths["fm_head"])
    ap.add_argument("--output", default="reports/mlflow_fm.json")
    args = ap.parse_args(argv)

    result = log(
        repo_path(args.metrics), repo_path(args.models),
        [repo_path(paths["fm_embeddings_summary"]),
         repo_path(paths["fm_embeddings_isolated_summary"]),
         repo_path(paths["fm_sequences_summary"])],
        params,
    )
    out = repo_path(args.output)
    out.write_text(json.dumps(result, indent=2))
    print(f"logged parent run {result['parent_run_id']} with "
          f"{len(result['child_runs'])} nested runs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
