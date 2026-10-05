"""Register a retrained GraphSAGE arm as @challenger -- after its 2019 replay.

    python -m fraud.models.registry_gnn_causal --run no_short_velocity-pw3

The order is the point (decision 23): a version enters the registry only with the
served 2019 number it was evaluated on, and is tagged with the `evaluated_model`
that number came from, so the gate can find its scores and evidence
(`promotion.evaluated`, decision 32). @challenger moves to the new version, so the
API starts shadow-scoring it on every request; @champion and @previous do not
move -- promotion is the gate's decision, never this script's.

The model is logged through `Predictor.from_disk`, i.e. WITH its feature mask
(`FraudGNN.txn_mask`), and the logged copy is loaded back and checked to carry
that mask before the alias moves.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import mlflow
import torch
from dotenv import load_dotenv

from fraud.config import REPO_ROOT, load_params, repo_path

ALIAS = "challenger"


def register(params: dict, run: str, replay: str, ablation: str) -> dict:
    from fraud.api.predictor import Predictor

    load_dotenv(REPO_ROOT / ".env")
    model_dir = repo_path(params["gnn_causal"]["models_dir"]) / run
    info = json.loads((model_dir / "train_info.json").read_text())
    served = json.loads(repo_path(replay).read_text())
    off = json.loads(repo_path(ablation).read_text())
    if served["model_version"] != info["model_name"]:
        raise RuntimeError(f"{replay} was produced by {served['model_version']}, "
                           f"not {info['model_name']} -- refusing to register")
    name = info["model_name"]
    if name not in params["promotion"]["evaluated"]:
        raise RuntimeError(f"add promotion.evaluated[{name!r}] first: the gate needs it")

    model = Predictor.from_disk(params, model_dir).model.eval()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    registered = params["mlflow"]["registered_model_name"]
    with mlflow.start_run(run_name=f"graphsage-causal-{info['arm']}") as r:
        mlflow.log_params({"model": "graphsage-hetero-causal", "arm": info["arm"],
                           "pos_weight": info["pos_weight"], "best_epoch": info["best_epoch"],
                           "masked_features": json.dumps(info.get("masked_features", [])),
                           **{k: v for k, v in params["gnn"].items()
                              if k in ("hidden_channels", "embedding_dim", "num_layers",
                                       "dropout", "learning_rate", "batch_size")}})
        m = served["metrics"]
        mlflow.log_metrics({"auc_pr": m["auc_pr"], "precision_at_100": m["precision_at_100"],
                            "val_2018_auc_pr": info["val_auc_pr"],
                            "no_neighbours_auc_pr": off["metrics"]["auc_pr"]})
        mlflow.log_dict(info, "train_info.json")
        mlflow.log_dict(served, "metrics_replay_2019.json")
        mlflow.log_dict(off, "metrics_replay_2019_no_neighbours.json")
        mlflow.set_tags({"evaluated_model": name,
                         "promotion": "registered as @challenger; promotion is the gate's"})
        # pickle: forward takes dicts keyed by node/edge type, which MLflow 3's
        # default "pt2" tracing cannot express (as registry_gnn.py).
        mlflow.pytorch.log_model(model, name="model", registered_model_name=registered,
                                 serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE)
        run_id = r.info.run_id

    back = mlflow.pytorch.load_model(f"runs:/{run_id}/model", map_location="cpu")
    want = None if model.txn_mask is None else model.txn_mask
    got = getattr(back, "txn_mask", None)
    if (want is None) != (got is None) or (want is not None and not torch.equal(want, got)):
        raise RuntimeError("the logged model lost its feature mask -- alias NOT moved")

    client = mlflow.tracking.MlflowClient()
    version = next(v.version for v in client.search_model_versions(f"name='{registered}'")
                   if v.run_id == run_id)
    client.set_registered_model_alias(registered, ALIAS, version)
    return {"run_id": run_id, "version": str(version), "alias": ALIAS,
            "evaluated_model": name, "auc_pr_2019": m["auc_pr"],
            "no_neighbours_auc_pr_2019": off["metrics"]["auc_pr"],
            "champion_version": client.get_model_version_by_alias(registered, "champion").version}


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="e.g. no_short_velocity-pw3")
    ap.add_argument("--replay", default="reports/metrics_replay_gnn_causal.json")
    ap.add_argument("--ablation", default="reports/metrics_replay_gnn_causal_no_neighbours.json")
    ap.add_argument("--output", default="reports/registry_gnn_causal.json")
    args = ap.parse_args(argv)
    result = register(params, args.run, args.replay, args.ablation)
    out = repo_path(args.output)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
