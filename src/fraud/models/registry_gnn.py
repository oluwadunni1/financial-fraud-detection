"""Stage `register_gnn`: put the GraphSAGE challenger in the registry.

Registered as `challenger`, not `champion`, despite winning on merit by 2x. The
`champion` alias drives the serving API, and the API cannot run this model until
Phase 4 switches to end-to-end graph scoring. Promoting it would point the alias
at something undeployable, which is exactly the failure the registry exists to
prevent.

Both evaluations are logged: the end-to-end score (the real result) and the
embeddings-as-features ablation (the reason promotion is blocked), so the
registry carries the full picture rather than just the flattering number.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import mlflow
import torch
from dotenv import load_dotenv
from torch_geometric.loader import NeighborLoader

from fraud.config import REPO_ROOT, load_params, repo_path
from fraud.models.gnn import TXN, FraudGNN, load_graph

ALIAS = "challenger"


def _flatten(prefix: str, payload: dict) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            flat |= _flatten(f"{prefix}{key}_", value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[f"{prefix}{key}"] = float(value)
    return flat


def register(
    graph_dir: pathlib.Path,
    model_dir: pathlib.Path,
    direct_metrics: pathlib.Path,
    ablation_metrics: pathlib.Path,
    params: dict,
) -> dict:
    load_dotenv(REPO_ROOT / ".env")
    for key in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME",
                "MLFLOW_TRACKING_PASSWORD"):
        if not os.getenv(key) or "<" in os.getenv(key, ""):
            raise RuntimeError(f"{key} missing or placeholder in .env")

    cfg = params["gnn"]
    info = json.loads((model_dir / "train_info.json").read_text())
    direct = json.loads(direct_metrics.read_text())
    ablation = json.loads(ablation_metrics.read_text())

    # Rebuild the architecture and materialise the lazy layers before loading.
    data = load_graph(graph_dir / "train")
    model = FraudGNN(data.metadata(), cfg)
    with torch.no_grad():
        batch = next(
            iter(
                NeighborLoader(
                    data,
                    num_neighbors=cfg["num_neighbors"],
                    input_nodes=TXN,
                    batch_size=16,
                    shuffle=False,
                )
            )
        )
        model(batch.x_dict, batch.edge_index_dict)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    name = params["mlflow"]["registered_model_name"]

    with mlflow.start_run(run_name="graphsage-challenger") as run:
        mlflow.log_params(
            {
                "model": "graphsage-hetero",
                **{k: v for k, v in cfg.items()},
                "best_epoch": info["best_epoch"],
                "train_nodes": info["train_nodes"],
                "feature_widths": json.dumps(info["feature_widths"]),
                "edge_types": json.dumps(info["edge_types"]),
            }
        )
        headline = direct["test_2019_only"]
        mlflow.log_metrics(
            {
                "auc_pr": headline["auc_pr"],
                "val_auc_pr": direct["val"]["auc_pr"],
                # The ablation, logged so the registry shows WHY this is a
                # challenger and not the champion.
                "ablation_base_auc_pr": ablation["headline"]["base_auc_pr"],
                "ablation_with_embeddings_auc_pr": ablation["headline"][
                    "base_plus_embeddings_auc_pr"
                ],
            }
        )
        for split, payload in direct.items():
            if isinstance(payload, dict):
                mlflow.log_metrics(_flatten(f"direct_{split}_", payload))

        mlflow.log_dict(direct, "metrics_gnn_direct.json")
        mlflow.log_dict(ablation, "metrics_gnn_ablation.json")
        mlflow.set_tag(
            "promotion_blocked",
            "Beats champion 2x end to end, but embeddings-as-features scores "
            "0.1351 vs a 0.2113 base -- ARCHITECTURE 3.1's serving design cannot "
            "deliver it. Phase 4 must serve the graph end to end first.",
        )
        # pickle, not MLflow 3's default "pt2": that format traces the graph by
        # executing forward with a tensor example, and this model's forward takes
        # dicts keyed by node and edge type, which cannot be expressed that way.
        mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name=name,
            serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE,
        )
        run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient()
    mine = [
        v for v in client.search_model_versions(f"name='{name}'") if v.run_id == run_id
    ]
    if not mine:
        raise RuntimeError(f"no registered version for run {run_id}")
    version = mine[0].version
    client.set_registered_model_alias(name, ALIAS, version)

    champion = client.get_model_version_by_alias(name, "champion")
    return {
        "run_id": run_id,
        "registered_model": name,
        "version": str(version),
        "alias": ALIAS,
        "model_uri": f"models:/{name}@{ALIAS}",
        "challenger_auc_pr": headline["auc_pr"],
        "champion_version": str(champion.version),
        "promotion": "deferred -- serving path cannot run this model yet",
    }


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default=params["paths"]["graph"])
    ap.add_argument("--model", default=params["paths"]["models"])
    ap.add_argument("--direct", default="reports/metrics_gnn_direct.json")
    ap.add_argument("--ablation", default=params["paths"]["metrics_gnn"])
    ap.add_argument("--output", default="reports/registry_gnn.json")
    args = ap.parse_args(argv)

    result = register(
        repo_path(args.graph),
        repo_path(args.model) / "gnn",
        repo_path(args.direct),
        repo_path(args.ablation),
        params,
    )
    out = repo_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(
        f"registered {result['registered_model']} v{result['version']} as "
        f"@{result['alias']} | AUC-PR {result['challenger_auc_pr']:.4f} "
        f"(champion is v{result['champion_version']})"
    )
    print(f"promotion: {result['promotion']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
