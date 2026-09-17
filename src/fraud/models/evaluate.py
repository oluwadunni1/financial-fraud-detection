"""Stage `evaluate`: score the champion, three ways.

The operating threshold is chosen on **validation** at the configured false
positive rate, then applied unchanged to test. Picking it on test would tune the
model to the thing being reported.

Test is reported twice, per the 2020 decision:

- `test_2019_only` -- the **headline AUC-PR**. 2019 is the only test year with
  positives, so this is the clean model-quality number to compare the GNN and
  the foundation model against later.
- `test_full` (2019 + 2020) -- operating-point metrics. 2020 contributes 336,500
  rows and zero fraud, which drops prevalence from 0.121% to 0.101% and makes
  precision look worse for reasons unrelated to the model. That is the honest
  alert-volume picture, so it is where precision@k and the confusion matrix come
  from.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import xgboost as xgb

from fraud.config import load_params, repo_path
from fraud.data.split import split_years
from fraud.models.metrics import evaluate_scores, threshold_at_fpr
from fraud.models.xgb import feature_names, load_split_matrix, matrix_files


def evaluate(matrix: pathlib.Path, model_dir: pathlib.Path, params: dict) -> dict:
    cfg = params["evaluate"]
    ks = list(cfg["precision_at_k"])
    target_fpr = float(cfg["target_false_positive_rate"])

    booster = xgb.Booster()
    booster.load_model(model_dir / "model.json")
    info = json.loads((model_dir / "train_info.json").read_text())
    best = info["best_iteration"]

    years = sorted(int(p.name.split("=")[1]) for p in matrix.glob("Year=*"))
    splits = split_years(params, years)
    names = feature_names(matrix_files(matrix, splits["train"])[0])
    if names != info["feature_names"]:
        raise ValueError(
            "feature names differ from training. The model and the matrix are "
            "out of step -- rerun the features and train_xgb stages."
        )

    def scores_for(split_years_: list[int]):
        x, y = load_split_matrix(matrix, split_years_, names)
        d = xgb.DMatrix(x, feature_names=names)
        return y, booster.predict(d, iteration_range=(0, best + 1))

    # Threshold comes from val and is then frozen.
    y_val, s_val = scores_for(splits["val"])
    threshold = threshold_at_fpr(y_val, s_val, target_fpr)

    test_years = splits["test"]
    headline_years = [min(test_years)]

    report = {
        "model": "xgboost",
        "best_iteration": best,
        "scale_pos_weight": info["scale_pos_weight"],
        "n_features": info["n_features"],
        "target_false_positive_rate": target_fpr,
        "threshold_chosen_on": "val",
        "threshold": float(threshold),
        "splits": {},
    }

    report["splits"]["val"] = {
        "years": splits["val"],
        **evaluate_scores(y_val, s_val, ks, target_fpr, threshold=threshold),
    }

    y, s = scores_for(headline_years)
    report["splits"]["test_2019_only"] = {
        "years": headline_years,
        "role": "headline AUC-PR -- clean model-quality comparison",
        **evaluate_scores(y, s, ks, target_fpr, threshold=threshold),
    }

    y, s = scores_for(test_years)
    report["splits"]["test_full"] = {
        "years": test_years,
        "role": "operating point -- precision@k, alert volume, recall at fixed FPR",
        **evaluate_scores(y, s, ks, target_fpr, threshold=threshold),
    }

    headline = report["splits"]["test_2019_only"]
    # A model at chance scores at the base rate; anything at or below it has
    # learned nothing, whatever the other numbers say.
    report["auc_pr_lift_over_base_rate"] = (
        headline["auc_pr"] / headline["base_rate"] if headline["base_rate"] else None
    )
    return report


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default=params["paths"]["feature_matrix"])
    ap.add_argument("--model", default=params["paths"]["models"])
    ap.add_argument("--report", default=params["paths"]["metrics_report"])
    args = ap.parse_args(argv)

    report = evaluate(
        repo_path(args.matrix), repo_path(args.model) / "xgb", params
    )

    out = repo_path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"threshold {report['threshold']:.6f} (chosen on val @ "
          f"{report['target_false_positive_rate']:.0%} FPR)\n")
    recall_key = f"recall_at_fpr_{report['target_false_positive_rate']}"
    for name, split in report["splits"].items():
        print(
            f"{name:<15} AUC-PR {split['auc_pr']:.4f} "
            f"| base {split['base_rate']:.5f} "
            f"| P@100 {split.get('precision_at_100', float('nan')):.3f} "
            f"| recall@thr {split[recall_key]:.3f}"
        )
    print(f"\nheadline AUC-PR lift over base rate: "
          f"{report['auc_pr_lift_over_base_rate']:.1f}x -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
