"""Does GraphSAGE need ITS neighbourhood, or just a neighbourhood?

Decision 28 showed that removing the card and merchant neighbourhoods drops 2019
AUC-PR from 0.4665 to 0.0418. But an empty neighbourhood is something the model
never saw in training, so that proves dependence, not mechanism. The shuffled
control (`replay --shuffle-neighbours`) gives every transaction a realistic
neighbourhood -- another, activity-matched entity's real and strictly earlier
history -- with its own velocity and identity, and breaks only whose it is.

This job puts every 2019 causal replay side by side on the same transactions:

    served       the champion as it runs              reports/metrics_replay.npz
    graph off    no neighbours                        metrics_replay_no_neighbours.npz
    shuffled     card / merchant / both borrowed      metrics_replay_shuffled_*.npz

and reports, for each control, the paired-bootstrap AUC-PR difference against the
served scores and the share of the graph-off loss it reproduces:

    share = (served - control) / (served - graph off)

~1 means another entity's history is as useless as none: the model reads the
relationships. ~0 means any plausible history will do: the ablation measured
distribution shift.

    python -m fraud.jobs.neighbour_controls
"""

from __future__ import annotations

import json
import sys

import numpy as np

from fraud.config import load_params, repo_path
from fraud.jobs.staleness import paired_bootstrap
from fraud.models.metrics import auc_pr, precision_at_k

ARMS = {
    "served": "reports/metrics_replay.npz",
    "graph_off": "reports/metrics_replay_no_neighbours.npz",
    "shuffled_card": "reports/metrics_replay_shuffled_card.npz",
    "shuffled_merchant": "reports/metrics_replay_shuffled_merchant.npz",
    "shuffled_both": "reports/metrics_replay_shuffled_both.npz",
}


def load_arms(paths: dict[str, str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Scores per arm, aligned on txn_id and checked to share one label vector."""
    arms: dict[str, np.ndarray] = {}
    reference = labels = None
    for name, rel in paths.items():
        path = repo_path(rel)
        if not path.exists():
            print(f"  (skipping {name}: {rel} not found)")
            continue
        z = np.load(path)
        order = np.argsort(z["txn_id"], kind="stable")
        ids, score, label = z["txn_id"][order], z["score"][order], z["label"][order]
        if reference is None:
            reference, labels = ids, label
        elif not (np.array_equal(ids, reference) and np.array_equal(label, labels)):
            raise ValueError(f"{name} does not cover the same transactions as {next(iter(arms))}")
        arms[name] = score
    if "served" not in arms:
        raise FileNotFoundError("the served replay is the baseline and is missing")
    return labels, arms


def compare(labels: np.ndarray, arms: dict[str, np.ndarray], bootstrap: int,
            top_n: int, seed: int = 0) -> dict:
    served = arms["served"]
    full_loss = (auc_pr(labels, served) - auc_pr(labels, arms["graph_off"])
                 if "graph_off" in arms else None)
    out = {}
    for name, score in arms.items():
        top = np.argsort(-score, kind="stable")[:top_n]
        row = {
            "auc_pr": auc_pr(labels, score),
            "precision_at_100": precision_at_k(labels, score, 100),
            f"frauds_in_top_{top_n}": int(labels[top].sum()),
        }
        if name != "served":
            row["vs_served"] = paired_bootstrap(labels, served, score, bootstrap, seed)
            if full_loss:
                row["share_of_graph_off_loss"] = -row["vs_served"]["delta"] / full_loss
        out[name] = row
    return out


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    cfg = params["neighbour_controls"]
    labels, arms = load_arms(ARMS)
    print(f"{labels.size:,} transactions of 2019, {int(labels.sum()):,} frauds; "
          f"{cfg['bootstrap']} paired resamples")
    result = compare(labels, arms, cfg["bootstrap"], cfg["top_n"], cfg["shuffle_seed"])
    report = {"rows": int(labels.size), "positives": int(labels.sum()),
              "bootstrap": cfg["bootstrap"], "arms": result}
    out = repo_path(cfg["output"])
    out.write_text(json.dumps(report, indent=2))

    top = f"frauds_in_top_{cfg['top_n']}"
    print(f"\n{'arm':<18} {'AUC-PR':>7} {'P@100':>6} {top:>18}   "
          "Δ vs served [95% CI]      share of graph-off loss")
    for name, r in result.items():
        line = f"{name:<18} {r['auc_pr']:7.4f} {r['precision_at_100']:6.2f} {r[top]:18,}"
        if "vs_served" in r:
            b = r["vs_served"]
            line += f"   {b['delta']:+.4f} [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]"
            if "share_of_graph_off_loss" in r:
                line += f"   {r['share_of_graph_off_loss']:.0%}"
        print(line)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
