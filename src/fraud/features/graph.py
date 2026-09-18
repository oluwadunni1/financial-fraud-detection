"""Stage `build_graph`: the tri-partite graph, ported from the NVIDIA blueprint.

`src/preprocess_TabFormer_np.py` is the reference (read-only; port its logic,
never import it). The shape it builds:

    User --> Transaction --> Merchant          no edge attributes
                  ^
             prediction target, carries the fraud label

The blueprint notes that a bipartite user-merchant graph with edge
classification would be the ideal model, but cuGraph could not do link
prediction, so it puts transactions in as nodes and does node classification.
We follow that.

Three node types, three different feature sets
----------------------------------------------
Easy to miss and important: `Merchant`, `Card` and `MCC` are *removed* from the
transaction features and become the user and merchant node features instead. The
graph, not the feature vector, carries merchant identity -- which is the whole
point of asking whether relational structure helps.

    transaction   the tabular features, minus Merchant and MCC
    user          binary code of its own card id      (identity)
    merchant      binary code of its own id, plus MCC (identity)

Per-split graphs with independent node ids
------------------------------------------
The reference writes `edges/` and `test_gnn/edges/` separately, and node ids
restart at 0 in each. That makes the setting *inductive*: message passing can
never carry a training transaction into a test prediction. Keep it.

Two deliberate deviations from the blueprint
--------------------------------------------
1. **Under-sampling is applied to the training graph only.** NVIDIA under-samples
   before splitting, so their val and test also sit at ~10% fraud instead of
   0.122%, which inflates every metric and makes them incomparable to the
   champion. Train like NVIDIA, evaluate on the real distribution.
2. **Identity codes are fitted on train only.** NVIDIA fits its id encoder on
   every card/merchant/MCC in the whole dataset, so test merchants always get a
   real code. We reuse the Phase 2 encoder, fitted on train, so a merchant first
   seen in 2019 gets the reserved all-zero code -- the same cold-start path the
   serving layer has to handle.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time

import polars as pl

from fraud.config import load_params, repo_path
from fraud.data.split import split_years
from fraud.features.encoders import Encoder, binary_code_exprs

LABEL = "Fraud"
SPLITS = ("train", "val", "test")

# Columns the blueprint moves out of the transaction and onto the id nodes.
NODE_ID_COLUMNS = ("Merchant", "MCC")

# The blueprint dedupes non-fraud rows on its *nominal* predictors before
# under-sampling, so near-identical legitimate rows do not dominate the sample.
# (User, Card) rather than the derived card_id: they are a bijection, and this
# lets the dedupe run in the lazy plan before card_id exists.
DEDUPE_SUBSET = (
    "User", "Card", "Merchant", "MCC", "City", "State", "Zip", "Errors", "Chip",
)

# Pulled from the processed data to build the graph. Beyond the ids and label,
# these exist only to form DEDUPE_SUBSET.
SOURCE_COLUMNS = (
    "txn_id", "User", "Card", "Merchant", "MCC",
    "City", "State", "Zip", "Errors", "Chip", LABEL,
)


def combined_card_id(n_card_values: int) -> pl.Expr:
    """`User * n_cards + Card`, the blueprint's card-level user id.

    A "user" node is really a *card*: the same person's two cards are two nodes,
    because fraud follows the card. 6,139 of these against 2,000 users.
    """
    return (pl.col("User") * n_card_values + pl.col("Card")).alias("card_id")


def select_training_ids(
    txns: pl.LazyFrame, fraud_ratio: float, seed: int
) -> list[int]:
    """Which txn_ids form the under-sampled training graph.

    The blueprint's recipe -- keep every fraud row, thin non-fraud until fraud is
    `fraud_ratio` -- and the reason the training graph is tractable: 20.6M
    transaction nodes become a few hundred thousand. Keeping *all* positives is
    what lets the GNN see every fraud XGBoost saw.

    Returns ids rather than rows, and does the heavy work in the lazy plan. An
    earlier version collected the whole training split (20.6M x 89, ~7.3 GB)
    before throwing 99% of it away, and was OOM-killed.
    """
    fraud_ids = (
        txns.filter(pl.col(LABEL) == 1)
        .select("txn_id")
        .collect(engine="streaming")
        .get_column("txn_id")
    )

    # Dedupe non-fraud on the nominal predictors, streaming, ids only.
    deduped = (
        txns.filter(pl.col(LABEL) == 0)
        .unique(subset=list(DEDUPE_SUBSET), keep="first")
        .select("txn_id")
        .collect(engine="streaming")
        .get_column("txn_id")
    )

    # Sort before sampling. `unique` under the streaming engine does not
    # guarantee row order, so sampling its raw output picks a different subset
    # on every run -- which would make `dvc repro` build a different training
    # graph each time while reporting the same config. Caught by
    # test_under_sampling_is_deterministic.
    deduped = deduped.sort()

    keep = min(len(deduped), int(len(fraud_ids) / fraud_ratio))
    sampled = deduped.sample(n=keep, seed=seed, shuffle=False)
    return sorted(pl.concat([fraud_ids, sampled]).to_list())


def build_split(
    transactions: pl.DataFrame,
    matrix: pl.DataFrame,
    encoder: Encoder,
    card_mapping: dict[str, int],
    n_card_values: int,
) -> dict[str, pl.DataFrame]:
    """Build one split's graph from rows already selected by the caller."""
    frame = transactions.with_columns(combined_card_id(n_card_values))

    # --- node ids, per split, starting at 0 ------------------------------
    # Never reuse txn_id here: that is the global Postgres key, and mixing the
    # two is exactly how a test transaction ends up in the training graph.
    frame = frame.sort("txn_id").with_row_index("transaction_id")

    users = (
        frame.select("card_id").unique().sort("card_id").with_row_index("user_id")
    )
    merchants = (
        frame.select("Merchant", "MCC")
        .unique(subset=["Merchant"], keep="first")
        .sort("Merchant")
        .with_row_index("merchant_id")
    )
    frame = frame.join(users, on="card_id", how="left").join(
        merchants.select("Merchant", "merchant_id"), on="Merchant", how="left"
    )

    # --- transaction node features ---------------------------------------
    # The Phase 2 matrix minus the columns that moved onto the id nodes.
    dropped = tuple(f"{c}_bin_" for c in NODE_ID_COLUMNS)
    txn_feature_cols = [
        c
        for c in matrix.columns
        if c not in ("txn_id", LABEL, "Year") and not c.startswith(dropped)
    ]
    transaction_nodes = (
        frame.select("transaction_id", "txn_id", LABEL)
        .join(matrix.select(["txn_id", *txn_feature_cols]), on="txn_id", how="left")
        .sort("transaction_id")
    )

    # --- id node features: identity codes --------------------------------
    user_nodes = users.select(
        "user_id",
        "card_id",
        *binary_code_exprs("card_id", card_mapping, prefix="card"),
    ).sort("user_id")

    merchant_nodes = merchants.select(
        "merchant_id",
        "Merchant",
        *binary_code_exprs("Merchant", encoder.binary["Merchant"]),
        *binary_code_exprs("MCC", encoder.binary["MCC"]),
    ).sort("merchant_id")

    # --- edges ------------------------------------------------------------
    edges_ut = frame.select(
        pl.col("user_id").alias("src"), pl.col("transaction_id").alias("dst")
    )
    edges_tm = frame.select(
        pl.col("transaction_id").alias("src"), pl.col("merchant_id").alias("dst")
    )

    return {
        "transaction_nodes": transaction_nodes,
        "user_nodes": user_nodes,
        "merchant_nodes": merchant_nodes,
        "edges_user_transaction": edges_ut,
        "edges_transaction_merchant": edges_tm,
    }


def build(
    processed: pathlib.Path,
    matrix_dir: pathlib.Path,
    encoder_path: pathlib.Path,
    out_dir: pathlib.Path,
    params: dict,
) -> dict:
    cfg = params["gnn"]
    seed = cfg["random_state"]
    encoder = Encoder.from_json(encoder_path)

    years = sorted(int(p.name.split("=")[1]) for p in processed.glob("Year=*"))
    if not years:
        raise FileNotFoundError(f"no Year=* partitions under {processed}")
    wanted = split_years(params, years)

    txns = pl.scan_parquet(processed / "**/*.parquet", hive_partitioning=True)
    # Global, so a card's node identity means the same thing in every split.
    n_card_values = (
        txns.select(pl.col("Card").n_unique()).collect(engine="streaming").item()
    )

    # Card identity codes fitted on TRAIN only (see module docstring).
    train_cards = (
        txns.filter(pl.col("Year").is_in(wanted["train"]))
        .select(combined_card_id(n_card_values))
        .unique()
        .collect(engine="streaming")
        .get_column("card_id")
        .sort()
        .to_list()
    )
    card_mapping = {str(c): i for i, c in enumerate(train_cards, start=1)}

    if out_dir.exists():
        shutil.rmtree(out_dir)
    started = time.perf_counter()
    summary: dict = {
        "n_card_values": int(n_card_values),
        "train_cards_encoded": len(card_mapping),
        "splits": {},
    }

    for split in SPLITS:
        split_years_ = wanted[split]
        split_txns = txns.filter(pl.col("Year").is_in(split_years_)).select(
            *SOURCE_COLUMNS
        )
        split_matrix = pl.scan_parquet(
            matrix_dir / "**/*.parquet", hive_partitioning=True
        ).filter(pl.col("Year").is_in(split_years_))

        # ONLY train is under-sampled. That is the deviation from the blueprint
        # that keeps val/test at the real 0.122% and the metrics comparable.
        if split == "train" and cfg["under_sample"]:
            keep_ids = select_training_ids(split_txns, cfg["fraud_ratio"], seed)
            selector = pl.col("txn_id").is_in(keep_ids)
            split_txns = split_txns.filter(selector)
            split_matrix = split_matrix.filter(selector)

        transactions = split_txns.collect(engine="streaming")
        matrix = split_matrix.collect(engine="streaming")

        frames = build_split(
            transactions, matrix, encoder, card_mapping, int(n_card_values)
        )

        target = out_dir / split
        target.mkdir(parents=True, exist_ok=True)
        for filename, frame in frames.items():
            frame.write_parquet(target / f"{filename}.parquet", compression="zstd")

        txn_nodes = frames["transaction_nodes"]
        frauds = int(txn_nodes.get_column(LABEL).sum())
        summary["splits"][split] = {
            "years": split_years_,
            "transaction_nodes": txn_nodes.height,
            "user_nodes": frames["user_nodes"].height,
            "merchant_nodes": frames["merchant_nodes"].height,
            "edges": frames["edges_user_transaction"].height
            + frames["edges_transaction_merchant"].height,
            "frauds": frauds,
            "fraud_rate": frauds / txn_nodes.height if txn_nodes.height else 0.0,
            "under_sampled": split == "train" and cfg["under_sample"],
            "txn_features": len(
                [c for c in txn_nodes.columns if c not in ("transaction_id", "txn_id", LABEL)]
            ),
            "user_features": len(
                [c for c in frames["user_nodes"].columns if c.startswith("card_bin_")]
            ),
            "merchant_features": len(
                [
                    c
                    for c in frames["merchant_nodes"].columns
                    if c.startswith(("Merchant_bin_", "MCC_bin_"))
                ]
            ),
        }
        print(
            f"  {split:<5} {txn_nodes.height:>9,} txn nodes | "
            f"{frames['user_nodes'].height:>6,} users | "
            f"{frames['merchant_nodes'].height:>7,} merchants | "
            f"{frauds:>6,} frauds ({summary['splits'][split]['fraud_rate']:.4f})",
            flush=True,
        )

    summary["seconds"] = round(time.perf_counter() - started, 1)
    return summary


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", default=params["paths"]["processed"])
    ap.add_argument("--matrix", default=params["paths"]["feature_matrix"])
    ap.add_argument("--encoder", default=params["paths"]["encoder"])
    ap.add_argument("--output", default=params["paths"]["graph"])
    ap.add_argument("--summary", default=params["paths"]["graph_summary"])
    args = ap.parse_args(argv)

    summary = build(
        repo_path(args.processed),
        repo_path(args.matrix),
        repo_path(args.encoder),
        repo_path(args.output),
        params,
    )

    out = repo_path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"graph built in {summary['seconds']}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
