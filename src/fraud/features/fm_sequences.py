"""Stage `fm_sequences`: transactions -> the token streams the model was trained on.

Sequence shape, taken from upstream's `clm_data.py` and notebook 02:

    <bos> T1(12 tokens) <sep> T2 <sep> ... Tn <eos>

so a transaction costs 13 tokens once its separator is counted, and a
4096-token window holds 315 of them. **4096, not 8192**: `config.json` allows
8192 positions but `configs/pretrain_financial_decoder.yaml` trained at 4096,
and RoPE extrapolation past the trained length is exactly the kind of quiet
degradation this project keeps finding. Stay in distribution.

Sequences are per `(User, Card)` in time order, matching upstream's
`sort_values(["user", "card", "time_full"])`. A card is the right grain: fraud
follows the card, and it is the same grain the GNN's user node uses.

Windows overlap on purpose
--------------------------
The model is causal, so a transaction's embedding sees only what precedes it
*within its window*. Chopping a card's history into abutting 315-transaction
blocks would leave the first transaction of every block with an empty context
and no way to tell -- the same shape of bug as velocity windows computed per
year partition, which silently zeroed at each boundary.

So windows advance by half a window, and a transaction is scored from the
window where it has the most preceding context. Only the first window emits
its whole span; later ones emit just their second half.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import polars as pl

from fraud.config import load_params, repo_path
from fraud.features.fm_tokenizer import (
    BOS_ID,
    EOS_ID,
    SEP_ID,
    TOKEN_COLUMNS,
    TOKENS_PER_TRANSACTION,
    tokenize,
)

# 1 (<bos>) + n*12 + (n-1) (<sep>) + 1 (<eos>) <= seq_length
TOKENS_PER_SLOT = TOKENS_PER_TRANSACTION + 1  # transaction plus its separator


def transactions_per_window(seq_length: int) -> int:
    """How many transactions fit, given the bos/sep/eos overhead."""
    return (seq_length - 2 + 1) // TOKENS_PER_SLOT


def token_position(index_in_window: int) -> int:
    """Where a transaction's LAST token (CUST) sits inside its sequence.

    <bos> is position 0, so transaction k spans [1 + 13k, 12 + 13k] and its
    final token -- the one whose hidden state becomes the embedding -- is at
    12 + 13k.
    """
    return TOKENS_PER_TRANSACTION + TOKENS_PER_SLOT * index_in_window


def build_sequence(token_rows: list[list[int]]) -> list[int]:
    out = [BOS_ID]
    for i, row in enumerate(token_rows):
        if i:
            out.append(SEP_ID)
        out.extend(row)
    out.append(EOS_ID)
    return out


def windows(n: int, per_window: int) -> list[tuple[int, int, int]]:
    """(start, stop, emit_from) spans over a card's `n` transactions.

    `emit_from` is where this window starts contributing embeddings: the whole
    span for the first window, the second half afterwards, so every emitted
    transaction carries at least half a window of context.
    """
    if n <= per_window:
        return [(0, n, 0)]
    stride = max(1, per_window // 2)
    spans, start, emitted = [], 0, 0
    while emitted < n:
        stop = min(start + per_window, n)
        # Emission resumes exactly where the previous window stopped, so the
        # windows PARTITION the transactions. Deriving it from the stride
        # instead double-emits the boundary transaction.
        spans.append((start, stop, emitted))
        emitted = stop
        if stop >= n:
            break
        start += stride
    return spans


def build(
    processed: pathlib.Path,
    out_dir: pathlib.Path,
    years: list[int],
    seq_length: int,
) -> dict:
    started = time.perf_counter()
    per_window = transactions_per_window(seq_length)

    frame = (
        pl.scan_parquet(processed / "**/*.parquet", hive_partitioning=True)
        .filter(pl.col("Year").is_in(years))
        .collect(engine="streaming")
        .sort(["User", "Card", "ts", "txn_id"])
    )
    tokens = tokenize(frame)
    # Tokenise once, then carry the 12 ids per row as a list column so the
    # windowing below is pure indexing.
    rows = tokens.select(
        "txn_id", pl.concat_list(TOKEN_COLUMNS).alias("tok")
    ).with_columns(frame.select("User", "Card"))

    sequences, index, seq_id = [], [], 0
    for (user, card), group in rows.group_by(["User", "Card"], maintain_order=True):
        tok = group.get_column("tok").to_list()
        ids = group.get_column("txn_id").to_list()
        for start, stop, emit_from in windows(len(tok), per_window):
            sequences.append(
                {
                    "sequence_id": seq_id,
                    "user": int(user),
                    "card": int(card),
                    "tokens": build_sequence(tok[start:stop]),
                }
            )
            for offset in range(emit_from - start, stop - start):
                index.append(
                    {
                        "txn_id": ids[start + offset],
                        "sequence_id": seq_id,
                        "position": token_position(offset),
                    }
                )
            seq_id += 1

    seq_frame = pl.DataFrame(
        sequences,
        schema={"sequence_id": pl.Int32, "user": pl.Int32, "card": pl.Int8,
                "tokens": pl.List(pl.Int16)},
    )
    index_frame = pl.DataFrame(
        index,
        schema={"txn_id": pl.Int64, "sequence_id": pl.Int32, "position": pl.Int32},
    )

    # Every transaction must be scored exactly once, or the join to the feature
    # matrix silently drops or duplicates rows.
    if index_frame.height != frame.height:
        raise RuntimeError(
            f"index covers {index_frame.height:,} transactions but the split has "
            f"{frame.height:,}. Every transaction must appear exactly once."
        )
    if index_frame.get_column("txn_id").n_unique() != frame.height:
        raise RuntimeError("a transaction appears more than once in the index")

    out_dir.mkdir(parents=True, exist_ok=True)
    seq_frame.write_parquet(out_dir / "sequences.parquet", compression="zstd")
    index_frame.write_parquet(out_dir / "index.parquet", compression="zstd")

    lengths = seq_frame.get_column("tokens").list.len()
    return {
        "transactions": frame.height,
        "sequences": seq_frame.height,
        "seq_length": seq_length,
        "transactions_per_window": per_window,
        "tokens_total": int(lengths.sum()),
        "tokens_max": int(lengths.max()),
        "cards": rows.select(["User", "Card"]).n_unique(),
        "seconds": round(time.perf_counter() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", default=params["paths"]["processed"])
    ap.add_argument("--output", default=params["paths"]["fm_sequences"])
    ap.add_argument("--summary", default=params["paths"]["fm_sequences_summary"])
    args = ap.parse_args(argv)

    cfg = params["fm"]
    summary = build(
        repo_path(args.processed),
        repo_path(args.output),
        list(range(cfg["min_year"], 2021)),
        cfg["seq_length"],
    )

    import json

    out = repo_path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(
        f"{summary['transactions']:,} transactions -> {summary['sequences']:,} "
        f"sequences ({summary['transactions_per_window']}/window, "
        f"{summary['tokens_total']:,} tokens) in {summary['seconds']}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
