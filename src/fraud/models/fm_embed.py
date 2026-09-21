"""Stage `fm_embeddings`: per-transaction embeddings from the frozen decoder.

The model is a causal decoder, so position *i* attends only to positions <= i.
That is the whole reason this phase reads a hidden state **per transaction**
rather than pooling one vector per user: a single forward pass over a card's
sequence yields, at every transaction's last token, a representation that has
seen that transaction's past and nothing after it. Causal by construction, and
almost free -- the alternative, one static vector per entity, is the shape that
measured 0.1351 against a 0.2113 base for the GNN (decision 17).

Upstream's `HuggingFaceDecoderInference._pool_embeddings` pools once per
*sequence* (last non-pad token). We need one per *transaction*, so this reads
`last_hidden_state` at the positions `fm_sequences` recorded. Same tensor,
different indexing.

Padding is on the RIGHT, which is safe here precisely because attention is
causal: a real token at position i attends only to <= i, all of which are real.
An attention mask is passed anyway so the arithmetic is right regardless.

The model is never trained, only run: `.eval()` under `torch.no_grad()`, with
frozen weights. Nothing in this stage updates a parameter.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import polars as pl
import torch

from fraud.config import load_params, repo_path
from fraud.features.fm_tokenizer import EXPECTED_VOCAB_SIZE, PAD_ID

_DTYPES = {"float16": torch.float16, "float32": torch.float32,
           "bfloat16": torch.bfloat16}


def load_model(path: pathlib.Path, device: str, dtype: str = "float32"):
    """The frozen feature extractor.

    `AutoModel` rather than `AutoModelForCausalLM`: the LM head is dead weight
    when only hidden states are wanted, and `last_hidden_state` is exactly
    `hidden_states[-1]` -- both are taken after the final norm.
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(path, dtype=_DTYPES[dtype])
    if model.get_input_embeddings().weight.shape[0] != EXPECTED_VOCAB_SIZE:
        raise RuntimeError(
            f"checkpoint embedding matrix is "
            f"{model.get_input_embeddings().weight.shape[0]} rows, but the "
            f"tokenizer emits ids for {EXPECTED_VOCAB_SIZE}. The tokenizer and "
            f"the weights disagree; every embedding would be meaningless."
        )
    return model.to(device).eval()


def _pad_batch(rows: list[list[int]], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(r) for r in rows)
    ids = np.full((len(rows), width), PAD_ID, dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=np.int64)
    for i, row in enumerate(rows):
        ids[i, : len(row)] = row
        mask[i, : len(row)] = 1
    return (
        torch.from_numpy(ids).to(device),
        torch.from_numpy(mask).to(device),
    )


@torch.no_grad()
def embed_sequences(
    model,
    sequences: pl.DataFrame,
    index: pl.DataFrame,
    batch_size: int,
    device: str,
    progress_every: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """(txn_id, embedding) for every transaction the index points at."""
    wanted = (
        index.group_by("sequence_id")
        .agg(pl.col("txn_id"), pl.col("position"))
        .sort("sequence_id")
    )
    lookup = {
        row["sequence_id"]: (row["txn_id"], row["position"])
        for row in wanted.iter_rows(named=True)
    }

    seq_ids = sequences.get_column("sequence_id").to_list()
    tokens = sequences.get_column("tokens").to_list()

    txn_out: list[np.ndarray] = []
    emb_out: list[np.ndarray] = []
    started = time.perf_counter()

    for start in range(0, len(tokens), batch_size):
        chunk_ids = seq_ids[start : start + batch_size]
        chunk = tokens[start : start + batch_size]
        ids, mask = _pad_batch(chunk, device)
        hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state

        # Flat gather: one (row, position) pair per transaction in the batch.
        rows_idx, pos_idx, txns = [], [], []
        for b, sid in enumerate(chunk_ids):
            entry = lookup.get(sid)
            if entry is None:
                continue
            t_ids, positions = entry
            rows_idx.extend([b] * len(positions))
            pos_idx.extend(positions)
            txns.extend(t_ids)
        if not txns:
            continue

        picked = hidden[
            torch.tensor(rows_idx, device=device),
            torch.tensor(pos_idx, device=device),
        ]
        emb_out.append(picked.to(torch.float32).cpu().numpy().astype(np.float16))
        txn_out.append(np.asarray(txns, dtype=np.int64))

        done = start // batch_size + 1
        if progress_every and done % progress_every == 0:
            rate = (start + len(chunk)) / (time.perf_counter() - started)
            remaining = (len(tokens) - start - len(chunk)) / max(rate, 1e-9)
            print(
                f"  {start + len(chunk):>7,} / {len(tokens):,} sequences "
                f"({rate:,.1f}/s, {remaining / 60:.1f} min left)",
                flush=True,
            )

    return np.concatenate(txn_out), np.concatenate(emb_out)


def build(
    sequences_dir: pathlib.Path,
    model_path: pathlib.Path,
    out_dir: pathlib.Path,
    batch_size: int,
    device: str,
    dtype: str,
    limit: int | None = None,
) -> dict:
    started = time.perf_counter()
    sequences = pl.read_parquet(sequences_dir / "sequences.parquet")
    index = pl.read_parquet(sequences_dir / "index.parquet")
    if limit:
        sequences = sequences.head(limit)
        keep = set(sequences.get_column("sequence_id").to_list())
        index = index.filter(pl.col("sequence_id").is_in(list(keep)))

    model = load_model(model_path, device, dtype)
    txn_ids, embeddings = embed_sequences(
        model, sequences, index, batch_size, device
    )

    if len(txn_ids) != index.height:
        raise RuntimeError(
            f"embedded {len(txn_ids):,} transactions but the index asked for "
            f"{index.height:,}. A sequence was skipped."
        )
    if len(set(txn_ids.tolist())) != len(txn_ids):
        raise RuntimeError("a transaction was embedded more than once")

    # .npy, not parquet: polars has no Float16, so a list column would store
    # float32 and double this artifact to 14.8 GB. Raw .npy keeps float16 at
    # 7.4 GB and memmaps, which is what the downstream head wants anyway --
    # it needs a dense 2-D array, not 512 list cells per row.
    order = np.argsort(txn_ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "txn_ids.npy", txn_ids[order])
    np.save(out_dir / "embeddings.npy", embeddings[order])

    return {
        "transactions": int(len(txn_ids)),
        "sequences": int(sequences.height),
        "dim": int(embeddings.shape[1]),
        "stored_dtype": str(embeddings.dtype),
        "megabytes": round(embeddings.nbytes / 1e6, 1),
        "device": device,
        "dtype": dtype,
        "batch_size": batch_size,
        "seconds": round(time.perf_counter() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    cfg = params["fm"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequences", default=params["paths"]["fm_sequences"])
    ap.add_argument("--model", default=params["paths"]["fm_checkpoint"])
    ap.add_argument("--output", default=params["paths"]["fm_embeddings"])
    ap.add_argument("--summary", default=params["paths"]["fm_embeddings_summary"])
    ap.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # fp16 on CPU is slow and, on some kernels, unsupported -- so a CPU run
    # (tests, a smoke check) silently falls back to fp32 rather than crawling.
    dtype = args.dtype or (cfg["dtype"] if device == "cuda" else "float32")
    if device == "cpu":
        print("WARNING: no CUDA device; this is a smoke run, not the real pass",
              file=sys.stderr)

    summary = build(
        repo_path(args.sequences), repo_path(args.model), repo_path(args.output),
        args.batch_size, device, dtype, args.limit,
    )
    out = repo_path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(
        f"embedded {summary['transactions']:,} transactions "
        f"({summary['dim']}-d, {summary['dtype']}) in {summary['seconds']}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
