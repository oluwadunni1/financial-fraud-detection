"""Stage `fm_embeddings_isolated`: NVIDIA's own downstream embedding, replicated.

Upstream's notebook 04 does not embed a transaction in its history. It encodes
each row ALONE -- `<bos> T(12 tokens) <eos>`, padded to 128 -- and takes the
hidden state at `<eos>` (`HuggingFaceDecoderInference`, pooling="last_token").
The model was pretrained on 315-transaction sequences, but this extraction
never shows it one.

This stage reproduces that exactly, so the contextual arm in `fm_embed` can be
measured against it rather than argued about: same checkpoint, same token ids,
same head downstream. Whatever separates the two is what sequence context is
worth.

Tokens are taken from `fm_sequences`' output rather than re-tokenised, so both
arms see byte-identical transaction tokens -- the comparison cannot be
confounded by a second tokenisation pass. Inputs are 14 tokens wide, not 128:
attention is causal, so right padding past `<eos>` cannot reach it, and
`tests/test_fm_isolated.py` proves the two widths agree.
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
from fraud.features.fm_tokenizer import BOS_ID, EOS_ID, TOKENS_PER_TRANSACTION
from fraud.models.fm_embed import flat_tokens, load_model, summarise, write_embeddings

# <bos> + 12 + <eos>; the embedding is read at the last of them.
INPUT_WIDTH = TOKENS_PER_TRANSACTION + 2
EOS_POSITION = INPUT_WIDTH - 1


def isolated_inputs(
    sequences: pl.DataFrame, index: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """(txn_ids, [n, 14] int64 inputs) -- each transaction framed on its own.

    `index.position` points at a transaction's last token (CUST), so its 12
    tokens are the slice ending there.
    """
    flat, offsets = flat_tokens(sequences)
    seq_row = {sid: i for i, sid in enumerate(sequences.get_column("sequence_id").to_list())}
    rows = np.fromiter(
        (seq_row[s] for s in index.get_column("sequence_id").to_list()),
        dtype=np.int64, count=index.height,
    )
    last = offsets[rows] + index.get_column("position").to_numpy().astype(np.int64)
    gather = last[:, None] + np.arange(-TOKENS_PER_TRANSACTION + 1, 1)[None, :]

    inputs = np.empty((index.height, INPUT_WIDTH), dtype=np.int64)
    inputs[:, 0] = BOS_ID
    inputs[:, 1:-1] = flat[gather]
    inputs[:, -1] = EOS_ID
    return index.get_column("txn_id").to_numpy().astype(np.int64), inputs


def iter_isolated(
    model, txn_ids: np.ndarray, inputs: np.ndarray, batch_size: int, device: str,
    progress_every: int = 200,
):
    """Yield (txn_ids, float16 embeddings at <eos>) per batch.

    Every row is the same width with no padding, so no attention mask is
    needed and `EOS_POSITION` is the last non-pad token for all of them --
    upstream's last-token pooling, exactly.
    """
    n = len(txn_ids)
    started = time.perf_counter()
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        ids = torch.from_numpy(inputs[start:stop]).to(device)
        hidden = model(input_ids=ids).last_hidden_state[:, EOS_POSITION]
        yield txn_ids[start:stop], hidden.to(torch.float16).cpu().numpy()

        done = start // batch_size + 1
        if progress_every and done % progress_every == 0:
            rate = stop / (time.perf_counter() - started)
            print(
                f"  {stop:>9,} / {n:,} transactions ({rate:,.0f}/s, "
                f"{(n - stop) / max(rate, 1e-9) / 60:.1f} min left)",
                flush=True,
            )


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

    txn_ids, inputs = isolated_inputs(sequences, index)
    del sequences
    order = np.argsort(txn_ids)
    txn_ids, inputs = txn_ids[order], inputs[order]

    model = load_model(model_path, device, dtype)
    with torch.no_grad():
        write_embeddings(
            iter_isolated(model, txn_ids, inputs, batch_size, device),
            txn_ids, model.config.hidden_size, out_dir,
        )

    return {
        **summarise(out_dir),
        "pooling": "<eos> after the transaction encoded alone (upstream notebook 04)",
        "input_width": INPUT_WIDTH,
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
    ap.add_argument("--output", default=params["paths"]["fm_embeddings_isolated"])
    ap.add_argument("--summary",
                    default=params["paths"]["fm_embeddings_isolated_summary"])
    ap.add_argument("--batch-size", type=int, default=cfg["isolated_batch_size"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
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
        f"embedded {summary['transactions']:,} transactions in isolation "
        f"({summary['dim']}-d, {summary['dtype']}) in {summary['seconds']}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
