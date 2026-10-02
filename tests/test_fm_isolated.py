"""The isolated arm must be NVIDIA's extraction exactly, and the streaming
writer must land every row where its txn_id says.

The replica is only useful as a control if it is faithful: a "NVIDIA baseline"
that differs from upstream in some unexamined way would make the
contextual-vs-isolated comparison measure that difference instead.
"""

from __future__ import annotations

import pathlib

import numpy as np
import polars as pl
import pytest

from fraud.features.fm_sequences import build_sequence, token_position
from fraud.features.fm_tokenizer import BOS_ID, EOS_ID, PAD_ID
from fraud.models.fm_embed import write_embeddings
from fraud.models.fm_embed_isolated import EOS_POSITION, INPUT_WIDTH, isolated_inputs

RNG = np.random.default_rng(1)


def transaction_tokens(n: int) -> list[list[int]]:
    return [RNG.integers(5, 6251, size=12).tolist() for _ in range(n)]


def frames(seqs: list[list[list[int]]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """sequences + index frames over several cards' transaction rows."""
    sequences = pl.DataFrame(
        {"sequence_id": list(range(len(seqs))),
         "tokens": [build_sequence(rows) for rows in seqs]},
        schema={"sequence_id": pl.Int32, "tokens": pl.List(pl.Int16)},
    )
    txn, sid, pos = [], [], []
    for s, rows in enumerate(seqs):
        for k in range(len(rows)):
            txn.append(1000 * s + k)
            sid.append(s)
            pos.append(token_position(k))
    index = pl.DataFrame(
        {"txn_id": txn, "sequence_id": sid, "position": pos},
        schema={"txn_id": pl.Int64, "sequence_id": pl.Int32, "position": pl.Int32},
    )
    return sequences, index


# --- input construction (no model needed) ---------------------------------

def test_each_transaction_is_framed_alone_as_upstream_does():
    """`<bos> T <eos>` -- notebook 04's `encode` row, minus the padding."""
    seqs = [transaction_tokens(5), transaction_tokens(3)]
    sequences, index = frames(seqs)
    txn_ids, inputs = isolated_inputs(sequences, index)
    assert inputs.shape == (8, INPUT_WIDTH)
    for t, row in zip(txn_ids, inputs, strict=True):
        s, k = divmod(int(t), 1000)
        assert row.tolist() == [BOS_ID, *seqs[s][k], EOS_ID]


def test_isolated_tokens_are_the_contextual_arms_tokens():
    """Both arms must read byte-identical transaction tokens, so the
    comparison cannot be confounded by a second tokenisation."""
    seqs = [transaction_tokens(20)]
    sequences, index = frames(seqs)
    _, inputs = isolated_inputs(sequences, index)
    flat = sequences.get_column("tokens").to_list()[0]
    for k, row in enumerate(inputs):
        end = token_position(k)
        assert row[1:-1].tolist() == flat[end - 11 : end + 1]


def test_eos_position_is_the_last_token():
    assert EOS_POSITION == INPUT_WIDTH - 1 == 13


# --- the streaming writer -------------------------------------------------

def test_writer_aligns_rows_to_sorted_txn_ids(tmp_path: pathlib.Path):
    ids = np.array([3, 10, 42, 99], dtype=np.int64)
    emb = {t: np.full(4, t, dtype=np.float16) for t in ids.tolist()}
    batches = [(np.array([42, 3]), np.stack([emb[42], emb[3]])),
               (np.array([99, 10]), np.stack([emb[99], emb[10]]))]
    assert write_embeddings(iter(batches), ids, 4, tmp_path) == 4
    out = np.load(tmp_path / "embeddings.npy")
    assert np.array_equal(np.load(tmp_path / "txn_ids.npy"), ids)
    assert out[:, 0].tolist() == [3, 10, 42, 99]


def test_writer_rejects_a_duplicate(tmp_path: pathlib.Path):
    ids = np.array([1, 2], dtype=np.int64)
    one = np.zeros((1, 2), np.float16)
    with pytest.raises(RuntimeError, match="more than once"):
        write_embeddings(iter([(np.array([1]), one), (np.array([1]), one)]),
                         ids, 2, tmp_path)


def test_writer_rejects_a_missing_row(tmp_path: pathlib.Path):
    ids = np.array([1, 2], dtype=np.int64)
    with pytest.raises(RuntimeError, match="skipped"):
        write_embeddings(iter([(np.array([1]), np.zeros((1, 2), np.float16))]),
                         ids, 2, tmp_path)


def test_writer_rejects_an_unlisted_txn(tmp_path: pathlib.Path):
    ids = np.array([1, 2], dtype=np.int64)
    with pytest.raises(RuntimeError, match="does not list"):
        write_embeddings(iter([(np.array([7]), np.zeros((1, 2), np.float16))]),
                         ids, 2, tmp_path)


# --- against the real weights ---------------------------------------------

CHECKPOINT = pathlib.Path("data/models/fm")
needs_model = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").exists(),
    reason="checkpoint absent -- run `dvc repro fetch_fm_checkpoint`",
)


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("transformers")
    from fraud.models.fm_embed import load_model
    return load_model(CHECKPOINT, "cpu", "float32")


def _isolated(model, seqs):
    torch = pytest.importorskip("torch")
    from fraud.models.fm_embed_isolated import iter_isolated
    sequences, index = frames(seqs)
    txn_ids, inputs = isolated_inputs(sequences, index)
    with torch.no_grad():
        parts = list(iter_isolated(model, txn_ids, inputs, 64, "cpu", 0))
    return (np.concatenate([t for t, _ in parts]),
            np.concatenate([e for _, e in parts]).astype(np.float32))


def _close(a, b, ulps=4):
    eps = float(np.finfo(np.float16).eps)
    tol = ulps * eps * max(float(np.abs(a).max()), 1.0)
    assert float(np.abs(a - b).max()) <= tol


@needs_model
def test_14_wide_equals_upstreams_128_wide_padded(model):
    """Upstream pads every row to 128 and masks; we skip the padding. Causal
    attention means the pads cannot reach <eos> -- proven, not assumed."""
    torch = pytest.importorskip("torch")
    seqs = [transaction_tokens(6)]
    _, ours = _isolated(model, seqs)

    sequences, index = frames(seqs)
    _, inputs = isolated_inputs(sequences, index)
    padded = np.full((len(inputs), 128), PAD_ID, dtype=np.int64)
    padded[:, :INPUT_WIDTH] = inputs
    ids = torch.from_numpy(padded)
    mask = (ids != PAD_ID).long()
    with torch.no_grad():
        hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
    # upstream's _pool_embeddings: last non-pad position per row
    last = mask.sum(dim=1) - 1
    theirs = hidden[torch.arange(len(ids)), last].to(torch.float16).float().numpy()
    _close(ours, theirs)


@needs_model
def test_isolated_ignores_history(model):
    """The defining property of upstream's extraction: the same transaction
    gets the same embedding whatever came before it."""
    shared = transaction_tokens(1)[0]
    a = [*transaction_tokens(4), shared]
    b = [*transaction_tokens(9), shared]
    ids, emb = _isolated(model, [a, b])
    by_id = dict(zip(ids.tolist(), emb, strict=True))
    _close(by_id[4], by_id[1000 + 9])


@needs_model
def test_the_two_arms_genuinely_differ(model):
    """With history, the contextual arm must NOT collapse to the isolated one
    -- otherwise the comparison is between two copies of one thing."""
    from fraud.models.fm_embed import embed_sequences
    seqs = [transaction_tokens(10)]
    sequences, index = frames(seqs)
    _, iso = _isolated(model, seqs)
    ids, ctx = embed_sequences(model, sequences, index, 4, "cpu", 0)
    ctx = ctx[np.argsort(ids)].astype(np.float32)
    assert float(np.abs(iso[9] - ctx[9]).max()) > 100 * float(np.finfo(np.float16).eps)


# --- the downstream head's plumbing ---------------------------------------

def test_store_maps_txn_ids_to_rows(tmp_path: pathlib.Path):
    from fraud.models.fm_head import EmbeddingStore
    ids = np.array([5, 9, 20], dtype=np.int64)
    np.save(tmp_path / "txn_ids.npy", ids)
    np.save(tmp_path / "embeddings.npy", np.arange(6, dtype=np.float16).reshape(3, 2))
    store = EmbeddingStore(tmp_path)
    assert store.rows_for(np.array([20, 5])).tolist() == [2, 0]


def test_store_refuses_a_row_without_an_embedding(tmp_path: pathlib.Path):
    """A silent nearest-neighbour match would pair a transaction with another
    transaction's embedding -- the searchsorted result must be checked."""
    from fraud.models.fm_head import EmbeddingStore
    np.save(tmp_path / "txn_ids.npy", np.array([5, 9], dtype=np.int64))
    np.save(tmp_path / "embeddings.npy", np.zeros((2, 2), np.float16))
    with pytest.raises(RuntimeError, match="no embedding"):
        EmbeddingStore(tmp_path).rows_for(np.array([7]))
