"""A transaction's embedding must not depend on anything that came after it.

This is the foundation model's equivalent of the causal replay, and it exists
for the same reason: leakage here would not raise. It would produce a slightly
better AUC-PR and a story about sequential structure that does not survive
serving -- exactly the 17% the GNN had to give back.

The argument is that a decoder's attention is causal, so the hidden state at a
transaction's last token has seen only that transaction's past. That is an
argument about the architecture. These tests check it against the actual
checkpoint, because the architecture being causal is not the same as OUR
indexing being right: reading one position too far, or pooling over the
sequence as upstream's own helper does, would quietly break it.

Run against the real weights rather than a random init -- a bug that cancels
out under random weights is exactly the kind that survives to production.
"""

from __future__ import annotations

import pathlib

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from fraud.features.fm_sequences import build_sequence, token_position  # noqa: E402
from fraud.features.fm_tokenizer import PAD_ID  # noqa: E402
from fraud.models.fm_embed import embed_sequences, load_model  # noqa: E402

CHECKPOINT = pathlib.Path("data/models/fm")
pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").exists(),
    reason="checkpoint absent -- run `dvc repro fetch_fm_checkpoint`",
)

RNG = np.random.default_rng(0)

# Embeddings are stored float16, and different sequence lengths or batch
# shapes change matmul dimensions, hence float accumulation order. So "the
# future did not leak" is NOT bit-equality -- measured, the residual is
# 2.4e-04, below float16's own epsilon (9.8e-04), while two DIFFERENT
# transactions differ by 4.3, some 17,000x larger. The assertion therefore
# pins the residual under one float16 ulp AND far under the signal scale;
# real leakage would move a value by the latter, not the former.
FP16_EPS = float(np.finfo(np.float16).eps)


def assert_same(a: np.ndarray, b: np.ndarray, ulps: int = 4) -> None:
    """Equal to within a few float16 ulps AT THE VALUES' MAGNITUDE.

    float16's epsilon is relative, so a single-ulp rounding difference on a
    value near 4 is four times larger than on a value near 1. A bare `< eps`
    bound rejects a legitimate one-ulp wobble, which is what it did here.
    Even at 4 ulps the bound stays ~250x below the between-transaction
    signal, so leakage cannot slip under it.
    """
    scale = max(float(np.abs(a).max()), 1.0)
    tolerance = ulps * FP16_EPS * scale
    diff = float(np.abs(a - b).max())
    assert diff <= tolerance, f"differs by {diff:.3e}, tolerance {tolerance:.3e}"


def assert_different(a: np.ndarray, b: np.ndarray) -> None:
    """Materially different, not merely unequal -- so a noise-level wobble
    cannot make a leakage test pass by accident."""
    diff = float(np.abs(a - b).max())
    assert diff > 100 * FP16_EPS, f"differs by only {diff:.3e}"


@pytest.fixture(scope="module")
def model():
    return load_model(CHECKPOINT, "cpu", "float32")


def transaction_tokens(n: int) -> list[list[int]]:
    """`n` transactions of 12 plausible in-range token ids."""
    return [RNG.integers(5, 6251, size=12).tolist() for _ in range(n)]


def embed(model, rows, wanted: list[int]) -> np.ndarray:
    """Embeddings for `wanted` transaction indices within one sequence."""
    sequences = pl.DataFrame(
        {"sequence_id": [0], "tokens": [build_sequence(rows)]},
        schema={"sequence_id": pl.Int32, "tokens": pl.List(pl.Int32)},
    )
    index = pl.DataFrame(
        {
            "txn_id": [int(i) for i in wanted],
            "sequence_id": [0] * len(wanted),
            "position": [token_position(i) for i in wanted],
        },
        schema={"txn_id": pl.Int64, "sequence_id": pl.Int32, "position": pl.Int32},
    )
    txn_ids, embeddings = embed_sequences(
        model, sequences, index, batch_size=4, device="cpu", progress_every=0
    )
    order = np.argsort(txn_ids)
    return embeddings[order].astype(np.float32)


# --- the invariant --------------------------------------------------------

def test_truncating_the_future_does_not_change_an_embedding(model):
    """The whole phase rests on this.

    Transaction k scored inside a 20-transaction sequence must equal
    transaction k scored in a sequence that stops at k. If these differ, the
    embedding has seen the future and every downstream number is inflated.
    """
    rows = transaction_tokens(20)
    k = 7
    full = embed(model, rows, [k])
    truncated = embed(model, rows[: k + 1], [k])
    assert_same(full, truncated)

    # ...and the residual must be tiny NEXT TO the signal, or the tolerance
    # above would be hiding a real leak rather than float noise.
    neighbours = embed(model, rows, [k, k + 1])
    residual = float(np.abs(full - truncated).max())
    signal = float(np.abs(neighbours[0] - neighbours[1]).max())
    assert signal > 1000 * residual, f"signal {signal:.3e} vs residual {residual:.3e}"


@pytest.mark.parametrize("k", [0, 1, 5, 12, 19])
def test_causality_holds_at_every_position(model, k):
    rows = transaction_tokens(20)
    assert_same(embed(model, rows, [k]), embed(model, rows[: k + 1], [k]))


def test_changing_a_later_transaction_leaves_earlier_ones_untouched(model):
    """The same invariant from the other side: perturb the future, not the past."""
    rows = transaction_tokens(12)
    before = embed(model, rows, [0, 3, 6])
    rows[9] = RNG.integers(5, 6251, size=12).tolist()
    after = embed(model, rows, [0, 3, 6])
    assert_same(before, after)


def test_changing_an_earlier_transaction_does_change_later_ones(model):
    """The converse, so the test above cannot pass by reading a constant."""
    rows = transaction_tokens(12)
    before = embed(model, rows, [9])
    rows[2] = RNG.integers(5, 6251, size=12).tolist()
    after = embed(model, rows, [9])
    assert_different(before, after)


# --- indexing and padding -------------------------------------------------

def test_embeddings_differ_between_positions(model):
    """Guards against reading the same position for every transaction."""
    rows = transaction_tokens(6)
    out = embed(model, rows, [0, 1, 2, 3, 4, 5])
    for i in range(1, len(out)):
        assert_different(out[0], out[i])


def test_right_padding_does_not_change_an_embedding(model):
    """Batching pads to the longest sequence; that must not move a result."""
    rows = transaction_tokens(4)
    alone = embed(model, rows, [3])

    long_rows = transaction_tokens(30)
    sequences = pl.DataFrame(
        {
            "sequence_id": [0, 1],
            "tokens": [build_sequence(rows), build_sequence(long_rows)],
        },
        schema={"sequence_id": pl.Int32, "tokens": pl.List(pl.Int32)},
    )
    index = pl.DataFrame(
        {"txn_id": [3], "sequence_id": [0], "position": [token_position(3)]},
        schema={"txn_id": pl.Int64, "sequence_id": pl.Int32, "position": pl.Int32},
    )
    _, batched = embed_sequences(
        model, sequences, index, batch_size=2, device="cpu", progress_every=0
    )
    assert_same(alone, batched.astype(np.float32))


def test_last_hidden_state_is_the_final_layer_after_norm(model):
    """Upstream reads hidden_states[-1]; we read last_hidden_state.

    They are the same tensor only because both are taken after the final norm.
    If that ever stops being true, the embeddings shift silently.
    """
    ids = torch.tensor([[1, 100, 200, 300, 2]])
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    assert torch.equal(out.last_hidden_state, out.hidden_states[-1])


def test_pad_id_is_the_one_the_checkpoint_expects(model):
    assert PAD_ID == 0
    assert model.config.pad_token_id == PAD_ID


def test_embedding_width_matches_the_hidden_size(model):
    out = embed(model, transaction_tokens(3), [0, 1, 2])
    assert out.shape == (3, model.config.hidden_size) == (3, 512)
