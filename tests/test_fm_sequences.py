"""Windowing must not lose, duplicate, or starve a transaction.

Three failure modes, none of which raises on its own:

* a transaction emitted twice silently duplicates rows in the join to the
  feature matrix
* a transaction never emitted silently drops from the training set
* a transaction emitted at the head of a window has NO preceding context, so
  its embedding is computed from nothing -- the same shape of bug as velocity
  windows computed per year partition, which zeroed at every boundary
"""

from __future__ import annotations

import pytest

from fraud.features.fm_sequences import (
    TOKENS_PER_SLOT,
    build_sequence,
    token_position,
    transactions_per_window,
    windows,
)
from fraud.features.fm_tokenizer import BOS_ID, EOS_ID, SEP_ID, TOKENS_PER_TRANSACTION


def test_window_size_matches_what_pretraining_used():
    """Upstream's config says '~315 transactions per sequence' at 4096."""
    assert transactions_per_window(4096) == 315


@pytest.mark.parametrize("n", [1, 2, 314, 315, 316, 700, 1000, 5000, 12000])
def test_windows_partition_every_transaction_exactly_once(n):
    per = transactions_per_window(4096)
    emitted = [i for start, stop, emit in windows(n, per) for i in range(emit, stop)]
    assert emitted == list(range(n))


@pytest.mark.parametrize("n", [316, 700, 5000])
def test_only_the_first_window_emits_without_context(n):
    per = transactions_per_window(4096)
    spans = windows(n, per)
    assert spans[0][2] == spans[0][0]                 # first window starts cold
    for start, _, emit in spans[1:]:
        assert emit - start >= per // 2               # the rest carry half a window


@pytest.mark.parametrize("n", [1, 100, 315, 1000])
def test_no_window_exceeds_the_trained_length(n):
    per = transactions_per_window(4096)
    for start, stop, _ in windows(n, per):
        assert stop - start <= per
        # 1 bos + k*12 + (k-1) sep + 1 eos
        k = stop - start
        assert 2 + k * TOKENS_PER_TRANSACTION + (k - 1) <= 4096


# --- sequence layout ------------------------------------------------------

def test_sequence_is_bos_transactions_separated_by_sep_then_eos():
    rows = [[10] * 12, [20] * 12, [30] * 12]
    seq = build_sequence(rows)
    assert seq[0] == BOS_ID
    assert seq[-1] == EOS_ID
    assert seq.count(SEP_ID) == len(rows) - 1
    assert len(seq) == 2 + len(rows) * 12 + (len(rows) - 1)


def test_token_position_points_at_the_transactions_last_token():
    """The embedding is read from the CUST position, so this must be exact."""
    rows = [[100 + i] * 12 for i in range(4)]
    seq = build_sequence(rows)
    for k in range(4):
        assert seq[token_position(k)] == 100 + k
        # ...and the next slot is a separator, except after the last one
        nxt = token_position(k) + 1
        assert seq[nxt] == (SEP_ID if k < 3 else EOS_ID)


def test_slot_accounting_matches_the_layout():
    assert TOKENS_PER_SLOT == TOKENS_PER_TRANSACTION + 1
