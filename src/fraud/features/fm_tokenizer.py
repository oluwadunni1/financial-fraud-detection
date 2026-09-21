"""The foundation model's tokenizer, ported from cuDF to polars.

Upstream: https://github.com/oluwadunni1/transaction-foundation-model
(`src/tokenizer/`). The constants and the id arithmetic below are theirs; only
the dataframe layer is rewritten, because their `tokenize()` returns a
`cudf.Series` and serving has no GPU -- a request tokenises the arriving
transaction on the API box.

**Fidelity, not correctness, is the goal here.** The checkpoint's embedding
matrix is indexed by these exact ids. A token that maps one bucket off does not
raise; it silently feeds the model a vector trained to mean something else. So
where upstream does something surprising, this file reproduces the surprise and
says why.

Two such quirks, both verified against their code:

1. `FixedVocabTokenizer.build_vocab` keys its table by the VALUE, not by a
   zero-based index: `{i: f"{prefix}_{i:0{pad}d}" for i in range(min, max+1)}`.
   Global id is `local + offset`, so a step whose `min_val` is not 0 leaves
   `offset+0` unused and runs one past its own block. Only MONTH starts at 1,
   so **`MONTH_12` and `CARD_0` are the same id (2179)** -- 6,251 token strings
   over 6,250 distinct ids. The checkpoint was pretrained that way.
2. The merchant bucket comes from cuDF's `Series.hash_values()`, which is
   MurmurHash3 x86_32 over the UTF-8 bytes with seed 0, taken **unsigned**. A
   signed modulo puts the same merchant in a different bucket (1001 vs 297 on
   the first row of 2016), so `signed=False` is load-bearing.

Vocabulary layout, which reproduces `global_vocab_size == 6251`:

    <pad> <bos> <eos> <sep> <unk>   0-4
    AMT 7 | MERCH 2000 | CAT 14 | MCC 110 | HOUR 24 | DOW 7 | MONTH 12
    CARD 10 | CHIP 4 | ZIP3 1000 | STATE 58 | CUST 3000
"""

from __future__ import annotations

import dataclasses

import mmh3
import polars as pl

# --- vendored constants (upstream src/tokenizer/financial_pipeline.py) -----

KNOWN_MCCS = [
    -1, 1711, 3000, 3001, 3005, 3006, 3007, 3008, 3009, 3058, 3066,
    3075, 3132, 3144, 3174, 3256, 3260, 3359, 3387, 3389, 3390, 3393,
    3395, 3405, 3504, 3509, 3596, 3640, 3684, 3722, 3730, 3771, 3775,
    3780, 4111, 4112, 4121, 4131, 4214, 4411, 4511, 4722, 4784, 4814,
    4829, 4899, 4900, 5045, 5094, 5192, 5193, 5211, 5251, 5261, 5300,
    5310, 5311, 5411, 5499, 5533, 5541, 5621, 5651, 5655, 5661, 5712,
    5719, 5722, 5732, 5733, 5812, 5813, 5814, 5815, 5816, 5912, 5921,
    5932, 5941, 5942, 5947, 5970, 5977, 6300, 7011, 7210, 7230, 7276,
    7349, 7393, 7531, 7538, 7542, 7549, 7801, 7802, 7832, 7922, 7995,
    7996, 8011, 8021, 8041, 8043, 8049, 8062, 8099, 8111, 8931, 9402,
]

INDUSTRY_RANGES = [
    (0, 1499, "AGRICULTURAL"), (1500, 2999, "CONTRACTED"),
    (3000, 3299, "AIRLINES"), (3300, 3499, "CAR_RENTAL"),
    (3500, 3999, "LODGING"), (4000, 4799, "TRANSPORTATION"),
    (4800, 4999, "UTILITIES"), (5000, 5599, "RETAIL"),
    (5600, 5699, "CLOTHING"), (5700, 7299, "MISC_STORES"),
    (7300, 7999, "BUSINESS"), (8000, 8999, "PROFESSIONAL"),
    (9000, 9999, "GOVERNMENT"),
]

CHIP_MAPPING = {
    "SWIPE TRANSACTION": "SWIPE",
    "CHIP TRANSACTION": "CHIP",
    "ONLINE TRANSACTION": "ONLINE",
}

ALL_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "VI", "GU", "AS", "MP", "XX", "ONLINE",
]

# The thresholds AMT counts; note 0 is never compared, so values are 0-6.
AMOUNT_THRESHOLDS = [10, 50, 100, 500, 1000, 5000]

MERCHANT_HASH_SIZE = 2000
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]
PAD_ID, BOS_ID, EOS_ID, SEP_ID, UNK_ID = range(len(SPECIAL_TOKENS))

# Pinned from the checkpoint's config.json. A drift here means the tokenizer
# and the weights disagree, which is unrecoverable and silent.
EXPECTED_VOCAB_SIZE = 6251

# The 12 columns, in the order a transaction's tokens are emitted.
TOKEN_COLUMNS = [
    "amt_val", "merch_hash", "mcc_int", "mcc_str", "hour", "dow",
    "month", "card", "chip_upper", "zip3", "state_clean", "cust",
]
TOKENS_PER_TRANSACTION = len(TOKEN_COLUMNS)


def _mapping_labels(values: list[str], default: str) -> list[str]:
    """Upstream MappingTokenizer order: sort, append default, dedupe."""
    labels = sorted(set(values))
    labels.append(default)
    return list(dict.fromkeys(labels))


CAT_LABELS = _mapping_labels([lbl for _, _, lbl in INDUSTRY_RANGES], "GENERAL")
MCC_LABELS = _mapping_labels([str(m) for m in KNOWN_MCCS], "-1")
CHIP_LABELS = _mapping_labels(list(CHIP_MAPPING.values()), "UNK")
STATE_LABELS = _mapping_labels(ALL_STATES, "XX")

# (name, vocab size). Order is upstream's `tokenizer_order` and fixes offsets.
_STEP_SIZES: list[tuple[str, int]] = [
    ("amt_val", 7), ("merch_hash", MERCHANT_HASH_SIZE),
    ("mcc_int", len(CAT_LABELS)), ("mcc_str", len(MCC_LABELS)),
    ("hour", 24), ("dow", 7), ("month", 12), ("card", 10),
    ("chip_upper", len(CHIP_LABELS)), ("zip3", 1000),
    ("state_clean", len(STATE_LABELS)), ("cust", 3000),
]


@dataclasses.dataclass(frozen=True)
class Vocab:
    offsets: dict[str, int]
    sizes: dict[str, int]
    size: int


def build_vocab() -> Vocab:
    offsets, sizes, cursor = {}, {}, len(SPECIAL_TOKENS)
    for name, size in _STEP_SIZES:
        offsets[name] = cursor
        sizes[name] = size
        cursor += size
    return Vocab(offsets=offsets, sizes=sizes, size=cursor)


VOCAB = build_vocab()


def merchant_bucket(merchant: str) -> int:
    """cuDF `hash_values()` equivalent: murmur3 x86_32, seed 0, UNSIGNED."""
    return mmh3.hash(merchant, 0, signed=False) % MERCHANT_HASH_SIZE


def _clean_merchant(series: pl.Series) -> pl.Series:
    """Upper, then drop anything outside [A-Z0-9 \\s -], as upstream does.

    TabFormer's merchant name is the string form of a hashed int64, so the
    regex is a no-op on real data -- but it is applied anyway, because a
    dataset where it is not a no-op must tokenise the same way theirs would.
    """
    return series.cast(pl.String).str.to_uppercase().str.replace_all(
        r"[^A-Z0-9\s\-]", ""
    )


def _category_expr(mcc: pl.Expr) -> pl.Expr:
    """MCC -> industry label via the range table, GENERAL outside every range."""
    expr = pl.lit(CAT_LABELS.index("GENERAL"), dtype=pl.Int32)
    for low, high, label in reversed(INDUSTRY_RANGES):
        expr = (
            pl.when((mcc >= low) & (mcc <= high))
            .then(pl.lit(CAT_LABELS.index(label), dtype=pl.Int32))
            .otherwise(expr)
        )
    return expr


def local_indices(frame: pl.DataFrame) -> pl.DataFrame:
    """Per-step LOCAL indices, before offsets are added.

    Reproduces upstream `preprocess()`. `hour`/`dow`/`month` come from our `ts`
    rather than re-parsing Year/Month/Day/Time -- the same instant, already
    materialised. `dow` is shifted because polars' `weekday()` is 1-7 Mon=1
    while cuDF's `dayofweek` is 0-6 Mon=0; without the shift every DOW token
    would be one day out and nothing would complain.
    """
    amt = pl.col("Amount").cast(pl.Float64)
    amt_val = pl.sum_horizontal(
        [(amt >= t).cast(pl.Int32) for t in AMOUNT_THRESHOLDS]
    )

    # Hash the distinct merchants once, then join -- 62,810 unique values in
    # 2016+ against 7.2M rows, so per-row Python hashing would dominate.
    merchants = frame.get_column("Merchant")
    cleaned = _clean_merchant(merchants)
    lookup = {m: merchant_bucket(m) for m in cleaned.unique().to_list()}

    mcc = pl.col("MCC").fill_null(-1).cast(pl.Int32)
    mcc_str_index = pl.col("MCC").fill_null(-1).cast(pl.Int64).cast(pl.String)

    zip_str = pl.col("Zip").fill_null(0).cast(pl.Int64).cast(pl.String)
    zip3 = zip_str.str.slice(0, 3).str.pad_start(3, "0").cast(pl.Int32)

    state = (
        pl.col("State").fill_null("XX").cast(pl.String)
        .str.to_uppercase().str.strip_chars()
    )
    state = pl.when(state == "").then(pl.lit("XX")).otherwise(state)

    chip = pl.col("Chip").cast(pl.String).str.to_uppercase()

    return frame.with_columns(_merch_clean=cleaned).select(
        pl.col("txn_id"),
        amt_val.clip(0, 6).alias("amt_val"),
        pl.col("_merch_clean")
        .replace_strict(lookup, default=0, return_dtype=pl.Int32)
        .alias("merch_hash"),
        _category_expr(mcc).alias("mcc_int"),
        mcc_str_index.replace_strict(
            {label: i for i, label in enumerate(MCC_LABELS)},
            default=MCC_LABELS.index("-1"),
            return_dtype=pl.Int32,
        ).alias("mcc_str"),
        pl.col("ts").dt.hour().cast(pl.Int32).alias("hour"),
        (pl.col("ts").dt.weekday() - 1).cast(pl.Int32).alias("dow"),
        # NOT month-1: upstream keys FixedVocab by value, so MONTH occupies
        # offset+1 .. offset+12 and collides with CARD_0. See the docstring.
        pl.col("ts").dt.month().cast(pl.Int32).alias("month"),
        pl.col("Card").cast(pl.Int32).clip(0, 9).alias("card"),
        chip.replace_strict(
            {k: CHIP_LABELS.index(v) for k, v in CHIP_MAPPING.items()},
            default=CHIP_LABELS.index("UNK"),
            return_dtype=pl.Int32,
        ).alias("chip_upper"),
        zip3.clip(0, 999).alias("zip3"),
        state.replace_strict(
            {label: i for i, label in enumerate(STATE_LABELS)},
            default=STATE_LABELS.index("XX"),
            return_dtype=pl.Int32,
        ).alias("state_clean"),
        pl.col("User").cast(pl.Int32).clip(0, 2999).alias("cust"),
    )


def tokenize(frame: pl.DataFrame, vocab: Vocab | None = None) -> pl.DataFrame:
    """`txn_id` plus the 12 GLOBAL token ids, one column each, in order."""
    vocab = vocab or VOCAB
    local = local_indices(frame)
    return local.select(
        pl.col("txn_id"),
        *[
            (pl.col(name) + vocab.offsets[name]).cast(pl.Int32).alias(name)
            for name in TOKEN_COLUMNS
        ],
    )
