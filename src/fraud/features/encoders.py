"""Categorical encoding, ported from the NVIDIA blueprint's logic.

`src/preprocess_TabFormer_lp.py` builds a scikit-learn `ColumnTransformer` of
`OneHotEncoder` (low cardinality), `category_encoders.BinaryEncoder` (high
cardinality) and `RobustScaler` (numerics). We reproduce the *behaviour*, not
the dependency: per CLAUDE.md the reference scripts are read-only and their
logic is ported into `src/fraud/features/`. Binary encoding is a deterministic
ordinal-to-bits mapping, and doing it in polars is both dependency-free and far
faster than a pandas/sklearn encoder over 20.6M rows.

Two properties matter more than the encoding scheme itself:

1. **Fitted on train only.** The mapping is part of the model. Re-fitting on a
   different split silently changes what every column means, and the resulting
   metrics are measuring something else.
2. **Unknown categories are a first-class case.** An unseen merchant encodes to
   the reserved all-zero code rather than raising. That is the same cold-start
   path the serving layer needs in Phase 4, so it is exercised from day one.

The fitted encoder serialises to JSON and ships with the model, because serving
must apply the identical mapping.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

# Ordinal 0 is reserved: it means "not seen during fit". In binary that is an
# all-zero code, and in one-hot it is all-zero too -- the two schemes agree on
# what "unknown" looks like, which keeps the cold-start vector consistent.
UNKNOWN_ORDINAL = 0


def binary_width(n_categories: int) -> int:
    """Bits needed for ordinals 0..n_categories (0 being the unknown code)."""
    return max(1, math.ceil(math.log2(n_categories + 1)))


def binary_code_exprs(
    column: str, mapping: dict[str, int], prefix: str | None = None
) -> list[pl.Expr]:
    """Binary-code a categorical column into one uint8 expression per bit.

    Shared by `Encoder.transform` and the graph builder, which needs the same
    codes for node identity features without carrying the rest of the encoder's
    columns. Unseen categories fall to UNKNOWN_ORDINAL -> an all-zero code.
    """
    prefix = prefix or column
    ordinal = (
        pl.col(column)
        .cast(pl.String)
        .replace_strict(mapping, default=UNKNOWN_ORDINAL, return_dtype=pl.UInt32)
    )
    return [
        ((ordinal // (2**bit)) % 2).cast(pl.UInt8).alias(f"{prefix}_bin_{bit}")
        for bit in range(binary_width(len(mapping)))
    ]


@dataclass
class Encoder:
    """A fitted encoder. Serialise with `to_json`, reload with `from_json`."""

    one_hot: dict[str, list[str]] = field(default_factory=dict)
    binary: dict[str, dict[str, int]] = field(default_factory=dict)
    numeric: dict[str, dict[str, float]] = field(default_factory=dict)
    one_hot_max_cardinality: int = 8

    # -- fitting ----------------------------------------------------------
    @classmethod
    def fit(
        cls,
        df: pl.DataFrame | pl.LazyFrame,
        categorical: list[str],
        numeric: list[str],
        one_hot_max_cardinality: int = 8,
    ) -> Encoder:
        """Fit on the training split only.

        Accepts a LazyFrame so the 20.6M-row training split never has to be
        materialised: each statistic is collected on its own, streaming.
        """
        lf = df.lazy() if isinstance(df, pl.DataFrame) else df
        enc = cls(one_hot_max_cardinality=one_hot_max_cardinality)

        for column in categorical:
            values = (
                lf.select(pl.col(column).cast(pl.String).unique().sort())
                .collect(engine="streaming")
                .to_series()
                .to_list()
            )
            values = [v for v in values if v is not None]
            if len(values) < one_hot_max_cardinality:
                enc.one_hot[column] = values
            else:
                # Ordinals start at 1; 0 stays reserved for unknown.
                enc.binary[column] = {v: i for i, v in enumerate(values, start=1)}

        for column in numeric:
            # RobustScaler: centre on the median, scale by the IQR. Chosen by the
            # blueprint and right for Amount, which spans refunds to outliers
            # and would let a single large value dominate a standard scaler.
            stats = lf.select(
                pl.col(column).median().alias("median"),
                pl.col(column).quantile(0.25).alias("q1"),
                pl.col(column).quantile(0.75).alias("q3"),
            ).collect(engine="streaming").row(0, named=True)
            iqr = float(stats["q3"] - stats["q1"])
            enc.numeric[column] = {
                "median": float(stats["median"]),
                # A constant column has IQR 0; scaling by 1 maps it to 0.0
                # rather than producing inf.
                "scale": iqr if iqr > 0 else 1.0,
            }

        return enc

    # -- applying ---------------------------------------------------------
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Encode a frame. Columns not fitted on are ignored."""
        missing = [
            c
            for c in (*self.one_hot, *self.binary, *self.numeric)
            if c not in df.columns
        ]
        if missing:
            raise ValueError(f"frame is missing encoded columns: {missing}")

        exprs: list[pl.Expr] = []

        for column, categories in self.one_hot.items():
            as_str = pl.col(column).cast(pl.String)
            for category in categories:
                exprs.append(
                    (as_str == category)
                    .fill_null(False)
                    .cast(pl.UInt8)
                    .alias(f"{column}_oh_{category}")
                )

        for column, mapping in self.binary.items():
            # Unseen categories fall to UNKNOWN_ORDINAL rather than raising.
            exprs += binary_code_exprs(column, mapping)

        for column, stats in self.numeric.items():
            exprs.append(
                ((pl.col(column).cast(pl.Float64) - stats["median"]) / stats["scale"])
                .cast(pl.Float32)
                .alias(column)
            )

        return df.select(exprs)

    # -- serving fast path -------------------------------------------------
    def transform_rows(self, rows: list[dict[str, Any]]) -> np.ndarray:
        """Encode a handful of rows without building a polars query.

        Identical output to `transform`, and `tests/test_subgraph.py` asserts
        that -- but ~100x faster for the one-to-twenty rows a single request
        carries.

        `transform` is the right tool for 24M rows: it builds 86 expressions,
        one of them a `replace_strict` holding 93,298 merchant codes, and lets
        polars execute them vectorised. Paying that construction cost to encode
        a single transaction took **110 ms**, which was over half the serving
        latency budget on its own. Here the same mappings are just dict lookups.
        """
        out = np.zeros((len(rows), len(self.feature_names())), dtype=np.float32)
        for row_index, row in enumerate(rows):
            column = 0
            for name, categories in self.one_hot.items():
                value = str(row[name])
                for category in categories:
                    out[row_index, column] = 1.0 if value == category else 0.0
                    column += 1
            for name, mapping in self.binary.items():
                # Unseen -> UNKNOWN_ORDINAL -> an all-zero code, exactly as the
                # batch path does. This is the cold-start case, not an error.
                ordinal = mapping.get(str(row[name]), UNKNOWN_ORDINAL)
                for bit in range(binary_width(len(mapping))):
                    out[row_index, column] = float((ordinal >> bit) & 1)
                    column += 1
            for name, stats in self.numeric.items():
                out[row_index, column] = (
                    float(row[name]) - stats["median"]
                ) / stats["scale"]
                column += 1
        return out

    # -- introspection ----------------------------------------------------
    def feature_names(self) -> list[str]:
        """Output column order. Must match `transform` exactly."""
        names: list[str] = []
        for column, categories in self.one_hot.items():
            names += [f"{column}_oh_{c}" for c in categories]
        for column, mapping in self.binary.items():
            names += [
                f"{column}_bin_{b}" for b in range(binary_width(len(mapping)))
            ]
        names += list(self.numeric)
        return names

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "one_hot": self.one_hot,
            "binary": self.binary,
            "numeric": self.numeric,
            "one_hot_max_cardinality": self.one_hot_max_cardinality,
            "feature_names": self.feature_names(),
        }

    def to_json(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Encoder:
        return cls(
            one_hot=payload["one_hot"],
            binary=payload["binary"],
            numeric=payload["numeric"],
            one_hot_max_cardinality=payload.get("one_hot_max_cardinality", 8),
        )

    @classmethod
    def from_json(cls, path: pathlib.Path) -> Encoder:
        return cls.from_dict(json.loads(pathlib.Path(path).read_text()))
