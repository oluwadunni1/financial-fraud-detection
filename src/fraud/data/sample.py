"""Stage `sample`: a small stratified subset, committed to git.

Exists so the whole DAG can run in seconds -- in tests, and in CI, where
pulling 2.35 GB to prove the pipeline works is not an option. Stratifying on
(Year, Fraud) keeps both the temporal span and the ~0.12% fraud rate, so a
sample-sized run exercises the same code paths and the same validation gates
as the full dataset.

Sampled per year partition rather than globally: it bounds memory, and it
guarantees every year survives into the sample.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import polars as pl

from fraud.config import load_params, repo_path


def build_sample(
    processed: pathlib.Path, n_rows: int, seed: int
) -> pl.DataFrame:
    parts = sorted(
        processed.glob("Year=*"), key=lambda p: int(p.name.split("=")[1])
    )
    if not parts:
        raise FileNotFoundError(f"no Year=* partitions under {processed}")

    total = sum(
        pl.scan_parquet(p / "**/*.parquet").select(pl.len()).collect().item()
        for p in parts
    )
    fraction = min(1.0, n_rows / total)

    frames: list[pl.DataFrame] = []
    for part in parts:
        year = int(part.name.split("=")[1])
        df = pl.read_parquet(part).with_columns(Year=pl.lit(year, pl.Int32))
        # Sample each class separately so the fraud rate is preserved rather
        # than left to chance at 1-in-819.
        for label in (0, 1):
            stratum = df.filter(pl.col("Fraud") == label)
            if stratum.height == 0:
                continue
            take = max(1, round(stratum.height * fraction))
            frames.append(
                stratum.sample(n=min(take, stratum.height), seed=seed, shuffle=False)
            )

    out = pl.concat(frames).sort("txn_id")
    return out


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=params["paths"]["processed"])
    ap.add_argument("--output", default=params["paths"]["sample"])
    args = ap.parse_args(argv)

    df = build_sample(
        repo_path(args.input),
        n_rows=params["sample"]["n_rows"],
        seed=params["sample"]["random_state"],
    )

    out = repo_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression="zstd")

    fraud = int(df["Fraud"].sum())
    size_mb = out.stat().st_size / 1e6
    print(
        f"sample: {df.height:,} rows | {fraud:,} fraud ({fraud / df.height:.5f}) | "
        f"years {df['Year'].min()}-{df['Year'].max()} | {size_mb:.1f} MB -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
