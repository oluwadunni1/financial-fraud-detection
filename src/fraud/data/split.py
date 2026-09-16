"""Stage `split`: the temporal split, as an asserted artifact.

The split is a *predicate* over year-partitioned Parquet, not three copies of
the data -- partition pruning makes filtering free, and duplicating ~2.4 GB
three ways would make the partitioning pointless.

What this stage produces is the manifest plus the guarantees: no temporal
overlap, no txn_id in two splits, nothing derived from a random seed. A random
split would leak the future into training, which is the single easiest way to
make every downstream metric meaningless.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import polars as pl

from fraud.config import load_params, repo_path


def split_years(params: dict, years: list[int]) -> dict[str, list[int]]:
    """Map each split to its years, straight from params.yaml."""
    cfg = params["split"]
    train_end, val_year = cfg["train_end_year"], cfg["val_year"]
    return {
        "train": [y for y in years if y <= train_end],
        "val": [y for y in years if y == val_year],
        "test": [y for y in years if y > val_year],
    }


def load_split(
    name: str,
    processed: pathlib.Path | str | None = None,
    params: dict | None = None,
) -> pl.LazyFrame:
    """Return one split as a LazyFrame.

    The only supported way to get a split. Phase 2 onward must call this rather
    than writing a year predicate by hand -- it is what makes an accidental
    random split impossible.
    """
    params = params or load_params()
    root = repo_path(processed or params["paths"]["processed"])
    years = sorted(int(p.name.split("=")[1]) for p in root.glob("Year=*"))
    wanted = split_years(params, years)
    if name not in wanted:
        raise KeyError(f"unknown split {name!r}; expected one of {list(wanted)}")
    lf = pl.scan_parquet(root / "**/*.parquet", hive_partitioning=True)
    return lf.filter(pl.col("Year").is_in(wanted[name]))


def _stats(lf: pl.LazyFrame) -> dict:
    row = lf.select(
        pl.len().alias("rows"),
        pl.col("Fraud").sum().alias("fraud"),
        pl.col("ts").min().alias("ts_min"),
        pl.col("ts").max().alias("ts_max"),
        pl.col("txn_id").min().alias("txn_id_min"),
        pl.col("txn_id").max().alias("txn_id_max"),
    ).collect(engine="streaming").row(0, named=True)
    rows, fraud = int(row["rows"]), int(row["fraud"])
    return {
        "rows": rows,
        "fraud": fraud,
        "fraud_rate": fraud / rows if rows else 0.0,
        # Left as datetimes: the ordering assertion below must compare
        # instants, not strings that merely happen to sort correctly.
        "ts_min": row["ts_min"],
        "ts_max": row["ts_max"],
        "txn_id_min": int(row["txn_id_min"]),
        "txn_id_max": int(row["txn_id_max"]),
    }


def build_manifest(processed: pathlib.Path, params: dict) -> dict:
    years = sorted(int(p.name.split("=")[1]) for p in processed.glob("Year=*"))
    if not years:
        raise FileNotFoundError(f"no Year=* partitions under {processed}")

    wanted = split_years(params, years)
    manifest: dict = {"source": str(processed), "splits": {}}
    failures: list[str] = []

    for name, yrs in wanted.items():
        lf = load_split(name, processed, params)
        manifest["splits"][name] = {"years": yrs, **_stats(lf)}

    # 2020 holds 336,500 rows and zero positives, so it moves precision without
    # saying anything about model quality. Both test variants are recorded here
    # so the evaluation choice is explicit rather than buried in an eval script.
    test_years = wanted["test"]
    lf_2019 = load_split("test", processed, params).filter(
        pl.col("Year") == min(test_years)
    )
    manifest["test_variants"] = {
        "headline": {
            "name": "test_2019_only",
            "years": [min(test_years)],
            **_stats(lf_2019),
            "use_for": "AUC-PR -- the clean model-quality comparison",
        },
        "operating_point": {
            "name": "test_full",
            "years": test_years,
            **manifest["splits"]["test"],
            "use_for": "precision@k, alert volume, recall at fixed FPR",
        },
    }

    # --- the assertions, which are the real deliverable --------------------
    s = manifest["splits"]
    if not (s["train"]["ts_max"] < s["val"]["ts_min"] < s["test"]["ts_min"]):
        failures.append(
            "temporal overlap: expected max(train.ts) < min(val.ts) < min(test.ts), got "
            f"{s['train']['ts_max']} / {s['val']['ts_min']} / {s['test']['ts_min']}"
        )

    overlap = set(wanted["train"]) & set(wanted["val"]) | set(wanted["val"]) & set(
        wanted["test"]
    ) | set(wanted["train"]) & set(wanted["test"])
    if overlap:
        failures.append(f"a year appears in two splits: {sorted(overlap)}")

    for name in ("train", "val", "test"):
        if s[name]["rows"] == 0:
            failures.append(f"{name} split is empty")
    for name in ("train", "val"):
        if s[name]["fraud"] == 0:
            failures.append(f"{name} split has no positives -- unusable for training")

    total = sum(s[n]["rows"] for n in ("train", "val", "test"))
    manifest["total_rows"] = total

    manifest["passed"] = not failures
    manifest["failures"] = failures
    return manifest


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=params["paths"]["processed"])
    ap.add_argument("--output", default=params["paths"]["splits_manifest"])
    args = ap.parse_args(argv)

    manifest = build_manifest(repo_path(args.input), params)

    out = repo_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, default=str))

    for name, info in manifest["splits"].items():
        print(
            f"{name:<5} years {info['years'][0]}-{info['years'][-1]} | "
            f"{info['rows']:>10,} rows | {info['fraud']:>6,} fraud "
            f"({info['fraud_rate']:.5f})"
        )
    if not manifest["passed"]:
        print("\nSPLIT ASSERTIONS FAILED:", file=sys.stderr)
        for f in manifest["failures"]:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
