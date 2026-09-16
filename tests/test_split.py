"""Temporal split tests.

The split's job is to make leakage impossible by construction. These tests
build tiny year-partitioned datasets on disk and check both that the mapping is
right and that the assertions actually fire when it is wrong -- an assertion
that never fails is decoration.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from fraud.data.split import build_manifest, load_split, split_years

PARAMS = {
    "split": {"train_end_year": 2017, "val_year": 2018},
    "paths": {"processed": "unused"},
}


def write_dataset(root, rows_by_year: dict[int, list[tuple[int, dt.datetime, int]]]):
    """Write Year=*/part-0.parquet from (txn_id, ts, Fraud) tuples."""
    for year, rows in rows_by_year.items():
        part = root / f"Year={year}"
        part.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "txn_id": pl.Series([r[0] for r in rows], dtype=pl.Int64),
                "ts": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us")),
                "Fraud": pl.Series([r[2] for r in rows], dtype=pl.Int8),
            }
        ).write_parquet(part / "part-0.parquet")
    return root


def healthy(root):
    """One row per year, 2016-2020, in chronological order."""
    return write_dataset(
        root,
        {
            2016: [(0, dt.datetime(2016, 6, 1), 1), (1, dt.datetime(2016, 7, 1), 0)],
            2017: [(2, dt.datetime(2017, 6, 1), 1)],
            2018: [(3, dt.datetime(2018, 6, 1), 1), (4, dt.datetime(2018, 7, 1), 0)],
            2019: [(5, dt.datetime(2019, 6, 1), 1)],
            # 2020 really does contain zero positives in the source data.
            2020: [(6, dt.datetime(2020, 1, 5), 0)],
        },
    )


def test_split_years_from_params():
    mapping = split_years(PARAMS, [2016, 2017, 2018, 2019, 2020])
    assert mapping["train"] == [2016, 2017]
    assert mapping["val"] == [2018]
    assert mapping["test"] == [2019, 2020]


def test_split_years_are_disjoint():
    mapping = split_years(PARAMS, list(range(1991, 2021)))
    all_years = mapping["train"] + mapping["val"] + mapping["test"]
    assert len(all_years) == len(set(all_years)) == 30


def test_load_split_returns_only_its_years(tmp_path):
    root = healthy(tmp_path / "processed")
    assert set(
        load_split("train", root, PARAMS).collect()["Year"].to_list()
    ) == {2016, 2017}
    assert set(load_split("val", root, PARAMS).collect()["Year"].to_list()) == {2018}
    assert set(
        load_split("test", root, PARAMS).collect()["Year"].to_list()
    ) == {2019, 2020}


def test_load_split_rejects_unknown_name(tmp_path):
    root = healthy(tmp_path / "processed")
    with pytest.raises(KeyError):
        load_split("holdout", root, PARAMS)


def test_manifest_passes_on_healthy_data(tmp_path):
    manifest = build_manifest(healthy(tmp_path / "processed"), PARAMS)
    assert manifest["passed"], manifest["failures"]
    assert manifest["splits"]["train"]["rows"] == 3
    assert manifest["splits"]["val"]["rows"] == 2
    assert manifest["splits"]["test"]["rows"] == 2
    assert manifest["total_rows"] == 7


def test_manifest_records_both_test_variants(tmp_path):
    """2020 adds negatives only, so the two variants must differ."""
    manifest = build_manifest(healthy(tmp_path / "processed"), PARAMS)
    headline = manifest["test_variants"]["headline"]
    operating = manifest["test_variants"]["operating_point"]
    assert headline["years"] == [2019]
    assert operating["years"] == [2019, 2020]
    # Adding a fraud-free year must dilute the rate.
    assert headline["fraud_rate"] > operating["fraud_rate"]


def test_temporal_overlap_is_caught(tmp_path):
    """A val row dated before the end of train is exactly the leakage case."""
    root = write_dataset(
        tmp_path / "processed",
        {
            2016: [(0, dt.datetime(2016, 6, 1), 1)],
            2017: [(1, dt.datetime(2017, 12, 31), 1)],
            # Backdated into the training period.
            2018: [(2, dt.datetime(2016, 1, 1), 1)],
            2019: [(3, dt.datetime(2019, 6, 1), 1)],
        },
    )
    manifest = build_manifest(root, PARAMS)
    assert not manifest["passed"]
    assert any("temporal overlap" in f for f in manifest["failures"])


def test_split_without_positives_is_caught(tmp_path):
    """Training on an all-negative split would silently produce a useless model."""
    root = write_dataset(
        tmp_path / "processed",
        {
            2016: [(0, dt.datetime(2016, 6, 1), 0)],
            2018: [(1, dt.datetime(2018, 6, 1), 0)],
            2019: [(2, dt.datetime(2019, 6, 1), 1)],
        },
    )
    manifest = build_manifest(root, PARAMS)
    assert not manifest["passed"]
    assert any("no positives" in f for f in manifest["failures"])


def test_missing_partitions_raise(tmp_path):
    empty = tmp_path / "processed"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        build_manifest(empty, PARAMS)
