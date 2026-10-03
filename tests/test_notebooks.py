"""The Phase 5 notebooks must still run against the code they demonstrate.

A notebook committed with outputs looks authoritative long after the code it
calls has moved on. Each one is executed here in sample mode
(FRAUD_NOTEBOOK_SAMPLE=1: small subsets, nothing written), in the kernel its
metadata names -- `fraud-monitoring` for NannyML, `fraud` for the boosters --
so a notebook that has rotted fails a test instead of misleading a reader.

Skipped where a kernel is not registered (e.g. CI before Phase 6 sets them up):
    .venv/bin/python -m ipykernel install --user --name fraud
    .venv-monitoring/bin/python -m ipykernel install --user --name fraud-monitoring
"""

from __future__ import annotations

import pathlib

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

NOTEBOOKS = sorted(pathlib.Path("notebooks/phase5").glob("*.ipynb"))


def _kernels() -> set[str]:
    from jupyter_client.kernelspec import KernelSpecManager

    return set(KernelSpecManager().find_kernel_specs())


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_executes_in_sample_mode(path: pathlib.Path, monkeypatch):
    nb = nbformat.read(path, as_version=4)
    kernel = nb.metadata["kernelspec"]["name"]
    if kernel not in _kernels():
        pytest.skip(f"kernel {kernel!r} not registered")
    monkeypatch.setenv("FRAUD_NOTEBOOK_SAMPLE", "1")
    monkeypatch.delenv("FRAUD_WRITE_SUPABASE", raising=False)
    client = nbclient.NotebookClient(
        nb, kernel_name=kernel, timeout=600,
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()   # raises CellExecutionError on the first failing cell


# NannyML notebooks need the monitoring env; everything else needs the boosters.
EXPECTED_KERNEL = {"01_monitoring": "fraud-monitoring",
                   "02_explainability": "fraud",
                   "03_staleness_and_operations": "fraud-monitoring"}


def test_committed_notebooks_carry_outputs_and_the_right_kernel():
    """They are committed executed, so the charts read on GitHub -- and they
    name the kernel they need. Saving 01 in the Studio's JupyterLab re-bound it
    to the default `python3` kernel (no NannyML), twice: the notebook still
    looked fine on GitHub and failed for anyone who ran it."""
    assert NOTEBOOKS, "no Phase 5 notebooks found"
    for path in NOTEBOOKS:
        nb = nbformat.read(path, as_version=4)
        assert nb.metadata["kernelspec"]["name"] == EXPECTED_KERNEL[path.stem], path.name
        code = [c for c in nb.cells if c.cell_type == "code"]
        assert all(c.get("outputs") for c in code[:3]), f"{path.name} is not executed"
