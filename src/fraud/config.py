"""Single entry point for `params.yaml`.

Repo convention: all config lives in params.yaml, nothing is hardcoded in src/.
Every stage reads through here so a config change invalidates the right DVC
stages and there is exactly one place to look.
"""

from __future__ import annotations

import functools
import pathlib
from typing import Any

import yaml

# src/fraud/config.py -> src/fraud -> src -> repo root
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO_ROOT / "params.yaml"


@functools.lru_cache(maxsize=4)
def load_params(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    """Parse params.yaml. Cached -- stages read it many times per run."""
    target = pathlib.Path(path) if path is not None else PARAMS_PATH
    with target.open() as fh:
        return yaml.safe_load(fh)


def repo_path(relative: str | pathlib.Path) -> pathlib.Path:
    """Resolve a params.yaml path (always repo-relative) to an absolute path.

    Stages must work regardless of the working directory they are invoked from,
    which DVC does not guarantee.
    """
    relative = pathlib.Path(relative)
    return relative if relative.is_absolute() else REPO_ROOT / relative
