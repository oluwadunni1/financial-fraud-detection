"""Stage `fetch_fm_checkpoint`: pull the pretrained decoder into the pipeline.

The checkpoint lives in the upstream repo behind Git LFS, which means a plain
`raw.githubusercontent.com` fetch returns a 133-byte pointer rather than the
weights -- silently, and the pointer is a perfectly valid file. GitHub serves
the real content from `media.githubusercontent.com` instead.

Bringing it into our own pipeline rather than depending on `git lfs` at run
time keeps the DAG self-contained and puts the weights in R2 with everything
else. It is 58 MB, so this costs nothing.

The safetensors file is verified against the sha256 recorded in the upstream
LFS pointer. That matters more than usual here: the tokenizer emits ids into
this exact embedding matrix, so a different checkpoint is not a worse model,
it is 6,251 rows of unrelated numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
import urllib.request

from fraud.config import load_params, repo_path

_RAW = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"
_LFS = "https://media.githubusercontent.com/media/{repo}/{ref}/{path}"

# Plain files come from raw; the weights come from the LFS media host.
PLAIN_FILES = ("config.json", "generation_config.json", "model.safetensors.index.json")
LFS_FILES = ("model-00001-of-00001.safetensors",)

# From the upstream LFS pointer.
EXPECTED = {
    "model-00001-of-00001.safetensors": (
        57_917_064,
        "71c5899c2ab0146c209207d67efc65dda882f4966ae09877b2580c14bcb62cbe",
    )
}

_CHUNK = 8 << 20


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, target: pathlib.Path) -> int:
    with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed https URLs
        target.write_bytes(response.read())
    return target.stat().st_size


def _ok(target: pathlib.Path, name: str) -> bool:
    if not target.exists():
        return False
    if name not in EXPECTED:
        return target.stat().st_size > 0
    size, digest = EXPECTED[name]
    return target.stat().st_size == size and sha256_of(target) == digest


def fetch(repo: str, ref: str, source_dir: str, out_dir: pathlib.Path) -> dict:
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    fetched, skipped = [], []

    for name, template in (
        *[(n, _RAW) for n in PLAIN_FILES],
        *[(n, _LFS) for n in LFS_FILES],
    ):
        target = out_dir / name
        if _ok(target, name):
            skipped.append(name)
            continue
        url = template.format(repo=repo, ref=ref, path=f"{source_dir}/{name}")
        print(f"  {name} ...", flush=True)
        _download(url, target)
        if not _ok(target, name):
            actual = target.stat().st_size
            raise RuntimeError(
                f"{name} failed verification (got {actual:,} bytes). If this is "
                f"133 bytes it is a Git LFS pointer, not the weights."
            )
        fetched.append(name)

    return {
        "repo": repo, "ref": ref,
        "fetched": fetched, "already_present": skipped,
        "seconds": round(time.perf_counter() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    cfg = params["fm"]["checkpoint"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=params["paths"]["fm_checkpoint"])
    args = ap.parse_args(argv)

    try:
        summary = fetch(
            cfg["repo"], cfg["ref"], cfg["path"], repo_path(args.output)
        )
    except Exception as exc:  # noqa: BLE001 - the reason is the useful part
        print(f"checkpoint fetch failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"checkpoint ready ({len(summary['fetched'])} fetched, "
        f"{len(summary['already_present'])} already present) "
        f"in {summary['seconds']}s"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
