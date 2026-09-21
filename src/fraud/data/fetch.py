"""Stage `fetch_raw`: download and unpack the TabFormer CSV.

This stage exists to close a trap rather than to add a capability. The raw CSV
was tracked by a standalone `.dvc` pointer, and DVC never collected that
pointer as a stage -- the index held 13 stages and 16 outs, none of them this
file. It was known only as a *dependency* of `ingest`, and dependencies are
never pushed or pulled. So `dvc push` silently skipped the 2.35 GB source while
uploading everything else, and `dvc status --cloud` reported "in sync" because
it agreed the file was not tracked. A fresh clone got every derived artifact
and no source data, with no error anywhere.

Making it a stage output puts it in the DAG, where blanket operations cover it.

Box serves a 278 MB `transactions.tgz`, not the CSV, so this downloads and
unpacks in one step. The archive is deleted afterwards: it is 278 MB of
redundancy next to the 2.35 GB it expands to, and re-downloading is cheap
compared with storing it twice.

The stage is idempotent and cheap to re-run. It verifies by size first and only
hashes when the size matches, because md5 over 2.35 GB costs several seconds
and the size check rejects almost every mismatch for free.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request

from fraud.config import load_params, repo_path

# Measured on the file this pipeline was built against. A mismatch means the
# upstream artifact changed, which must fail loudly: txn_id is the row's index
# in this exact file, so a different byte stream silently renumbers every
# transaction -- and predictions, labels and transaction_events all key on it.
EXPECTED_SIZE = 2_354_626_737
EXPECTED_MD5 = "5d0f027f333e8ec7d58969a7b8a206f8"

_CHUNK = 8 << 20


def md5_of(path: pathlib.Path, chunk: int = _CHUNK) -> str:
    digest = hashlib.md5()  # noqa: S324 - integrity check, not a security boundary
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def verify(path: pathlib.Path, size: int, md5: str) -> tuple[bool, str]:
    """(ok, reason). Size first -- it rejects nearly every mismatch for free."""
    if not path.exists():
        return False, "missing"
    actual_size = path.stat().st_size
    if actual_size != size:
        return False, f"size {actual_size:,} != expected {size:,}"
    actual_md5 = md5_of(path)
    if actual_md5 != md5:
        return False, f"md5 {actual_md5} != expected {md5}"
    return True, "size and md5 match"


def _download(url: str, target: pathlib.Path) -> int:
    with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed https URL
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with target.open("wb") as handle:
            while block := response.read(_CHUNK):
                handle.write(block)
                done += len(block)
                if total:
                    print(
                        f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB "
                        f"({done / total:.0%})",
                        end="",
                        flush=True,
                    )
        print()
    return done


def _extract_csv(archive: pathlib.Path, target: pathlib.Path) -> None:
    """Pull the one CSV out of the tarball.

    The member is located by suffix rather than by a hardcoded name, so a
    change in the archive's internal layout surfaces as a clear error here
    instead of a confusing one three stages downstream.
    """
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.lower().endswith(".csv")]
        if len(members) != 1:
            raise RuntimeError(
                f"expected exactly one .csv in {archive.name}, found "
                f"{[m.name for m in members]}. The upstream archive layout changed."
            )
        member = members[0]
        print(f"  extracting {member.name} ({member.size / 1e9:.2f} GB)")
        source = tar.extractfile(member)
        if source is None:
            raise RuntimeError(f"{member.name} is not a regular file")
        with source, target.open("wb") as handle:
            shutil.copyfileobj(source, handle, _CHUNK)


def fetch(
    url: str,
    target: pathlib.Path,
    size: int = EXPECTED_SIZE,
    md5: str = EXPECTED_MD5,
    force: bool = False,
) -> dict:
    started = time.perf_counter()
    target.parent.mkdir(parents=True, exist_ok=True)

    if not force:
        ok, reason = verify(target, size, md5)
        if ok:
            print(f"already present and verified ({reason}) -- nothing to do")
            return {"downloaded": False, "verified": True, "seconds": 0.0}

    print(f"downloading {url.split('?')[0]} ...")
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        archive = pathlib.Path(tmp) / "transactions.tgz"
        downloaded = _download(url, archive)
        print(f"  {downloaded / 1e6:,.0f} MB archive -> unpacking")
        # Extract beside the target, then move into place, so an interrupted
        # run never leaves a half-written CSV that looks real to the next stage.
        staged = pathlib.Path(tmp) / target.name
        _extract_csv(archive, staged)
        ok, reason = verify(staged, size, md5)
        if not ok:
            raise RuntimeError(
                f"downloaded file failed verification: {reason}. The upstream "
                f"artifact has changed -- txn_id is this file's row index, so a "
                f"different byte stream renumbers every transaction."
            )
        staged.replace(target)

    seconds = round(time.perf_counter() - started, 1)
    print(f"verified {target} ({size:,} bytes) in {seconds}s")
    return {"downloaded": True, "verified": True, "seconds": seconds}


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=params["fetch"]["raw_url"])
    ap.add_argument("--output", default=params["paths"]["raw_csv"])
    ap.add_argument("--size", type=int, default=params["fetch"]["expected_size"])
    ap.add_argument("--md5", default=params["fetch"]["expected_md5"])
    ap.add_argument(
        "--force", action="store_true", help="re-download even if verification passes"
    )
    args = ap.parse_args(argv)

    try:
        fetch(args.url, repo_path(args.output), args.size, args.md5, args.force)
    except Exception as exc:  # noqa: BLE001 - surface the reason, not a traceback
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
