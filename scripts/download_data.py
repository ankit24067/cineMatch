"""Fetch the MovieLens ml-latest-small dataset into data/raw/.

The MovieLens licence does not permit redistribution, so the raw CSVs are not
committed (see README section 6). This script is the reproducible substitute:
run it once after cloning.

    python scripts/download_data.py

Integrity is checked against the MD5 published alongside the archive. The
observed digest is written to data/raw/CHECKSUM so a later run can detect a
silently changed upstream file.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DATASET = "ml-latest-small"
BASE_URL = f"https://files.grouplens.org/datasets/movielens/{DATASET}"
ARCHIVE_URL = f"{BASE_URL}.zip"
MD5_URL = f"{BASE_URL}.zip.md5"

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

# Files we actually consume downstream. Anything else in the archive is dropped
# so data/raw/ stays an explicit contract rather than "whatever upstream ships".
REQUIRED_FILES = ("ratings.csv", "movies.csv", "tags.csv", "links.csv")

CHUNK = 1 << 16


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - integrity check, not a security boundary
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with dest.open("wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, disable=not total
        ) as bar:
            for chunk in response.iter_content(CHUNK):
                fh.write(chunk)
                bar.update(len(chunk))


def _expected_md5() -> str | None:
    """Read the checksum GroupLens publishes next to the archive.

    Note this is same-origin: it guards against a truncated or corrupted
    download, not against a compromised host. That is the threat we care about.
    """
    try:
        response = requests.get(MD5_URL, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ! could not fetch published MD5 ({exc}); skipping verification")
        return None
    # Format: "MD5 (ml-latest-small.zip) = <hex>"
    text = response.text.strip()
    return text.rsplit("=", 1)[-1].strip() if "=" in text else text.split()[0]


def _extract(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        members = {Path(n).name: n for n in zf.namelist() if not n.endswith("/")}
        missing = [f for f in REQUIRED_FILES if f not in members]
        if missing:
            raise RuntimeError(f"archive is missing expected files: {missing}")
        for filename in REQUIRED_FILES:
            with zf.open(members[filename]) as src, (dest / filename).open("wb") as out:
                shutil.copyfileobj(src, out)
            print(f"  extracted {filename}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if data/raw/ is populated"
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if all((RAW_DIR / f).exists() for f in REQUIRED_FILES) and not args.force:
        print(f"data/raw/ already populated -> {RAW_DIR}")
        print("re-run with --force to refresh")
        return 0

    archive = RAW_DIR / f"{DATASET}.zip"
    print(f"downloading {ARCHIVE_URL}")
    _download(ARCHIVE_URL, archive)

    observed = _md5(archive)
    expected = _expected_md5()
    if expected and observed != expected:
        archive.unlink()
        print(f"  ! MD5 mismatch: expected {expected}, got {observed}", file=sys.stderr)
        return 1
    print(f"  md5 {observed}" + (" (verified)" if expected else " (unverified)"))

    print("extracting")
    _extract(archive, RAW_DIR)
    archive.unlink()

    (RAW_DIR / "CHECKSUM").write_text(f"{observed}  {DATASET}.zip\n", encoding="utf-8")

    total = sum((RAW_DIR / f).stat().st_size for f in REQUIRED_FILES)
    print(f"done -> {RAW_DIR} ({total / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
