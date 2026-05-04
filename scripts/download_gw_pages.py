"""Download original George Washington Papers page scans from the Library of Congress.

The FKI Washington dataset (data/raw/iam_gw/washingtondb-v1.0/) ships line images
only — no full page scans. We need page images to send Claude for visual context
during post-correction. The original scans are public at LoC under
"George Washington Papers, Series 2, Letterbook 1" (mgw2.001).

FKI uses 20 pages: 270-279 and 300-309 (two blocks of 10 with a gap). The
page list is derived from the FKI transcription so we don't over-download.
FKI page N maps directly to LoC sp=N with no offset (verified visually — FKI's
ground truth for line 270-01 reads "270. Letters, Orders and Instructions.
October 1755", which matches the header at the top of LoC sp=270).

URL pattern is not cleanly predictable (LoC's filenames embed an internal image
ID that differs from the sp number for some pages), so we hit the LoC item JSON
API once to get the canonical URL list, then download mechanically.

Idempotent: skips files that already exist with non-zero size.

Run: python scripts/download_gw_pages.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

LOC_ITEM_JSON = "https://www.loc.gov/item/mgw2.001/?fo=json"
DEFAULT_OUTPUT = Path("data/raw/iam_gw/loc_pages")
DEFAULT_TRANSCRIPTION = Path(
    "data/raw/iam_gw/washingtondb-v1.0/ground_truth/transcription.txt"
)
USER_AGENT = "historical-doc-extractor/0.1 (research; narayananv10@github)"
HEADERS = {"User-Agent": USER_AGENT}


def _fetch_loc_metadata() -> list:
    """Returns the per-scan files list from the LoC item JSON API.

    files[i] is itself a list of file variants (different sizes / formats) for
    scan number i+1. We pick the largest JPEG variant per scan downstream.
    """
    print(f"[loc] fetching {LOC_ITEM_JSON}", file=sys.stderr)
    resp = requests.get(LOC_ITEM_JSON, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    files = resp.json()["resources"][0]["files"]
    print(f"[loc] item has {len(files)} scans", file=sys.stderr)
    return files


def _pick_jpeg_url(file_variants: list) -> str | None:
    """Pick the largest JPEG variant from the variants list for one scan."""
    jpegs = [
        f for f in file_variants
        if "image/jpeg" in f.get("mimetype", "") and f.get("url")
    ]
    if not jpegs:
        return None
    return max(jpegs, key=lambda f: f.get("size", 0))["url"]


def _download(url: str, dest: Path, *, max_attempts: int = 3) -> int:
    """Download url -> dest with simple retry on transient errors.

    LoC's CDN occasionally returns 525 (SSL handshake failure) or drops the
    connection mid-stream; both are routinely fixed by a second attempt.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return len(resp.content)
        except (requests.RequestException, OSError) as exc:
            last_exc = exc
            if attempt < max_attempts:
                wait = 2 ** attempt  # 2, 4, 8 seconds
                print(f"    retry {attempt}/{max_attempts - 1} in {wait}s ({exc!r})",
                      file=sys.stderr)
                time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def _read_page_ids_from_transcription(path: Path) -> list[int]:
    """Extract the unique page IDs FKI actually labels (FKI's coverage is
    270-279 and 300-309 — not contiguous — so deriving the list keeps us
    from over-downloading)."""
    page_ids: set[int] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        line_id = raw.split(" ", 1)[0]
        try:
            page_ids.add(int(line_id.split("-")[0]))
        except ValueError:
            continue
    return sorted(page_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Directory to write JPGs into (one per FKI page id)",
    )
    parser.add_argument(
        "--transcription", type=Path, default=DEFAULT_TRANSCRIPTION,
        help="FKI transcription.txt; page IDs are derived from it",
    )
    parser.add_argument(
        "--pages", type=str, default=None,
        help="Override: comma-separated page numbers (e.g. '270,271,300') "
             "instead of deriving from --transcription",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.3,
        help="Seconds to sleep between downloads (be polite to LoC)",
    )
    args = parser.parse_args()

    if args.pages:
        page_nums = sorted({int(x) for x in args.pages.split(",") if x.strip()})
    else:
        if not args.transcription.exists():
            sys.exit(
                f"transcription file not found: {args.transcription}\n"
                f"Pass --pages '270,271,...' to override, or --transcription "
                f"<path> to point at your FKI extraction."
            )
        page_nums = _read_page_ids_from_transcription(args.transcription)
    print(f"[plan] {len(page_nums)} pages to fetch: "
          f"{page_nums[:3]}...{page_nums[-3:]}", file=sys.stderr)

    args.output.mkdir(parents=True, exist_ok=True)
    files = _fetch_loc_metadata()
    n_downloaded = 0
    n_skipped = 0
    n_failed = 0
    total_bytes = 0

    for page_num in page_nums:
        dest = args.output / f"{page_num}.jpg"
        if dest.exists() and dest.stat().st_size > 0:
            n_skipped += 1
            continue

        if page_num > len(files):
            print(f"[skip] page {page_num} exceeds item file count ({len(files)})",
                  file=sys.stderr)
            n_failed += 1
            continue

        url = _pick_jpeg_url(files[page_num - 1])
        if url is None:
            print(f"[skip] page {page_num}: no JPEG variant found", file=sys.stderr)
            n_failed += 1
            continue

        try:
            n_bytes = _download(url, dest)
        except Exception as exc:
            print(f"[error] page {page_num}: {exc!r}", file=sys.stderr)
            n_failed += 1
            continue

        n_downloaded += 1
        total_bytes += n_bytes
        print(f"  page {page_num} -> {dest.name}  ({n_bytes/1024:.0f} KB)",
              file=sys.stderr)
        time.sleep(args.sleep)

    print(
        f"\n[done] {n_downloaded} downloaded, {n_skipped} already present, "
        f"{n_failed} failed  ({total_bytes/1024:.0f} KB total) -> {args.output}",
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
