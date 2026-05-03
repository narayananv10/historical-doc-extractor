"""Fetch a curated set of historical handwritten document scans from the
Library of Congress for the demo + flagger holdout.

Targets the loc.gov JSON API for these collections:
  - abraham-lincoln-papers       (letters, 1840s-1865)
  - mary-church-terrell-papers   (letters and diaries, 1880s-1950s)

Saves images to data/raw/loc/{collection-slug}/ as JPEGs.

Usage:
    python scripts/download_loc.py                      # default 15/collection
    python scripts/download_loc.py --per-collection 10
    python scripts/download_loc.py --dry-run            # preview, don't download

The loc.gov API occasionally changes shape; if downloads fail, run with
--debug to dump a sample item record for inspection.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

LOC_BASE = "https://www.loc.gov"

COLLECTIONS = {
    "abraham-lincoln-papers": "Lincoln-era letters (1840s-1865)",
    "mary-church-terrell-papers": "Letters and diaries (1880s-1950s)",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "data" / "raw" / "loc"

REQUEST_TIMEOUT = 90
DOWNLOAD_TIMEOUT = 90
POLITE_DELAY_SEC = 0.5


def fetch_collection_items(collection_slug: str, n_request: int) -> list[dict]:
    """Hit loc.gov JSON API; over-fetch since not every record has a usable image."""
    url = f"{LOC_BASE}/collections/{collection_slug}/"
    params = {
        "fo": "json",
        "c": str(n_request * 3),
        "fa": "online-format:image",
    }
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", [])


def best_image_url(item: dict) -> str | None:
    """Pick the highest-resolution image URL from a loc.gov item record.

    `image_url` is typically a list ordered low to high resolution.
    """
    urls = item.get("image_url") or []
    if isinstance(urls, str):
        return urls
    if not urls:
        return None
    return urls[-1]


def safe_filename(raw_id_or_url: str) -> str:
    """Build a local filename from an item ID or image URL.

    Strips URL fragments (#h=...&w=...) and query strings, keeps the
    last path component, ensures a .jpg extension.
    """
    s = raw_id_or_url.split("#", 1)[0].split("?", 1)[0].rstrip("/")
    base = s.rsplit("/", 1)[-1]
    base = base.replace(":", "_")
    if not base.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
        base += ".jpg"
    return base


def download(url: str, dest: Path) -> bool:
    try:
        with requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"  ! download failed: {e}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        return False


def process_collection(slug: str, desc: str, target_n: int, dry_run: bool, debug: bool) -> int:
    print(f"\n[{slug}] {desc}")
    try:
        items = fetch_collection_items(slug, target_n)
    except requests.RequestException as e:
        print(f"  ! API call failed: {e}")
        return 0

    if debug and items:
        sample_keys = sorted(items[0].keys())
        print(f"  (debug) sample item keys: {sample_keys}")

    out_dir = OUT_ROOT / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
    for item in items:
        if n_done >= target_n:
            break
        url = best_image_url(item)
        if not url:
            continue
        item_id = item.get("id") or url
        dest = out_dir / safe_filename(str(item_id))
        if dest.exists():
            print(f"  - skip (exists): {dest.name}")
            n_done += 1
            continue
        if dry_run:
            print(f"  - would download: {url} -> {dest.name}")
            n_done += 1
            continue
        print(f"  - downloading: {dest.name}")
        if download(url, dest):
            n_done += 1
            time.sleep(POLITE_DELAY_SEC)

    print(f"  fetched {n_done} (target {target_n})")
    return n_done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-collection", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--debug", action="store_true",
                    help="Dump sample item keys to help diagnose API changes")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    total = 0
    for slug, desc in COLLECTIONS.items():
        total += process_collection(slug, desc, args.per_collection, args.dry_run, args.debug)

    print(f"\nTotal: {total} files in {OUT_ROOT}")
    if total == 0 and not args.dry_run:
        print("No files fetched. Try --debug to see what the API returned, or check loc.gov status.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
