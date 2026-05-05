"""Extract per-line crops + pipeline outputs from LoC scans for OOD labeling.

Walks data/hand_labeled/source/ for image files, runs the full pipeline
(preprocess → TrOCR → Claude vision post-correction → flagger) on each, then
saves:

  - One PNG crop per line at data/hand_labeled/crops/{doc_id}-{line_id:03d}.png
  - One CSV row per line at data/hand_labeled/loc_ood.csv with the columns:
      doc_id, line_id, trocr_text, corrected_text, llm_confidence,
      prob_wrong, gt

The `gt` column is left empty — populated by the user via
scripts/label_for_ood.py. The other columns are reproducible from the pipeline
so the user only needs to type the actual ground-truth transcription.

Idempotent: skips documents already represented in the CSV. Re-running after
adding new source images only processes the new ones.

Cost: ~$0.02 per document in Claude API credits.

Run:
  python scripts/extract_for_ood_labeling.py
  python scripts/extract_for_ood_labeling.py --source data/raw/loc/abraham-lincoln-papers
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Allow running as `python scripts/extract_for_ood_labeling.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from src.pipeline import process

DEFAULT_SOURCE = Path("data/hand_labeled/source")
DEFAULT_OUTPUT = Path("data/hand_labeled/loc_ood.csv")
DEFAULT_CROPS = Path("data/hand_labeled/crops")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".heif"}

CSV_COLUMNS = [
    "doc_id",
    "line_id",
    "trocr_text",
    "corrected_text",
    "llm_confidence",
    "prob_wrong",
    "gt",
]


def _walk_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _existing_doc_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["doc_id"] for row in reader if row.get("doc_id")}


def _crop_line(page_image_path: Path, bbox: tuple[int, int, int, int]) -> Image.Image:
    """Crop a line from the original page image using the bbox from preprocess."""
    page = Image.open(page_image_path).convert("RGB")
    x, y, w, h = bbox
    # Defensive clamp — preprocess sometimes returns slightly out-of-bounds boxes
    x = max(0, x)
    y = max(0, y)
    right = min(page.width, x + w)
    bottom = min(page.height, y + h)
    return page.crop((x, y, right, bottom))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE,
        help="Folder of images to process (recursive)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="CSV path to append rows to (one per line)",
    )
    parser.add_argument(
        "--crops", type=Path, default=DEFAULT_CROPS,
        help="Directory to write per-line PNG crops",
    )
    parser.add_argument(
        "--no-api", action="store_true",
        help="Skip Claude post-correction (corrected_text == trocr_text). "
             "Defaults off because OOD validation needs the real corrected output.",
    )
    args = parser.parse_args()

    if not args.source.is_dir():
        sys.exit(
            f"source folder not found: {args.source}\n"
            f"Stage some LoC scans there first, e.g.:\n"
            f"  mkdir -p {args.source}\n"
            f"  cp data/raw/loc/abraham-lincoln-papers/*.jpg {args.source}/\n"
        )

    images = _walk_images(args.source)
    if not images:
        sys.exit(f"no image files in {args.source}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.crops.mkdir(parents=True, exist_ok=True)

    already_done = _existing_doc_ids(args.output)
    pending = [p for p in images if p.stem not in already_done]
    if already_done:
        print(f"[resume] {len(already_done)} docs already in {args.output}; "
              f"{len(pending)} new", file=sys.stderr)
    if not pending:
        print("[done] nothing new to extract")
        return 0

    write_header = not args.output.exists()
    n_done, n_lines, n_failed = 0, 0, 0
    t_start = time.monotonic()

    for i, path in enumerate(pending, start=1):
        doc_id = path.stem
        t0 = time.monotonic()
        try:
            result = process(path, no_api=args.no_api)
        except Exception as exc:
            n_failed += 1
            print(f"[{i}/{len(pending)}] {doc_id}  FAILED: {exc!r}", file=sys.stderr)
            continue

        with args.output.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
                write_header = False
            for line in result.lines:
                # Save the line crop so the labeler can show it visually
                try:
                    crop = _crop_line(path, line.bbox)
                    crop_path = args.crops / f"{doc_id}-{line.line_id:03d}.png"
                    crop.save(crop_path, "PNG")
                except Exception as exc:
                    print(f"  WARN: crop failed for {doc_id}-{line.line_id}: {exc!r}",
                          file=sys.stderr)
                writer.writerow({
                    "doc_id": doc_id,
                    "line_id": line.line_id,
                    "trocr_text": line.trocr_text,
                    "corrected_text": line.corrected_text,
                    "llm_confidence": (
                        round(line.llm_confidence, 3)
                        if line.llm_confidence is not None else ""
                    ),
                    "prob_wrong": round(line.prob_wrong, 3),
                    "gt": "",
                })
                n_lines += 1

        n_done += 1
        elapsed = time.monotonic() - t0
        print(f"[{i}/{len(pending)}] {doc_id}  -> {len(result.lines)} lines, "
              f"{elapsed:.1f}s", file=sys.stderr)

    total = time.monotonic() - t_start
    print(
        f"\n[done] {n_done} docs, {n_lines} lines, {n_failed} failed  "
        f"({total:.0f}s)\nCSV: {args.output}\nCrops: {args.crops}/"
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
