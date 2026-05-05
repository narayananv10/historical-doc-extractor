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

from PIL import Image, ImageDraw

from src.pipeline import process

# Vertical context above/below the bbox in the labeling crop. Bigger = more
# context for ambiguous segmentation, but a busier image. ~30px ≈ one line of
# typical handwritten text at LoC scan resolution.
CROP_CONTEXT_PADDING = 40
HIGHLIGHT_COLOR = "red"
HIGHLIGHT_WIDTH = 4

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


def _make_contextual_crop(
    page_image: Image.Image,
    bbox: tuple[int, int, int, int],
) -> Image.Image:
    """Crop a vertical band around the bbox (full page width, +/- padding) and
    draw a coloured rectangle highlighting the actual line. The labeler shows
    this so the user can see WHICH line is being labelled even when doctr's
    segmentation is ambiguous (e.g., when the bbox is tall enough to span
    multiple visual lines, or when neighbouring lines bleed into the crop)."""
    x, y, w, h = bbox
    # Vertical context — clamp to page bounds
    top = max(0, y - CROP_CONTEXT_PADDING)
    bottom = min(page_image.height, y + h + CROP_CONTEXT_PADDING)
    # Full page width — gives the user lateral context too
    crop = page_image.crop((0, top, page_image.width, bottom)).copy()

    # Translate the bbox into crop coordinates and draw the highlight
    rect_left = max(0, x)
    rect_right = min(crop.width, x + w)
    rect_top = y - top
    rect_bottom = rect_top + h
    draw = ImageDraw.Draw(crop)
    draw.rectangle(
        [rect_left, rect_top, rect_right, rect_bottom],
        outline=HIGHLIGHT_COLOR,
        width=HIGHLIGHT_WIDTH,
    )
    return crop


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

        # Open the page once for cropping all lines from it
        page_image = Image.open(path).convert("RGB")

        with args.output.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
                write_header = False
            for line in result.lines:
                # Save a contextual crop with the line bbox highlighted in red
                # so the labeler can show "this exact line" within its surroundings.
                try:
                    crop = _make_contextual_crop(page_image, line.bbox)
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
