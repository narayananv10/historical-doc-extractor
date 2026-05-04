"""Batch runner: process a folder of images, write catalogue.csv.

One row per document. Sequential processing — the per-doc Claude vision
call dominates wall time and multiprocessing risks PyTorch / HuggingFace
fork hazards for marginal gain at 30-50 docs.

Idempotent: if --output already exists, files already represented in it
are skipped. The CSV is written incrementally (header on first row, append
after each doc) so an interrupted run leaves a usable partial catalogue.

Usage:
    python -m src.batch data/raw/loc/                          # uses ./catalogue.csv
    python -m src.batch data/raw/loc/ -o /tmp/catalogue.csv
    python -m src.batch data/raw/loc/abraham-lincoln-papers --no-api
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from src.ner import Entity
from src.pipeline import DocumentResult, process

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

CSV_COLUMNS = [
    "filename",
    "doc_type",
    "doc_type_confidence",
    "n_lines",
    "n_review_lines",
    "mean_prob_wrong",
    "sender",
    "recipient",
    "signed_date",
    "amount",
    "persons",
    "dates",
    "places",
]


def _walk_images(folder: Path) -> list[Path]:
    """Return image paths sorted, recursive."""
    paths: list[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(p)
    return sorted(paths)


def _existing_filenames(csv_path: Path) -> set[str]:
    """Read filename column from existing CSV (for resume)."""
    if not csv_path.exists():
        return set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["filename"] for row in reader if row.get("filename")}


def _entities_by_label(entities: list[Entity], labels: set[str]) -> list[str]:
    """Unique entity texts where label matches one of `labels`, in first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for e in entities:
        if e.label in labels and e.text not in seen:
            seen.add(e.text)
            out.append(e.text)
    return out


def _first_entity(entities: list[Entity], labels: set[str]) -> str:
    """Text of the first entity matching one of `labels`, or empty string."""
    for e in entities:
        if e.label in labels:
            return e.text
    return ""


def _result_to_row(result: DocumentResult) -> dict[str, str | int | float]:
    e = result.entities
    return {
        "filename": result.image_path.name,
        "doc_type": result.classification.doc_type,
        "doc_type_confidence": round(result.classification.confidence, 3),
        "n_lines": len(result.lines),
        "n_review_lines": result.n_review_lines,
        "mean_prob_wrong": round(result.mean_prob_wrong, 3),
        "sender": _first_entity(e, {"SENDER"}),
        "recipient": _first_entity(e, {"RECIPIENT"}),
        "signed_date": _first_entity(e, {"SIGNED_DATE"}),
        "amount": _first_entity(e, {"AMOUNT"}),
        "persons": "; ".join(_entities_by_label(e, {"PERSON", "REFERENCED_PERSON"})),
        "dates": "; ".join(_entities_by_label(e, {"DATE", "SIGNED_DATE"})),
        "places": "; ".join(_entities_by_label(e, {"GPE", "LOC", "REFERENCED_PLACE"})),
    }


def _write_row(csv_path: Path, row: dict, *, write_header: bool) -> None:
    mode = "w" if write_header else "a"
    with csv_path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Folder of images to process")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("catalogue.csv"),
        help="Output CSV path (default: ./catalogue.csv)",
    )
    parser.add_argument(
        "--no-api", action="store_true",
        help="Skip Claude calls (post-correction, classify, extract); spaCy still runs",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process at most this many files",
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        sys.exit(f"folder not found: {args.folder}")

    paths = _walk_images(args.folder)
    if not paths:
        sys.exit(f"no image files in {args.folder}")
    print(f"[batch] found {len(paths)} images under {args.folder}", file=sys.stderr)

    already = _existing_filenames(args.output)
    if already:
        print(f"[batch] resuming — {len(already)} files already in {args.output}",
              file=sys.stderr)
    pending = [p for p in paths if p.name not in already]
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        print("[batch] nothing to do", file=sys.stderr)
        return 0

    write_header = not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_failed = 0
    t_start = time.monotonic()
    for i, path in enumerate(pending, start=1):
        t0 = time.monotonic()
        try:
            result = process(path, no_api=args.no_api)
        except Exception as exc:
            n_failed += 1
            print(f"[{i}/{len(pending)}] {path.name}  FAILED: {exc!r}",
                  file=sys.stderr)
            continue

        row = _result_to_row(result)
        _write_row(args.output, row, write_header=write_header)
        write_header = False
        n_done += 1
        elapsed = time.monotonic() - t0
        print(
            f"[{i}/{len(pending)}] {path.name}  -> "
            f"{row['doc_type']} ({row['doc_type_confidence']:.2f}), "
            f"{row['n_review_lines']}/{row['n_lines']} flagged, "
            f"{elapsed:.1f}s",
            file=sys.stderr,
        )

    total = time.monotonic() - t_start
    print(
        f"\n[done] wrote {args.output}  "
        f"({n_done} processed, {n_failed} failed, {total:.0f}s total)"
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
