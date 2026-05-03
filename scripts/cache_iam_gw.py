"""Cache TrOCR + Claude vision post-correction over the IAM-HistDB GW subset.

Writes data/parquet_cache/iam_gw_pipeline.parquet — the training set for the
flagger. One row per ground-truth line. Columns include TrOCR confidence
features, TrOCR/post-correction agreement features, the ground truth, and the
binary label `is_still_wrong = (CER(corrected, gt) > 0)`.

Strategy:
  - Use the GW dataset's pre-segmented line images for TrOCR (one call/line).
    This guarantees 1:1 alignment with the dataset's line-level ground truth.
  - Use the page image for the Claude vision post-correction call (one
    call/page). The model sees the same visual context a real production run
    would see.
  - Idempotent: skips pages whose rows are already in the output parquet.

Dataset layout assumed (adjust _discover_gw if your extraction differs):
    {root}/data/line_images_normalized/{line_id}.png
    {root}/data/pages/{page_id}.png   (or pages_normalized, page_images)
    {root}/ground_truth/transcription.txt   ("line_id word|word|word ...")

Run:
    python scripts/cache_iam_gw.py --root data/raw/iam_gw/washingtondb-v1.0
    python scripts/cache_iam_gw.py --dry-run        # just discover & print
    python scripts/cache_iam_gw.py --limit 2        # process first 2 pages
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Allow running as `python scripts/cache_iam_gw.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from jiwer import cer
from PIL import Image
from rapidfuzz.distance import Levenshtein

from src.ocr_trocr import Line, transcribe_one
from src.postcorrect import post_correct

DEFAULT_GW_ROOT = Path("data/raw/iam_gw/washingtondb-v1.0")
DEFAULT_OUTPUT = Path("data/parquet_cache/iam_gw_pipeline.parquet")

# GW transcriptions encode special characters as "s_xx" tokens. This map covers
# the common ones; extend if you hit unmapped tokens (you'll see them verbatim
# in the gt column with an `s_` prefix).
GW_SPECIAL_CHARS = {
    "s_pt": ".", "s_cm": ",", "s_mi": "-", "s_qo": "'", "s_qc": "'",
    "s_qt": '"', "s_sq": ";", "s_cl": ":", "s_GW": "G.W.", "s_lb": "£",
    "s_bsl": "/", "s_et": "&", "s_br": ")", "s_bl": "(", "s_pl": "+",
    "s_eq": "=", "s_no": "No.", "s_pa": ",", "s_5s": "5s", "s_qu": "?",
    "s_excl": "!", "s_s": "s",
}

# Candidate page-image directories (different versions of the GW release name
# this differently). First hit wins.
PAGE_DIR_CANDIDATES = ["pages", "page_images", "page_images_normalized"]
LINE_DIR_CANDIDATES = ["line_images_normalized", "lines", "line_images"]
PAGE_EXT_CANDIDATES = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]


@dataclass
class GWDoc:
    page_id: str
    page_image: Path
    lines: list[tuple[str, Path, str]]  # (line_id, line_image_path, gt_text)


def _decode_gw_word(word: str) -> str:
    chars = word.split("-")
    return "".join(GW_SPECIAL_CHARS.get(c, c) for c in chars)


def _decode_gw_transcription(text_part: str) -> str:
    return " ".join(_decode_gw_word(w) for w in text_part.split("|"))


def _parse_transcription(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split(" ", 1)
        if len(parts) != 2:
            continue
        line_id, text_part = parts
        out[line_id] = _decode_gw_transcription(text_part)
    return out


def _resolve_dir(root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        cand = root / "data" / name
        if cand.is_dir():
            return cand
    return None


def _resolve_image(directory: Path, stem: str) -> Path | None:
    for ext in PAGE_EXT_CANDIDATES:
        cand = directory / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def _discover_gw(root: Path) -> list[GWDoc]:
    transcription = root / "ground_truth" / "transcription.txt"
    if not transcription.exists():
        sys.exit(
            f"transcription file not found: {transcription}\n"
            f"Pass --root pointing at your extracted GW dataset, or adjust "
            f"DEFAULT_GW_ROOT in this file."
        )

    page_dir = _resolve_dir(root, PAGE_DIR_CANDIDATES)
    line_dir = _resolve_dir(root, LINE_DIR_CANDIDATES)
    if page_dir is None:
        sys.exit(f"No page-image directory found under {root / 'data'} "
                 f"(tried: {PAGE_DIR_CANDIDATES})")
    if line_dir is None:
        sys.exit(f"No line-image directory found under {root / 'data'} "
                 f"(tried: {LINE_DIR_CANDIDATES})")

    print(f"[discover] page images: {page_dir}", file=sys.stderr)
    print(f"[discover] line images: {line_dir}", file=sys.stderr)

    gt = _parse_transcription(transcription)
    print(f"[discover] {len(gt)} ground-truth lines", file=sys.stderr)

    pages: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for line_id, text in gt.items():
        page_id = line_id.split("-")[0]
        pages[page_id].append((line_id, text))

    docs: list[GWDoc] = []
    for page_id in sorted(pages.keys()):
        page_image = _resolve_image(page_dir, page_id)
        if page_image is None:
            print(f"[skip page {page_id}] no image found", file=sys.stderr)
            continue
        lines = sorted(pages[page_id], key=lambda t: t[0])
        line_paths: list[tuple[str, Path, str]] = []
        for line_id, gt_text in lines:
            line_img = _resolve_image(line_dir, line_id)
            if line_img is None:
                print(f"[skip line {line_id}] no image", file=sys.stderr)
                continue
            line_paths.append((line_id, line_img, gt_text))
        if line_paths:
            docs.append(GWDoc(page_id=page_id, page_image=page_image, lines=line_paths))

    print(f"[discover] {len(docs)} pages with at least one usable line",
          file=sys.stderr)
    return docs


def _logprob_features(token_logprobs: list[float]) -> dict:
    if not token_logprobs:
        return {
            "n_tokens": 0,
            "mean_logprob": 0.0,
            "min_logprob": 0.0,
            "std_logprob": 0.0,
            "length_normalized_logprob": 0.0,
        }
    arr = np.asarray(token_logprobs, dtype=np.float64)
    return {
        "n_tokens": int(arr.size),
        "mean_logprob": float(arr.mean()),
        "min_logprob": float(arr.min()),
        "std_logprob": float(arr.std()),
        "length_normalized_logprob": float(arr.sum() / max(arr.size, 1)),
    }


def _agreement_features(trocr_text: str, corrected_text: str) -> dict:
    distance = Levenshtein.distance(trocr_text, corrected_text)
    base_len = max(len(trocr_text), 1)
    return {
        "edit_distance_trocr_vs_corrected": int(distance),
        "n_chars_changed": int(distance),
        "frac_chars_changed": float(distance / base_len),
    }


def _process_doc(doc: GWDoc, *, no_api: bool) -> list[dict]:
    print(f"[page {doc.page_id}] {len(doc.lines)} lines", file=sys.stderr)

    trocr_lines: list[Line] = []
    for line_id, line_image_path, _gt in doc.lines:
        img = np.array(Image.open(line_image_path).convert("RGB"))
        text, logprobs, token_ids = transcribe_one(img)
        trocr_lines.append(
            Line(
                text=text,
                bbox=(0, 0, img.shape[1], img.shape[0]),
                token_logprobs=logprobs,
                token_ids=token_ids,
            )
        )

    corrected = post_correct(doc.page_image, trocr_lines, no_api=no_api)

    rows: list[dict] = []
    for (line_id, _path, gt_text), trocr_line, corrected_line in zip(
        doc.lines, trocr_lines, corrected
    ):
        gt_cer = cer(reference=gt_text, hypothesis=corrected_line.corrected)
        row = {
            "doc_id": doc.page_id,
            "line_id": line_id,
            "trocr_text": trocr_line.text,
            "trocr_token_logprobs": list(trocr_line.token_logprobs),
            **_logprob_features(trocr_line.token_logprobs),
            "corrected_text": corrected_line.corrected,
            "llm_confidence": corrected_line.llm_confidence,
            "changed": corrected_line.changed,
            **_agreement_features(trocr_line.text, corrected_line.corrected),
            "line_height_px": int(trocr_line.bbox[3]),
            "line_width_px": int(trocr_line.bbox[2]),
            "gt": gt_text,
            "gt_cer": float(gt_cer),
            "is_still_wrong": bool(gt_cer > 0),
        }
        rows.append(row)

    return rows


def _existing_doc_ids(parquet_path: Path) -> set[str]:
    if not parquet_path.exists():
        return set()
    table = pq.read_table(parquet_path, columns=["doc_id"])
    return set(table["doc_id"].to_pylist())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_GW_ROOT,
                        help="Path to extracted GW dataset")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Parquet path to write")
    parser.add_argument("--no-api", action="store_true",
                        help="Skip Claude post-correction (dev mode; the resulting "
                             "parquet is useless for training the flagger)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many pages (testing only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover the dataset and exit without running the pipeline")
    args = parser.parse_args()

    docs = _discover_gw(args.root)
    if args.dry_run:
        for doc in docs[: (args.limit or len(docs))]:
            print(f"  {doc.page_id}: {len(doc.lines)} lines, "
                  f"page={doc.page_image.name}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    already_done = _existing_doc_ids(args.output)
    if already_done:
        print(f"[resume] skipping {len(already_done)} pages already in {args.output}",
              file=sys.stderr)

    pending = [d for d in docs if d.page_id not in already_done]
    if args.limit is not None:
        pending = pending[: args.limit]

    if not pending:
        print("[done] nothing to do", file=sys.stderr)
        return 0

    existing_rows: list[dict] = []
    if args.output.exists():
        existing_rows = pq.read_table(args.output).to_pylist()

    new_rows: list[dict] = []
    for i, doc in enumerate(pending, start=1):
        try:
            rows = _process_doc(doc, no_api=args.no_api)
        except Exception as exc:
            print(f"[error page {doc.page_id}] {exc!r}", file=sys.stderr)
            continue
        new_rows.extend(rows)
        df = pd.DataFrame(existing_rows + new_rows)
        df.to_parquet(args.output, index=False)
        print(f"[saved] {i}/{len(pending)}  rows total: {len(df)}",
              file=sys.stderr)

    print(f"\n[done] wrote {args.output}  ({len(existing_rows) + len(new_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
