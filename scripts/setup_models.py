"""Pre-download the model files the pipeline needs at runtime.

Specifically:
  - doctr's db_resnet50 detector. The library's built-in download path hits
    a 308 redirect that urllib doesn't follow, leaving a 0-byte cache file.
    We work around it with curl which follows redirects properly.
  - TrOCR-base-handwritten (encoder + decoder). HuggingFace handles redirects
    correctly so we just trigger the standard download.

Run once after `pip install -r requirements.txt`. Idempotent.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

DOCTR_MODEL_URL = (
    "https://doctr-static.mindee.com/models?id=v0.7.0/db_resnet50-79bd7d70.pt&src=0"
)
DOCTR_CACHE = Path.home() / ".cache" / "doctr" / "models" / "db_resnet50-79bd7d70.pt"
MIN_DOCTR_BYTES = 50 * 1024 * 1024  # ~97 MB on disk; anything below 50 MB is suspect


def _ensure_doctr() -> None:
    DOCTR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if DOCTR_CACHE.exists() and DOCTR_CACHE.stat().st_size > MIN_DOCTR_BYTES:
        print(f"[doctr] cached: {DOCTR_CACHE} ({DOCTR_CACHE.stat().st_size / 1e6:.0f} MB)")
        return
    if not shutil.which("curl"):
        sys.exit(
            "curl not found on PATH. Install curl, or download the doctr model "
            f"manually from {DOCTR_MODEL_URL} to {DOCTR_CACHE}"
        )
    print(f"[doctr] downloading via curl -> {DOCTR_CACHE}")
    result = subprocess.run(
        ["curl", "-fsSL", DOCTR_MODEL_URL, "-o", str(DOCTR_CACHE)],
    )
    if result.returncode != 0:
        sys.exit(f"curl failed with exit code {result.returncode}")
    size = DOCTR_CACHE.stat().st_size
    if size < MIN_DOCTR_BYTES:
        sys.exit(
            f"Downloaded file is suspiciously small ({size} bytes). "
            "Inspect the URL/network and retry."
        )
    print(f"[doctr] ok ({size / 1e6:.0f} MB)")


def _ensure_trocr() -> None:
    """Trigger HuggingFace's downloader for TrOCR by importing the module."""
    print("[trocr] loading microsoft/trocr-base-handwritten (will download on first run)")
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    print("[trocr] ok")


def main() -> int:
    _ensure_doctr()
    _ensure_trocr()
    print("\nAll models ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
