"""TrOCR with per-token logprobs.

Uses microsoft/trocr-base-handwritten. Returns transcribed text plus the raw
per-token logprobs needed by the flagger's feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_ID = "microsoft/trocr-base-handwritten"
MAX_LEN = 128


@dataclass
class Line:
    """A transcribed line with bounding box and per-token logprobs."""

    text: str
    bbox: tuple[int, int, int, int]
    token_logprobs: list[float] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def _load() -> tuple[TrOCRProcessor, VisionEncoderDecoderModel, str]:
    """Load TrOCR processor + model once. Cached at module level."""
    device = _pick_device()
    processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return processor, model, device


def _to_pil(image: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    if image.shape[2] == 3:
        # OpenCV is BGR; flip to RGB
        return Image.fromarray(image[..., ::-1]).convert("RGB")
    return Image.fromarray(image).convert("RGB")


def transcribe_one(
    line_image: np.ndarray | Image.Image,
) -> tuple[str, list[float], list[int]]:
    """Transcribe a single line image; return (text, per-token logprobs, token ids).

    Logprobs are over the *generated* tokens (excluding the BOS); EOS is not
    included in the returned lists.
    """
    processor, model, device = _load()
    pil = _to_pil(line_image)
    pixel_values = processor(images=pil, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            output_scores=True,
            return_dict_in_generate=True,
            max_length=MAX_LEN,
        )

    text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

    eos_id = processor.tokenizer.eos_token_id
    generated_ids = outputs.sequences[0, 1:]  # skip the start token
    token_logprobs: list[float] = []
    token_ids: list[int] = []
    for step, score in enumerate(outputs.scores):
        log_probs = F.log_softmax(score[0], dim=-1)
        tid = int(generated_ids[step].item())
        if tid == eos_id:
            break
        token_logprobs.append(float(log_probs[tid].item()))
        token_ids.append(tid)

    return text, token_logprobs, token_ids


def transcribe(line_crops: Sequence) -> list[Line]:
    """Transcribe a sequence of LineCrop objects produced by preprocess.preprocess.

    Loops sequentially; for the demo set sizes (~30-50 docs * ~30 lines) the
    Claude vision post-correction call dominates total wall time anyway.
    """
    out: list[Line] = []
    for crop in line_crops:
        text, logprobs, token_ids = transcribe_one(crop.image)
        out.append(
            Line(
                text=text,
                bbox=crop.bbox,
                token_logprobs=logprobs,
                token_ids=token_ids,
            )
        )
    return out
