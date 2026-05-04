"""Image preprocessing: deskew, binarize, line segmentation.

Default line detector is doctr (robust on cursive); horizontal projection
profile is a fallback for clean printed pages. Returns line image crops with
bounding boxes for downstream OCR.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from deskew import determine_skew
from PIL import Image, ImageOps

# Register HEIF/HEIC support with PIL so iPhone photos load through the PIL
# fallback path. Many iPhone-exported files have a .jpg/.jpeg extension but
# HEIC contents — without this, PIL.Image.open() fails with UnidentifiedImageError.
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass  # pillow-heif is optional; absence just means HEIC files won't load


@dataclass
class LineCrop:
    """A cropped line image plus its bounding box in the original (deskewed) page."""

    image: np.ndarray
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    deskew_angle: float = 0.0


@lru_cache(maxsize=1)
def _load_doctr_detector():
    """Lazy-load doctr's text detector. Heavy on first import (~500 MB model)."""
    from doctr.models import detection_predictor

    return detection_predictor("db_resnet50", pretrained=True, assume_straight_pages=True)


def _read_image(image_path: Path) -> np.ndarray:
    """Read an image as a BGR numpy array.

    cv2.imread is fast but silently returns None on JPEG variants it doesn't
    handle (HEIC-derived files exported by iPhone Photos, unusual ICC profiles,
    progressive JPEGs with non-standard markers). We fall back to PIL, which
    handles those, and also apply EXIF orientation since phone cameras store
    rotation in metadata rather than rotating pixels.
    """
    image = cv2.imread(str(image_path))
    if image is not None:
        return image
    try:
        pil = Image.open(image_path)
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        rgb = np.array(pil)
        # cv2 expects BGR ordering downstream
        return rgb[:, :, ::-1].copy()
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not read image: {image_path} "
            f"(cv2.imread returned None; PIL fallback failed: {exc!r})"
        )


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate skew angle and rotate the image to correct it."""
    gray = _to_grayscale(image)
    angle = determine_skew(gray)
    if angle is None or abs(angle) < 0.1:
        return image, 0.0
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(angle), 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
    )
    return rotated, float(angle)


def _segment_lines_doctr(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Use doctr to find word boxes, then cluster vertically into line bboxes."""
    detector = _load_doctr_detector()
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    result = detector([rgb])
    if not result:
        return []
    page = result[0]
    # doctr returns either a dict with "words" or a numpy array of (N, 5)
    if isinstance(page, dict):
        words = page.get("words", [])
    else:
        words = page
    if len(words) == 0:
        return []
    # Convert normalized [0,1] coords to absolute pixel coords
    abs_boxes: list[tuple[int, int, int, int]] = []
    for box in words:
        x0, y0, x1, y1 = box[:4]
        abs_boxes.append((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
    return _cluster_words_to_lines(abs_boxes)


def _cluster_words_to_lines(
    word_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Group word boxes by vertical center; return one bbox per line."""
    if not word_boxes:
        return []
    word_boxes = sorted(word_boxes, key=lambda b: (b[1] + b[3]) / 2)
    clusters: list[list[tuple[int, int, int, int]]] = [[word_boxes[0]]]
    for box in word_boxes[1:]:
        last_cluster = clusters[-1]
        ref_y = sum((b[1] + b[3]) / 2 for b in last_cluster) / len(last_cluster)
        ref_height = max((b[3] - b[1]) for b in last_cluster)
        yc = (box[1] + box[3]) / 2
        if abs(yc - ref_y) < 0.5 * ref_height:
            last_cluster.append(box)
        else:
            clusters.append([box])

    line_bboxes: list[tuple[int, int, int, int]] = []
    for cluster in clusters:
        x0 = min(b[0] for b in cluster)
        y0 = min(b[1] for b in cluster)
        x1 = max(b[2] for b in cluster)
        y1 = max(b[3] for b in cluster)
        line_bboxes.append((x0, y0, x1 - x0, y1 - y0))
    # Sort top-to-bottom for reading order
    line_bboxes.sort(key=lambda b: b[1])
    return line_bboxes


def _segment_lines_projection(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Fallback line segmentation via horizontal projection profile.

    Works well on clean printed text; less reliable on cursive.
    """
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    h, w = binary.shape
    row_sums = (binary > 0).sum(axis=1)
    ink_threshold = w * 0.01
    in_line = False
    start = 0
    boxes: list[tuple[int, int, int, int]] = []
    min_height = 5
    for y, s in enumerate(row_sums):
        if s > ink_threshold:
            if not in_line:
                start = y
                in_line = True
        else:
            if in_line and y - start > min_height:
                boxes.append((0, start, w, y - start))
            in_line = False
    if in_line and h - start > min_height:
        boxes.append((0, start, w, h - start))
    return boxes


def preprocess(
    image_path: str | Path,
    *,
    use_doctr: bool = True,
    min_line_width: int = 20,
    min_line_height: int = 8,
) -> list[LineCrop]:
    """Load an image, deskew it, segment into line crops in reading order.

    Returns a list of LineCrop objects whose `bbox` is in the deskewed-page
    coordinate frame. Tiny artifacts below the size thresholds are discarded.
    """
    image_path = Path(image_path)
    image = _read_image(image_path)

    deskewed, angle = _deskew(image)
    gray = _to_grayscale(deskewed)

    if use_doctr:
        try:
            line_bboxes = _segment_lines_doctr(deskewed)
            if not line_bboxes:
                print(
                    f"[preprocess] doctr returned no boxes for {image_path}; "
                    "falling back to projection profile",
                    file=sys.stderr,
                )
                line_bboxes = _segment_lines_projection(gray)
        except Exception as e:
            print(
                f"[preprocess] doctr failed ({e!r}); "
                "falling back to projection profile. "
                "Run `python scripts/setup_models.py` if the doctr model file is missing.",
                file=sys.stderr,
            )
            line_bboxes = _segment_lines_projection(gray)
    else:
        line_bboxes = _segment_lines_projection(gray)

    crops: list[LineCrop] = []
    for x, y, w, h in line_bboxes:
        if w < min_line_width or h < min_line_height:
            continue
        crop = deskewed[y : y + h, x : x + w].copy()
        crops.append(LineCrop(image=crop, bbox=(x, y, w, h), deskew_angle=angle))
    return crops
