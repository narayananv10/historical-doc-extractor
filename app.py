"""Streamlit demo: upload a scan, see TrOCR + Claude post-correction +
flagger + classify + NER all in one page.

Run from the repo root:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from src.batch import _entities_by_label, _first_entity
from src.flagger import describe
from src.pipeline import DocumentResult, process

# Hosted-mode detection: HF Spaces sets SPACE_ID automatically; HOSTED_DEMO=1
# is the manual override for any other host (Streamlit Cloud, Render, etc.).
# When hosted, the --no-api toggle defaults ON to keep API spend bounded against
# uncontrolled visitor uploads, and a banner explains the trade-off.
HOSTED_DEMO = bool(os.environ.get("SPACE_ID") or os.environ.get("HOSTED_DEMO"))
GITHUB_URL = "https://github.com/narayananv10/historical-doc-extractor"

st.set_page_config(
    page_title="Historical Document Extractor",
    layout="wide",
    page_icon="📜",
)


def _bbox_color(prob: float) -> tuple[int, int, int]:
    """Map prob in [0, 1] to an RGB tuple: green -> yellow -> red."""
    p = max(0.0, min(1.0, prob))
    if p < 0.5:
        return int(p * 2 * 255), 255, 0
    return 255, int((1 - p) * 2 * 255), 0


def _annotate(image_path: Path, lines, threshold: float) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for line in lines:
        x, y, w, h = line.bbox
        color = _bbox_color(line.prob_wrong)
        width = 4 if line.prob_wrong >= threshold else 2
        draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
    return img


def _render_image_tab(result: DocumentResult, image_path: Path, threshold: float):
    st.subheader("Original scan with line bounding boxes")
    st.caption(
        "Box colour: green = low `prob_wrong`, red = high. "
        "Box width: thicker = flagged at the current threshold."
    )
    st.image(_annotate(image_path, result.lines, threshold), use_container_width=True)


def _render_transcription_tab(
    result: DocumentResult, image_path: Path, threshold: float
):
    if result.no_api:
        st.warning(
            "⚠ TrOCR raw output (Claude post-correction skipped — every line "
            "auto-flagged for review below)"
        )
    else:
        st.caption("Corrected transcript — TrOCR + Claude vision post-correction")
    st.markdown(result.full_text)
    st.download_button(
        "Download transcript (.txt)",
        result.full_text,
        file_name=f"{image_path.stem}.txt",
        mime="text/plain",
    )

    st.divider()

    st.subheader(f"Per-line breakdown ({len(result.lines)} lines)")
    df = pd.DataFrame(
        [
            {
                "line": line.line_id,
                "prob_wrong": round(line.prob_wrong, 3),
                "flagged": "🚩" if line.prob_wrong >= threshold else "",
                "TrOCR raw": line.trocr_text,
                "Corrected": line.corrected_text,
                "changed": "✓" if line.changed else "",
                "llm_conf": (
                    round(line.llm_confidence, 2)
                    if line.llm_confidence is not None
                    else None
                ),
            }
            for line in result.lines
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_structured_tab(result: DocumentResult, image_path: Path):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            "Document type",
            result.classification.doc_type,
            f"confidence {result.classification.confidence:.2f}",
        )
    with col2:
        if result.classification.reasoning:
            st.markdown("**Reasoning**")
            st.info(result.classification.reasoning)

    st.subheader(f"Entities ({len(result.entities)})")
    if result.entities:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "label": e.label,
                        "text": e.text,
                        "source": e.source,
                        "confidence": (
                            round(e.confidence, 2) if e.confidence is not None else None
                        ),
                    }
                    for e in result.entities
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No entities extracted.")

    json_str = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    st.download_button(
        "Download as JSON",
        json_str,
        file_name=f"{image_path.stem}.json",
        mime="application/json",
    )


def _summary_narrative(result: DocumentResult, threshold: float) -> str:
    """Compose a 1-3 sentence prose summary of the pipeline run."""
    n = len(result.lines)
    n_flagged = sum(1 for line in result.lines if line.prob_wrong >= threshold)
    pct = (n_flagged / n) if n else 0.0

    if result.no_api:
        return (
            f"Document type was not classified (`--no-api` mode). "
            f"Pipeline transcribed {n} lines; **all {n} flagged for human review** "
            f"because Claude post-correction was skipped — the transcription is "
            f"unverified TrOCR output."
        )

    parts: list[str] = [
        f"This document was classified as a **{result.classification.doc_type}** "
        f"(confidence {result.classification.confidence:.0%})."
    ]
    sender = _first_entity(result.entities, {"SENDER"})
    recipient = _first_entity(result.entities, {"RECIPIENT"})
    if sender or recipient:
        parts.append(
            f"Sender: {sender or '—'}. Recipient: {recipient or '—'}."
        )
    signed_date = _first_entity(result.entities, {"SIGNED_DATE"})
    if signed_date:
        parts.append(f"Dated {signed_date}.")
    parts.append(
        f"Pipeline transcribed {n} lines, of which {n_flagged} ({pct:.0%}) "
        f"need human review at the current threshold."
    )
    return " ".join(parts)


def _render_summary_tab(result: DocumentResult, threshold: float):
    st.markdown(_summary_narrative(result, threshold))

    st.divider()

    st.subheader("Key fields")
    cols = st.columns(2)
    cols[0].metric("Sender", _first_entity(result.entities, {"SENDER"}) or "—")
    cols[0].metric(
        "Signed date",
        _first_entity(result.entities, {"SIGNED_DATE"}) or "—",
    )
    cols[1].metric(
        "Recipient", _first_entity(result.entities, {"RECIPIENT"}) or "—"
    )
    cols[1].metric("Amount", _first_entity(result.entities, {"AMOUNT"}) or "—")

    persons = _entities_by_label(
        result.entities, {"PERSON", "REFERENCED_PERSON"}
    )
    places = _entities_by_label(
        result.entities, {"GPE", "LOC", "REFERENCED_PLACE"}
    )
    if persons or places:
        st.markdown("**Other named entities**")
        if persons:
            st.markdown(f"- People: {', '.join(persons)}")
        if places:
            st.markdown(f"- Places: {', '.join(places)}")

    st.divider()

    st.subheader("Pipeline quality")
    qcols = st.columns(3)
    qcols[0].metric("Lines", len(result.lines))
    qcols[1].metric(
        "Flagged @ threshold",
        sum(1 for line in result.lines if line.prob_wrong >= threshold),
    )
    qcols[2].metric("Mean prob_wrong", f"{result.mean_prob_wrong:.2f}")

    reason_counts: Counter[str] = Counter()
    for line in result.lines:
        if line.prob_wrong >= threshold:
            for r in line.reasons:
                reason_counts[r] += 1
    if reason_counts:
        st.subheader("Why lines were flagged")
        for reason, count in reason_counts.most_common(5):
            st.markdown(f"- **{count}×** {describe(reason)}")


def _render_review_tab(result: DocumentResult, image_path: Path, threshold: float):
    flagged = [line for line in result.lines if line.prob_wrong >= threshold]
    st.subheader(f"{len(flagged)} flagged for review (threshold = {threshold:.2f})")

    if not flagged:
        st.success("Nothing flagged at this threshold. Lower the threshold to see more.")
        return

    original = Image.open(image_path).convert("RGB")
    for line in flagged:
        with st.expander(
            f"Line {line.line_id}  —  prob_wrong = {line.prob_wrong:.2f}",
            expanded=True,
        ):
            col_img, col_text = st.columns([1, 2])
            with col_img:
                x, y, w, h = line.bbox
                if w > 0 and h > 0:
                    crop = original.crop((x, y, x + w, y + h))
                    st.image(crop, use_container_width=True)
                else:
                    st.caption("(no bbox crop)")
            with col_text:
                st.markdown(f"**TrOCR raw:** {line.trocr_text}")
                st.markdown(f"**Corrected:** {line.corrected_text}")
                if line.llm_confidence is not None:
                    st.caption(f"LLM confidence: {line.llm_confidence:.2f}")
                if line.reasons:
                    st.markdown("**Why flagged:**")
                    for r in line.reasons:
                        st.markdown(f"- {describe(r)}")


def main() -> None:
    st.title("📜 Historical Document Extractor")
    st.caption(
        "Upload a handwritten or printed scan. Pipeline: preprocess → TrOCR → "
        "Claude vision post-correction → flagger → classify → NER. "
        "Confidence-aware review queue with per-line probabilities and reason codes."
    )

    if HOSTED_DEMO:
        st.info(
            "**Hosted demo** — `Skip Claude API` is on by default to control API costs. "
            "TrOCR + spaCy NER still run end-to-end and every line is flagged for review. "
            "For the full pipeline (Claude vision post-correction + classify + custom-entity "
            "extraction), clone the repo and add your own `ANTHROPIC_API_KEY`.",
            icon="ℹ️",
        )

    with st.sidebar:
        st.header("Input")
        uploaded = st.file_uploader(
            "Upload a scan",
            type=["jpg", "jpeg", "png", "tif", "tiff", "webp", "heic", "heif"],
        )
        no_api = st.toggle(
            "Skip Claude API",
            value=HOSTED_DEMO,
            help=(
                "Skip post-correction, classification, and Claude entity "
                "extraction. spaCy NER still runs. The flagger falls back to "
                "rule-based mode (no learned-model context). Defaults ON in "
                "hosted demos to control API spend."
            ),
        )
        threshold = st.slider(
            "Flagger threshold",
            0.0, 1.0, 0.5, 0.01,
            help="Lines with `prob_wrong` above this are flagged.",
        )
        run = st.button("Process", type="primary", disabled=not uploaded)

        st.divider()
        st.markdown(
            f"[![View source on GitHub]"
            f"(https://img.shields.io/badge/GitHub-View_Source-181717?logo=github&style=for-the-badge)]"
            f"({GITHUB_URL})"
        )

    if run and uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.read())
            temp_path = Path(tmp.name)
        with st.spinner("Running pipeline... (~30-60s; longer on first call due to model load)"):
            result = process(temp_path, no_api=no_api)
        st.session_state["result"] = result
        st.session_state["image_path"] = temp_path
        st.session_state["filename"] = uploaded.name

    if "result" not in st.session_state:
        st.info("Upload a scan in the sidebar, then click **Process** to begin.")
        return

    result: DocumentResult = st.session_state["result"]
    image_path: Path = st.session_state["image_path"]

    st.markdown(f"**File:** `{st.session_state.get('filename', image_path.name)}`")
    summary_cols = st.columns(4)
    summary_cols[0].metric("lines", len(result.lines))
    summary_cols[1].metric(
        "flagged @ threshold",
        sum(1 for line in result.lines if line.prob_wrong >= threshold),
    )
    summary_cols[2].metric("doc type", result.classification.doc_type)
    summary_cols[3].metric("entities", len(result.entities))

    tab_image, tab_transcription, tab_structured, tab_summary, tab_review = st.tabs(
        ["Image", "Transcription", "Structured", "Summary", "Review queue"]
    )
    with tab_image:
        _render_image_tab(result, image_path, threshold)
    with tab_transcription:
        _render_transcription_tab(result, image_path, threshold)
    with tab_structured:
        _render_structured_tab(result, image_path)
    with tab_summary:
        _render_summary_tab(result, threshold)
    with tab_review:
        _render_review_tab(result, image_path, threshold)


if __name__ == "__main__":
    main()
