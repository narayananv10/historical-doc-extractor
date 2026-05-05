"""Streamlit labeling UI for the LoC out-of-distribution holdout.

Reads data/hand_labeled/loc_ood.csv (produced by extract_for_ood_labeling.py),
shows one line at a time with the actual handwriting crop alongside the
TrOCR + Claude-corrected text, lets the user type the true ground truth,
and persists after every action.

Resumable: skips rows that already have a non-empty `gt` value, so you can
quit + restart and pick up where you left off.

Run locally (NOT deployed to HF Spaces):
  streamlit run scripts/label_for_ood.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running via `streamlit run scripts/label_for_ood.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from PIL import Image

CSV_PATH = Path("data/hand_labeled/loc_ood.csv")
CROPS_DIR = Path("data/hand_labeled/crops")
SKIP_SENTINEL = "[SKIPPED]"

st.set_page_config(
    page_title="OOD Labeling — Historical Doc Extractor",
    layout="wide",
    page_icon="🏷️",
)


def _load_csv() -> pd.DataFrame:
    if not CSV_PATH.exists():
        st.error(
            f"CSV not found: `{CSV_PATH}`. Run "
            "`python scripts/extract_for_ood_labeling.py` first to generate it."
        )
        st.stop()
    df = pd.read_csv(CSV_PATH, dtype={"gt": str}).fillna({"gt": ""})
    return df


def _save_csv(df: pd.DataFrame) -> None:
    df.to_csv(CSV_PATH, index=False)


def _is_unlabelled(value: str) -> bool:
    return value == "" or value is None or pd.isna(value)


def _next_unlabelled_index(df: pd.DataFrame, start_from: int = 0) -> int | None:
    for i in range(start_from, len(df)):
        if _is_unlabelled(df.iloc[i]["gt"]):
            return i
    # wrap around
    for i in range(0, start_from):
        if _is_unlabelled(df.iloc[i]["gt"]):
            return i
    return None


def main() -> None:
    df = _load_csv()
    n_total = len(df)
    n_labelled = (~df["gt"].apply(_is_unlabelled)).sum()
    n_skipped = (df["gt"] == SKIP_SENTINEL).sum()
    n_actual = n_labelled - n_skipped

    st.title("🏷️ OOD line labeling")
    st.caption(
        "Label the LoC holdout for out-of-distribution flagger validation. "
        "Edit the text box to the true ground-truth transcription of the line shown, "
        "then click **Save & next**. Progress persists after every save."
    )

    # Sidebar: progress + nav
    with st.sidebar:
        st.header("Progress")
        st.progress(n_labelled / max(n_total, 1))
        st.metric("Labelled", f"{n_labelled} / {n_total}")
        st.metric("Skipped (unreadable)", n_skipped)
        st.metric("Real labels", n_actual)

        st.divider()
        st.subheader("Filter by document")
        doc_ids = ["(all)"] + sorted(df["doc_id"].astype(str).unique().tolist())
        doc_filter = st.selectbox("Document", doc_ids, index=0)

        st.divider()
        if n_labelled == n_total:
            st.success("All rows labelled! 🎉")

    # Determine which rows are eligible based on filter
    if doc_filter == "(all)":
        eligible = df
    else:
        eligible = df[df["doc_id"].astype(str) == doc_filter]

    eligible_unlabelled_idx = [
        i for i in eligible.index if _is_unlabelled(df.iloc[i]["gt"])
    ]

    if not eligible_unlabelled_idx:
        st.success(
            f"No more unlabelled rows for `{doc_filter}`. "
            "Pick another document in the sidebar, or you're done."
        )
        with st.expander("Preview first 30 labelled rows"):
            st.dataframe(
                df.head(30)[["doc_id", "line_id", "corrected_text", "gt"]],
                use_container_width=True,
            )
        return

    # Track current row in session state so re-runs (e.g. typing) don't reset it
    if "current_idx" not in st.session_state or st.session_state["current_idx"] not in eligible_unlabelled_idx:
        st.session_state["current_idx"] = eligible_unlabelled_idx[0]

    idx = st.session_state["current_idx"]
    row = df.iloc[idx]

    # Header row with metadata
    h1, h2, h3 = st.columns([2, 1, 1])
    h1.markdown(f"**Document:** `{row['doc_id']}`")
    h2.markdown(f"**Line:** `{int(row['line_id']):03d}`")
    h3.markdown(f"**prob_wrong:** `{row['prob_wrong']:.2f}`")

    # Line image
    crop_path = CROPS_DIR / f"{row['doc_id']}-{int(row['line_id']):03d}.png"
    if crop_path.exists():
        st.image(str(crop_path), use_container_width=True)
    else:
        st.warning(f"Line image not found at `{crop_path}`. Use the texts below.")

    # Read-only context: TrOCR + corrected (so user knows what the model said)
    cc1, cc2 = st.columns(2)
    cc1.markdown("**TrOCR raw**")
    cc1.code(row["trocr_text"] or "(empty)", language=None)
    cc2.markdown("**Claude-corrected**")
    cc2.code(row["corrected_text"] or "(empty)", language=None)

    st.divider()

    # The actual labeling input — pre-populated with the corrected text so easy
    # lines are one Cmd+Enter
    gt_input = st.text_area(
        "Ground truth (edit to match what's actually written in the line image above)",
        value=row["corrected_text"] or "",
        height=80,
        key=f"gt_input_{idx}",
        help="Preserve original spelling, capitalisation, and punctuation as written. "
             "Use [illegible] for unreadable spans. Press Save & next to commit.",
    )

    # Action buttons
    b1, b2, b3, _ = st.columns([1, 1, 1, 3])

    if b1.button("💾 Save & next", type="primary", use_container_width=True):
        df.at[idx, "gt"] = gt_input.strip()
        _save_csv(df)
        # advance
        next_idx = _next_unlabelled_index(df, start_from=idx + 1)
        if next_idx is not None:
            st.session_state["current_idx"] = next_idx
        st.rerun()

    if b2.button("⏭️ Skip / unreadable", use_container_width=True):
        df.at[idx, "gt"] = SKIP_SENTINEL
        _save_csv(df)
        next_idx = _next_unlabelled_index(df, start_from=idx + 1)
        if next_idx is not None:
            st.session_state["current_idx"] = next_idx
        st.rerun()

    if b3.button("◀ Previous", use_container_width=True):
        # find the previous index (labelled or not) within eligible
        prev_candidates = [i for i in eligible.index if i < idx]
        if prev_candidates:
            st.session_state["current_idx"] = prev_candidates[-1]
            st.rerun()


if __name__ == "__main__":
    main()
