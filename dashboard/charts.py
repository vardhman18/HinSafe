"""Display helpers for the HinSafe Streamlit dashboard."""

import html
import re

import pandas as pd
import streamlit as st

BAND_ORDER = ("SAFE", "OFFENSIVE", "HATE SPEECH")

# Escape markdown-sensitive characters before rendering highlighted text.
MARKDOWN_SPECIALS = str.maketrans({
    "*": "&#42;", "_": "&#95;", "#": "&#35;", "`": "&#96;",
    "[": "&#91;", "]": "&#93;", "~": "&#126;", "$": "&#36;",
})


# Single-comment display.

def _normalize_token(token: str) -> str:
    """Match a displayed token to the model's normalized SHAP token."""
    return re.sub(r"[^\w\s]", "", token.lower())


def show_severity(score: int, band: str) -> None:
    """Severity score, progress bar, and a colored band label."""
    st.metric("Severity score", f"{score} / 100")
    st.progress(score / 100)

    if band == "SAFE":
        st.success(band)
    elif band == "OFFENSIVE":
        st.warning(band)
    else:
        st.error(band)


def show_highlighted_comment(
    comment: str,
    flagged_words: dict[str, float],
) -> None:
    """Display the comment with influential words highlighted."""
    normalized_scores = {}
    for word, score in flagged_words.items():
        normalized_word = _normalize_token(word)
        if normalized_word:
            normalized_scores[normalized_word] = max(
                score,
                normalized_scores.get(normalized_word, 0.0),
            )

    max_score = max(normalized_scores.values(), default=0.0)
    if max_score <= 0:
        st.write(comment)
        return

    pieces = []

    for token in comment.split():
        key = _normalize_token(token)
        safe_token = html.escape(token, quote=False).translate(MARKDOWN_SPECIALS)

        if key in normalized_scores:
            strength = normalized_scores[key] / max_score
            alpha = 0.25 + 0.55 * strength
            pieces.append(
                f'<span style="background-color: rgba(220, 50, 50, {alpha:.2f});'
                f' padding: 2px 4px; border-radius: 4px;">{safe_token}</span>'
            )
        else:
            pieces.append(safe_token)

    st.markdown(" ".join(pieces), unsafe_allow_html=True)


def show_flagged_words_chart(flagged_words: dict[str, float]) -> None:
    """Bar chart of SHAP scores for the flagged words."""
    if not flagged_words:
        st.caption("No individual word pushed this comment toward offensive.")
        return

    st.caption("Influence of each word on the offensive prediction (higher = stronger push)")
    chart_data = pd.DataFrame(
        {"SHAP score": list(flagged_words.values())},
        index=list(flagged_words.keys()),
    )
    st.bar_chart(chart_data)


# Batch charts for successfully analyzed rows.

def show_band_counts(ok_rows: pd.DataFrame) -> None:
    """How many comments fell into each severity band."""
    st.caption("Comments per severity band")
    counts = ok_rows["severity_band"].value_counts().reindex(BAND_ORDER, fill_value=0)
    st.bar_chart(counts)


def show_severity_distribution(ok_rows: pd.DataFrame) -> None:
    """Histogram-style view of severity scores."""
    st.caption("Severity score distribution")

    bins = [-1, 19, 39, 59, 79, 100]
    labels = ["0-19", "20-39", "40-59", "60-79", "80-100"]

    counts = pd.cut(
        ok_rows["severity_score"].astype(int), bins=bins, labels=labels
    ).value_counts().sort_index()
    counts.index = counts.index.astype(str)

    st.bar_chart(counts)


def show_category_counts(ok_rows: pd.DataFrame) -> None:
    """How often each category was detected."""
    st.caption("Detected categories")

    categories = (
        ok_rows["categories"]
        .fillna("")
        .astype(str)
        .str.split(", ")
        .explode()
    )
    categories = categories[categories.str.len() > 0]

    if categories.empty:
        st.caption("No categories detected in this batch.")
        return

    st.bar_chart(categories.value_counts())