"""HinSafe Streamlit dashboard."""

import os
from typing import Any

import pandas as pd
import streamlit as st

from api_client import check_health, predict_comment
from charts import (
    show_band_counts,
    show_category_counts,
    show_flagged_words_chart,
    show_highlighted_comment,
    show_severity,
    show_severity_distribution,
)

# Dashboard settings.
DEFAULT_API_URL = os.getenv("HINSAFE_API_URL", "http://localhost:8000")
MAX_COMMENT_LENGTH = 1000
MAX_BATCH_ROWS = 10

EXAMPLES = [
    ("Threat", "dekh lena, main tujhe maar dunga"),
    ("Insult", "tu ek number ka bewakoof hai"),
    ("Friendly", "thank you yaar, tu bahut accha hai"),
    ("Neutral", "kal college mein milte hain"),
    ("Negation", "tu bilkul bekaar nahi hai, tu accha hai"),
]

# Render API settings and dashboard help.
def render_sidebar() -> str:
    with st.sidebar:
        st.header("HinSafe")
        st.caption("Hinglish hate speech and cyberbullying detection")

        api_url = st.text_input("API URL", value=DEFAULT_API_URL).strip().rstrip("/")

        if st.button("Check API connection"):
            health, error = check_health(api_url)
            if error:
                st.error(error)
            elif health is not None and health.get("model_loaded") is True:
                st.success("API online, model loaded")
            else:
                st.warning("API online, but the model is still loading")

        st.divider()
        st.subheader("How to read the results")
        st.markdown(
            "- **Severity score:** the model's confidence that the comment is "
            "offensive, scaled to 0-100\n"
            "- **Categories:** keyword rules, applied only to comments the model flags\n"
            "- **Highlighted words:** the words the model weighted most (SHAP)"
        )

    return api_url


# Store an example comment in the text box.
def set_example(text: str) -> None:
    """Button callback: put an example comment into the text box."""
    st.session_state["comment_input"] = text


# Display the result for one analyzed comment.
def show_single_result(result: dict[str, Any], explain_used: bool) -> None:
    st.divider()
    left, right = st.columns([1, 2])

    with left:
        show_severity(result["severity_score"], result["severity_band"])

    with right:
        label_text = "Offensive" if result["label"] == 1 else "Not offensive"
        categories = result["categories"]

        st.markdown(f"**Label:** {label_text}")
        category_text = ", ".join(categories) if categories else "None detected"
        st.markdown(f"**Categories:** {category_text}")

        if explain_used:
            st.markdown("**Comment with influential words highlighted**")
            show_highlighted_comment(result["comment"], result["flagged_words"])
        else:
            st.caption("Word highlighting was turned off.")

    if explain_used:
        show_flagged_words_chart(result["flagged_words"])


# Render the single-comment analysis tab.
def render_single_tab(api_url: str) -> None:
    st.session_state.setdefault("comment_input", "")

    st.caption("Try an example:")
    example_columns = st.columns(len(EXAMPLES))
    for column, (label, text) in zip(example_columns, EXAMPLES):
        column.button(
            label,
            on_click=set_example,
            args=(text,),
            key=f"example_{label}",
        )

    comment = st.text_area(
        "Hinglish comment",
        key="comment_input",
        height=120,
        max_chars=MAX_COMMENT_LENGTH,
        placeholder="Type or paste a comment...",
    )
    explain = st.checkbox("Highlight influential words (SHAP, slower)", value=True)

    if st.button("Analyze", type="primary"):
        if not comment.strip():
            st.warning("Please enter a comment first.")
            st.session_state["single"] = None
        else:
            with st.spinner("Analyzing..."):
                result, error = predict_comment(api_url, comment.strip(), explain)

            if error:
                st.error(error)
                st.session_state["single"] = None
            else:
                st.session_state["single"] = (result, explain)

    # Keep the last result visible after widget interactions.
    saved = st.session_state.get("single")
    if saved:
        show_single_result(*saved)


# Read a CSV file selected by the user.
def read_uploaded_csv(uploaded: Any) -> pd.DataFrame | None:
    try:
        return pd.read_csv(uploaded)
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as error:
        st.error(f"Could not read the CSV file: {error}")
        return None


# Analyze each usable CSV comment with optional SHAP explanations.
def run_batch(api_url: str, texts: list[Any], explain: bool) -> None:
    health, error = check_health(api_url)
    if error:
        st.error(error)
        return
    if health is None or not health["model_loaded"]:
        st.warning("The API is online, but the model is still loading.")
        return

    rows: list[dict[str, Any]] = []
    skipped = 0
    total = len(texts)
    progress = st.progress(0.0, text="Starting...")

    for position, raw in enumerate(texts, start=1):
        text = "" if pd.isna(raw) else str(raw).strip()[:MAX_COMMENT_LENGTH]

        if text == "":
            skipped += 1
        else:
            result, error = predict_comment(api_url, text, explain=explain)

            if error:
                rows.append(
                    {
                        "comment": text,
                        "label": None,
                        "severity_score": None,
                        "severity_band": "",
                        "categories": "",
                        "flagged_words": "",
                        "error": error,
                    }
                )
            else:
                rows.append(
                    {
                        "comment": text,
                        "label": result["label"],
                        "severity_score": result["severity_score"],
                        "severity_band": result["severity_band"],
                        "categories": ", ".join(result["categories"]),
                        "flagged_words": ", ".join(result["flagged_words"].keys()),
                        "error": "",
                    }
                )

        progress.progress(
            position / total,
            text=f"Processed {position} of {total}",
        )

    progress.empty()

    if not rows:
        st.warning("No usable comments were found in the file.")
        st.session_state.pop("batch_results", None)
        st.session_state.pop("batch_skipped", None)
        return

    st.session_state["batch_results"] = pd.DataFrame(rows).convert_dtypes()
    st.session_state["batch_skipped"] = skipped


# Display batch metrics, charts, results, and download controls.
def show_batch_results(results: pd.DataFrame, skipped: int) -> None:
    ok_rows = results[results["error"] == ""]
    failed = len(results) - len(ok_rows)

    if ok_rows.empty:
        st.error(
            "No comments were analyzed successfully. "
            "See the error column below."
        )
        st.dataframe(results, use_container_width=True)
        return

    offensive = int((ok_rows["label"] == 1).sum())
    share = offensive / len(ok_rows) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Analyzed", len(ok_rows))
    col2.metric("Offensive", offensive)
    col3.metric("Offensive share", f"{share:.1f}%")
    col4.metric("Failed", failed)

    if skipped:
        st.caption(f"{skipped} empty rows were skipped.")

    chart_left, chart_middle, chart_right = st.columns(3)
    with chart_left:
        show_band_counts(ok_rows)
    with chart_middle:
        show_severity_distribution(ok_rows)
    with chart_right:
        show_category_counts(ok_rows)

    st.subheader("Detailed results")
    st.dataframe(results, use_container_width=True)

    st.download_button(
        "Download results as CSV",
        data=results.to_csv(index=False).encode("utf-8-sig"),   # utf-8-sig opens cleanly in Excel
        file_name="hinsafe_batch_results.csv",
        mime="text/csv",
    )


# Render the CSV batch-analysis tab.
def render_batch_tab(api_url: str) -> None:
    st.write(
        "Upload a CSV file with a column named **text**. "
        "SHAP explanations make batch analysis slower."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is not None:
        df_input = read_uploaded_csv(uploaded)

        if df_input is not None:
            if "text" not in df_input.columns:
                columns = ", ".join(map(str, df_input.columns))
                st.error(f"The CSV needs a column named 'text'. Found: {columns}")
            else:
                st.write(f"{len(df_input)} rows found.")
                if len(df_input) > MAX_BATCH_ROWS:
                    st.warning(
                        f"Only the first {MAX_BATCH_ROWS} rows will be analyzed."
                    )

                explain = st.checkbox(
                    "Include SHAP explanations (slower)",
                    help="Adds the influential words to the batch results.",
                )
                if st.button("Run batch analysis", type="primary"):
                    run_batch(
                        api_url,
                        df_input["text"].head(MAX_BATCH_ROWS).tolist(),
                        explain,
                    )

    results = st.session_state.get("batch_results")
    if results is not None:
        show_batch_results(results, st.session_state.get("batch_skipped", 0))


# Render the complete HinSafe dashboard.
def main() -> None:
    st.set_page_config(page_title="HinSafe", layout="wide")

    api_url = render_sidebar()

    st.title("HinSafe")
    st.write(
        "Detect hate speech and cyberbullying in Hinglish (mixed Hindi and English) "
        "comments, with an explanation of which words drove the result."
    )

    tab_single, tab_batch = st.tabs(["Single comment", "Batch analysis"])
    with tab_single:
        render_single_tab(api_url)
    with tab_batch:
        render_batch_tab(api_url)


if __name__ == "__main__":
    main()
