"""
PDF Document Summarizer — Hugging Face version (local model, no API key)

Run with: streamlit run app.py
"""

import streamlit as st
from summarizer import summarize_pdf

st.set_page_config(page_title="PDF Summarizer (Local)", page_icon="📄", layout="centered")

st.title("📄 PDF Document Summarizer")
st.caption("Runs entirely locally using an open-source Hugging Face model — no API key needed.")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Model",
        options=[
            "facebook/bart-large-cnn",
            "sshleifer/distilbart-cnn-12-6",
        ],
        index=0,
        help=(
            "bart-large-cnn: higher quality, slower, larger download (~1.6GB).\n"
            "distilbart-cnn-12-6: faster, smaller, slightly lower quality — "
            "good for quick demos."
        ),
    )
    st.info(
        "First run will download the model (one-time). "
        "Summarization runs on CPU by default, so expect ~10-30s per chunk."
    )

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("Summarize", type="primary"):
        with st.spinner("Reading PDF and generating summary locally... this may take a bit."):
            try:
                result = summarize_pdf(uploaded_file, model_name=model_name)

                st.success(
                    f"Done — processed {result['num_pages']} page(s) "
                    f"in {result['num_chunks']} chunk(s)."
                )

                st.subheader("Summary")
                st.write(result["summary"])

                st.subheader("Key Points")
                for point in result["key_points"]:
                    st.markdown(f"- {point}")

            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Something went wrong: {e}")

st.divider()
st.caption(
    "Tip: works best on text-based PDFs (reports, articles, papers). "
    "Scanned/image-only PDFs aren't supported without OCR. "
    "Model runs locally via Hugging Face transformers — first summary will be slower "
    "while the model loads into memory."
)