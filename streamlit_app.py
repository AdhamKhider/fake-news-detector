"""
Fake News Detector - Streamlit App
Deploy for free on Streamlit Community Cloud (share.streamlit.io)
"""
import re
import joblib
import streamlit as st

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


@st.cache_resource
def load_model():
    model = joblib.load("models/fake_news_model.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    return model, vectorizer


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_news(text: str, model, vectorizer):
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    decision_score = model.decision_function(features)[0]
    confidence = 1 / (1 + pow(2.718281828, -abs(decision_score)))
    prediction = model.predict(features)[0]
    return prediction, confidence


model, vectorizer = load_model()

st.title("📰 Fake News Detector")
st.markdown(
    "An NLP model (TF-IDF + Passive Aggressive Classifier) trained on 6,300+ news "
    "articles to detect whether a news text is likely **REAL** or **FAKE**, with "
    "**95% accuracy** on test data."
)
st.caption(
    "⚠️ This analyzes linguistic patterns, not real-time fact-checking. "
    "Use it as a helpful initial indicator, not a replacement for verifying trusted sources."
)

text_input = st.text_area(
    "Paste a news article or headline here:",
    height=200,
    placeholder="e.g. WASHINGTON (Reuters) - The U.S. Senate on Thursday approved a bipartisan spending bill...",
)

col1, col2 = st.columns(2)
with col1:
    if st.button("Try a REAL example"):
        text_input = (
            "WASHINGTON (Reuters) - The U.S. Senate on Thursday approved a bipartisan "
            "spending bill after weeks of negotiations between Republican and Democratic leaders."
        )
        st.session_state["example_text"] = text_input
with col2:
    if st.button("Try a FAKE example"):
        text_input = (
            "BREAKING anonymous government insider reveals shocking secret plot that "
            "mainstream media refuses to report share this before they delete it wake up sheeple"
        )
        st.session_state["example_text"] = text_input

if "example_text" in st.session_state:
    text_input = st.session_state["example_text"]

if st.button("Analyze", type="primary"):
    if not text_input or not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        prediction, confidence = predict_news(text_input, model, vectorizer)
        st.divider()
        if prediction == "REAL":
            st.success(f"✅ **REAL** — {confidence*100:.1f}% confidence")
            st.write("This article appears to be REAL based on its linguistic pattern.")
        else:
            st.error(f"🚨 **FAKE** — {confidence*100:.1f}% confidence")
            st.write(
                "This article appears to be FAKE/misleading based on its linguistic "
                "pattern. We recommend verifying with trusted sources."
            )

st.divider()
st.caption("Model performance: 94.95% accuracy | Trained on the ISOT/Kdnuggets Fake News Dataset (6,335 articles)")
