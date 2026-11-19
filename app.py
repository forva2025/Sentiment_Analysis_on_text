from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from nltk.tokenize import wordpunct_tokenize


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "sentiment_model.joblib"


st.set_page_config(
    page_title="Sentiment Studio",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    """Load the trained classifier once per Streamlit process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The model file 'sentiment_model.joblib' was not found next to app.py."
        )
    return joblib.load(MODEL_PATH)


def extract_features(text: str) -> tuple[dict[str, bool], list[str]]:
    """Tokenize text in a way that resembles the NLTK movie-review corpus."""
    tokens = wordpunct_tokenize(text)
    return {token: True for token in tokens}, tokens


def analyze(text: str, classifier) -> dict:
    features, tokens = extract_features(text)
    distribution = classifier.prob_classify(features)
    probabilities = {
        label: float(distribution.prob(label)) for label in classifier.labels()
    }
    prediction = max(probabilities, key=probabilities.get)
    return {
        "prediction": prediction,
        "confidence": probabilities[prediction],
        "probabilities": probabilities,
        "tokens": tokens,
        "unique_tokens": len(features),
    }


def friendly_label(label: str) -> str:
    return {"pos": "Positive", "neg": "Negative"}.get(label, label.title())


def render_result(result: dict) -> None:
    positive = result["prediction"] == "pos"
    color = "#38d996" if positive else "#ff6b7a"
    icon = "😊" if positive else "😕"

    st.markdown(
        f"""
        <div class="result-card" style="--accent:{color}">
            <div class="result-icon">{icon}</div>
            <div>
                <div class="eyebrow">MODEL VERDICT</div>
                <div class="result-title">{friendly_label(result['prediction'])}</div>
                <div class="result-subtitle">
                    {result['confidence']:.1%} model confidence
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    probability_rows = [
        {
            "Sentiment": friendly_label(label),
            "Probability": probability,
        }
        for label, probability in sorted(
            result["probabilities"].items(), key=lambda item: item[1], reverse=True
        )
    ]
    chart_data = pd.DataFrame(probability_rows).set_index("Sentiment")
    st.bar_chart(chart_data, horizontal=True, color="#7c6cff", height=210)


st.markdown(
    """
    <style>
        :root {
            --ink: #f7f7fb;
            --muted: #a9abc0;
            --panel: rgba(28, 29, 48, .78);
            --line: rgba(255, 255, 255, .09);
        }
        .stApp {
            background:
                radial-gradient(circle at 12% 8%, rgba(124,108,255,.19), transparent 30rem),
                radial-gradient(circle at 90% 18%, rgba(56,217,150,.10), transparent 28rem),
                #0c0d18;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: rgba(17, 18, 32, .94);
            border-right: 1px solid var(--line);
        }
        .block-container { max-width: 1180px; padding-top: 2.5rem; }
        h1, h2, h3 { letter-spacing: -.025em; }
        .hero-kicker, .eyebrow {
            color: #9b8fff;
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .14em;
        }
        .hero-title {
            color: var(--ink);
            font-size: clamp(2.5rem, 6vw, 4.8rem);
            font-weight: 850;
            letter-spacing: -.055em;
            line-height: .98;
            margin: .55rem 0 1rem;
        }
        .hero-copy {
            color: var(--muted);
            font-size: 1.08rem;
            line-height: 1.7;
            max-width: 720px;
            margin-bottom: 1.7rem;
        }
        .result-card {
            align-items: center;
            background: linear-gradient(135deg, var(--panel), rgba(36,37,61,.55));
            border: 1px solid var(--line);
            border-left: 4px solid var(--accent);
            border-radius: 18px;
            display: flex;
            gap: 1.25rem;
            margin: .35rem 0 1.25rem;
            padding: 1.4rem 1.5rem;
        }
        .result-icon { font-size: 2.7rem; }
        .result-title { color: var(--ink); font-size: 2rem; font-weight: 800; }
        .result-subtitle { color: var(--muted); font-size: .95rem; }
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 15px;
            padding: .9rem 1rem;
        }
        .small-note {
            color: var(--muted);
            font-size: .88rem;
            line-height: 1.55;
        }
        .stButton > button {
            border-radius: 999px;
            min-height: 2.8rem;
        }
        [data-testid="stTextArea"] textarea { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)


try:
    classifier = load_model()
except Exception as error:
    st.error(f"The sentiment model could not be loaded: {error}")
    st.info("Place sentiment_model.joblib in the same folder as app.py, then restart.")
    st.stop()


with st.sidebar:
    st.markdown("## 🎭 Sentiment Studio")
    st.caption("A classic NLP demo powered by NLTK Naive Bayes.")
    st.divider()
    st.markdown("**Model at a glance**")
    st.markdown("Movie-review classifier  ")
    st.markdown("Two outcomes: positive / negative  ")
    st.markdown("Runs locally—no text is sent to an API")
    st.divider()
    st.markdown(
        '<p class="small-note">Best suited to English, review-like text. '
        "Confidence is the model's probability estimate, not a guarantee.</p>",
        unsafe_allow_html=True,
    )


st.markdown('<div class="hero-kicker">INTERACTIVE NLP LAB</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">What mood is<br>hiding in your words?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">Type a review, comment, or short passage. The trained '
    "classifier will label its sentiment, show its probability split, and reveal "
    "the clues behind this classic machine-learning model.</div>",
    unsafe_allow_html=True,
)

analyze_tab, batch_tab, model_tab = st.tabs(
    ["✦ Try the model", "▦ Compare texts", "◉ Inside the model"]
)

with analyze_tab:
    st.write("")
    examples = {
        "Warm & positive": "This film was wonderful, moving, and beautifully acted.",
        "Sharp & negative": "The story was boring, predictable, and a complete waste of time.",
        "Mixed review": "The performances were great, but the plot felt slow and uneven.",
    }
    st.caption("Need a starting point?")
    example_columns = st.columns(3)
    for column, (name, example) in zip(example_columns, examples.items()):
        if column.button(name, use_container_width=True):
            st.session_state["review_text"] = example

    text = st.text_area(
        "Text to analyze",
        key="review_text",
        height=170,
        max_chars=5000,
        placeholder="Example: The movie started slowly, but the ending was absolutely brilliant...",
    )
    analyze_clicked = st.button(
        "Analyze sentiment", type="primary", use_container_width=True
    )

    if analyze_clicked:
        if not text.strip():
            st.warning("Enter some text before running the analysis.")
        else:
            result = analyze(text, classifier)
            left, right = st.columns([1.05, 1], gap="large")
            with left:
                render_result(result)
            with right:
                st.markdown("### Text snapshot")
                metric_columns = st.columns(3)
                metric_columns[0].metric("Words / tokens", len(result["tokens"]))
                metric_columns[1].metric("Unique clues", result["unique_tokens"])
                metric_columns[2].metric("Characters", len(text))
                with st.expander("See the tokens the model received"):
                    st.write(result["tokens"])
                st.caption(
                    "This model treats each distinct token as present or absent; "
                    "it does not understand word order or context like a modern language model."
                )

with batch_tab:
    st.write("")
    st.markdown("### Compare several messages at once")
    st.caption("Enter one review per line. Blank lines are ignored (up to 50 entries).")
    batch_text = st.text_area(
        "Reviews",
        height=230,
        placeholder="I loved every minute of it!\nThe ending was disappointing.\nGood cast, weak story.",
        label_visibility="collapsed",
    )
    if st.button("Analyze all", key="analyze_batch", type="primary"):
        rows = [line.strip() for line in batch_text.splitlines() if line.strip()][:50]
        if not rows:
            st.warning("Add at least one line of text.")
        else:
            results = []
            for row in rows:
                output = analyze(row, classifier)
                results.append(
                    {
                        "Text": row,
                        "Prediction": friendly_label(output["prediction"]),
                        "Confidence": output["confidence"],
                        "Positive": output["probabilities"].get("pos", 0.0),
                        "Negative": output["probabilities"].get("neg", 0.0),
                    }
                )
            frame = pd.DataFrame(results)
            st.dataframe(
                frame,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence", min_value=0.0, max_value=1.0, format="percent"
                    ),
                    "Positive": st.column_config.NumberColumn(format="percent"),
                    "Negative": st.column_config.NumberColumn(format="percent"),
                },
            )
            summary = frame["Prediction"].value_counts()
            positive_count = int(summary.get("Positive", 0))
            negative_count = int(summary.get("Negative", 0))
            summary_columns = st.columns(3)
            summary_columns[0].metric("Texts analyzed", len(frame))
            summary_columns[1].metric("Positive", positive_count)
            summary_columns[2].metric("Negative", negative_count)

with model_tab:
    st.write("")
    st.markdown("### What the classifier learned")
    st.write(
        "The model is a Naive Bayes classifier trained on NLTK's labeled movie "
        "reviews. It compares how strongly individual words are associated with "
        "positive or negative reviews."
    )
    count = st.slider("Informative features to display", 5, 30, 12)
    try:
        informative = classifier.most_informative_features(count)
        feature_frame = pd.DataFrame(
            {
                "Rank": range(1, len(informative) + 1),
                "Word / feature": [str(feature) for feature in informative],
            }
        )
        st.dataframe(feature_frame, hide_index=True, use_container_width=True)
    except Exception:
        st.info("Informative-feature details are unavailable for this model file.")

    st.markdown("#### How a prediction is made")
    st.markdown(
        "1. Your text is split into word and punctuation tokens.\n"
        "2. Each unique token becomes a yes/no feature.\n"
        "3. The classifier estimates positive and negative probabilities.\n"
        "4. The higher-probability label is shown as the result."
    )
    st.info(
        "Limitations: this educational model was trained on movie reviews. It may "
        "struggle with sarcasm, mixed emotions, domain-specific language, spelling "
        "variants, and sentences whose meaning depends heavily on word order."
    )

st.divider()
st.caption("Built with Streamlit · NLTK Naive Bayes · Predictions run on this machine")
