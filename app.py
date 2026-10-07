# ============================================================
# CineSentiment AI
# Premium Movie Review Sentiment Dashboard
# Streamlit + TensorFlow/Keras + LSTM
# ============================================================

import os
import re

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 10000
MAX_LENGTH = 200

MODEL_PATH = "Final Model.keras"
BANNER_PATH = os.path.join("assets", "movie-banner.png")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CineSentiment AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PREMIUM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 5%,
                rgba(100, 70, 255, 0.18),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 15%,
                rgba(255, 50, 130, 0.12),
                transparent 28%
            ),
            radial-gradient(
                circle at 50% 100%,
                rgba(70, 40, 180, 0.10),
                transparent 30%
            ),
            #070711;

        animation: pageFade 0.8s ease-out;
    }


    .block-container {
        max-width: 1280px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }


    @keyframes pageFade {

        from {
            opacity: 0;
            transform: translateY(12px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }

    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #0b0b17 0%,
                #0d0d1d 50%,
                #080811 100%
            );

        border-right:
            1px solid rgba(255,255,255,0.07);
    }


    /* ======================================================
       HEADINGS
       ====================================================== */

    h1, h2, h3 {

        animation:
            titleReveal 0.7s ease-out;

    }


    @keyframes titleReveal {

        from {
            opacity: 0;
            transform: translateX(-20px);
        }

        to {
            opacity: 1;
            transform: translateX(0);
        }

    }


    /* ======================================================
       BANNER
       ====================================================== */

    [data-testid="stImage"] img {

        border-radius: 22px;

        border:
            1px solid rgba(255,255,255,0.10);

        box-shadow:
            0 20px 70px rgba(0,0,0,0.50);

        transition:
            transform 0.5s ease,
            filter 0.5s ease,
            box-shadow 0.5s ease;

        animation:
            bannerReveal 1s ease-out;
    }


    [data-testid="stImage"] img:hover {

        transform: scale(1.012);

        filter:
            brightness(1.07)
            saturate(1.10);

        box-shadow:
            0 25px 80px rgba(90,60,255,0.18);
    }


    @keyframes bannerReveal {

        from {
            opacity: 0;
            transform: scale(0.985);
        }

        to {
            opacity: 1;
            transform: scale(1);
        }

    }


    /* ======================================================
       METRIC CARDS
       ====================================================== */

    [data-testid="stMetric"] {

        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.055),
                rgba(255,255,255,0.025)
            );

        border:
            1px solid rgba(255,255,255,0.08);

        border-radius: 17px;

        padding: 16px;

        box-shadow:
            0 10px 30px rgba(0,0,0,0.18);

        transition:
            transform 0.3s ease,
            border-color 0.3s ease,
            box-shadow 0.3s ease;

        animation:
            cardReveal 0.7s ease-out;
    }


    [data-testid="stMetric"]:hover {

        transform:
            translateY(-6px);

        border-color:
            rgba(135,105,255,0.45);

        box-shadow:
            0 18px 40px
            rgba(80,60,220,0.20);
    }


    @keyframes cardReveal {

        from {
            opacity: 0;
            transform: translateY(18px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }

    }


    /* ======================================================
       TEXT AREA
       ====================================================== */

    textarea {

        background:
            #10101c !important;

        color:
            #ffffff !important;

        border:
            1px solid rgba(255,255,255,0.12)
            !important;

        border-radius:
            16px !important;

        font-size:
            15px !important;

        line-height:
            1.65 !important;

        transition:
            border-color 0.3s ease,
            box-shadow 0.3s ease,
            transform 0.3s ease;
    }


    textarea:focus {

        border-color:
            rgba(130,100,255,0.75)
            !important;

        box-shadow:
            0 0 28px
            rgba(100,75,255,0.17)
            !important;

        transform:
            translateY(-1px);
    }


    /* ======================================================
       ANALYZE BUTTON
       ====================================================== */

    .stButton > button {

        width: 100%;

        height: 54px;

        border: none;

        border-radius: 15px;

        color: white;

        font-size: 16px;

        font-weight: 800;

        background:
            linear-gradient(
                90deg,
                #5e49dc,
                #8d4df5,
                #c14cff,
                #5e49dc
            );

        background-size:
            300% 100%;

        box-shadow:
            0 12px 35px
            rgba(105,70,255,0.27);

        animation:
            gradientMove 5s ease infinite;

        transition:
            transform 0.25s ease,
            box-shadow 0.25s ease;
    }


    .stButton > button:hover {

        transform:
            translateY(-3px)
            scale(1.01);

        box-shadow:
            0 18px 45px
            rgba(105,70,255,0.42);
    }


    .stButton > button:active {

        transform:
            scale(0.98);
    }


    @keyframes gradientMove {

        0% {
            background-position: 0% 50%;
        }

        50% {
            background-position: 100% 50%;
        }

        100% {
            background-position: 0% 50%;
        }

    }


    /* ======================================================
       NATIVE CONTAINERS
       ====================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {

        background:
            rgba(18,18,32,0.60);

        border:
            1px solid rgba(255,255,255,0.08);

        border-radius:
            18px;

        transition:
            transform 0.3s ease,
            border-color 0.3s ease;
    }


    [data-testid="stVerticalBlockBorderWrapper"]:hover {

        transform:
            translateY(-3px);

        border-color:
            rgba(130,100,255,0.25);
    }


    /* ======================================================
       ALERT / RESULT
       ====================================================== */

    [data-testid="stAlert"] {

        border-radius:
            16px;

        animation:
            resultReveal 0.7s ease-out;
    }


    @keyframes resultReveal {

        0% {
            opacity: 0;
            transform:
                scale(0.92)
                translateY(15px);
        }

        70% {
            transform:
                scale(1.02);
        }

        100% {
            opacity: 1;
            transform:
                scale(1)
                translateY(0);
        }

    }


    /* ======================================================
       PROGRESS BAR
       ====================================================== */

    [data-testid="stProgressBar"] {

        animation:
            progressReveal 0.8s ease-out;
    }


    @keyframes progressReveal {

        from {
            opacity: 0;
            transform: translateX(-20px);
        }

        to {
            opacity: 1;
            transform: translateX(0);
        }

    }


    /* ======================================================
       DIVIDER
       ====================================================== */

    hr {

        border-color:
            rgba(255,255,255,0.08);
    }


    /* ======================================================
       CAPTION
       ====================================================== */

    .stCaption {

        color:
            #858297 !important;
    }


    /* ======================================================
       REDUCED MOTION ACCESSIBILITY
       ====================================================== */

    @media (prefers-reduced-motion: reduce) {

        * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_sentiment_model():

    return load_model(MODEL_PATH)


# ============================================================
# LOAD IMDB WORD INDEX
# ============================================================

@st.cache_data
def load_word_index():

    return imdb.get_word_index()


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "❌ Final Model.keras was not found."
    )

    st.info(
        "Make sure Final Model.keras is in the "
        "same folder as app.py."
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_sentiment_model()

except Exception as error:

    st.error(
        "❌ Unable to load the trained LSTM model."
    )

    st.exception(error)

    st.stop()


# ============================================================
# LOAD WORD INDEX
# ============================================================

try:

    word_index = load_word_index()

except Exception as error:

    st.error(
        "❌ Unable to load the IMDB word dictionary."
    )

    st.exception(error)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎬 CineSentiment AI")

    st.caption(
        "Premium Movie Review Intelligence"
    )

    st.divider()

    st.subheader("🧠 LSTM Architecture")

    st.write("📝 Review Text")
    st.write("↓")
    st.write("🔢 Tokenization")
    st.write("↓")
    st.write("📏 Sequence Padding")
    st.write("↓")
    st.write("🔤 32D Embedding")
    st.write("↓")
    st.write("🔄 LSTM — 64 Units")
    st.write("↓")
    st.write("🎯 Sigmoid")
    st.write("↓")
    st.write("😊 / 😞 Sentiment")

    st.divider()

    st.subheader("⚙️ Configuration")

    st.write("**Vocabulary:** 10,000")
    st.write("**Sequence:** 200 tokens")
    st.write("**Embedding:** 32 dimensions")
    st.write("**LSTM:** 64 units")
    st.write("**Output:** Binary")

    st.divider()

    st.success("Model Ready")


# ============================================================
# HERO BANNER
# ============================================================

if os.path.exists(BANNER_PATH):

    st.image(
        BANNER_PATH,
        use_container_width=True
    )

else:

    st.warning(
        "⚠️ movie-banner.png was not found "
        "inside the assets folder."
    )


st.caption(
    "🎬 CineSentiment AI  •  Deep Learning NLP  •  LSTM"
)


# ============================================================
# HERO INTRO
# ============================================================

st.title(
    "🤖 AI-Powered Movie Review Analysis"
)

st.write(
    """
    Transform movie reviews into actionable sentiment insights
    using a trained **Long Short-Term Memory (LSTM)** neural network.

    Enter a review below and the model will classify it as
    **Positive** or **Negative**, together with its confidence
    and probability distribution.
    """
)


# ============================================================
# MODEL OVERVIEW
# ============================================================

st.subheader("📊 Model Overview")

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.metric(
        "🎬 IMDB Reviews",
        "50K+"
    )

with m2:

    st.metric(
        "📚 Vocabulary",
        "10K"
    )

with m3:

    st.metric(
        "🔄 LSTM Units",
        "64"
    )

with m4:

    st.metric(
        "🧠 Embedding",
        "32D"
    )


st.divider()


# ============================================================
# MAIN WORKSPACE
# ============================================================

input_column, pipeline_column = st.columns(
    [1.35, 0.65],
    gap="large"
)


# ============================================================
# REVIEW INPUT
# ============================================================

with input_column:

    st.subheader(
        "📝 Review Analyzer"
    )

    st.caption(
        "Paste or write a complete movie review."
    )

    review = st.text_area(
        "Movie Review",
        placeholder=(
            "Example:\n\n"
            "This movie was absolutely fantastic. "
            "The performances were brilliant, "
            "the story was engaging, and the ending "
            "was incredibly satisfying..."
        ),
        height=260,
        label_visibility="collapsed"
    )

    st.write("")

    analyze_button = st.button(
        "🚀 Analyze Sentiment"
    )


# ============================================================
# AI PIPELINE
# ============================================================

with pipeline_column:

    st.subheader(
        "🔬 AI Pipeline"
    )

    with st.container(border=True):

        st.write("📝 **Review Text**")
        st.write("↓")

        st.write("🔢 **Tokenization**")
        st.write("↓")

        st.write("📏 **Padding — 200 Tokens**")
        st.write("↓")

        st.write("🧠 **Embedding — 32D**")
        st.write("↓")

        st.write("🔄 **LSTM — 64 Units**")
        st.write("↓")

        st.write("🎯 **Sigmoid Classification**")


# ============================================================
# REVIEW ENCODING
# ============================================================

def encode_review(text):

    words = re.findall(
        r"[a-zA-Z0-9']+",
        text.lower()
    )

    encoded_review = []

    for word in words:

        index = word_index.get(
            word,
            2
        )

        imdb_index = index + 3

        if imdb_index >= VOCAB_SIZE:

            imdb_index = 2

        encoded_review.append(
            imdb_index
        )

    return encoded_review


# ============================================================
# PREDICTION
# ============================================================

if analyze_button:

    if not review.strip():

        st.warning(
            "⚠️ Please enter a movie review first."
        )

    else:

        # ----------------------------------------------------
        # PREPROCESSING + INFERENCE
        # ----------------------------------------------------

        with st.spinner(
            "🧠 CineSentiment AI is analyzing..."
        ):

            encoded_review = encode_review(
                review
            )

            padded_review = pad_sequences(
                [encoded_review],
                maxlen=MAX_LENGTH
            )

            prediction = model.predict(
                padded_review,
                verbose=0
            )

            probability = float(
                prediction[0][0]
            )


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        if probability >= 0.5:

            sentiment = "POSITIVE"

            confidence = probability

            positive_probability = probability

            negative_probability = 1 - probability

        else:

            sentiment = "NEGATIVE"

            confidence = 1 - probability

            positive_probability = probability

            negative_probability = 1 - probability


        st.divider()


        # ====================================================
        # MAIN RESULT
        # ====================================================

        st.subheader(
            "🎯 Prediction Result"
        )

        if sentiment == "POSITIVE":

            st.success(
                f"😊 POSITIVE SENTIMENT\n\n"
                f"Confidence: {confidence:.2%}"
            )

        else:

            st.error(
                f"😞 NEGATIVE SENTIMENT\n\n"
                f"Confidence: {confidence:.2%}"
            )


        # ====================================================
        # PROBABILITY DISTRIBUTION
        # ====================================================

        st.subheader(
            "📊 Sentiment Probability"
        )

        positive_col, negative_col = st.columns(2)


        with positive_col:

            st.write(
                "🟢 **Positive**"
            )

            st.progress(
                positive_probability
            )

            st.caption(
                f"{positive_probability:.2%}"
            )


        with negative_col:

            st.write(
                "🔴 **Negative**"
            )

            st.progress(
                negative_probability
            )

            st.caption(
                f"{negative_probability:.2%}"
            )


        st.divider()


        # ====================================================
        # PREDICTION DETAILS
        # ====================================================

        st.subheader(
            "🧠 Prediction Intelligence"
        )

        d1, d2, d3, d4 = st.columns(4)


        with d1:

            st.metric(
                "Prediction",
                sentiment
            )


        with d2:

            st.metric(
                "Confidence",
                f"{confidence:.2%}"
            )


        with d3:

            st.metric(
                "Model Output",
                f"{probability:.4f}"
            )


        with d4:

            st.metric(
                "Processed Tokens",
                len(encoded_review)
            )


        # ====================================================
        # REVIEW STATISTICS
        # ====================================================

        st.subheader(
            "📋 Review Statistics"
        )

        s1, s2, s3 = st.columns(3)


        with s1:

            st.metric(
                "Original Words",
                len(review.split())
            )


        with s2:

            st.metric(
                "Processed Tokens",
                len(encoded_review)
            )


        with s3:

            st.metric(
                "Maximum Sequence",
                MAX_LENGTH
            )


        # ====================================================
        # MODEL INTERPRETATION
        # ====================================================

        st.subheader(
            "💡 Model Interpretation"
        )

        if confidence >= 0.90:

            st.success(
                f"Very strong {sentiment.lower()} "
                "sentiment detected."
            )

        elif confidence >= 0.75:

            st.info(
                f"Strong {sentiment.lower()} "
                "sentiment detected."
            )

        elif confidence >= 0.60:

            st.warning(
                f"Moderate {sentiment.lower()} "
                "sentiment detected."
            )

        else:

            st.warning(
                "The prediction is close to the "
                "0.50 decision threshold, indicating "
                "lower model confidence."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎬 CineSentiment AI  |  "
    "Python • TensorFlow • Keras • NLP • LSTM • Streamlit"
)