import re
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Job Role Category Classifier", page_icon="💼", layout="centered")

MAX_LEN = 40

MODEL_OPTIONS = {
    "Logistic Regression": {
        "type": "sklearn",
        "model_file": "role_classifier.pkl",
        "vectorizer_file": "tfidf_vectorizer.pkl",
    },
    "SimpleRNN": {
        "type": "keras",
        "model_file": "simplernn_role_model.h5",
    },
    "LSTM": {
        "type": "keras",
        "model_file": "lstm_role_model.h5",
    },
    "GRU": {
        "type": "keras",
        "model_file": "gru_role_model.h5",
    },
}


@st.cache_resource
def load_label_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_sklearn_artifacts(model_file, vectorizer_file):
    with open(model_file, "rb") as f:
        clf = pickle.load(f)
    with open(vectorizer_file, "rb") as f:
        vectorizer = pickle.load(f)
    return clf, vectorizer


@st.cache_resource
def load_keras_artifacts(model_file):
    model = load_model(model_file)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


# ---- Same cleaning function used during training ----
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_sklearn(clf, vectorizer, label_encoder, combined_text):
    cleaned = clean_text(combined_text)
    vec = vectorizer.transform([cleaned])
    probs = clf.predict_proba(vec)[0]
    return probs


def predict_keras(model, tokenizer, combined_text):
    cleaned = clean_text(combined_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    return probs


def top_predictions(probs, label_encoder, top_n=5):
    top_idx = np.argsort(probs)[::-1][:top_n]
    return [(label_encoder.inverse_transform([i])[0], float(probs[i])) for i in top_idx]


# ---------------- Streamlit UI ----------------
st.title("💼 Job Role Category Classifier")
st.write("Predict a job's Role Category from Title, Key Skills, and Location.")

model_choice = st.selectbox("Choose a model:", list(MODEL_OPTIONS.keys()))

job_title = st.text_input("Job Title:", placeholder="e.g. Senior Software Engineer")
key_skills = st.text_area("Key Skills (pipe or comma separated):", height=100, placeholder="e.g. Python, Django, REST API")
location = st.text_input("Location:", placeholder="e.g. Bangalore")

if st.button("Predict Role Category"):
    if job_title.strip() == "" and key_skills.strip() == "":
        st.warning("Please enter at least a job title or key skills.")
    else:
        combined_text = f"{job_title} {key_skills} {location}"
        label_encoder = load_label_encoder()
        config = MODEL_OPTIONS[model_choice]

        try:
            if config["type"] == "sklearn":
                clf, vectorizer = load_sklearn_artifacts(config["model_file"], config["vectorizer_file"])
                probs = predict_sklearn(clf, vectorizer, label_encoder, combined_text)
            else:
                model, tokenizer = load_keras_artifacts(config["model_file"])
                probs = predict_keras(model, tokenizer, combined_text)

            results = top_predictions(probs, label_encoder, top_n=5)
            top_label, top_confidence = results[0]

            st.success(f"**{top_label}** (confidence: {top_confidence:.2%})")
            st.progress(top_confidence)
            st.caption(f"Model used: {model_choice} — Top 5 predicted categories:")
            for cat, prob in results:
                st.write(f"- {cat}: {prob:.2%}")

        except FileNotFoundError as e:
            st.error(
                f"Missing model file for **{model_choice}**: `{e.filename}`. "
                "Make sure all model/vectorizer/tokenizer/label encoder files "
                "are in the same folder as this app."
            )