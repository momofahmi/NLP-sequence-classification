import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    task: str  # "Sarcasm" or "Sentiment"


DEFAULT_SPECS = {
    # TODO: replace with your best-performing model IDs or adapter-backed models
    "en-UK": ModelSpec(model_id="cardiffnlp/twitter-roberta-base-sentiment-latest", task="Sentiment"),
    "en-AU": ModelSpec(model_id="cardiffnlp/twitter-roberta-base-sentiment-latest", task="Sentiment"),
    "en-IN": ModelSpec(model_id="cardiffnlp/twitter-roberta-base-sentiment-latest", task="Sentiment"),
}


@st.cache_resource(show_spinner=False)
def load_hf_model(model_id: str) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    return tok, model


def predict(text: str, tokenizer, model) -> Tuple[int, np.ndarray]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits.detach().cpu().numpy()[0]
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        pred = int(probs.argmax())
    return pred, probs


def main():
    st.set_page_config(page_title="BESSTIE Demo", page_icon="📝", layout="centered")

    st.title("BESSTIE: Sentiment & Sarcasm Demo")
    st.caption("Select an English variety and run the model used for that context.")

    variety = st.selectbox("English variety", ["en-UK", "en-AU", "en-IN"], index=0)
    task = st.selectbox("Task", ["Sentiment", "Sarcasm"], index=0)

    text = st.text_area(
        "Input text",
        value="This place is absolutely fantastic... said no one ever.",
        height=120,
    )

    st.divider()

    # Choose model spec (simple default routing)
    spec = DEFAULT_SPECS[variety]
    if spec.task != task:
        st.warning(
            "This demo is currently routed to a placeholder Sentiment model for all varieties.\n"
            "Update `DEFAULT_SPECS` to point to your fine-tuned/adapter models for each task/variety."
        )

    model_id = spec.model_id

    col1, col2 = st.columns([1, 2])
    with col1:
        do_run = st.button("Predict", type="primary", use_container_width=True)
    with col2:
        st.write(f"**Model**: `{model_id}`")

    if do_run:
        if not text.strip():
            st.error("Please enter some text.")
            return

        with st.spinner("Loading model and running inference..."):
            tok, model = load_hf_model(model_id)
            t0 = time.perf_counter()
            pred, probs = predict(text, tok, model)
            dt_ms = (time.perf_counter() - t0) * 1000

        st.subheader("Prediction")
        st.write(f"**Predicted class index**: `{pred}`")
        st.write(f"**Probabilities**: `{probs.round(4).tolist()}`")
        st.caption(f"Inference time: {dt_ms:.1f} ms")


if __name__ == "__main__":
    main()

