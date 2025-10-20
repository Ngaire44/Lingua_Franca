import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd

MODEL_PATH = "sentiment_model"

@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tok, model

tok, model = load_model()

st.set_page_config(page_title="Tech Sentiment Analyzer", layout="centered")
st.title("📊 Tech Product Sentiment Analyzer")
st.markdown("Analyze tweets about Apple & Google using a fine-tuned DistilBERT model.")

# Single text input
text = st.text_area("Enter a tweet or short text:", height=120)

if st.button("Analyze Sentiment"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        inputs = tok(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]
            pred = int(np.argmax(probs))
        label_map = {0: "Negative", 1: "Positive"}
        st.subheader("Result:")
        st.write(f"**Sentiment:** {label_map[pred]}")
        st.progress(float(probs[pred]))
        st.write(f"Confidence: {probs[pred]*100:.2f}%")
