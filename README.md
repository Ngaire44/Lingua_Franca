
---
title: Sentiment App
emoji: 🌖
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# TechSent — Social Media Sentiment Intelligence

A Natural Language Processing (NLP) project analyzing public sentiment toward Apple and Google using DistilBERT and deployed via Streamlit for real-time brand monitoring.

## 📊 Project overview

TechSent (Technology Sentiment Intelligence) explores how people express emotions about major tech brands on Twitter/X.
By applying transformer models, the project uncovers public sentiment patterns and provides a deployable web dashboard for live prediction and analysis.

### Business goals

- Measure and compare public emotions toward Apple and Google
- Identify the key topics driving positive and negative sentiment
- Enable real-time sentiment tracking for marketing and PR teams
- Deploy an interactive Streamlit dashboard for stakeholders

## ⚙️ Features

- Sentiment classification using *DistilBERT-base-uncased*
- Real-time text input or batch CSV uploads
- Streamlit web dashboard for interactive use
- Model deployment via Hugging Face
- Insight generation and brand comparison visualizations

## 🧩 Tech stack

| Category | Tools |
| -------- | ----- |
| *Language* | Python 3.10+ |
| *Libraries* | transformers, torch, pandas, numpy, scikit-learn, streamlit |
| *Model* | DistilBERT (fine-tuned for sentiment classification) |
| *Deployment* | Streamlit / Hugging Face Spaces |
| *EDA & Visualization* | Matplotlib, WordCloud |
| *Environment* | Jupyter Notebook |

## 🧪 Workflow overview

1. **Data collection**
   - Dataset: [CrowdFlower “Brand and Product Emotions”](https://data.world/crowdflower/brands-and-product-emotions)
   - Filtered for Apple and Google mentions only

2. **Data cleaning**
   - Lowercasing, punctuation & stopword removal
   - Removal of duplicates, emojis, and URLs

3. **EDA**
   - Sentiment distribution by brand
   - Word frequency and keyword analysis

4. **Modeling**
   - Baseline: TF-IDF + Logistic Regression
   - Advanced: Fine-tuned DistilBERT transformer
   - Evaluation metrics: Accuracy, Precision, Recall, F1

5. **Deployment**
   - Saved model with:

     ```python
     model.save_pretrained("sentiment_model")
     tok.save_pretrained("sentiment_model")
     ```

   - Streamlit dashboard for real-time analysis

## 🚀 Deployment guide

### Option 1: Local run

```bash
git clone https://github.com/Ngaire44/techsent.git
cd techsent
pip install -r requirements.txt
streamlit run app.py
```

Check out the configuration reference at [Hugging Face Spaces config reference](https://huggingface.co/docs/hub/spaces-config-reference)
