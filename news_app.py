


import streamlit as st
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # Fixed typo here

# ==== 1. Load models and classifier ====
@st.cache_resource
def load_model_and_classifier():
    # Load summarization model
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # Load dataset
    df = pd.read_csv("allsides_balanced_news_headlines-texts.csv")
    df.dropna(subset=['text', 'bias_rating'], inplace=True)

    # Prepare bias classifier
    X = df['text']
    y = df['bias_rating'].str.lower()

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_vec, y)

    return tokenizer, model, vectorizer, clf

# Load resources
tokenizer, model, vectorizer, clf = load_model_and_classifier()

# ==== 2. Summarize article ====
def summarize(text):
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    output = model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ==== 3. Detect bias ====
def detect_bias(text):  # Fixed function name typo
    vec = vectorizer.transform([text])
    prediction = clf.predict(vec)[0]
    return prediction.capitalize()

# ==== 4. Streamlit UI ====
st.set_page_config(page_title="AutoBrief", layout="centered")
st.title("📰 AutoBrief: News Summarizer + Bias Detector")
st.write("Paste a news article below to get a quick summary and predict its political bias.")

input_article = st.text_area("✍️ Paste News Article:", height=300)

if st.button("Analyze"):
    if input_article.strip() == "":
        st.warning("Please enter an article.")
    else:
        with st.spinner("Processing..."):
            summary = summarize(input_article)
            bias = detect_bias(input_article)  # Fixed function name here

        st.subheader("📝 Summary")
        st.success(summary)

        st.subheader("🏷️ Predicted Bias")
        st.info(f"**{bias}**")

st.markdown("---")
st.caption("Made with ❤️ using HuggingFace Transformers and Streamlit.")