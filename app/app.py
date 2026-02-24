import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# ✅ Correct path
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

st.title("AI Fake News Detection System")

news = st.text_area("Enter News Text")

if st.button("Predict"):
    
    if news.strip() == "":
        st.warning("Please enter news text")
        
    else:
        cleaned = clean_text(news)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)

        if prediction[0] == 0:
            st.error("Fake News")
        else:
            st.success("Real News")