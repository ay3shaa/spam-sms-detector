# spam_detector_app.py
import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf.pkl")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Streamlit UI
st.title("📩 Spam Message Detector")
st.write("Enter a text message and see if it's spam or not.")

user_input = st.text_area("Enter your message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned_text = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_input)[0]

        if prediction == 1:
            st.error("🚫 This message is **SPAM**.")
        else:
            st.success("✅ This message is **NOT spam**.")
