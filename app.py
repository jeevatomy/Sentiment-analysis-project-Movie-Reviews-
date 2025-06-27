import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if not present
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load model and vectorizer
model = joblib.load('models/logistic_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Preprocess input text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)

# Streamlit App UI
st.title("🎬 Sentiment Analysis App")
st.write("Enter a movie review and predict if it's Positive or Negative.")

user_input = st.text_area("💬 Your Review", height=150)

if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        cleaned = preprocess_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0].max()

        if prediction == "positive":
            st.success(f"✅ Positive ({probability*100:.2f}% confidence)")
        else:
            st.error(f"❌ Negative ({probability*100:.2f}% confidence)")
