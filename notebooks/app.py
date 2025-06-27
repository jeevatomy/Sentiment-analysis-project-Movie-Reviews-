import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the model and vectorizer
model = joblib.load('models/logistic_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Text preprocessing
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

# 🌟 Streamlit UI Layout
st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
    .big-font {
        font-size: 25px !important;
    }
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

# 🎥 Movie Banner
st.image("https://cdn.wallpapersafari.com/59/75/QeW4Os.jpg", use_column_width=True)

# 📝 Title and Description
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Welcome to your very own **Sentiment Analysis Web App**! Enter any movie review below and find out whether it’s 🔥 Positive or 💔 Negative — powered by Machine Learning.")

# 🎞 Movie Posters
cols = st.columns(5)
movie_posters = [
    "https://m.media-amazon.com/images/I/81nCIXVCKML._AC_UF894,1000_QL80_.jpg",
    "https://m.media-amazon.com/images/I/51oDs3fR4HL._AC_SY580_.jpg",
    "https://m.media-amazon.com/images/I/81n1OdoC3yL._AC_SY741_.jpg",
    "https://m.media-amazon.com/images/I/71NIpJt2TxL._AC_SY679_.jpg",
    "https://m.media-amazon.com/images/I/61cZJvF3NBL._AC_UF1000,1000_QL80_.jpg"
]
for i, col in enumerate(cols):
    col.image(movie_posters[i], use_column_width=True)

st.markdown("---")

# 💬 Input Area
review = st.text_area("✍️ Enter your movie review:", height=200)

# 🎯 Predict Button
if st.button("🔍 Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        processed = preprocess_text(review)
        vect_input = vectorizer.transform([processed])
        prediction = model.predict(vect_input)[0]
        confidence = model.predict_proba(vect_input)[0].max() * 100

        if prediction == "positive":
            st.success(f"🎉 Positive Review ({confidence:.2f}% confident)")
            st.balloons()
        else:
            st.error(f"💔 Negative Review ({confidence:.2f}% confident)")
            st.snow()

# 📌 Footer
st.markdown("---")
st.markdown("Made with ❤️ by **Jeeva Mariya Tomy** | Powered by Logistic Regression + Streamlit")
