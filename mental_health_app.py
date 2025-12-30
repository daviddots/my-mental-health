
import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    WordNetLemmatizer()
except LookupError:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load the trained model and TF-IDF vectorizer
try:
    model = joblib.load('model_and_vectorizer/best_model.pkl')
    tfidf_vectorizer = joblib.load('model_and_vectorizer/tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model or TF-IDF vectorizer not found. Please ensure 'best_model.pkl' and 'tfidf_vectorizer.pkl' are in the 'model_and_vectorizer' directory.")
    st.stop()

# Define the label map
label_map = {0: 'Stress', 1: 'Depression', 2: 'Bipolar disorder', 3: 'Personality disorder', 4: 'Anxiety'}

st.title("Mental Health Condition Predictor")
st.write("Enter text to predict the mental health condition.")

user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input:
        # Clean the input text
        cleaned_input = clean_text(user_input)

        # Transform the cleaned text using the loaded TF-IDF vectorizer
        input_vectorized = tfidf_vectorizer.transform([cleaned_input])

        # Make a prediction
        prediction = model.predict(input_vectorized)
        predicted_label_index = prediction[0]
        predicted_label = label_map.get(predicted_label_index, "Unknown")

        st.success(f"The predicted mental health condition is: **{predicted_label}**")
    else:
        st.warning("Please enter some text for prediction.")

st.markdown('''
<style>
.stTextArea>label {
    font-size: 1.2em;
    font-weight: bold;
}
.stButton>button {
    font-size: 1.1em;
    padding: 0.5em 1em;
    background-color: #4CAF50;
    color: white;
    border-radius: 5px;
    border: none;
    cursor: pointer;
}
.stButton>button:hover {
    background-color: #45a049;
}
</style>
''', unsafe_allow_html=True)
