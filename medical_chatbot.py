import nltk
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import re
import random

NLTK_DATA_PATH = 'C:/nltk_data' 
nltk.data.path.append(NLTK_DATA_PATH)

def check_nltk_data():
    """
    Check if required NLTK data is available locally.
    Raises:
        FileNotFoundError: If data is missing with instructions.
    """
    required_data = ['punkt', 'stopwords', 'wordnet']
    for data in required_data:
        try:
            nltk.data.find(f'tokenizers/{data}' if data == 'punkt' else f'corpora/{data}')
        except LookupError:
            raise FileNotFoundError(
                f"NLTK data '{data}' not found at {NLTK_DATA_PATH}. "
                "Download it from https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml "
                "and extract to the specified path."
            )
        

# Preprocess text
def preprocess_text(text):
    """
    Preprocess user input: tokenize, remove stopwords, and lemmatize.
    Args:
        text (str): User input text.
    Returns:
        str: Processed text or empty string if preprocessing fails.
    """
    try:
        lemmatizer = nltk.WordNetLemmatizer()
        stop_words = set(nltk.corpus.stopwords.words('english'))
        tokens = nltk.word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in text preprocessing: {e}")
        return ""

KNOWLEDGE_BASE = {
    "symptoms_diabetes": {
        "keywords": ["thirsty", "fatigue", "blurred vision", "weight loss", "urinate frequently", "tired", "vision blurry"],
        "response": "You may be experiencing diabetes symptoms, such as increased thirst, fatigue, blurred vision, or frequent urination. Visit a clinic for a blood glucose test."
    },
    "symptoms_hypertension": {
        "keywords": ["headache", "dizziness", "chest pain", "shortness of breath", "nosebleed"],
        "response": "Symptoms like headaches, dizziness, chest pain, or shortness of breath may indicate hypertension. Please check your blood pressure at a clinic."
    },
    "medication_diabetes": {
        "keywords": ["medication", "insulin", "metformin", "diabetes medicine", "drugs for diabetes"],
        "response": "Common diabetes medications include metformin and insulin. Follow your doctor's prescription and monitor blood sugar regularly."
    },
    "medication_hypertension": {
        "keywords": ["medication", "blood pressure medicine", "hypertension medicine", "drugs for hypertension"],
        "response": "Hypertension medications may include ACE inhibitors or diuretics. Consult your doctor for proper dosage and monitoring."
    },
    "diet_diabetes": {
        "keywords": ["diet", "food", "eating", "diabetes", "what to eat"],
        "response": "For diabetes, eat a balanced diet with low sugar, high fiber, and lean proteins (e.g., beans, yam, fish). Avoid sugary drinks and processed foods like biscuits. Consult a nutritionist."
    },
    "diet_hypertension": {
        "keywords": ["diet", "food", "eating", "hypertension", "blood pressure diet"],
        "response": "For hypertension, follow a low-sodium diet with fruits, vegetables (e.g., ugu, spinach), and whole grains. Limit salt and processed foods like maggi cubes. Consult a healthcare provider."
    },
    "general_advice": {
        "keywords": ["healthy", "lifestyle", "prevent", "stay healthy"],
        "response": "To prevent NCDs, maintain a healthy weight, exercise (e.g., 30 minutes daily walking), avoid smoking, and get regular check-ups. Consult a healthcare professional for advice."
    }
}        