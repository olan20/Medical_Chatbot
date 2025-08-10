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
        

        