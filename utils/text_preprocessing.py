import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download once (safe if already present)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    """
    Cleans and preprocesses text for ML classification.
    Used for all inputs: txt, csv, pdf, docx, OCR output.
    """

    if not text:
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove non-alphabet characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    words = text.split()

    # Remove stopwords + stemming
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(words)
