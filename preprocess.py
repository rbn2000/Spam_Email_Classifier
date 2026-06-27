import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK resources
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# Load stop words
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """
    Cleans the input text by:
    1. Converting to lowercase
    2. Removing numbers
    3. Removing punctuation
    4. Removing stopwords
    5. Applying stemming
    """

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    words = text.split()

    # Remove stopwords & stem
    cleaned_words = []

    for word in words:
        if word not in stop_words:
            cleaned_words.append(ps.stem(word))

    return " ".join(cleaned_words)