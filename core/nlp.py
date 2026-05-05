import re
import nltk
from nltk.corpus import stopwords
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords

# Download NLTK stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Load English stopwords from NLTK
ENGLISH_STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess text using PyThaiNLP:
    1. Remove special characters
    2. Tokenize using 'newmm'
    3. Remove Thai and English stopwords
    """
    # Remove special characters but keep Thai and English letters and numbers
    text = re.sub(r'[^ก-๙a-zA-Z0-9\s]', '', text)
    
    # Word tokenization
    tokens = word_tokenize(text, engine='newmm')
    
    # Stopwords removal
    thai_stops = list(thai_stopwords())
    filtered_tokens = []
    
    for token in tokens:
        token_lower = token.lower()
        if (token_lower not in thai_stops and 
            token_lower not in ENGLISH_STOPWORDS and 
            token.strip()):
            filtered_tokens.append(token)
            
    return filtered_tokens
