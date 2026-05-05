import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from core.data_loader import load_csv_data
from core.nlp import preprocess_text
from core.model import RocchioClassifier

@st.cache_resource
def train_models():
    """
    Train 4 separate NearestCentroid models (Rocchio) 
    using Cosine Similarity on Tfidf vectors.
    """
    df, df_clean, blacklist = load_csv_data()
    
    # Initialize TF-IDF Vectorizer with Unigrams and Bigrams
    vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=preprocess_text,
        preprocessor=lambda x: x,
        token_pattern=None,
        ngram_range=(1, 2)
    )
    
    # Fit and transform the clean training text
    X_train = vectorizer.fit_transform(df_clean['text'])
    
    # 4 Dimensions based on IEEE 1044-2009 (Priority replaced by Mode)
    dimensions = ["type", "effect", "severity", "mode"]
    models = {}
    
    for dim in dimensions:
        # Train custom Rocchio classifier (Cosine Similarity Centroid)
        clf = RocchioClassifier()
        clf.fit(X_train, df_clean[dim])
        models[dim] = clf
        
    return vectorizer, models, blacklist

@st.cache_data
def evaluate_models():
    """
    Evaluates the Rocchio Algorithm using an 80/20 train/test split.
    """
    df, df_clean, blacklist = load_csv_data()
    
    # 80/20 split
    X_text = df_clean['text']
    # Use index to keep track of splits
    X_train_text, X_test_text, train_idx, test_idx = train_test_split(
        X_text, df_clean.index, test_size=0.2, random_state=42
    )
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=preprocess_text,
        preprocessor=lambda x: x,
        token_pattern=None,
        ngram_range=(1, 2)
    )
    
    # Fit and transform
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    
    dimensions = ["type", "effect", "severity", "mode"]
    metrics_data = {}
    
    for dim in dimensions:
        y_train = df_clean.loc[train_idx, dim]
        y_test = df_clean.loc[test_idx, dim]
        
        clf = RocchioClassifier()
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        
        metrics_data[dim] = {
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm,
            "classes": clf.classes_
        }
        
    return metrics_data, len(train_idx), len(test_idx)
