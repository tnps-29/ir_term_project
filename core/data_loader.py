import pandas as pd
import streamlit as st
import os

@st.cache_data
def load_csv_data():
    """
    Load dataset from data/defects_classification.csv
    and combine text columns for feature extraction.
    """
    # Assuming the data folder is at the root, one level up from core/
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'defects_classification.csv')
    df = pd.read_csv(csv_path)
    
    # Combine text columns to form the 'text' for NLP
    text_cols = ['Title', 'Issue found', 'Step to reproduce', 'Actual result', 'Expected result']
    df['text'] = df[text_cols].fillna('').agg(' '.join, axis=1)
    
    # We will predict: IEEE_Type, IEEE_Effect, Severity, IEEE_Mode
    # Rename for consistency
    df = df.rename(columns={
        'IEEE_Type': 'type',
        'IEEE_Effect': 'effect',
        'Severity': 'severity',
        'IEEE_Mode': 'mode'
    })
    
    # Drop rows with missing labels in the dimensions we want to predict
    df = df.dropna(subset=['type', 'effect', 'severity', 'mode'])
    
    # Blacklist dictionary to filter out Tasks/Enhancements
    blacklist = ["เพิ่ม field", "add ", "enhance ", "ปรับปรุง"]
    
    def is_defect(text):
        text_lower = text.lower()
        return not any(word in text_lower for word in blacklist)
        
    # Apply rule-based filtering
    df_clean = df[df["text"].apply(is_defect)].copy()
    
    return df, df_clean, blacklist
