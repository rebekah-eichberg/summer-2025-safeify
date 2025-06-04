import nltk
nltk.download('punkt')       
nltk.download('stopwords') 
nltk.download('punkt_tab')
nltk.download('vader_lexicon')
from transformers import pipeline
import pandas as pd
import numpy as np

def load_clean_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the first line that contains all expected headers (starts with 'Report No.')
    header_index = next(i for i, line in enumerate(lines) if 'Report No.' in line)

    # Load CSV from that line forward
    return pd.read_csv(path, skiprows=header_index)

labels = ['harmful', 'toxic', 'hazardous', 'dangerous', 'caused injury', 'accident', 'choke']


def classify_batch_all_scores(texts):
    classifier = pipeline("zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",
    device=0)
    
    results = classifier(texts, candidate_labels=labels, truncation=True)
    if isinstance(results, dict):  # single input returns dict
        results = [results]
    
    # Return a dict per input with all label scores
    return [
        dict(zip(r["labels"], r["scores"]))
        for r in results
    ]
