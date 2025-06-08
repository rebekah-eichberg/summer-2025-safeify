import nltk
from transformers import pipeline
import pandas as pd
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from collections import Counter

# Run this once to download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

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

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

# Function to clean and lemmatize
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())  
    tokens = [word for word in tokens if word.isalpha()]  
    tokens = [word for word in tokens if word not in stop_words]  
    pos_tags = pos_tag(tokens)  # POS tagging
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    return lemmatized


