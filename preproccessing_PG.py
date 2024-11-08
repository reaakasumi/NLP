# Required Imports
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import stanza
import re
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Load Datasets
train_df = pd.read_json('train.model-agnostic.json')
test_df = pd.read_json('test.model-agnostic.json')
val_df = pd.read_json('val.model-agnostic.json')

# Filter rows for specific tasks in each dataset
train_df = train_df[train_df['task'].isin(['PG'])]
# Remove unnecessary columns for preprocessing
train_df = train_df.drop(columns=['tgt', 'model', 'ref', 'task'])

test_df = test_df[test_df['task'].isin(['PG'])]
# Remove unnecessary columns for preprocessing
test_df = test_df.drop(columns=['tgt', 'task', 'labels', 'label', 'p(Hallucination)', 'id'])

val_df = val_df[val_df['task'].isin(['PG'])]
# Remove unnecessary columns for preprocessing
val_df = val_df.drop(columns=['tgt', 'model', 'ref', 'task', 'labels', 'label', 'p(Hallucination)'])

# Reset indices for easier handling
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Load Stanza NLP models
stanza.download('en')
nlp_en = stanza.Pipeline('en', processors='tokenize,lemma')

# Text Segmentation and Normalization Function
def normalize_text(text):
    expanded_text = contractions.fix(text).lower()  # Expand contractions and lowercase
    return re.sub(r'[^\w\s]', '', expanded_text)  # Remove punctuation

# Lemmatization Function
def preprocess_text(text):
    doc = nlp_en(text)
    return [word.lemma for sentence in doc.sentences for word in sentence.words]

# Apply Preprocessing Steps to Dataset
def preprocess_dataset(df):
    df['hyp_normalized'] = df['hyp'].apply(normalize_text)
    df['src_normalized'] = df['src'].apply(normalize_text)
    
    # Sentence Segmentation
    df['hyp_sentences'] = df['hyp_normalized'].apply(sent_tokenize)
    df['src_sentences'] = df['src_normalized'].apply(sent_tokenize)
    
    # Tokenization
    df['hyp_tokens'] = df['hyp_sentences'].apply(lambda sentences: [word_tokenize(sentence) for sentence in sentences])
    df['src_tokens'] = df['src_sentences'].apply(lambda sentences: [word_tokenize(sentence) for sentence in sentences])
    
    # Lemmatization
    df['hyp_lemmas'] = df['hyp_normalized'].apply(preprocess_text)
    df['src_lemmas'] = df['src_normalized'].apply(preprocess_text)
    
    return df

# Preprocess each dataset
train_df = preprocess_dataset(train_df)
test_df = preprocess_dataset(test_df)
val_df = preprocess_dataset(val_df)

# Function for Similarity Analysis
def calculate_similarity(df):
    # Join lemmas back into text format
    hyp_lemmas_text = df['hyp_lemmas'].apply(lambda x: ' '.join(x))
    src_lemmas_text = df['src_lemmas'].apply(lambda x: ' '.join(x))
    
    # Cosine Similarity using TF-IDF
    vectorizer = TfidfVectorizer()
    hyp_tfidf = vectorizer.fit_transform(hyp_lemmas_text)
    src_tfidf = vectorizer.transform(src_lemmas_text)
    df['cosine_similarity'] = cosine_similarity(hyp_tfidf, src_tfidf).diagonal()
    
    # Semantic Similarity using Sentence Transformers
    model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    src_embeddings = model.encode(src_lemmas_text.tolist(), convert_to_tensor=True)
    hyp_embeddings = model.encode(hyp_lemmas_text.tolist(), convert_to_tensor=True)
    df['semantic_similarity'] = [sim.item() for sim in util.pytorch_cos_sim(src_embeddings, hyp_embeddings).diag()]
    
    return df

# Apply similarity calculations to each dataset
train_df = calculate_similarity(train_df)
test_df = calculate_similarity(test_df)
val_df = calculate_similarity(val_df)

# Display results for each dataset
print("Train Data:")
print(train_df.head())
print("\nTest Data:")
print(test_df.head())
print("\nValidation Data:")
print(val_df.head())

# Filter rows where cosine_similarity is not 1.0 for further analysis
filtered_train_non_one_similarity = train_df[train_df['semantic_similarity'] < 0.9]
filtered_test_non_one_similarity = test_df[test_df['semantic_similarity'] < 0.9]
filtered_val_non_one_similarity = val_df[val_df['semantic_similarity'] < 0.9]

# Display filtered rows
print("\nTrain Data (Cosine Similarity < 1):")
print(filtered_train_non_one_similarity.head())
print("\nTest Data (Cosine Similarity < 1):")
print(filtered_test_non_one_similarity.head())
print("\nValidation Data (Cosine Similarity < 1):")
print(filtered_val_non_one_similarity.head())