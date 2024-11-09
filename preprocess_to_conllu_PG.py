import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import stanza
import re
import contractions
from stanza.utils.conll import CoNLL

# Initialize Stanza NLP model
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,lemma,pos')

# Load Datasets and Filter for PG Task
def load_and_filter_data(file_path):
    df = pd.read_json(file_path)
    df = df[df['task'].isin(['PG'])].reset_index(drop=True)
    columns_to_drop = [col for col in ['tgt', 'model', 'ref', 'task', 'labels', 'label', 'p(Hallucination)', 'id'] if col in df.columns]
    df = df.drop(columns=columns_to_drop)
    return df

train_df = load_and_filter_data('data/train.model-agnostic.json')
test_df = load_and_filter_data('data/test.model-agnostic.json')
val_df = load_and_filter_data('data/val.model-agnostic.json')

# Text Segmentation and Normalization Function
def normalize_text(text):
    expanded_text = contractions.fix(text).lower()  # Expand contractions and lowercase
    return re.sub(r'[^\w\s]', '', expanded_text)  # Remove punctuation

# Lemmatization Function
def lemmatize_text(text):
    doc = nlp(text)
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
    df['hyp_lemmas'] = df['hyp_normalized'].apply(lemmatize_text)
    df['src_lemmas'] = df['src_normalized'].apply(lemmatize_text)
    
    return df

train_df = preprocess_dataset(train_df)
test_df = preprocess_dataset(test_df)
val_df = preprocess_dataset(val_df)

# Helper Function to Save to CoNLL Format
def save_column_to_conllu(df, column_name, output_file):
    texts = ". ".join(df[column_name])
    doc = nlp(texts)
    CoNLL.write_doc2conll(doc, output_file)

# Export Normalized and Lemmatized Columns to CoNLL Format
def export_datasets_to_conllu(df, prefix):
    save_column_to_conllu(df, "hyp_normalized", f"conllu_files/pg_{prefix}_hyp.conllu")
    save_column_to_conllu(df, "src_normalized", f"conllu_files/pg_{prefix}_src.conllu")

export_datasets_to_conllu(train_df, "train")
export_datasets_to_conllu(test_df, "test")
export_datasets_to_conllu(val_df, "val")
