

import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import stanza
import re
from nltk.corpus import stopwords
import contractions
from stanza.utils.conll import CoNLL

# Initialize Stanza NLP model
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,lemma,pos')

# Load Datasets 
train_df = pd.read_json('data/train.model-agnostic.json')
test_df = pd.read_json('data/test.model-agnostic.json')
val_df = pd.read_json('data/val.model-agnostic.json')
train_labeled = pd.read_json('data/labeled-train-model-agnostic.json')
#test_labeled = pd.read_json('data/labeled-test-model-agnostic.json')

# Filter for Specific Task
def load_and_filter_data(df, task_type, column_to_keep):
    # Filter for the specified task type
    df = df[df['task'].isin([task_type])].reset_index(drop=True)
    
    # Define columns to drop, keeping only "hyp" and the specified column ("src" or "tgt")
    columns_to_drop = [col for col in df.columns if col not in ['hyp','label', column_to_keep]]
    df = df.drop(columns=columns_to_drop)
    
    return df

# Load train, test, and validation datasets with specified task type and column to keep
pg_train_df = load_and_filter_data(train_df, 'PG', 'src')
pg_train_label_df = load_and_filter_data(train_labeled, 'PG', 'src')
pg_test_df = load_and_filter_data(test_df, 'PG', 'src')
pg_val_df = load_and_filter_data(val_df, 'PG', 'src')

mt_train_df = load_and_filter_data(train_df, 'MT', 'tgt')
mt_train_label_df = load_and_filter_data(train_labeled, 'MT', 'tgt')
mt_test_df = load_and_filter_data(test_df, 'MT', 'tgt')
mt_val_df = load_and_filter_data(val_df, 'MT', 'tgt')

dm_train_df = load_and_filter_data(train_df, 'DM', 'tgt')
dm_train_label_df = load_and_filter_data(train_labeled, 'DM', 'tgt')
dm_test_df = load_and_filter_data(test_df, 'DM', 'tgt')
dm_val_df = load_and_filter_data(val_df, 'DM', 'tgt')

# Text Segmentation, Normalization, and Stopword Removal Function
def normalize_text(text, remove_stopwords=False):
    expanded_text = contractions.fix(text).lower()  # Expand contractions and lowercase
    text_no_punctuation = re.sub(r'[^\w\s]', '', expanded_text) # Remove punctuation
    
    # Optionally remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        # Tokenize to remove stopwords and then join back to a single string
        text_no_stopwords = ' '.join(word for word in text_no_punctuation.split() if word not in stop_words)
        return text_no_stopwords
    
    return text_no_punctuation

# Lemmatization Function
def lemmatize_text(text):
    doc = nlp(text)
    return [word.lemma for sentence in doc.sentences for word in sentence.words]

# Apply Preprocessing Steps to Dataset
def preprocess_dataset(df, column):
    # Normalize text
    df['hyp_normalized'] = df['hyp'].apply(normalize_text)
    df[f'{column}_normalized'] = df[column].apply(normalize_text)
    
    # Sentence Segmentation
    df['hyp_sentences'] = df['hyp_normalized'].apply(sent_tokenize)
    df[f'{column}_sentences'] = df[f'{column}_normalized'].apply(sent_tokenize)
    
    # Tokenization
    df['hyp_tokens'] = df['hyp_sentences'].apply(lambda sentences: [word_tokenize(sentence) for sentence in sentences])
    df[f'{column}_tokens'] = df[f'{column}_sentences'].apply(lambda sentences: [word_tokenize(sentence) for sentence in sentences])
    
    # Lemmatization
    # df['hyp_lemmas'] = df['hyp_normalized'].apply(lemmatize_text)
    # df[f'{column}_lemmas'] = df[f'{column}_normalized'].apply(lemmatize_text)
    
    return df

pg_train_df = preprocess_dataset(pg_train_df, 'src')
pg_train_label_df = preprocess_dataset(pg_train_label_df, 'src')
pg_test_df = preprocess_dataset(pg_test_df, 'src')
pg_val_df = preprocess_dataset(pg_val_df, 'src')

mt_train_df = preprocess_dataset(mt_train_df, 'tgt')
mt_train_label_df = preprocess_dataset(mt_train_label_df, 'tgt')
mt_test_df = preprocess_dataset(mt_test_df, 'tgt')
mt_val_df = preprocess_dataset(mt_val_df, 'tgt')

dm_train_df = preprocess_dataset(dm_train_df, 'tgt')
dm_train_label_df = preprocess_dataset(dm_train_label_df, 'tgt')
dm_test_df = preprocess_dataset(dm_test_df, 'tgt')
dm_val_df = preprocess_dataset(dm_val_df, 'tgt')

# Helper Function to Save to CoNLL Format
def save_column_to_conllu(df, column_name, output_file):
    texts = ". ".join(df[column_name])
    doc = nlp(texts)
    CoNLL.write_doc2conll(doc, output_file)

# Export Normalized and Lemmatized Columns to CoNLL Format
def export_datasets_to_conllu(df, prefix, column):
    save_column_to_conllu(df, "hyp_normalized", f"conllu_files/{prefix}_hyp.conllu")
    save_column_to_conllu(df, f"{column}_normalized", f"conllu_files/{prefix}_src.conllu")

#export_datasets_to_conllu(pg_train_df, "pg_train", 'src')
#export_datasets_to_conllu(pg_test_df, "pg_test", 'src')
#export_datasets_to_conllu(pg_val_df, "pg_val", 'src')
#print('PG Files created')

#export_datasets_to_conllu(mt_train_df, "mt_train", 'tgt')
#export_datasets_to_conllu(mt_test_df, "mt_test", 'tgt')
#export_datasets_to_conllu(mt_val_df, "mt_val", 'tgt')
#print('MT Files created')

#export_datasets_to_conllu(dm_train_df, "dm_train", 'tgt')
#export_datasets_to_conllu(dm_test_df, "dm_test", 'tgt')
#export_datasets_to_conllu(dm_val_df, "dm_val", 'tgt')
#print('DM Files created')

mt_test_df.to_csv('mt_test_label.csv', index = False)
mt_val_df.to_csv('mt_val_label.csv', index = False)
mt_train_label_df.to_csv('mt_train_label.csv', index = False)
dm_train_label_df.to_csv('dm_train_label.csv', index = False)
pg_train_label_df.to_csv('pg_train_label.csv', index = False)

dm_val_df.to_csv('dm_val_label.csv', index = False)
dm_test_df.to_csv('dm_test_label.csv', index = False)

pg_val_df.to_csv('pg_val_label.csv', index = False)
pg_test_df.to_csv('pg_test_label.csv', index= False)
