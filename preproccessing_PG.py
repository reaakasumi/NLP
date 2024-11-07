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

# Load Dataset
train_df = pd.read_json('train.model-agnostic.json')

# Select rows where the 'task' column is either 'MT' or 'PG'
filtered_df = train_df[train_df['task'].isin(['PG'])]

# Reset the index 
filtered_df = filtered_df.reset_index(drop=True)

# Load Stanza NLP models
stanza.download('en')
nlp_en = stanza.Pipeline('en', processors='tokenize,lemma')


# Text Segmentation and Normalization
def normalize_text(text):
    # Expand contractions (e.g., "you're" to "you are")
    expanded_text = contractions.fix(text)
    # Lowercase the expanded text
    expanded_text = expanded_text.lower()  
    # Remove punctuation
    expanded_text = re.sub(r'[^\w\s]', '', expanded_text)
    return expanded_text

# Apply normalization
filtered_df['hyp_normalized'] = filtered_df['hyp'].apply(normalize_text)
filtered_df['src_normalized'] = filtered_df['src'].apply(normalize_text)

# Sentence Segmentation
filtered_df['hyp_sentences'] = filtered_df['hyp_normalized'].apply(sent_tokenize)
filtered_df['src_sentences'] = filtered_df['src_normalized'].apply(sent_tokenize)

# Tokenization
filtered_df['hyp_tokens'] = filtered_df['hyp_sentences'].apply(lambda sentences: [word_tokenize(sentence) for sentence in sentences])
filtered_df['src_tokens'] = filtered_df['src_sentences'].apply(lambda sentences: [word_tokenize(sentence) for sentence in sentences])

# Lemmatization
def preprocess_text(text):
    doc = nlp_en(text)  
    lemmas = [word.lemma for sentence in doc.sentences for word in sentence.words]
    return lemmas

# Apply lemmatization 
filtered_df['hyp_lemmas'] = filtered_df['hyp_normalized'].apply(preprocess_text)
filtered_df['src_lemmas'] = filtered_df['src_normalized'].apply(preprocess_text)

# Join lemmas back into text format for each row only within the similarity analysis functions
hyp_lemmas_text = filtered_df['hyp_lemmas'].apply(lambda x: ' '.join(x))
src_lemmas_text = filtered_df['src_lemmas'].apply(lambda x: ' '.join(x))

# Similarity Analysis
# Cosine Similarity using TF-IDF
vectorizer = TfidfVectorizer()
hyp_tfidf = vectorizer.fit_transform(hyp_lemmas_text)
src_tfidf = vectorizer.transform(src_lemmas_text)
filtered_df['cosine_similarity'] = cosine_similarity(hyp_tfidf, src_tfidf).diagonal()

# Semantic Similarity using Sentence Transformers
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
src_embeddings = model.encode(src_lemmas_text.tolist(), convert_to_tensor=True)
hyp_embeddings = model.encode(hyp_lemmas_text.tolist(), convert_to_tensor=True)
filtered_df['semantic_similarity'] = [sim.item() for sim in util.pytorch_cos_sim(src_embeddings, hyp_embeddings).diag()]

# Display results
filtered_df.head()

# Filter rows where cosine_similarity is not 1.0
filtered_non_one_similarity = filtered_df[filtered_df['cosine_similarity'] < 1]

# Display the filtered rows
filtered_non_one_similarity.head()