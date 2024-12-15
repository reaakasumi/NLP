import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import spacy
import numpy as np
import re

# Load the dataset
data = pd.read_csv('data/mt_train_label.csv')

# Drop rows with missing values
data = data.dropna()

# Load the spaCy model for preprocessing and NER
nlp = spacy.load("en_core_web_sm")

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize and lemmatize using spaCy
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)

# Apply preprocessing to both hypothesis and target text
data['hyp_normalized'] = data['hyp_normalized'].apply(preprocess_text)
data['tgt_normalized'] = data['tgt_normalized'].apply(preprocess_text)

# Combine the text fields for TF-IDF feature extraction
data['combined_text'] = data['hyp_normalized'] + " " + data['tgt_normalized']

# Function to extract named entities
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Apply NER to the dataset
data['hyp_entities'] = data['hyp_normalized'].apply(lambda x: extract_entities(x))
data['tgt_entities'] = data['tgt_normalized'].apply(lambda x: extract_entities(x))

# Calculate entity overlap ratio
def entity_overlap(hyp, tgt):
    hyp_entities = {ent[0] for ent in hyp}  # Set of entities in hypothesis
    tgt_entities = {ent[0] for ent in tgt}  # Set of entities in target
    overlap = len(hyp_entities & tgt_entities)  # Intersection of entities
    total = len(hyp_entities | tgt_entities)   # Union of entities
    return overlap / total if total > 0 else 0

# Add entity overlap as a feature
data['entity_overlap'] = data.apply(lambda row: entity_overlap(row['hyp_entities'], row['tgt_entities']), axis=1)

# Split the data into train and test sets
X_train_text, X_test_text, y_train, y_test = train_test_split(
    data['combined_text'], data['label'], test_size=0.2, random_state=42
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# Convert TF-IDF sparse matrix to dense
X_train_tfidf_dense = X_train_tfidf.toarray()
X_test_tfidf_dense = X_test_tfidf.toarray()

# Add the entity_overlap feature to the dense matrix
X_train = np.hstack((X_train_tfidf_dense, data.loc[X_train_text.index, 'entity_overlap'].values.reshape(-1, 1)))
X_test = np.hstack((X_test_tfidf_dense, data.loc[X_test_text.index, 'entity_overlap'].values.reshape(-1, 1)))

# Train a Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Predict on the test set
y_pred = logistic_model.predict(X_test)

# Evaluate the model
print("Logistic Regression Results with Preprocessing and NER Features")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
