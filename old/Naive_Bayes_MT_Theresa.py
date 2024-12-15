import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import category_encoders as ce
import spacy
import re

# Load the spaCy model for preprocessing
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

# Load the dataset
data = pd.read_csv('data/mt_train_label.csv')

# Drop rows with missing values
data = data.dropna()

# Apply preprocessing to hypothesis and target text
data['hyp_normalized'] = data['hyp_normalized'].apply(preprocess_text)
data['tgt_normalized'] = data['tgt_normalized'].apply(preprocess_text)

# Combine the text fields for further processing
data['combined_text'] = data['hyp_normalized'] + " " + data['tgt_normalized']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data[['hyp_normalized', 'tgt_normalized']], data['label'], test_size=0.2, random_state=42
)

# One-hot encode the text features
encoder = ce.OneHotEncoder(cols=['hyp_normalized', 'tgt_normalized'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# Train a Naive Bayes classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)

# Predict
y_pred = naive_bayes.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
