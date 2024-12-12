import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
import ast
import nltk

nltk.download('stopwords')

def tune_hyperparameters(X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV."""
    # Define the pipeline for preprocessing and model training
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stopwords.words('english'))),  # Vectorizer with stopword removal
        ('nb', MultinomialNB())       # Naive Bayes classifier
    ])

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'tfidf__max_features': [2000, 3000, 5000, 10000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__min_df': [1, 2, 3],
        'nb__alpha': [1.0, 0.5, 0.1, 0.01]
    }

    # Perform grid search with stratified cross-validation
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_cv, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Return the best parameters and the best model from grid search
    return grid_search.best_params_, grid_search.best_estimator_

def preprocess_data(train_df, test_df):
    """Preprocess training and testing datasets."""
    # Drop missing values
    train_data = train_df.dropna()
    test_data = test_df.dropna()

    # Convert string representations of lists to actual lists
    train_data['hyp_lemmas'] = train_data['hyp_lemmas'].apply(ast.literal_eval)
    train_data['tgt_src_lemmas'] = train_data['tgt_src_lemmas'].apply(ast.literal_eval)
    test_data['hyp_lemmas'] = test_data['hyp_lemmas'].apply(ast.literal_eval)
    test_data['tgt_src_lemmas'] = test_data['tgt_src_lemmas'].apply(ast.literal_eval)

    # Combine `hyp_lemmas` and `res_lemmas` into a single text feature
    train_data['combined_text'] = train_data['hyp_lemmas'].apply(lambda x: " ".join(x)) + " " + train_data['tgt_src_lemmas'].apply(lambda x: " ".join(x))
    test_data['combined_text'] = test_data['hyp_lemmas'].apply(lambda x: " ".join(x)) + " " + test_data['tgt_src_lemmas'].apply(lambda x: " ".join(x))

    # Extract features (text) and labels
    X_train_full = train_data['combined_text']
    y_train_full = train_data['label']

    X_test = test_data['combined_text']
    y_test = test_data['label']

    return X_train_full, y_train_full, X_test, y_test, test_data

def train_and_evaluate(X_train_full, y_train_full, X_test, y_test, test_data):
    """Train and evaluate the model with hyperparameter tuning."""
    # Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    # Vectorize text data using TF-IDF with stopword removal
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words=stopwords.words('english'))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)

    # Perform hyperparameter tuning to find the best model
    best_params, best_model = tune_hyperparameters(X_train, y_train)

    # Evaluate the best model on the validation set
    y_val_pred = best_model.predict(X_val)
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))

    # Retrain the best model on the full training data (including validation)
    best_model.fit(X_train_full, y_train_full)

    # Evaluate the retrained model on the test set
    y_test_pred = best_model.predict(X_test)
    print(f"Best Parameters: {best_params}")
    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred))

    # Add predicted labels to the test data for manual inspection
    test_data['predicted_label'] = y_test_pred
    print("Test data with actual and predicted labels:")
    print(test_data[['combined_text', 'label', 'predicted_label']])

    # Save the test data with predictions to a CSV file
    test_data.to_csv("test_data_with_predictions.csv", index=False)
    print("Predictions saved to 'test_data_with_predictions.csv'.")

# Main process
if __name__ == "__main__":
    train_df = pd.read_csv('data/labeled_data/preprocessed/train_preprocessed.csv')
    test_df = pd.read_csv('data/labeled_data/preprocessed/test_preprocessed.csv')

    # Preprocess data and extract features and labels
    X_train_full, y_train_full, X_test, y_test, test_data = preprocess_data(train_df, test_df)

    # Train the model and evaluate its performance
    train_and_evaluate(X_train_full, y_train_full, X_test, y_test, test_data)
