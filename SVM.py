import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

###################### Preparing Data #########################

# Load validation and testing data
df_mt_val = pd.read_csv('data/mt_val_label.csv')
df_mt_test = pd.read_csv('data/mt_test_label.csv')

df_dm_val = pd.read_csv('data/dm_val_label.csv')
df_dm_test = pd.read_csv('data/dm_test_label.csv')

df_pg_val = pd.read_csv('data/pg_val_label.csv')
df_pg_test = pd.read_csv('data/pg_test_label.csv')

# Convert labels to binary
def preprocess_labels(df):
    df['true_labels'] = df['label'].apply(lambda x: 0 if x == "Not Hallucination" else 1)
    return df

# Prepare each dataset
df_mt_val = preprocess_labels(df_mt_val)
df_mt_test = preprocess_labels(df_mt_test)
df_dm_val = preprocess_labels(df_dm_val)
df_dm_test = preprocess_labels(df_dm_test)
df_pg_val = preprocess_labels(df_pg_val)
df_pg_test = preprocess_labels(df_pg_test)

# Ensure all datasets have the 'tgt_normalized' column
for df in [df_mt_val, df_mt_test, df_dm_val, df_dm_test, df_pg_val, df_pg_test]:
    if 'tgt_normalized' not in df.columns:
        df['tgt_normalized'] = ''  # Fill with empty strings if missing

###################### Function to Train and Test SVM Model #########################

def train_and_evaluate_svm(df_train, df_test, task_name):
    # Combine hypothesis and target columns as input features
    df_train['combined_text'] = df_train['hyp_normalized'] + " " + df_train['tgt_normalized']
    df_test['combined_text'] = df_test['hyp_normalized'] + " " + df_test['tgt_normalized']

    # Handle missing values by filling them with an empty string
    df_train['combined_text'] = df_train['combined_text'].fillna('')
    df_test['combined_text'] = df_test['combined_text'].fillna('')

    # Use TfidfVectorizer to convert text to numerical features
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(df_train['combined_text']).toarray()
    X_test = vectorizer.transform(df_test['combined_text']).toarray()

    y_train = df_train['true_labels'].values
    y_test = df_test['true_labels'].values

    # Train an SVM classifier
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm_model.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Results for {task_name} Task:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("\n")

    # Save predictions
    df_test[f'predicted_svm_{task_name.lower()}'] = y_pred
    df_test.to_csv(f"data/test_results_svm_{task_name.lower()}.csv", index=False)
    print(f"Results saved for {task_name} task in 'data/test_results_svm_{task_name.lower()}.csv'.")

###################### Train and Evaluate for Each Task #########################

# Train and evaluate separately for MT, DM, and PG tasks
train_and_evaluate_svm(df_mt_val, df_mt_test, "MT")
train_and_evaluate_svm(df_dm_val, df_dm_test, "DM")
train_and_evaluate_svm(df_pg_val, df_pg_test, "PG")
