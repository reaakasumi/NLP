import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, GlobalMaxPool1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


###################### Function to Preprocess Data #########################

def preprocess_labels(df):
    """Convert labels to binary."""
    df['true_labels'] = df['label'].apply(lambda x: 0 if x == "Not Hallucination" else 1)
    return df


def combine_text(df):
    """Combine hypothesis and target columns."""
    df['combined_text'] = df['hyp_normalized'] + " " + df['tgt_normalized']
    df['combined_text'] = df['combined_text'].fillna('')
    return df


###################### Function to Train and Evaluate Model #########################

def train_and_evaluate(task_name, df_val, df_test):
    """Train and evaluate BiLSTM model for a specific task."""
    print(f"Training and evaluating for task: {task_name}")

    # Preprocess data
    df_val = preprocess_labels(df_val)
    df_test = preprocess_labels(df_test)

    if 'tgt_normalized' not in df_val.columns:
        df_val['tgt_normalized'] = ''
    if 'tgt_normalized' not in df_test.columns:
        df_test['tgt_normalized'] = ''

    df_val = combine_text(df_val)
    df_test = combine_text(df_test)

    # Tokenization and Padding
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df_val['combined_text'])

    X_train = tokenizer.texts_to_sequences(df_val['combined_text'])
    X_test = tokenizer.texts_to_sequences(df_test['combined_text'])

    max_length = 100
    X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')
    X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')

    y_train = df_val['true_labels'].values
    y_test = df_test['true_labels'].values

    # Define BiLSTM Model
    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=max_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPool1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())

    # Train the Model
    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate the Model
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    def evaluate_model(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"Results for {task_name} Task:")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1)
        return accuracy, precision, recall, f1

    evaluate_model(y_test, y_pred)

    # Save the Model
    model.save(f"bilstm_hallucination_model_{task_name.lower()}.h5")
    print(f"Model saved as 'bilstm_hallucination_model_{task_name.lower()}.h5'.\n")


###################### Load Data and Run for Each Task #########################

# Load validation and testing data
df_mt_val = pd.read_csv('mt_val_label.csv')
df_mt_test = pd.read_csv('mt_test_label.csv')

df_dm_val = pd.read_csv('dm_val_label.csv')
df_dm_test = pd.read_csv('dm_test_label.csv')

df_pg_val = pd.read_csv('pg_val_label.csv')
df_pg_test = pd.read_csv('pg_test_label.csv')

# Train and evaluate separately for MT, DM, and PG tasks
train_and_evaluate("MT", df_mt_val, df_mt_test)
train_and_evaluate("DM", df_dm_val, df_dm_test)
train_and_evaluate("PG", df_pg_val, df_pg_test)
