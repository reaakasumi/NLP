# Import required libraries
import pandas as pd
import nltk
import stanza
from nltk.corpus import stopwords
import subprocess

# Download necessary resources
nltk.download('punkt_tab')

# Load the data
train_df = pd.read_json('data/train.model-agnostic.json')

# Check for missing values
print("Missing values in the dataset:")
print(train_df.isnull().sum())

# Analyze categorical columns
categorical_columns = ['task', 'ref', 'model']
for column in categorical_columns:
    print(f"Unique values in '{column}' column:", train_df[column].unique())

# Distribution of tasks
print("Task Distribution:")
print(train_df['task'].value_counts())

# Analyze text length distributions
for col in ['hyp', 'tgt', 'src']:
    train_df[f'{col}_length'] = train_df[col].apply(lambda x: len(x.split()))
    print(f"{col.capitalize()} Text Length Stats:")
    print(train_df[f'{col}_length'].describe())

# Reference usage by task
model_reference_usage = train_df.groupby('task')['ref'].unique().reset_index()
print("Reference types used by each task:")
print(model_reference_usage)

# Sentence analysis function
def analyze_sentences(df, column_name):
    print(f'\nAnalyzing sentences in {column_name.upper()} column:')
    df[f'{column_name}_sentence'] = df[column_name].apply(nltk.sent_tokenize)

    # Check max sentences and tasks with more than one sentence
    maxima = df[f'{column_name}_sentence'].apply(len).max()
    tasks_with_multiple_sentences = df.loc[df[f'{column_name}_sentence'].apply(len) > 1, 'task'].unique()
    count_multiple_sentences = (df[f'{column_name}_sentence'].apply(len) > 1).sum()

    print(f"Task categories with more than 1 sentence: {tasks_with_multiple_sentences}")
    print(f"Maximal number of sentences in {column_name}: {maxima}")
    print(f"Number of entries with more than 1 sentence: {count_multiple_sentences}")

    # Distribution of multiple sentences by task
    df[f'count_{column_name}'] = df[f'{column_name}_sentence'].apply(len)
    df[f'has_multiple_{column_name}_sentences'] = df[f'count_{column_name}'] > 1
    print(f"Number of entries with more than 1 sentence by task:")
    print(df.groupby('task')[f'has_multiple_{column_name}_sentences'].sum())

# Analyze sentences in 'hyp', 'tgt', and 'src' columns
for column in ['hyp', 'tgt', 'src']:
    analyze_sentences(train_df, column)

# Data preprocessing with subprocess (external script execution)
print("\nRunning data preprocessing scripts...")
preprocessing_scripts = [
    'data_preprocessing_conllu.py',  # Replace this with the actual file name if different
    # 'MT_preprocessing.py',
    # 'preprocess_to_conllu_PG.py',
    # 'processing_dm.py'
]

for script in preprocessing_scripts:
    subprocess.run(['python', script])
    print(f"{script} executed successfully.")

print("Preprocessing completed!")
