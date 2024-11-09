import json
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from conllu import TokenList
import spacy


# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def load_json_to_dataframe(file_path, columns):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return pd.DataFrame(data, columns=columns)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[.,!?:"\'-]', '', text)  # Remove punctuation except ;
    text = text.strip()
    return text


def segment_text(text):
    segments = [segment.strip() for segment in text.split(';') if segment.strip()]
    return segments


def tokenize_text(segments):
    tokenized_segments = [word_tokenize(segment) for segment in segments]
    return tokenized_segments


def normalize_text(tokens, remove_stopwords=True):
    normalized_tokens = []
    lemmas = []
    for token in tokens:
        if remove_stopwords and token in stop_words:
            continue
        lemma = lemmatizer.lemmatize(token)
        normalized_tokens.append(token)
        lemmas.append(lemma)
    return list(zip(normalized_tokens, lemmas))


def convert_to_conllu_with_spacy(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in data:
            # Reconstruct the sentence text
            text = ' '.join([token for token, _ in sentence])
            doc = nlp(text)
            conllu_sentence = []
            for token in doc:
                # Correct assignment of dependency roles and head
                head = token.head.i + 1 if token.head != token else 0  # Root has head = 0
                deprel = token.dep_ if token.dep_ != 'ROOT' else 'root'
                # Create a dictionary for each token following the CoNLL-U format
                token_entry = {
                    'id': token.i + 1,
                    'form': token.text,
                    'lemma': token.lemma_,
                    'upostag': token.pos_,
                    'xpostag': None,
                    'feats': None,
                    'head': head,
                    'deprel': deprel,
                    'deps': None,
                    'misc': None
                }
                conllu_sentence.append(token_entry)
            token_list = TokenList(conllu_sentence)
            f.write(token_list.serialize())
            f.write("\n")  # Blank line after each sentence


def main():
    file_path = 'train.model-agnostic.json'
    output_path = 'output.conllu'
    columns = ['hyp', 'tgt', 'src', 'ref', 'task', 'model']

    # Load data
    df = load_json_to_dataframe(file_path, columns)

    # Drop unnecessary columns
    df = df.drop(columns=['ref', 'task', 'model'])

    # Preprocess, segment, tokenize, and normalize text in each column, storing results for CoNLL-U format
    conllu_data = []
    for col in ['hyp', 'tgt', 'src']:
        df[col] = df[col].apply(preprocess_text)  # Preprocess text
        df[col] = df[col].apply(segment_text)  # Segment by semicolon
        df[col] = df[col].apply(tokenize_text)  # Tokenize each segment
        # Normalize text with lemmatization and stop word removal, and prepare for CoNLL-U format
        df[col] = df[col].apply(lambda segments: [normalize_text(segment) for segment in segments])

        # Flatten sentences and append to conllu_data for each column
        for row in df[col]:
            for sentence in row:
                conllu_data.append(sentence)

    # Convert to CoNLL-U format and save using spaCy
    convert_to_conllu_with_spacy(conllu_data, output_path)
    print("Data saved successfully to", output_path)


if __name__ == "__main__":
    main()
