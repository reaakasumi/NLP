import json
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import stanza
from stanza.utils.conll import CoNLL
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# loading data
data_train = open("data/train.model-agnostic.json")
data_train = json.load(data_train)
data_test = open("data/test.model-agnostic.json")
data_test = json.load(data_test)
data_val = open("data/val.model-agnostic.json")
data_val = json.load(data_val)


# converting to dataframe
df_train = pd.DataFrame(data_train)
df_test = pd.DataFrame(data_test)
df_val = pd.DataFrame(data_val)


# function for removing stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(sentence):

    # Tokenize the text
    words = word_tokenize(sentence)

    # Filter out stopwords
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Join the filtered words back into a single string
    filtered_text = " ".join(filtered_words)
    return filtered_text


# stanza pipeline for tokenization and lemmatization
nlp = stanza.Pipeline('en', processors='tokenize,lemma,pos')


# function for preprocessing one column
def process_column(column, stopwords = False):

    # bringing everything to lower case
    column = column.str.lower()
    
    # removing anything that are not words
    column = column.replace(to_replace=r'[^\w\s]', value='', regex=True)

    if stopwords == True:
        column = column.apply(remove_stopwords)

    list_input = list(column)
    output_doc = nlp(". ".join(list_input))

    return output_doc



# creating conllu file out of column

# training data
hyp_dm_doc = process_column(df_train["hyp"])
CoNLL.write_doc2conll(hyp_dm_doc,"data/train_hyp_dm.conllu")

tgt_dm_doc = process_column(df_train["tgt"])
CoNLL.write_doc2conll(tgt_dm_doc,"data/train_tgt_dm.conllu")

# testing data
hyp_dm_doc_test = process_column(df_test["hyp"])
CoNLL.write_doc2conll(hyp_dm_doc_test,"data/test_hyp_dm.conllu")

tgt_dm_doc_test = process_column(df_test["tgt"])
CoNLL.write_doc2conll(tgt_dm_doc_test,"data/test_tgt_dm.conllu")

# testing data
hyp_dm_doc_val = process_column(df_val["hyp"])
CoNLL.write_doc2conll(hyp_dm_doc_val,"data/val_hyp_dm.conllu")

tgt_dm_doc_val = process_column(df_val["tgt"])
CoNLL.write_doc2conll(tgt_dm_doc_val,"data/val_tgt_dm.conllu")

