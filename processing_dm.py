
# imports
import json
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import stanza
from stanza.utils.conll import CoNLL

# loading data
data = open("data/train.model-agnostic.json")
data = json.load(data)


# converting to dataframe
df = pd.DataFrame(data)

# getting rid of column model
df = df.drop( columns = ["model"])


# only processing DM
df_dm = df[df["task"] == "DM"]


# function for removing stop words

# removing stopwords
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



# function for creating conllu file out of column

hyp_dm_doc = process_column(df_dm["hyp"])
CoNLL.write_doc2conll(hyp_dm_doc,"data/train_hyp_dm.conllu")

tgt_dm_doc = process_column(df_dm["tgt"])
CoNLL.write_doc2conll(tgt_dm_doc,"data/train_tgt_dm.conllu")

