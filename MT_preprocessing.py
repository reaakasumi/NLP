
# function for removing stop words

# removing stopwords
#import json
import pandas as pd
#import re
import nltk
from nltk.tokenize import word_tokenize
import stanza
from stanza.utils.conll import CoNLL
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

#data = open("train.model-agnostic.json")
#data = json.load(data)

df_mt = pd.read_csv('df_mt.csv')
val_mt = pd.read_csv('val_mt.csv')
test_mt = pd.read_csv('test_mt.csv')





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
#train
hyp_mt_doc = process_column(df_mt["hyp"])
CoNLL.write_doc2conll(hyp_mt_doc,"Data/train_hyp_mt.conllu")

tgt_mt_doc = process_column(df_mt["tgt"])
CoNLL.write_doc2conll(tgt_mt_doc,"Data/train_tgt_mt.conllu")
#val 
hyp_val_mt_doc = process_column(val_mt["hyp"])
CoNLL.write_doc2conll(hyp_val_mt_doc,"Data/val_hyp_mt.conllu")

tgt_val_mt_doc = process_column(val_mt["tgt"])
CoNLL.write_doc2conll(tgt_val_mt_doc,"Data/val_tgt_mt.conllu")
#test 
hyp_test_mt_doc = process_column(test_mt["hyp"])
CoNLL.write_doc2conll(hyp_mt_doc,"Data/test_hyp_mt.conllu")

tgt_test_mt_doc = process_column(test_mt["tgt"])
CoNLL.write_doc2conll(tgt_test_mt_doc,"Data/test_tgt_mt.conllu")