# 

import pandas as pd
#import json
#import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import stanza
nltk.download('punkt')
nltk.download('stopwords')
stanza.download('en')
stanza.download('de')

#Read the data 
train_df = pd.read_json('train.model-agnostic.json')
#train_df2 = pd.read_json('train.model-aware.v2.json')

val_df = pd.read_json('val.model-agnostic.json')

test_df = pd.read_json('test.model-agnostic.json')



#First information on data set 
# task differentiates between 3 different categories: 
#DM is defintion modelling 
#MT is machine translation 
#PG is paraphrase generation 
#MT and PG are dual-referential (are target or source referential)
#DM is target-referential


#now we do the same approach for 3 columns hyp,tgt, src: 
#bringing all into lowercase 
#removing non-words and non-whitespace characters 
#we split the text into sentences 
# then we split the sentences into words 
#We will do stopword removal and store it in an extra dataset for comparison in the later model evaluation 
#Next we normalize the text --> already done with lowercase and removing non-white space 
#Then we do Lemmatization and Stemming 


#First we start with the hyp columm : 
    #hyp = a hypothesis, the actual model predicts


#lowercase 
train_df['hyp'] = train_df['hyp'].str.lower()
#replace non-words and non-whitespace
train_df['hyp'] = train_df['hyp'].replace(to_replace=r'[^\w\s]', value='', regex=True)
#split text into sentences by using ;
train_df['hyp_sentence'] = train_df['hyp'].apply(nltk.sent_tokenize)

#With this code you can see, that there are some entries with more than 1 sentence
#for i in range(0,len(train_df['hyp_sentence'])): 
#    if len(train_df['hyp_sentence'][i]) > 1: 
#        print(train_df['hyp_sentence'][i])


#Next split it into words
train_df['hyp_words'] = train_df['hyp_sentence'].apply(lambda sentences: [word_tokenize(sentence) for sentence in sentences])

#Ansatz zwei: ohne sentence splitting, only into words 
train_df['hyp_words2'] = train_df['hyp'].apply(nltk.word_tokenize)

#Now we split the dataset into 2: with stopword removal, and without 


stop_words = set(stopwords.words('english'))
train_df_stop = train_df.copy()

#alternative approach 
#with not splitting into sente
train_df_stop['hyp_stop'] = train_df_stop['hyp_words2'].apply(lambda x: [word for word in x if word not in stop_words])


#Lemmatization and Stemmation
stemmer = PorterStemmer()
nlp = stanza.Pipeline('en', processors='tokenize,lemma,pos')
train_df['hyp_stemmed'] = train_df['hyp_words2'].apply(lambda x: [stemmer.stem(word) for word in x])
train_df_stop['hyp_stemmed'] = train_df_stop['hyp_stop'].apply(lambda x: [stemmer.stem(word) for word in x])


#Lemmatization isnt working, the program starts running and doesnt stop (maybe its working but it takes longer than 20 min)

#train_df['hyp_end'] = train_df['hyp_stemmed'].apply(lambda x: [nlp(word) for word in x])
#tgt: a target the intended reference 'gold' text that the model ought to generate 




