#install packages via pip
#pip install pandas numpy scipy torch scikit-learn nltk stanza re contractions


#import all the packages 
import pandas as pd
#from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
#from transformers import pipeline
import nltk
import stanza
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))




nltk.download('punkt')
nltk.download('stopwords')
stanza.download('en')
stanza.download('de')


#read the data
train_df = pd.read_json('data/train.model-agnostic.json')


#HERE BEGINS DATA ANALYSIS

# Check for missing values
print(train_df.isnull().sum())

# Get unique values in categorical columns
print("Unique values in 'task' column:", train_df['task'].unique())
print("Unique values in 'ref' column:", train_df['ref'].unique())
print("Unique values in 'model' column:", train_df['model'].unique())


# Check data distribution in the 'task' column
print("Task Distribution:\n", train_df['task'].value_counts())

# Look at text length distributions for hyp, tgt, and src columns
train_df['hyp_length'] = train_df['hyp'].apply(lambda x: len(x.split()))
train_df['tgt_length'] = train_df['tgt'].apply(lambda x: len(x.split()))
train_df['src_length'] = train_df['src'].apply(lambda x: len(x.split()))

# Summary statistics of lengths
print("Hypothesis Text Length Stats:\n", train_df['hyp_length'].describe())
print("Target Text Length Stats:\n", train_df['tgt_length'].describe())
print("Source Text Length Stats:\n", train_df['src_length'].describe())


# Group by model and list unique reference types for each model
model_reference_usage = train_df.groupby('task')['ref'].unique().reset_index()

# Display reference types used by each model
print("Reference types used by each task:")
print(model_reference_usage)


#Analysis on sentences, to decide if sentence split is necessary
#First we started with the *hyp* column 

print('HYP COLUMN')
train_df['hyp_sentence'] = train_df['hyp'].apply(nltk.sent_tokenize)

#Analysis on sentences 
#for i in range(0,len(train_df['hyp_sentence'])): 
#    if len(train_df['hyp_sentence'][i]) > 1: 
#        print(train_df['hyp_sentence'][i])

#
maxima = 0
list = []
for i in range(0,len(train_df['hyp_sentence'])): 
    if len(train_df['hyp_sentence'][i]) > 1: 
        if len(train_df['hyp_sentence'][i]) > maxima: 
            maxima = len(train_df['hyp_sentence'][i])
        list.append(train_df['task'][i])
print('Task categories where more than 1 sentence occur:')
print(set(list))    
print('Maximal number of sentences in hyp column: ' + str(maxima))
#We can see that there occur more than 1 sentence in every task category


count = 0
for i in range(0,len(train_df['hyp_sentence'])): 
    if len(train_df['hyp_sentence'][i]) > 1: 
        count = count +1
print('Number of columns with more than 1 sentence: ' +str(count))

#Because there are only 243 entries with more than 1 sentence we decide to skip sentence splitting 

#we take a look at the distribution of number of sentences according to task
print('Number of entries with more than one sentences split into task categories')
train_df['count_hyp'] = train_df['hyp_sentence'].apply(len)
train_df['count_hyp2'] = train_df['count_hyp'] > 1
print(train_df.groupby('task')['count_hyp2'].sum())

#Next we take a look at *tgt* 

print('TGT COLUMN')
train_df['tgt_sentence'] = train_df['tgt'].apply(nltk.sent_tokenize)

#
maxima = 0
list = []
for i in range(0,len(train_df['tgt_sentence'])): 
    if len(train_df['tgt_sentence'][i]) > 1: 
        if len(train_df['tgt_sentence'][i]) > maxima: 
            maxima = len(train_df['tgt_sentence'][i])
        list.append(train_df['task'][i])
print('Task categories where more than 1 sentence occur:')
print(set(list))    
print('Maximal number of sentences in tgt column: ' + str(maxima))
#We can see that there occur more than 1 sentence in every task category


count = 0
for i in range(0,len(train_df['tgt_sentence'])): 
    if len(train_df['tgt_sentence'][i]) > 1: 
        count = count +1
print('Number of columns with more than 1 sentence: ' +str(count))

#we take a look at the distribution of number of sentences according to task
print('Number of entries with more than one sentences split into task categories')
train_df['count_tgt'] = train_df['tgt_sentence'].apply(len)
train_df['count_tgt2'] = train_df['count_tgt'] > 1
print(train_df.groupby('task')['count_tgt2'].sum())

#Now we look at *src*

print('SRC COLUMN')
train_df['src_sentence'] = train_df['src'].apply(nltk.sent_tokenize)

#
maxima = 0
list = []
for i in range(0,len(train_df['src_sentence'])): 
    if len(train_df['src_sentence'][i]) > 1: 
        if len(train_df['src_sentence'][i]) > maxima: 
            maxima = len(train_df['src_sentence'][i])
        list.append(train_df['task'][i])
print('Task categories where more than 1 sentence occur:')
print(set(list))    
print('Maximal number of sentences in src column: ' + str(maxima))
#We can see that there occur more than 1 sentence in every task category


count = 0
for i in range(0,len(train_df['src_sentence'])): 
    if len(train_df['src_sentence'][i]) > 1: 
        count = count +1
print('Number of columns with more than 1 sentence: ' +str(count))

#we take a look at the distribution of number of sentences according to task
print('Number of entries with more than one sentences split into task categories')
train_df['count_src'] = train_df['src_sentence'].apply(len)
train_df['count_src2'] = train_df['count_src'] > 1
print(train_df.groupby('task')['count_src2'].sum())


#We can ignore here the entries for src, because for DM we decided to take the target as reference attribute 
#so we can ignore the 2000 entries with more than 1 sentence 
#and do not have to do sentence splitting with this column



#HERE BEGINS DATA PREPROCESSING VIA subprocess separate file for each task
#for precise explanation look at the file 'data_preprocessing_conllu.py' 


import subprocess

# running other file using run()
subprocess.run(['python', 'data_preprocessing_conllu.py'])
#subprocess.run(['python', 'MT_preprocessing.py'])
#print('MT conllu files created')
#subprocess.run(['python','preprocess_to_conllu_PG.py'])
#print('PG conllu files created')
#subprocess.run(['python','processing_dm.py'])
#print('Dm conllu files created')

