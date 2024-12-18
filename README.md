# NLP

## Topic 1 Hallucination Detection 

## Goal: Classification of short utterances to determine whether they are hallucinations or not (binary classification task) using the SHROOM dataset 

## Data Description 

a task indicating what objective the model was optimized for --> 'task'  
a source the input passed to the models for generation --> 'src'   
a target the intended reference 'gold' text that the model ought to generate --> 'tgt'  
a hypothesis the actual model predicts --> 'hyp'  
a set of per annotator labels indicating whether each individual annotator thought this datapoint constituted a hallucination or not --> 'label'  
model name --> 'model' 

## Data Analysis 

The analysis starts with reading a JSON dataset (train.model-agnostic.json) into a Pandas DataFrame (train_df). The data includes columns like 'hyp', 'tgt', 'src', and 'task', which represent different aspects of the text, such as hypotheses, targets, and sources for various natural language processing tasks.
We can find the data analysis in main_analysis.py file. 

## Preprocessing 

Goal: create a label through similarity analysis.
Due to the difference in optimization purpose, we split the dataset accordingly and consider each accordingly. For maschine translation (MT) and definition modelling (DM) we use column hyp and tgt to check their similarity. For paraphrase generation we use hyp and src. 
For each column we follow the same pattern. 
1. lower all characters
2. remove non-words and non-white spaces
3. (optional) removing stopwords
4. Lemmatization

We performed some analysis and decided to skip step 3, as it has shown no improvement to the task. 
You can find the preprocessing in the file: preprocessing_conllu.py.

## Export to CoNLL-U format

The processed datasets are exported to the CoNLL-U format using the export_datasets_to_conllu() function, which saves the normalized text in .conllu files for further linguistic analysis. 
You can find the generated CoNLL-U files with separate files for each dataset and task type in: conllu_files folder. 

## Requirements

Before running the code, ensure the following Python packages are installed:

pandas, numpy, scipy, torch, sklearn, nltk, stanza, json, re, contractions

You can install them via pip:
pip install pandas numpy scipy torch scikit-learn nltk stanza contractions

## Usage

To run the code:

Make sure you have installed all the necessary dependencies listed above. 
Place the train.model-agnostic.json, val.model-agnostic.json, test.model-agnostic.json files in the same directory as your script.
Execute the script to perform the data analysis and preliminary preprocessing steps.

## Milestone 2 

For Milestone 2 take a look at the branch milestone2 and its added README file. 
