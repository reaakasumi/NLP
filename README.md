# NLP

## Topic 1 Hallucination Detection 

## Goal: Classification of short utterances to determine whether they are hallucinations or not (binary classification task) using the SHROOM dataset 

The Management Summary summarizes what our project and can be found in the file NLP_group2.pdf.

## Data Description 

a task indicating what objective the model was optimized for --> 'task'  
a source the input passed to the models for generation --> 'src'   
a target the intended reference 'gold' text that the model ought to generate --> 'tgt'  
a hypothesis the actual model predicts --> 'hyp'  
a set of per annotator labels indicating whether each individual annotator thought this datapoint constituted a hallucination or not --> 'label'  
model name --> 'model' 

## If you want to run the whole repository, run the file main_pipeline.py 
Otherwise for a individual approach you can follow the steps below. 


## Milestone 1: Data Analysis and Preprocessing 
### Data Analysis 

The analysis starts with reading a JSON dataset (train.model-agnostic.json) into a Pandas DataFrame (train_df). The data includes columns like 'hyp', 'tgt', 'src', and 'task', which represent different aspects of the text, such as hypotheses, targets, and sources for various natural language processing tasks.
We can find the data analysis in main_analysis.py file. 

### Preprocessing 

Goal: create a label through similarity analysis.
Due to the difference in optimization purpose, we split the dataset accordingly and consider each accordingly. For maschine translation (MT) and definition modelling (DM) we use column hyp and tgt to check their similarity. For paraphrase generation we use hyp and src. 
For each column we follow the same pattern. 
1. lower all characters
2. remove non-words and non-white spaces
3. (optional) removing stopwords
4. Lemmatization

We performed some analysis and decided to skip step 3, as it has shown no improvement to the task. 
You can find the preprocessing in the file: preprocessing_conllu.py.

### Export to CoNLL-U format

The processed datasets are exported to the CoNLL-U format using the export_datasets_to_conllu() function, which saves the normalized text in .conllu files for further linguistic analysis. 
You can find the generated CoNLL-U files with separate files for each dataset and task type in: conllu_files folder. 

## Milestone 2: Creation of two baseline models for our classification task

### Naive Bayes 
For the Naive Bayes baseline we created the file naive_bayes.py for an analysis on the overall performance. Also a more precise analysis on the Naive Bayes model is found in the file naive_bayes_analysis.py. For an approach on the separate tasks take a look at the file nb_function_with_outputs.py

### Vectara
For the Vectara model we created one file vectara_improved.py which computes the model for each task separately. 

### Model Evaluation
For the model evaluation we created a jupyter notebook named model_evaluation.ipynb, which computes all results also shown in the report.

## Final Solution: Additional models and finetuning for improvement of accuracy 

### BertScore 
To enhance the BERTScore model's performance, fine-tuning is performed in BERTScore_fine_tuned.py using a custom loss function to handle class imbalance, with the model evaluated on various datasets. The results are analyzed in bertscore_fine_tuned_analysis.py, which includes metrics like precision, recall, F1-score, and confusion matrix, and further semantic similarity analysis is done in bertscore_semantic_similarity_analysis.py.

### Feature-based Classifier 
To improve a ML model, we first extracted features which is done in create_features.py. These features are then used in feature_classifier.py to train a Random Forest model for all tasks together. 

### Majority voting 
Moreover, a majority voting implementation that uses the three better performing models (Vectara, BERTScore, Feature-based Classifier) with added weights can be found in majority_models_with_weights.py. 

## Requirements

Before running the code, ensure the following Python packages are installed:

pandas, numpy, scipy, torch, sklearn, nltk, stanza, json, re, contractions, matplotlib, os, imblearn, ast 

You can install them via pip:
pip install pandas nltk stanza subprocess re contractions sklearn imblearn ast numpy matplotlib sentence_transfomers spacy transformers datasets bert_score random torch

## Usage

Make sure you have installed all the necessary dependencies listed above. Then you can run the main_pipeline.py file to run all the scripts mentioned above. 





