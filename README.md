# NLP

#Topic 1 Hallucination Detection 

#Goal: Classification of short utterances to determine whether they are hallucinations or not (binary classification task) using the SHROOM dataset 

#Data Description 

#a task indicating what objective the model was optimized for --> 'task'  
#a source the input passed to the models for generation --> 'src'   
#a target the intended reference 'gold' text that the model ought to generate --> 'tgt'  
#a hypothesis the actual model predicts --> 'hyp'  
#a set of per annotator labels indicating whether each individual annotator thought this datapoint constituted a hallucination or not --> 'label'  
#model name --> 'model' 

# Preprocessing 
#Goal: create a label through similarity analysis.
#Due to the difference in optimization purpose, we split the dataset accordingly and consider each accordingly. For maschine translation (MT) and definition modelling (DM) we use column hyp and tgt to check their similarity. For paraphrase generation we use hyp and src. 
For each column we follow the same pattern. 
1. lower all characters
2. remove non-words and non-white spaces
3. (optional) removing stopwords
4. Lemmatization

#We performed some analysis and decided to skip step 3, as it has shown no improvement to the task
