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

#Theresas Attempt: 
#I did the whole preprocessing step with only the column 'hyp' and I created extra columns to compare in and output, 
#better approach is to change only the 'hyp' column, otherwise with tgt and src we would get to many columns 
#
#first I started with lowering all entries and removing non-words and non-whitespace characters 
#then I split the entries into sentences and then words --> but I got a problem with removing stopwords and stemming, because after sentence splitting, I have entries with lists in a list 
#so maybe better approach would be to only do word splitting, because there are 2 sentences at most 
#then I created an additional dataset for stopword removing, and did Stemming 
#Lemmatization did take too long, so I did not include it till now. 
