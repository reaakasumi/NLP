import pandas as pd 
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

#Create function for detailed analysis (can also be seen in milestone 2 evaluation)

def detailed_analysis(df_results, true_label_col, pred_label_col):
    # Extracting true and predicted labels
    true_labels = df_results[true_label_col]
    pred_labels = df_results[pred_label_col]

    # Generating classification report
    print("Classification Report:")
    print(classification_report(true_labels, pred_labels))

    # Generating confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(true_labels, pred_labels)
    print(cm)

    # Plotting confusion matrix
    plt.matshow(cm, cmap='Greens')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Calculating precision and recall
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0

    recall = tp / (tp + fn)
    accuracy = (tn + tp) / len(true_labels)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

    return {
        'classification_report': classification_report(true_labels, pred_labels, output_dict=True),
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall
    }



#read the test data with predicted labels for BertScore, Vectara and Feature Based Classifier 
data_vectara = pd.read_csv('data/generated/test_labeled_vectara.csv')
data_bert_mt = pd.read_csv('data/generated/mt_bertscore_finetuned_predictions.csv')
data_bert_dm = pd.read_csv('data/generated/dm_bertscore_finetuned_predictions.csv')
data_bert_pg = pd.read_csv('data/generated/pg_bertscore_finetuned_predictions.csv')

data_fc = pd.read_csv('data/generated/test_labeled_fc.csv')


#compare it taskwise 
data_vectara_mt = data_vectara[data_vectara['task'] == 'MT'].reset_index(drop = True)
data_vectara_dm = data_vectara[data_vectara['task'] == 'DM'].reset_index(drop = True)
data_vectara_pg = data_vectara[data_vectara['task'] == 'PG'].reset_index(drop = True)

data_fc_mt = data_fc[data_fc['task'] == 'MT'].reset_index(drop = True)
data_fc_dm = data_fc[data_fc['task'] == 'DM'].reset_index(drop = True)
data_fc_pg = data_fc[data_fc['task'] == 'PG'].reset_index(drop = True)


#MT 

data_mt = pd.DataFrame({'tgt': data_vectara_mt['tgt'],'hyp_bert': data_bert_mt['hyp'], 'hyp_vectara': data_vectara_mt['hyp'],
                       'hyp_fc': data_fc_mt['hyp'],'label': data_fc_mt['label'],
                       'prediction_bert': data_bert_mt['is_hallucination'],
                       'prediction_vectara': data_vectara_mt['predicted_vectara'],
                       'prediction_fc': data_fc_mt['predicted_fc']})

data_mt['prediction_bert'] = data_mt['prediction_bert'].apply(lambda x: 0 if x == False else 1)
data_mt['label'] = data_mt['label'].apply(lambda x:0 if x == 'Not Hallucination' else 1)
data_mt['prediction_fc'] = data_mt['prediction_fc'].apply(lambda x:0 if x == 'Not Hallucination' else 1)


#bert: False = Not Hallucination 
#Vectara is 0 Not Hallucination 


#Majority Vote 

data_mt['majority'] = data_mt['prediction_bert'] + data_mt['prediction_vectara'] + data_mt['prediction_fc']
data_mt['majority_vote'] = data_mt['majority'].apply(lambda x: 1 if x >=2 else 0)

print('RESULTS FOR MT')
results_mt = detailed_analysis(data_mt, "label", "majority_vote")


#DM 

data_dm = pd.DataFrame({'tgt': data_vectara_dm['tgt'],'hyp_bert': data_bert_dm['hyp'], 'hyp_vectara': data_vectara_dm['hyp'],
                       'hyp_fc': data_fc_dm['hyp'], 'label': data_fc_dm['label'],
                       'prediction_bert': data_bert_dm['is_hallucination'],
                       'prediction_vectara': data_vectara_dm['predicted_vectara'],
                       'prediction_fc': data_fc_dm['predicted_fc']})

data_dm['prediction_bert'] = data_dm['prediction_bert'].apply(lambda x: 0 if x == False else 1)
data_dm['label'] = data_dm['label'].apply(lambda x:0 if x == 'Not Hallucination' else 1)
data_dm['prediction_fc'] = data_dm['prediction_fc'].apply(lambda x:0 if x == 'Not Hallucination' else 1)


#bert: False = Not Hallucination 

#Vectara is 0 Not Hallucination 


#Majority Vote 

data_dm['majority'] = data_dm['prediction_bert'] + data_dm['prediction_vectara'] + data_dm['prediction_fc']
data_dm['majority_vote'] = data_dm['majority'].apply(lambda x: 1 if x >=2 else 0)

print('RESULTS FOR DM')
results_dm = detailed_analysis(data_dm, "label", "majority_vote")

#PG

data_pg = pd.DataFrame({'tgt': data_vectara_pg['tgt'],'hyp_bert': data_bert_pg['hyp'], 'hyp_vectara': data_vectara_pg['hyp'],
                       'hyp_fc': data_fc_pg['hyp'], 'label': data_fc_pg['label'], 
                       'prediction_bert': data_bert_pg['is_hallucination'],
                       'prediction_vectara': data_vectara_pg['predicted_vectara'],
                       'prediction_fc': data_fc_pg['predicted_fc']})

data_pg['prediction_bert'] = data_pg['prediction_bert'].apply(lambda x: 0 if x == False else 1)
data_pg['label'] = data_pg['label'].apply(lambda x:0 if x == 'Not Hallucination' else 1)
data_pg['prediction_fc'] = data_pg['prediction_fc'].apply(lambda x:0 if x == 'Not Hallucination' else 1)


#bert: False = Not Hallucination 

#Vectara is 0 Not Hallucination 


#Majority Vote 




data_pg['majority'] = data_pg['prediction_bert'] + data_pg['prediction_vectara'] + data_pg['prediction_fc']
data_pg['majority_vote'] = data_pg['majority'].apply(lambda x: 1 if x >=2 else 0)

print('RESULTS FOR PG')
results_pg = detailed_analysis(data_pg, "label", "majority_vote")


#comparison in total 

#include weights 

def majority_weights(data_all, weight_bert, weight_vectara, weight_fc, threshold): 
    #normalize the weights
    weight_sum = weight_bert + weight_vectara + weight_fc
    weight_bert = weight_bert/weight_sum
    weight_vectara = weight_vectara/weight_sum
    weight_fc = weight_fc/weight_sum
    data_all['majority'] = weight_bert * data_all['prediction_bert'] + weight_vectara * data_all['prediction_vectara'] + weight_fc * data_all['prediction_fc']
    data_all['majority_vote'] = data_all['majority'].apply(lambda x: 1 if x >=threshold else 0)
    result = detailed_analysis(data_all, "label", "majority_vote")
    return(result)
    
#we set threshold to 0.66, so if at least two models say its an hallucination then it is labeled as an hallucination 
threshold = 0.66

data_all = pd.concat([data_dm, data_mt, data_pg], ignore_index = True)

print('RESULTS FOR ALL')
results_standard = majority_weights(data_all, 1, 1,1, threshold)

#now we can test variants, 
#V1 is where as soon as vectara predicts yes, then the vote is yes 
#--> expected accuracy same as with only feature based classifier 
results_v1 = majority_weights(data_all, 1, 1, 4, threshold)

#V2 is when vectara and fc are equal and bert is less 
#-> performance is worse then when all are equal 
results_v2 = majority_weights(data_all, 1, 4, 4, threshold)

#V3 same as V2 only with bert and fc 
#-> performance even worse
results_v3 = majority_weights(data_all, 4, 1, 4, threshold)

#V4  Combination of Bert and Vectara
#-> even worse outcome 
results_v4 = majority_weights(data_all, 4, 4, 1, threshold)

#As comparison performance of only feature based classifier 
results_v5 = majority_weights(data_all, 0, 0, 1, threshold)

#for comparison we take a look at when the threshold is set lower 
threshold = 0.33 

results_v6 = majority_weights(data_all, 1, 1, 1, threshold)

#Same performance as fc
