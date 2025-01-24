import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


###################### preparing data #########################

# loading validation  and testing data

#load mt data
df_mt_val = pd.read_csv('data/mt_val_label.csv')
df_mt_test = pd.read_csv('data/mt_test_label.csv')

#load dm data
df_dm_val = pd.read_csv('data/dm_val_label.csv')
df_dm_test = pd.read_csv('data/dm_test_label.csv')

#load pg data
df_pg_val = pd.read_csv('data/pg_val_label.csv')
df_pg_test = pd.read_csv('data/pg_test_label.csv')


# Rename columns of PG to match the other tasks
df_pg_val.rename(columns=lambda col: col.replace('src', 'tgt', 1) if col.startswith('src') else col, inplace=True)
df_pg_test.rename(columns=lambda col: col.replace('src', 'tgt', 1) if col.startswith('src') else col, inplace=True)


# extracting true labels
def extract_truelabels(df):

    true_labels= []

    for label in df["label"]:

        # 0 for not Hallucination, 1 for Hallucination

        if label == "Not Hallucination":
            true_labels.append(0)

        else:
            true_labels.append(1)

    df["true_labels"] = true_labels
    return true_labels

true_labels_val = extract_truelabels(df_mt_val)
true_labels_test = extract_truelabels(df_mt_test)



################### Functions to Choose and Evaluate Model ############################


# function to assess performance of classifier

def assess_performance(true_labels, pred_labels):

    # initialize counter for the correct vs. incorrect classifications
    tp = 0
    tn = 0
    fp = 0
    fn = 0


    for true, pred in zip(true_labels, pred_labels):

        if true == 1 and pred == 1:
            tp += 1
        elif true == 1 and pred == 0:
            fn += 1
        elif true == 0 and pred == 0:
            tn += 1
        else:
            fp += 1

    accuracy = (tp + tn) / len(true_labels)
    try:
        precision = tp / (tp + fp)
    except:
        precision = 0
    recall = tp / (tp + fn)
    try:
       f1 = 2* (precision * recall) / (precision + recall)
    except:
        f1 = 0

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

    return accuracy, precision, recall, f1


# function for finding best threshold on validation data

def finding_threshold(similarities,labels_train):

    # now testing which threshold leads to the highest accuracy
    accs = []
    recs = []
    precs = []
    f1s = []

    # defining thresholds
    thresholds = np.linspace(0, 1, 20).tolist()


    for thresh in thresholds:

        # convert similarities to a label based on threshold
        pred_label = []

        for sim in similarities:

            if sim > thresh:
                pred_label.append(0)

            else:
                pred_label.append(1)

        # calculate performance based on threshold
        acc, prec, rec, f1 = assess_performance(labels_train, pred_label)
        accs.append(acc)
        recs.append(rec)
        precs.append(prec)
        f1s.append(f1)
    
    # return threshold with highest accuracy
    max_f1 = thresholds[f1s.index(max(f1s))]

    # if determined threshold is not good, we use fixed threshold
    if max_f1 < 0.2 or max_f1 > 0.7:
        max_f1 = 0.35

    print("highest f1 at a threshold of:", max_f1)
    
    plt.plot(thresholds, accs, color = "red", label = "accuracy")
    plt.plot(thresholds, precs, color = "green", label = "precision")
    plt.plot(thresholds, recs, color = "blue", label = "recall")
    plt.plot(thresholds, f1s, color = "purple", label = "f1")
    plt.legend()
    plt.axvline(max_f1, color = "grey")
    plt.show()


    return(max_f1)



# Function to transform probabilities into labels
 
def get_prediction(probs, thresh):

    pred_labels = []

    for sim in probs:

        if sim > thresh:
            pred_labels.append(0)

        else:
            pred_labels.append(1)

    return pred_labels




######################### Defining Model ##############################

def model_vectara(df_train, df_test): 

    true_labels_train = extract_truelabels(df_train)
    true_labels_test = extract_truelabels(df_test)
    pairs_train = [(hyp, tgt) for hyp, tgt in zip(df_train['hyp_normalized'], df_train['tgt_normalized'])]

    # loading model
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
         'vectara/hallucination_evaluation_model', trust_remote_code=True)

    # finding best model based on validation data
    pred_vectara = model.predict(pairs_train)
    thresh_vectara = finding_threshold(pred_vectara, true_labels_train)

    # now using model on testing data
    pairs_test = [(hyp,tgt) for hyp, tgt in zip(df_test['hyp_normalized'], df_test['tgt_normalized'])]
    pred_vectara_test = model.predict(pairs_test)
    labels_vectara_test = get_prediction(pred_vectara_test, thresh_vectara)

    #returns Dataframe with true labels and predictions 
    result = pd.DataFrame({'label': true_labels_test, 'prediction_vectara': labels_vectara_test})
    return(result)
    
    
# predicting for the different tasks
data_result_mt = model_vectara(df_mt_val, df_mt_test)
data_result_dm = model_vectara(df_dm_val, df_dm_test)
data_result_pg = model_vectara(df_pg_val, df_pg_test)

print('results for MT')
assess_performance(data_result_mt['label'], data_result_mt['prediction_vectara'])
print('Results for DM')
assess_performance(data_result_dm['label'], data_result_dm['prediction_vectara'])
print('Results for PG')
assess_performance(data_result_pg['label'], data_result_pg['prediction_vectara'])

# add predicted label and task
df_mt_test["predicted_vectara"] = data_result_mt['prediction_vectara']
df_mt_test["task"] = "MT"
df_dm_test["predicted_vectara"] = data_result_dm['prediction_vectara']
df_dm_test["task"] = "DM"
df_pg_test["predicted_vectara"] = data_result_pg['prediction_vectara']
df_pg_test["task"] = "PG"

# create labeled dataframe and store it
df_test_labeled = pd.concat([df_mt_test, df_dm_test, df_pg_test])
df_test_labeled.to_csv("data/test_labeled_vectara.csv")
