import json
import pandas as pd
import numpy as np
import torch
import transformers
import matplotlib.pyplot as plt


###################### preparing data #########################

# loading validation data
data_val = open("data/val.model-agnostic.json")
data = json.load(data_val)
df_val = pd.DataFrame(data)
df_dm = df_val[df_val["task"] == "DM"]


# loading test data
data_test = open("data/test.model-agnostic.json")
data = json.load(data_test)
df_test = pd.DataFrame(data)
df_dm_test = df_test[df_test["task"] == "DM"]


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

true_labels_val = extract_truelabels(df_dm)
true_labels_test = extract_truelabels(df_dm_test)



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

def finding_threshold(similarities):

    # now testing which threshold leads to the highest accuracy
    accs = []
    recs = []
    precs = []

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
        acc, prec, rec, f1 = assess_performance(true_labels_val, pred_label)
        accs.append(acc)
        recs.append(rec)
        precs.append(prec)


    
    # return threshold with highest accuracy
    max_acc = thresholds[accs.index(max(accs))]
    print("highest accuracy at a threshold of:", max_acc)
    
    plt.plot(thresholds, accs, color = "red", label = "accuracy")
    plt.plot(thresholds, precs, color = "green", label = "precision")
    plt.plot(thresholds, recs, color = "blue", label = "recall")
    plt.legend()
    plt.axvline(max_acc, color = "grey")
    plt.show()


    return(max_acc)



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

# prepating input
sentences_hyp = df_dm["hyp"]
sentences_tgt = df_dm["tgt"]

pairs_val = [(hyp, tgt) for hyp, tgt in zip(sentences_hyp, sentences_tgt)]

# loading model
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    'vectara/hallucination_evaluation_model', trust_remote_code=True)

# finding best model based on validation data
pred_vectara_val = model.predict(pairs_val)
thresh_vectara = finding_threshold(pred_vectara_val)


# now testing model
sentences_hyp_test = df_dm_test["hyp"]
sentences_tgt_test = df_dm_test["tgt"]

pairs_test = [(hyp, tgt) for hyp, tgt in zip(sentences_hyp_test, sentences_tgt_test)]

pred_vectara_test = model.predict(pairs_test)

labels_vectara_test = get_prediction(pred_vectara_test, thresh_vectara)
print("Performance of Vectara on Test Data:")
assess_performance(true_labels_test, labels_vectara_test)
