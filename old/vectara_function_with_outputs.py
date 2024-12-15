import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

###################### preparing data #########################

# loading validation data

# load mt data
df_mt_train = pd.read_csv('mt_train_label.csv')
df_mt_val = pd.read_csv('mt_val_label.csv')
df_mt_test = pd.read_csv('mt_test_label.csv')

# load dm data
df_dm_val = pd.read_csv('dm_val_label.csv')
df_dm_test = pd.read_csv('dm_test_label.csv')

# load pg data
df_pg_val = pd.read_csv('pg_val_label.csv')
df_pg_test = pd.read_csv('pg_test_label.csv')

df_pg_val = df_pg_val.rename(columns={'src_normalized': 'tgt_normalized'})
df_pg_test = df_pg_test.rename(columns={'src_normalized': 'tgt_normalized'})


# extracting true labels
def extract_truelabels(df):
    true_labels = []

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
        f1 = 2 * (precision * recall) / (precision + recall)
    except:
        f1 = 0

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

    return accuracy, precision, recall, f1


# function for finding best threshold on validation data

def finding_threshold(similarities, labels_train):
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
        acc, prec, rec, f1 = assess_performance(labels_train, pred_label)
        accs.append(acc)
        recs.append(rec)
        precs.append(prec)

    # return threshold with highest accuracy
    max_acc = thresholds[accs.index(max(accs))]
    print("highest accuracy at a threshold of:", max_acc)

    plt.plot(thresholds, accs, color="red", label="accuracy")
    plt.plot(thresholds, precs, color="green", label="precision")
    plt.plot(thresholds, recs, color="blue", label="recall")
    plt.legend()
    plt.axvline(max_acc, color="grey")
    plt.show()

    return (max_acc)


# Function to transform probabilities into labels

def get_prediction(probs, thresh):
    pred_labels = []

    for sim in probs:

        if sim > thresh:
            pred_labels.append(0)

        else:
            pred_labels.append(1)

    return pred_labels


######################### Defining Analysis Function ##############################
def detailed_analysis(df_results, true_label_col, pred_label_col, hyp_col, tgt_col):
    """
    Perform detailed analysis for classification results including:
    - Classification report
    - Confusion matrix
    - Average length of hypothesis and target entries
    - Precision and Recall calculations
    """

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
    plt.matshow(cm, cmap='coolwarm')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Computing average length of hypothesis and target entries
    hyp_lengths = df_results[hyp_col].apply(len)
    tgt_lengths = df_results[tgt_col].apply(len)

    avg_hyp_length = hyp_lengths.mean()
    avg_tgt_length = tgt_lengths.mean()

    print(f"Average Hypothesis Length: {avg_hyp_length:.2f}")
    print(f"Average Target Length: {avg_tgt_length:.2f}")

    # Calculating precision and recall
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0

    recall = tp / (tp + fn)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    return {
        'classification_report': classification_report(true_labels, pred_labels, output_dict=True),
        'confusion_matrix': cm,
        'avg_hyp_length': avg_hyp_length,
        'avg_tgt_length': avg_tgt_length,
        'precision': precision,
        'recall': recall
    }


######################### Defining Model ##############################
# need to change column name for normalized entries to generic form

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
    pairs_test = [(hyp, tgt) for hyp, tgt in zip(df_test['hyp_normalized'], df_test['tgt_normalized'])]
    pred_vectara_test = model.predict(pairs_test)
    labels_vectara_test = get_prediction(pred_vectara_test, thresh_vectara)
    # returns Dataframe with true labels and predictions
    result = pd.DataFrame({'label': true_labels_test, 'prediction_vectara': labels_vectara_test})
    return (result)


data_result_mt = model_vectara(df_mt_val, df_mt_test)
data_result_dm = model_vectara(df_dm_val, df_dm_test)
data_result_pg = model_vectara(df_pg_val, df_pg_test)

print('results for MT')
#assess_performance(data_result_mt['label'], data_result_mt['prediction_vectara'])
analysis_mt = detailed_analysis(data_result_mt, 'label', 'prediction_vectara', 'hyp_normalized', 'tgt_normalized')

print('Results for DM')
#assess_performance(data_result_dm['label'], data_result_dm['prediction_vectara'])
analysis_dm = detailed_analysis(data_result_dm, 'label', 'prediction_vectara', 'hyp_normalized', 'tgt_normalized')

print('Results for PG')
#assess_performance(data_result_pg['label'], data_result_pg['prediction_vectara'])
analysis_pg = detailed_analysis(data_result_pg, 'label', 'prediction_vectara', 'hyp_normalized', 'tgt_normalized')