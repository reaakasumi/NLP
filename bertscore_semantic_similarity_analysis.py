import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load datasets
pg_test_data = pd.read_csv("data/generated/pg_bertscore_predictions.csv")
mt_test_data = pd.read_csv("data/generated/mt_bertscore_predictions.csv")
dm_test_data = pd.read_csv("data/generated/dm_bertscore_predictions.csv")

# Ensure labels and predictions are of the same type for all datasets
for dataset in [pg_test_data, mt_test_data, dm_test_data]:
    dataset['label'] = dataset['label'].apply(lambda x: 1 if x == "Hallucination" else 0)
    dataset['is_hallucination'] = dataset['is_hallucination'].astype(int)

# Define functions for analysis
def count_labels_predictions(data):
    """Count hallucinations and non-hallucinations in labels and predictions."""
    label_counts = data['label'].value_counts()
    predicted_counts = data['is_hallucination'].value_counts()
    return pd.DataFrame({
        "Type": ["Hallucination", "Not Hallucination"],
        "Count in Label": [label_counts.get(1, 0), label_counts.get(0, 0)],
        "Count in Predicted": [predicted_counts.get(1, 0), predicted_counts.get(0, 0)]
    })

def task_wise_analysis(data, task_column="task", label_column="label", predicted_column="is_hallucination"):
    """Perform task-wise analysis."""
    results = {
        "Task Type": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": []
    }

    grouped = data.groupby(task_column) if task_column in data.columns else [("All", data)]

    for task, group in grouped:
        accuracy = accuracy_score(group[label_column], group[predicted_column])
        report = classification_report(group[label_column], group[predicted_column], output_dict=True)
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        f1_score = report["weighted avg"]["f1-score"]

        results["Task Type"].append(task)
        results["Accuracy"].append(accuracy)
        results["Precision"].append(precision)
        results["Recall"].append(recall)
        results["F1-Score"].append(f1_score)

    return pd.DataFrame(results)

def evaluate_predictions(data, label_column="label", predicted_column="is_hallucination", text_column="hypotheses"):
    """Evaluate predictions with classification metrics and confusion matrix."""
    results = {}

    # Classification report
    print("Classification Report:")
    report = classification_report(data[label_column], data[predicted_column], output_dict=True)
    results["classification_report"] = report
    print(classification_report(data[label_column], data[predicted_column]))

    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(data[label_column], data[predicted_column])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Hallucination", "Hallucination"])
    disp.plot(cmap="Blues")
    results["confusion_matrix"] = cm

    # Average text length
    data["text_length"] = data[text_column].apply(len)
    avg_length_correct = data[data[label_column] == data[predicted_column]]["text_length"].mean()
    avg_length_incorrect = data[data[label_column] != data[predicted_column]]["text_length"].mean()
    results["average_length_correct"] = avg_length_correct
    results["average_length_incorrect"] = avg_length_incorrect
    print(f"Average length of correct predictions: {avg_length_correct:.2f}")
    print(f"Average length of incorrect predictions: {avg_length_incorrect:.2f}")

    # Precision and recall
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    results["precision"] = precision
    results["recall"] = recall
    print(f"Precision for Hallucination: {precision:.2f}")
    print(f"Recall for Hallucination: {recall:.2f}")

    return results

# Analyze PG, MT, and DM datasets
datasets = {
    "pg": pg_test_data,
    "mt": mt_test_data,
    "dm": dm_test_data
}

for dataset_name, dataset in datasets.items():
    print(f"\n=== {dataset_name.upper()} Dataset Analysis ===")
    
    # Count labels and predictions
    label_pred_counts = count_labels_predictions(dataset)
    print(label_pred_counts)

    # Task-wise analysis (if applicable)
    if "task" in dataset.columns:
        task_results = task_wise_analysis(dataset)
        print(task_results)

    # Evaluate predictions
    evaluate_predictions(dataset, label_column="label", predicted_column="is_hallucination", text_column="hypotheses")
