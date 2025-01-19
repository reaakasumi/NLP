import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Function to count hallucinations and non-hallucinations in labels and predictions
def count_labels_predictions(data):
    label_counts = data['label'].value_counts()
    predicted_counts = data['is_hallucination'].value_counts()
    return pd.DataFrame({
        "Type": ["Hallucination", "Not Hallucination"],
        "Count in Label": [label_counts.get(1, 0), label_counts.get(0, 0)],
        "Count in Predicted": [predicted_counts.get(1, 0), predicted_counts.get(0, 0)]
    })

# Function to evaluate predictions and provide metrics
def evaluate_predictions(data):
    results = {}

    # Classification report
    print("Classification Report:")
    report = classification_report(data['label'], data['is_hallucination'], target_names=["Not Hallucination", "Hallucination"])
    results["classification_report"] = report
    print(report)

    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(data['label'], data['is_hallucination'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Hallucination", "Hallucination"])
    disp.plot(cmap="Blues")
    plt.show()
    results["confusion_matrix"] = cm

    # Average text length for correct and incorrect predictions
    data["text_length"] = data["hyp"].apply(len)
    avg_length_correct = data[data['label'] == data['is_hallucination']]["text_length"].mean()
    avg_length_incorrect = data[data['label'] != data['is_hallucination']]["text_length"].mean()
    results["average_length_correct"] = avg_length_correct
    results["average_length_incorrect"] = avg_length_incorrect
    print(f"Average length of correct predictions: {avg_length_correct:.2f}")
    print(f"Average length of incorrect predictions: {avg_length_incorrect:.2f}")

    # Precision and recall for hallucinations
    report_dict = classification_report(data['label'], data['is_hallucination'], output_dict=True)
    precision = report_dict["1"]["precision"]
    recall = report_dict["1"]["recall"]
    results["precision_hallucination"] = precision
    results["recall_hallucination"] = recall
    print(f"Precision for Hallucination: {precision:.2f}")
    print(f"Recall for Hallucination: {recall:.2f}")

    return results

# Analyze the dataset
def analyze_dataset(data, dataset_name):
    print(f"\n=== Analysis for {dataset_name.upper()} Dataset ===")
    
    # Count labels and predictions
    label_pred_counts = count_labels_predictions(data)
    print("\nLabel and Prediction Counts:")
    print(label_pred_counts)

    # Evaluate predictions
    results = evaluate_predictions(data)
    return results

# File paths for datasets
file_paths = {
    "pg": "data/generated/pg_bertscore_finetuned_predictions.csv",
    "mt": "data/generated/mt_bertscore_finetuned_predictions.csv",
    "dm": "data/generated/dm_bertscore_finetuned_predictions.csv"
}

# Analyze all datasets
results = {}
for dataset_name, file_path in file_paths.items():
    print(f"Loading dataset for {dataset_name.upper()}...")
    data = pd.read_csv(file_path)
    
    # Analyze dataset and store results
    results[dataset_name] = analyze_dataset(data, dataset_name)

print("\n=== Analysis Completed for All Datasets ===")
