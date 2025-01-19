import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Function to count hallucinations and non-hallucinations in labels and predictions
def count_labels_predictions(data):
    label_counts = data['label'].value_counts()
    predicted_counts = data['predicted_label'].value_counts()
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
    report = classification_report(data['label'], data['predicted_label'], target_names=["Not Hallucination", "Hallucination"])
    results["classification_report"] = report
    print(report)

    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(data['label'], data['predicted_label'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Hallucination", "Hallucination"])
    disp.plot(cmap="Blues")
    plt.show()
    results["confusion_matrix"] = cm

    # Average text length for correct and incorrect predictions
    data["text_length"] = data["hyp"].apply(len)
    avg_length_correct = data[data['label'] == data['predicted_label']]["text_length"].mean()
    avg_length_incorrect = data[data['label'] != data['predicted_label']]["text_length"].mean()
    results["average_length_correct"] = avg_length_correct
    results["average_length_incorrect"] = avg_length_incorrect
    print(f"Average length of correct predictions: {avg_length_correct:.2f}")
    print(f"Average length of incorrect predictions: {avg_length_incorrect:.2f}")

    # Precision and recall for hallucinations
    report_dict = classification_report(data['label'], data['predicted_label'], output_dict=True)
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

# Load the naive_bayes dataset
naive_bayes_data = pd.read_csv('data/generated/naive_bayes_predictions.csv')

# Analyze the dataset
naive_bayes_results = analyze_dataset(naive_bayes_data, "naive_bayes")