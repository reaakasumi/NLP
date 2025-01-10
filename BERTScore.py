import pandas as pd
from bert_score import score
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Model configuration
MODEL_TYPE = "roberta-large"

# Paths to datasets
DATA_PATHS = {
    "pg": {"val": "data/pg_val_label.csv", "test": "data/pg_test_label.csv"},
    "mt": {"val": "data/mt_val_label.csv", "test": "data/mt_test_label.csv"},
    "dm": {"val": "data/dm_val_label.csv", "test": "data/dm_test_label.csv"}
}

# Load model and tokenizer
model = AutoModel.from_pretrained(MODEL_TYPE, use_safetensors=False)
tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)

# Function to load and process a dataset
def process_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return None, None, None

    # Replace NaN values and ensure strings
    if 'hyp_normalized' in data.columns:
        data['hyp_normalized'] = data['hyp_normalized'].fillna('').astype(str)
    if 'src_normalized' in data.columns:
        data['src_normalized'] = data['src_normalized'].fillna('').astype(str)
    if 'tgt_normalized' in data.columns:
        data['tgt_normalized'] = data['tgt_normalized'].fillna('').astype(str)
    
    references = (
        data['src_normalized'].tolist() 
        if 'src_normalized' in data.columns 
        else data['tgt_normalized'].tolist()
    )
    hypotheses = data['hyp_normalized'].tolist()
    return data, hypotheses, references

# Function to compute BERTScore
def compute_bertscore(hypotheses, references):
    hypotheses = [str(hyp) for hyp in hypotheses]
    references = [str(ref) for ref in references]
    return score(hypotheses, references, lang="en", model_type=MODEL_TYPE, verbose=True)

# Function to evaluate thresholds
def evaluate_thresholds(df, true_labels_col, f1_col, thresholds):
    results = []
    true_labels = df[true_labels_col].apply(lambda x: 1 if x == "Hallucination" else 0)
    for threshold in thresholds:
        predictions = (df[f1_col] < threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="binary")
        accuracy = accuracy_score(true_labels, predictions)
        results.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy})
    return pd.DataFrame(results)

# Function to process and evaluate a dataset
def process_dataset_with_evaluation(dataset_name, val_file, test_file, best_threshold=None):
    # Process validation data
    val_data, val_hypotheses, val_references = process_dataset(val_file)
    P_val, R_val, F1_val = compute_bertscore(val_hypotheses, val_references)
    val_results = pd.DataFrame({
        "hypotheses": val_hypotheses,
        "references": val_references,
        "precision": P_val.tolist(),
        "recall": R_val.tolist(),
        "f1": F1_val.tolist(),
        "label": val_data['label']
    })

    # Determine the best threshold if not provided
    if best_threshold is None:
        thresholds = np.linspace(0.5, 0.9, 20)
        threshold_results = evaluate_thresholds(val_results, "label", "f1", thresholds)
        best_threshold = threshold_results.loc[threshold_results["f1"].idxmax()]["threshold"]
        logging.info(f"Best threshold for {dataset_name}: {best_threshold}")

    # Apply threshold to validation results
    val_results["is_hallucination"] = val_results["f1"] < best_threshold

    # Process test data
    test_data, test_hypotheses, test_references = process_dataset(test_file)
    P_test, R_test, F1_test = compute_bertscore(test_hypotheses, test_references)
    test_results = pd.DataFrame({
        "hypotheses": test_hypotheses,
        "references": test_references,
        "precision": P_test.tolist(),
        "recall": R_test.tolist(),
        "f1": F1_test.tolist(),
        "label": test_data['label']
    })
    test_results["is_hallucination"] = test_results["f1"] < best_threshold

    # Calculate metrics for test data
    true_labels = test_results["label"].apply(lambda x: 1 if x == "Hallucination" else 0)
    predictions = test_results["is_hallucination"]
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="binary")
    accuracy = accuracy_score(true_labels, predictions)
    test_metrics = {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

    # Save results
    #val_results.to_csv(f"data/{dataset_name}_val_bertscore_results_with_labels.csv", index=False)
    #test_results.to_csv(f"data/{dataset_name}_test_bertscore_results_with_labels.csv", index=False)
    logging.info(f"{dataset_name.upper()} results saved. Test Metrics: {test_metrics}")

    return best_threshold

# Process all datasets
best_pg_threshold = None
for dataset_name, paths in DATA_PATHS.items():
    best_pg_threshold = process_dataset_with_evaluation(dataset_name, paths["val"], paths["test"], best_pg_threshold)
