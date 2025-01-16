import pandas as pd
import numpy as np
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from bert_score import score
import random
from torch.nn import CrossEntropyLoss

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configurations for training
MODEL_TYPE = "roberta-base"  # Use smaller model for faster training
MAX_LENGTH = 256  # Reduce max token length for faster processing
BATCH_SIZE = 16  # Increase batch size for better efficiency
EPOCHS = 5  # Increased number of epochs
LEARNING_RATE = 5e-5
LOG_DIR = "./logs"

# Custom loss function with class weights for handling imbalance
class_weight = torch.tensor([1.0, 7607 / 2393]).to(device)
loss_fn = CrossEntropyLoss(weight=class_weight)

# Function to load and preprocess the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['label'] = df['label'].apply(lambda x: 1 if x == "Hallucination" else 0)  # Convert labels to 0/1
    return df

# Function to create Hugging Face Dataset
def create_hf_dataset(df, tokenizer, max_length, text_col="src"):
    hf_dataset = Dataset.from_pandas(df, preserve_index=False)
    hf_dataset = hf_dataset.map(
        lambda examples: {
            **tokenizer(
                examples["hyp"],
                examples[text_col],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            ),
            "labels": examples["label"],
        },
        batched=True,
    )
    return hf_dataset

# Function to compute evaluation metrics during training
def compute_metrics(pred):
    logits = pred.predictions
    preds = np.argmax(logits, axis=1)
    labels = pred.label_ids
    preds = preds if isinstance(preds, np.ndarray) else preds.cpu().numpy()
    labels = labels if isinstance(labels, np.ndarray) else labels.cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    accuracy = accuracy_score(labels, preds)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Custom Trainer class to override compute_loss
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override compute_loss to apply custom loss with class weights.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Function to fine-tune the model
def fine_tune_model(train_dataset, val_dataset, tokenizer, output_dir):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_TYPE, num_labels=2).to(device)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir=LOG_DIR,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),  # Enable mixed precision if using GPU
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return model

# BERTScore-based thresholding
def compute_bertscore(hypotheses, references, model_type):
    return score(
        hypotheses,
        references,
        lang="en",
        model_type=model_type,
        verbose=True,
        device=device.type,  # Use GPU if available
    )

def evaluate_bertscore_threshold(df, f1_col, label_col):
    thresholds = np.linspace(0.5, 0.9, 20)
    results = []

    true_labels = df[label_col]
    for threshold in thresholds:
        predictions = (df[f1_col] < threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="binary")
        accuracy = accuracy_score(true_labels, predictions)
        results.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy})
    best_threshold = max(results, key=lambda x: x['f1'])['threshold']
    return pd.DataFrame(results), best_threshold

# Unified workflow for all datasets
def process_dataset(train_file, val_file, test_file, output_dir, text_col="src", output_test_file=None):
    # Load datasets
    train_data = load_data(train_file)
    val_data = load_data(val_file)
    test_data = load_data(test_file)

    # Tokenizer and Hugging Face Dataset creation
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)
    train_dataset = create_hf_dataset(train_data, tokenizer, MAX_LENGTH, text_col=text_col)
    val_dataset = create_hf_dataset(val_data, tokenizer, MAX_LENGTH, text_col=text_col)

    # Fine-tune the model
    model = fine_tune_model(train_dataset, val_dataset, tokenizer, output_dir)

    # Save the fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Compute BERTScore on the test dataset
    test_hypotheses = test_data["hyp"].tolist()
    test_references = test_data[text_col].tolist()
    P, R, F1 = compute_bertscore(test_hypotheses, test_references, MODEL_TYPE)

    # Create results DataFrame
    test_data["precision"] = P.tolist()
    test_data["recall"] = R.tolist()
    test_data["f1"] = F1.tolist()

    # Evaluate thresholds and find the best threshold
    threshold_results, best_threshold = evaluate_bertscore_threshold(test_data, f1_col="f1", label_col="label")
    print(f"Best Threshold: {best_threshold}")

    # Test classification report
    test_data["is_hallucination"] = (test_data["f1"] < best_threshold).astype(int)
    print("Test Classification Report:")
    print(classification_report(test_data["label"], test_data["is_hallucination"], target_names=["Not Hallucination", "Hallucination"]))

    # Confusion Matrix
    cm = confusion_matrix(test_data["label"], test_data["is_hallucination"])
    print("Confusion Matrix:")
    print(cm)

    # Save the labeled test dataset
    if output_test_file:
        test_data.to_csv(output_test_file, index=False)
        print(f"Labeled test data saved to {output_test_file}")

    return test_data

# Main workflow for all datasets
if __name__ == "__main__":
    # Process PG dataset
    print("Processing PG dataset...")
    process_dataset(
        "pg_train_label.csv", 
        "pg_val_label.csv", 
        "pg_test_label.csv", 
        "./fine_tuned_model_pg", 
        text_col="src", 
        output_test_file="pg_test_labeled.csv"
    )

    # Process MT dataset
    print("Processing MT dataset...")
    process_dataset(
        "mt_train_label.csv", 
        "mt_val_label.csv", 
        "mt_test_label.csv", 
        "./fine_tuned_model_mt", 
        text_col="hyp", 
        output_test_file="mt_test_labeled.csv"
    )

    # Process DM dataset
    print("Processing DM dataset...")
    process_dataset(
        "dm_train_label.csv", 
        "dm_val_label.csv", 
        "dm_test_label.csv", 
        "./fine_tuned_model_dm", 
        text_col="hyp", 
        output_test_file="dm_test_labeled.csv"
    )
