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
MODEL_TYPE = "roberta-base"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 5e-5
OUTPUT_DIR = "./fine_tuned_model"
LOG_DIR = "./logs"

# Custom loss function with class weights for handling imbalance
class_weight = torch.tensor([1.0, 7607 / 2393]).to(device)
loss_fn = CrossEntropyLoss(weight=class_weight)

# Function to load and preprocess the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['label'] = df['label'].apply(lambda x: 1 if x == "Hallucination" else 0)
    for col in ['hyp_normalized', 'src_normalized', 'tgt_normalized']:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df

# Function to create Hugging Face Dataset
def create_hf_dataset(df, tokenizer, max_length, reference_column):
    hf_dataset = Dataset.from_pandas(df, preserve_index=False)
    hf_dataset = hf_dataset.map(
        lambda examples: {
            **tokenizer(
                examples["hyp_normalized"],
                examples[reference_column],
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
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

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
def fine_tune_model(train_dataset, val_dataset, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_TYPE, num_labels=2).to(device)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
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
        fp16=torch.cuda.is_available(),
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

# Function to compute BERTScore
def compute_bertscore(hypotheses, references, model_type):
    return score(
        hypotheses,
        references,
        lang="en",
        model_type=model_type,
        verbose=True,
        device=device.type,
    )

# Main workflow
if __name__ == "__main__":
    # Paths to datasets
    DATA_PATHS = {
        "pg": {"train": "data/preprocessed/pg_train_label.csv", "val": "data/preprocessed/pg_val_label.csv", "test": "data/preprocessed/pg_test_label.csv"},
        "mt": {"train": "data/preprocessed/mt_train_label.csv", "val": "data/preprocessed/mt_val_label.csv", "test": "data/preprocessed/mt_test_label.csv"},
        "dm": {"train": "data/preprocessed/dm_train_label.csv", "val": "data/preprocessed/dm_val_label.csv", "test": "data/preprocessed/dm_test_label.csv"},
    }

    for task, paths in DATA_PATHS.items():
        print(f"\nProcessing task: {task.upper()}")
        reference_column = "src_normalized" if task == "pg" else "tgt_normalized"

        # Load datasets
        train_data = load_data(paths["train"])
        val_data = load_data(paths["val"])
        test_data = load_data(paths["test"])

        # Tokenizer and Hugging Face Dataset creation
        tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)
        train_dataset = create_hf_dataset(train_data, tokenizer, MAX_LENGTH, reference_column)
        val_dataset = create_hf_dataset(val_data, tokenizer, MAX_LENGTH, reference_column)

        # Fine-tune the model
        model = fine_tune_model(train_dataset, val_dataset, tokenizer)

        # Save the fine-tuned model and tokenizer
        model.save_pretrained(f"{OUTPUT_DIR}/{task}")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}/{task}")
        print(f"Model for task {task.upper()} saved successfully.")

        # Compute BERTScore for test data
        test_hypotheses = test_data["hyp_normalized"].tolist()
        test_references = test_data[reference_column].tolist()
        P, R, F1 = compute_bertscore(test_hypotheses, test_references, MODEL_TYPE)

        # Add BERTScore results to the test dataset
        test_data["precision"] = P.tolist()
        test_data["recall"] = R.tolist()
        test_data["f1"] = F1.tolist()

        # Determine the best threshold using validation data
        thresholds = np.linspace(0.5, 0.9, 20)
        val_hypotheses = val_data["hyp_normalized"].tolist()
        val_references = val_data[reference_column].tolist()
        P_val, R_val, F1_val = compute_bertscore(val_hypotheses, val_references, MODEL_TYPE)
        val_data["f1"] = F1_val.tolist()

        best_threshold = max(
            [
                {
                    "threshold": threshold,
                    "f1": precision_recall_fscore_support(
                        val_data["label"], (val_data["f1"] < threshold).astype(int), average="binary"
                    )[2],
                }
                for threshold in thresholds
            ],
            key=lambda x: x["f1"],
        )["threshold"]

        print(f"Best threshold for task {task.upper()}: {best_threshold}")

        # Apply the best threshold to the test data
        test_data["is_hallucination"] = (test_data["f1"] < best_threshold).astype(int)

        # Save the test results
        output_file = f"data/generated/{task}_bertscore_finetuned_predictions.csv"
        test_data.to_csv(output_file, index=False)
        print(f"Test results for {task.upper()} saved to: {output_file}")
