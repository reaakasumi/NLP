import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
import os

train_df = pd.read_csv('mt_train_label.csv')
validation_df = pd.read_csv('mt_val_label.csv')  

# 2. Tokenizer laden
model_name = "meta-llama/Llama-2-7b-chat-hf"


hf_token = 'yourtoken'
os.environ["HF_ACCESS_TOKEN"] = hf_token

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# 3. Preprocessing-Funktion

def preprocess_function(examples):
    # Kombiniere `hyp` und `tgt` als Eingabe für das Modell
    return tokenizer(
        examples["hyp"],
        examples["tgt"],
        truncation=True,
        padding="max_length",
        max_length=128,  # Begrenze die Sequenzlänge
    )

# 4. Tokenisierung der Trainings- und Validierungsdaten
# Konvertiere Pandas DataFrame zu Hugging Face Dataset und wende die Tokenisierung an
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)

tokenized_train_data = train_dataset.map(preprocess_function, batched=True)
tokenized_validation_data = validation_dataset.map(preprocess_function, batched=True)

# 5. Modell laden
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, use_auth_token=hf_token)  # Zwei Klassen: 0 und 1

# 6. Trainingsargumente
training_args = TrainingArguments(
    output_dir="./results",  # Verzeichnis für Ergebnisse
    evaluation_strategy="epoch",  # Evaluation nach jeder Epoche
    learning_rate=2e-5,  # Lernrate
    per_device_train_batch_size=4,  # Batch-Größe pro Gerät
    per_device_eval_batch_size=4,
    num_train_epochs=3,  # Anzahl der Epochen
    weight_decay=0.01,  # Gewichtung für Regularisierung
    save_total_limit=2,  # Maximalanzahl gespeicherter Modelle
    logging_dir="./logs",  # Logging-Verzeichnis
    logging_steps=10,  # Logging-Häufigkeit
    fp16=torch.cuda.is_available(),  # Mixed Precision für GPUs
    save_strategy="epoch",  # Modell nach jeder Epoche speichern
)

# 7. Trainer einrichten
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_validation_data,
    tokenizer=tokenizer,
)

# 8. Modell trainieren
trainer.train()

# 9. Modell speichern
model.save_pretrained("./llama2_label_model")
tokenizer.save_pretrained("./llama2_label_model")

print("Training abgeschlossen und Modell gespeichert!")
