# ===========================
# train_mbbert.py
# ===========================

import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch
from huggingface_hub import login

# ===========================
# 🔹 Data Files
# ===========================
SYNTHETIC_TRAIN_PATH = "synthetic_train.csv"
SYNTHETIC_TEST_PATH = "synthetic_test.csv"
YELP_TRAIN_PATH = "yelp_sample_train.csv"
YELP_TEST_PATH = "yelp_sample_test.csv"

# ===========================
# 🔹 Data Preprocessing Function
# ===========================
def df_preprocessing(input_file):
    """Loads and preprocesses a CSV file containing dish and diet info."""
    # 1️⃣ Load dataset
    df = pd.read_csv(input_file)  # Must have: dish_name, description, cuisine, diet

    # 2️⃣ Handle missing description
    df["description"] = df["description"].fillna("")

    # 3️⃣ Create full text input including cuisine
    df["text"] = df["cuisine"] + " dish: " + df["dish_name"] + " - " + df["description"]
    df["text"] = df["text"].str.strip(" - ")  # Remove trailing dash if description was empty

    # 4️⃣ Map labels: Vegetarian -> 1, Non-Vegetarian -> 0
    label_mapping = {"Vegetarian": 1, "Non-Vegetarian": 0}
    df["label"] = df["diet"].map(label_mapping)
    return df

# ===========================
# 🔹 Load and preprocess datasets
# ===========================
synthetic_train_df = df_preprocessing(SYNTHETIC_TRAIN_PATH)
synthetic_test_df = df_preprocessing(SYNTHETIC_TEST_PATH)
yelp_train_df = df_preprocessing(YELP_TRAIN_PATH)
yelp_test_df = df_preprocessing(YELP_TEST_PATH)

# ===========================
# 🔹 Convert to HuggingFace Datasets
# ===========================
synthetic_train_dataset = Dataset.from_pandas(synthetic_train_df)
yelp_train_dataset = Dataset.from_pandas(yelp_train_df)
synthetic_test_dataset = Dataset.from_pandas(synthetic_test_df)
yelp_test_dataset = Dataset.from_pandas(yelp_test_df)

# ===========================
# 🔹 Authenticate to HuggingFace
# ===========================
login()  # You’ll be prompted to enter your HF token

# ===========================
# 🔹 Tokenization
# ===========================
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

synthetic_train_dataset = synthetic_train_dataset.map(tokenize_fn, batched=True)
yelp_train_dataset = yelp_train_dataset.map(tokenize_fn, batched=True)
synthetic_test_dataset = synthetic_test_dataset.map(tokenize_fn, batched=True)
yelp_test_dataset = yelp_test_dataset.map(tokenize_fn, batched=True)

# Remove unused columns
columns_to_remove = ["dish_name", "description", "cuisine", "diet", "text"]
for dataset in [
    synthetic_train_dataset,
    yelp_train_dataset,
    synthetic_test_dataset,
    yelp_test_dataset,
]:
    dataset = dataset.remove_columns(
        [col for col in columns_to_remove if col in dataset.column_names]
    )

print(type(synthetic_train_dataset))
print(synthetic_train_dataset.column_names)

# ===========================
# 🔹 Metrics Function
# ===========================
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

# ===========================
# 🔹 Initialize result containers
# ===========================
synthetic_acc_scores = [0.9805013927576601, 0.9637883008356546]
synthetic_f1_scores = [0.9824561403508771, 0.962536023054755]
yelp_acc_scores = [0.9106145251396648, 0.8876404494382022]
yelp_f1_scores = [0.92, 0.8947368421052632]

# ===========================
# 🔹 K-Fold Cross Validation
# ===========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, ((train_idx_synth, test_idx_synth), (train_idx_yelp, test_idx_yelp)) in enumerate(
    zip(kf.split(synthetic_train_dataset), kf.split(yelp_train_dataset)), 1
):
    print(f"\n===== Fold {fold} =====")

    if fold != 3:
        continue  # Only run fold 3 (adjust as needed)

    # Create train/test splits
    synthetic_train_split = synthetic_train_dataset.select(train_idx_synth)
    synthetic_test_split = synthetic_train_dataset.select(test_idx_synth)
    yelp_train_split = yelp_train_dataset.select(train_idx_yelp)
    yelp_test_split = yelp_train_dataset.select(test_idx_yelp)

    # Combine synthetic + yelp train splits
    combined_train_split = concatenate_datasets(
        [synthetic_train_split, yelp_train_split]
    )

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=2
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results_fold_{fold+1}",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"./logs_fold_{fold+1}",
        load_best_model_at_end=False,
        logging_steps=50,
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_train_split,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()

    # Save model & tokenizer
    save_path = f"./mbbert_fold_{fold}"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ Model and tokenizer saved at {save_path}")

    # Evaluate on synthetic test
    synthetic_preds = trainer.predict(synthetic_test_split)
    synthetic_metrics = synthetic_preds.metrics
    synthetic_acc_scores.append(synthetic_metrics["test_accuracy"])
    synthetic_f1_scores.append(synthetic_metrics["test_f1"])

    print(f"Synthetic Test - Accuracy: {synthetic_metrics['test_accuracy']:.4f}, "
          f"F1_score: {synthetic_metrics['test_f1']:.4f}")

    # Evaluate on yelp test
    yelp_preds = trainer.predict(yelp_test_split)
    yelp_metrics = yelp_preds.metrics
    yelp_acc_scores.append(yelp_metrics["test_accuracy"])
    yelp_f1_scores.append(yelp_metrics["test_f1"])

    print(f"Yelp Test - Accuracy: {yelp_metrics['test_accuracy']:.4f}, "
          f"F1_score: {yelp_metrics['test_f1']:.4f}")

# ===========================
# 🔹 Print Average Results
# ===========================
print("\n===== AVERAGE METRICS ACROSS FOLDS =====")
print("Synthetic Test Set:")
print(f"Average Accuracy: {np.mean(synthetic_acc_scores):.4f}")
print(f"Average F1_score: {np.mean(synthetic_f1_scores):.4f}")

print("\nYelp Test Set:")
print(f"Average Accuracy: {np.mean(yelp_acc_scores):.4f}")
print(f"Average F1_score: {np.mean(yelp_f1_scores):.4f}")

# ===========================
# 🔹 Load Saved Model
# ===========================
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("./mbbert_fold_1")
tokenizer = AutoTokenizer.from_pretrained("./mbbert_fold_1")

print("\n✅ Model and tokenizer loaded successfully.")



