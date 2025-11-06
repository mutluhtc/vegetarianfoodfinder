import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset 
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import warnings

# Suppress UserWarning related to MPS and pin_memory if running on Apple Silicon
warnings.filterwarnings("ignore", ".*'pin_memory' argument is set as true but not supported on MPS.*")


# --- 1. File Path Definition and Constants ---

# Training/CV Data Paths (These will be combined for final training)
SYNTHETIC_CSV_PATH = 'synthetic_train.csv'
YELP_CSV_PATH = 'yelp_sample_train.csv'

# FINAL HELD-OUT TEST DATA PATHS
FINAL_TEST_SYNTHETIC_PATH = 'synthetic_test.csv'
FINAL_TEST_YELP_PATH = 'yelp_sample_test.csv'

# Output Directories
FINAL_OUTPUT_DIR = './final_model_dir'

# Define label map and its inverse
label_map = {'Vegetarian': 0, 'Non-Vegetarian': 1}
id_to_label = {0: 'Vegetarian', 1: 'Non-Vegetarian'}
SEED = 42

# Define the common new tokens for BERT
NEW_TOKENS = [
    # American (30 tokens)
    "brisket", "sourdough", "buttermilk", "cobbler", "roux", "clam chowder", "jambalaya", "po'boy", "griddle", "slaw",
    "barbecue", "gravy", "hushpuppies", "pecan", "gumbo", "chili", "coleslaw", "patty", "pulled pork", "cornbread",
    "hushpuppy", "biscuit", "cheddar", "buffalo", "key lime", "succotash", "apple pie", "chili con carne", "catfish", "meatloaf",
    # Indian (30 tokens)
    "paneer", "masala", "ghee", "korma", "bharta", "tandoor", "vindaloo", "raita", "naan", "biryani",
    "chapati", "paratha", "aloo", "palak", "tikka", "samosa", "chana", "jalfrezi", "dosa", "curried",
    "mutter", "rogan josh", "gulab jamun", "chutney", "sambar", "idli", "vada", "thali", "papadum", "kochuri",
    # Chinese (30 tokens)
    "wok", "dim sum", "char siu", "bok choy", "hoisin", "mapo tofu", "congee", "szechuan", "xiao long bao", "lo mein",
    "bao", "shu mai", "peking", "jiaozi", "wonton", "chow mein", "ma po", "kung pao", "chop suey", "scallion",
    "zongzi", "dan dan", "hot pot", "shaokao", "siu yeh", "dan tat", "fung zao", "jianbing", "char kway teow", "mala",
    # Thai (30 tokens)
    "galangal", "kaffir lime", "lemongrass", "nam pla", "tom kha", "pad see ew", "satay", "sticky rice", "larb", "phanaeng",
    "chili paste", "fish sauce", "massaman", "green curry", "red curry", "basil", "prik", "pad krapow", "mango sticky rice", "som tam",
    "khao soi", "pad prik", "tom saap", "hor mok", "pla pao", "kanom krok", "met mamuang", "pad woon sen", "kluay tod", "gang som",
    # Italian (30 tokens)
    "burrata", "prosciutto", "pesto", "ragu", "gnocchi", "bolognese", "osso buco", "focaccia", "polenta", "tiramisu",
    "mozzarella", "parmesan", "cannoli", "calzone", "bruschetta", "caprese", "lasagna", "risotto", "marinara", "chianti",
    "arancini", "cacio e pepe", "pancetta", "guanciale", "pecorino", "zabaglione", "panna cotta", "minestrone", "sicilian", "spritz",
    # Mediterranean (30 tokens)
    "hummus", "falafel", "tzatziki", "tagine", "couscous", "tahini", "souvlaki", "pita", "baba ghanoush", "moussaka",
    "halloumi", "dolmades", "kebabs", "gyro", "feta", "borek", "baklava", "spanakopita", "tabouleh", "kibbeh",
    "saganaki", "labneh", "za'atar", "basturma", "kibbe", "merguez", "avgolemono", "bourekas", "pita bread", "shish kebab",
    # Japanese (30 tokens)
    "sushi", "sashimi", "miso", "dashi", "tempura", "ramen", "udon", "yakitori", "tonkatsu", "teriyaki",
    "wasabi", "edamame", "gyoza", "shoyu", "nigiri", "mochi", "soba", "katsu", "onigiri", "tamagoyaki",
    "unagi", "ebi", "tori", "teppanyaki", "okazu", "kaiseki", "ikura", "nori", "tataki", "kombu"
]


# --- 2. Data Loading and Preparation Functions ---

def load_data_and_preprocess(csv_filepath, label_map, keep_metadata=False):
    """Loads a single CSV and preprocesses it into a DataFrame, optionally keeping metadata."""
    print(f"--- Loading data from: {csv_filepath} ---")
    df_loaded = pd.read_csv(csv_filepath)
    df_loaded['labels'] = df_loaded['diet'].map(label_map) 
    
    # Ensure the labels are explicitly cast to the required integer type (np.int64) 
    df_loaded['labels'] = df_loaded['labels'].astype(np.int64) 
    
    # Handle NaN values and concatenate text fields
    df_loaded['dish_name'] = df_loaded['dish_name'].fillna('').astype(str)
    df_loaded['description'] = df_loaded['description'].fillna('').astype(str)
    df_loaded['text'] = df_loaded['dish_name'] + " - " + df_loaded['description']
    
    if keep_metadata:
        # Return columns needed for model input AND for final CSV export
        return df_loaded[['dish_name', 'description', 'text', 'labels']]
    else:
        # Return only columns needed for model training (text and labels)
        return df_loaded[['text', 'labels']]

def load_final_test_data(synthetic_path, yelp_path, label_map):
    """Loads and preprocesses the final, held-out test data, returning two separate DataFrames."""
    
    # Load and preprocess Synthetic test data, keeping metadata
    df_synth = load_data_and_preprocess(synthetic_path, label_map, keep_metadata=True)
    
    # Load and preprocess Yelp test data, keeping metadata
    df_yelp = load_data_and_preprocess(yelp_path, label_map, keep_metadata=True)
    
    print(f"Final Test Set 1 (Synthetic) Size: {len(df_synth)}")
    print(f"Final Test Set 2 (Yelp) Size: {len(df_yelp)}")
    
    # Return two separate DataFrames
    return df_synth, df_yelp 

def tokenize_function(examples, tokenizer):
    """Tokenizes the text column for BERT input."""
    return tokenizer(list(examples['text']), truncation=True, padding='max_length', max_length=128)

def compute_metrics(eval_pred):
    """Computes accuracy and F1 score using sklearn."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    # Using 'binary' F1 score for the 2-class problem
    f1 = f1_score(labels, predictions, average='binary', pos_label=1) 

    return {
        'eval_accuracy': accuracy,
        'eval_f1': f1,
    }

def get_final_training_args(output_dir, seed):
    """Returns TrainingArguments for final training using optimal settings."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,                     # Optimal Epochs from CV
        learning_rate=2e-5,                     # Optimal Learning Rate from CV
        per_device_train_batch_size=4,          
        per_device_eval_batch_size=4,           
        warmup_steps=100,                       
        weight_decay=0.01,                      
        logging_dir=os.path.join('./logs_final', os.path.basename(output_dir)),
        logging_steps=10,
        eval_strategy="no",                     
        save_strategy="epoch",                  
        load_best_model_at_end=False,           
        seed=seed
    )

# --- 3. Final Training and Testing Orchestration ---

def run_final_training_and_testing(train_df_list, final_test_synth_df, final_test_yelp_df, new_tokens, output_dir=FINAL_OUTPUT_DIR, seed=SEED):
    
    print(f"\n=======================================================")
    print(f"--- Starting Final Model Training ---")
    print(f"=======================================================")

    # 1. Prepare Final Training Dataset (Combine ALL synthetic_train and yelp_sample_train data)
    final_train_df = pd.concat(train_df_list, ignore_index=True)
    print(f"Final Training Set Size (All initial data): {len(final_train_df)}")
    
    # Handle the edge case where the combined dataframe might be empty despite the size output
    if final_train_df.empty:
        raise ValueError("Combined training DataFrame is empty. Check your input CSV files.")

    # 2. Model Initialization
    MODEL_NAME = 'bert-base-uncased'
    NUM_LABELS = 2
    
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    
    # Re-apply custom tokens and resize embeddings
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # --- Dataset Cleanup and Formatting Function (Refactored for robustness) ---
    
    # The columns the model and Trainer expect after tokenization are: 
    # 'input_ids', 'attention_mask', 'token_type_ids', and 'label'.
    required_trainer_cols = ['input_ids', 'attention_mask', 'token_type_ids', 'label']

    def clean_and_format_dataset(df, tokenizer):
        """Tokenizes, renames 'labels' to 'label', removes unnecessary columns, and sets format to 'torch'."""
        
        # Convert DataFrame to Dataset and Tokenize (adds input_ids, attention_mask, token_type_ids)
        dataset = Dataset.from_pandas(df, preserve_index=False).map(
            lambda examples: tokenize_function(examples, tokenizer), batched=True)
        
        # Rename the 'labels' column (from pandas DF) to 'label' (required by Trainer)
        if 'labels' in dataset.column_names:
            dataset = dataset.rename_column("labels", "label")
        
        # Determine which columns to remove (all non-required columns)
        cols_to_remove = [col for col in dataset.column_names if col not in required_trainer_cols]
        
        # Remove extraneous columns to keep the Dataset clean
        dataset = dataset.remove_columns(cols_to_remove)
        
        # Set format to PyTorch tensors
        dataset.set_format('torch') 
        
        return dataset

    # 3. Tokenization and Dataset Conversion using the new robust function
    
    # Final Training Dataset
    tokenized_final_train_dataset = clean_and_format_dataset(final_train_df, tokenizer)

    # Final Test Datasets (Separate)
    tokenized_final_synth_test_dataset = clean_and_format_dataset(final_test_synth_df, tokenizer)
    tokenized_final_yelp_test_dataset = clean_and_format_dataset(final_test_yelp_df, tokenizer)

    # 4. Trainer Setup and Training
    os.makedirs(output_dir, exist_ok=True)
    final_training_args = get_final_training_args(output_dir, seed)

    trainer = Trainer(
        model=model,
        args=final_training_args,
        train_dataset=tokenized_final_train_dataset, # Should now be correctly formatted
        compute_metrics=compute_metrics, 
    )

    trainer.train()
    
    # Save the FINAL trained model after 3 epochs
    final_model_path = os.path.join(output_dir, 'final_model')
    trainer.save_model(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # 5. Final Evaluation on Held-Out Test Sets
    
    print(f"\n=======================================================")
    print(f"--- FINAL EVALUATION on Held-Out Test Sets ---")
    print(f"=======================================================")
    
    # Evaluate on Synthetic Test Data
    synthetic_eval_results = trainer.evaluate(tokenized_final_synth_test_dataset)
    
    # Evaluate on Yelp Test Data
    yelp_eval_results = trainer.evaluate(tokenized_final_yelp_test_dataset)
    
    
    # 6. Report and Export Results
    print("\n--- Final Test Results (Synthetic Data) ---")
    synth_acc = synthetic_eval_results.get('eval_accuracy')
    synth_f1 = synthetic_eval_results.get('eval_f1')
    print(f"Accuracy: {synth_acc:.4f}")
    print(f"F1 Score: {synth_f1:.4f}")

    print("\n--- Final Test Results (Yelp Data) ---")
    yelp_acc = yelp_eval_results.get('eval_accuracy')
    yelp_f1 = yelp_eval_results.get('eval_f1')
    print(f"Accuracy: {yelp_acc:.4f}")
    print(f"F1 Score: {yelp_f1:.4f}")

    
    # Store all results in a single dictionary to return
    all_final_results = {
        'Synthetic_Accuracy': synth_acc,
        'Synthetic_F1': synth_f1,
        'Yelp_Accuracy': yelp_acc,
        'Yelp_F1': yelp_f1,
    }
    
    # --- EXPORTING DETAILED PREDICTIONS ---
    
    # 6a. Synthetic Predictions
    synth_predictions = trainer.predict(tokenized_final_synth_test_dataset)
    synth_pred_ids = np.argmax(synth_predictions.predictions, axis=1)
    # NOTE: The original DF (final_test_synth_df) still has the 'labels' column
    synth_output_df = final_test_synth_df.copy() 
    synth_output_df['Predicted Label ID'] = synth_pred_ids
    synth_output_df['Source'] = 'Synthetic'
    
    # 6b. Yelp Predictions
    yelp_predictions = trainer.predict(tokenized_final_yelp_test_dataset)
    yelp_pred_ids = np.argmax(yelp_predictions.predictions, axis=1)
    # NOTE: The original DF (final_test_yelp_df) still has the 'labels' column
    yelp_output_df = final_test_yelp_df.copy()
    yelp_output_df['Predicted Label ID'] = yelp_pred_ids
    yelp_output_df['Source'] = 'Yelp'
    
    # Combine and finalize output
    final_predictions_output = pd.concat([synth_output_df, yelp_output_df], ignore_index=True)
    # Note: We use the original 'labels' column in the DataFrame for mapping
    final_predictions_output['Actual Diet'] = final_predictions_output['labels'].map(id_to_label) 
    final_predictions_output['Predicted Diet'] = final_predictions_output['Predicted Label ID'].map(id_to_label)
    
    # Define the final columns for the CSV export
    output_cols = [
        'Source', 
        'dish_name', 
        'description', 
        'text', 
        'labels',          # Original Label ID
        'Actual Diet',     # Original Label Name
        'Predicted Label ID', 
        'Predicted Diet'   # Predicted Label Name
    ]
    csv_output_path = os.path.join(output_dir, 'final_test_predictions_detailed.csv')
    final_predictions_output[output_cols].to_csv(csv_output_path, index=False)
    print(f"\nFinal detailed test predictions exported to: {csv_output_path}")

    return all_final_results

# --- 4. Main Execution ---
if __name__ == '__main__':
    
    # 1. Load all training data (text and labels only, metadata not needed for training)
    synthetic_train_df = load_data_and_preprocess(SYNTHETIC_CSV_PATH, label_map, keep_metadata=False)
    yelp_train_df = load_data_and_preprocess(YELP_CSV_PATH, label_map, keep_metadata=False)
    final_train_df_list = [synthetic_train_df, yelp_train_df]
    
    # 2. Load the completely separate final test files (KEEPING metadata for export)
    print("\n--- Loading Final Test Data for Separate Evaluation and Detailed Export ---")
    final_test_synth_data, final_test_yelp_data = load_final_test_data(
        FINAL_TEST_SYNTHETIC_PATH, 
        FINAL_TEST_YELP_PATH, 
        label_map
    )
    
    # 3. Run the final training and testing
    final_results = run_final_training_and_testing(
        final_train_df_list, 
        final_test_synth_data, 
        final_test_yelp_data,  
        NEW_TOKENS
    )
    
    # 4. Save the final performance metrics
    results_path = os.path.join(FINAL_OUTPUT_DIR, 'final_test_metrics_split.txt')
    with open(results_path, 'w') as f:
        f.write(f"Synthetic Test Accuracy: {final_results['Synthetic_Accuracy']}\n")
        f.write(f"Synthetic Test F1 Score: {final_results['Synthetic_F1']}\n")
        f.write(f"Yelp Test Accuracy: {final_results['Yelp_Accuracy']}\n")
        f.write(f"Yelp Test F1 Score: {final_results['Yelp_F1']}\n")
    print(f"\nFinal split test metrics saved to: {results_path}")
