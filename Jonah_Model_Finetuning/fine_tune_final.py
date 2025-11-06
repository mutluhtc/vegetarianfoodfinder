import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset 
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import warnings
from dotenv import load_dotenv 
from huggingface_hub import login, HfFolder 

# Suppress UserWarning related to MPS and pin_memory if running on Apple Silicon
warnings.filterwarnings("ignore", ".*'pin_memory' argument is set as true but not supported on MPS.*")

# --- 0. Hugging Face Setup ---

# Load environment variables from .env file
load_dotenv() 

# Get the Hugging Face token from environment variables
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") 

# Define the name for your model repository on Hugging Face Hub (e.g., your_username/model_name)
HF_REPO_ID = 'ytarasov/diet-classifier-bert-v1' 


# --- 1. File Path Definition and Constants (ADJUSTED) ---

# All Data Paths
SYNTHETIC_TRAIN_PATH = 'synthetic_train.csv'
YELP_TRAIN_PATH = 'yelp_sample_train.csv'
FINAL_TEST_SYNTHETIC_PATH = 'synthetic_test.csv'
FINAL_TEST_YELP_PATH = 'yelp_sample_test.csv'

# Output Directory (RENAMED)
FINAL_OUTPUT_DIR = './final_production_model'

# Define label map and its inverse
label_map = {'Vegetarian': 0, 'Non-Vegetarian': 1}
id_to_label = {0: 'Vegetarian', 1: 'Non-Vegetarian'}
SEED = 42

# Define the common new tokens for BERT (omitted for brevity)
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


# --- 2. Data Loading and Preparation Functions (load_final_test_data function REMOVED) ---

def load_data_and_preprocess(csv_filepath, label_map, keep_metadata=False):
    """Loads a single CSV and preprocesses it into a DataFrame, returning only text and labels."""
    print(f"--- Loading data from: {csv_filepath} ---")
    df_loaded = pd.read_csv(csv_filepath)
    df_loaded['labels'] = df_loaded['diet'].map(label_map) 
    df_loaded['labels'] = df_loaded['labels'].astype(np.int64) 
    
    # Handle NaN values and concatenate text fields
    df_loaded['dish_name'] = df_loaded['dish_name'].fillna('').astype(str)
    df_loaded['description'] = df_loaded['description'].fillna('').astype(str)
    df_loaded['text'] = df_loaded['dish_name'] + " - " + df_loaded['description']
    
    # For a final production model, we only need 'text' and 'labels'
    return df_loaded[['text', 'labels']]

def tokenize_function(examples, tokenizer):
    """Tokenizes the text column for BERT input."""
    return tokenizer(list(examples['text']), truncation=True, padding='max_length', max_length=128)

def compute_metrics(eval_pred):
    """Computes accuracy and F1 score using sklearn. (Kept for consistency, though unused in final training)"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary', pos_label=1) 

    return {
        'accuracy': accuracy,
        'f1': f1,
    }

def get_final_training_args(output_dir, seed, hf_repo_id):
    """Returns TrainingArguments for final training using optimal settings."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,                     
        learning_rate=2e-5,                     
        per_device_train_batch_size=4,          
        per_device_eval_batch_size=4,           
        warmup_steps=100,                       
        weight_decay=0.01,                      
        logging_dir=os.path.join('./logs_final', os.path.basename(output_dir)),
        logging_steps=10,
        eval_strategy="no",                     # No evaluation needed for final production training
        save_strategy="epoch",                  
        load_best_model_at_end=False,           
        seed=seed,
        # --- Hugging Face Hub Integration ---
        push_to_hub=True,                      
        hub_model_id=hf_repo_id.split('/')[1], 
        hub_strategy="end",                    
        hub_token=os.getenv("HUGGINGFACE_TOKEN"),
        # ------------------------------------
    )

# --- 3. Final Production Training Orchestration (SIMPLIFIED) ---

def run_production_training(full_train_df, new_tokens, hf_repo_id, output_dir=FINAL_OUTPUT_DIR, seed=SEED):
    
    print(f"\n=======================================================")
    print(f"--- Starting FINAL Production Model Training ---")
    print(f"Total Samples: {len(full_train_df)}")
    print(f"Output Directory: {output_dir}")
    print(f"Hugging Face Repo: {hf_repo_id}")
    print(f"=======================================================")

    # 1. Model Initialization
    MODEL_NAME = 'bert-base-uncased'
    NUM_LABELS = 2
    
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS,
        id2label=id_to_label, 
        label2id=label_map     
    )
    
    # Re-apply custom tokens and resize embeddings
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # --- Dataset Formatting ---
    required_trainer_cols = ['input_ids', 'attention_mask', 'token_type_ids', 'label']

    def clean_and_format_dataset(df, tokenizer):
        """Tokenizes, renames 'labels' to 'label', removes unnecessary columns, and sets format to 'torch'."""
        
        dataset = Dataset.from_pandas(df, preserve_index=False).map(
            lambda examples: tokenize_function(examples, tokenizer), batched=True)
        
        if 'labels' in dataset.column_names:
            dataset = dataset.rename_column("labels", "label")
        
        cols_to_remove = [col for col in dataset.column_names if col not in required_trainer_cols]
        dataset = dataset.remove_columns(cols_to_remove)
        dataset.set_format('torch') 
        
        return dataset

    # 2. Tokenization and Dataset Conversion
    tokenized_full_train_dataset = clean_and_format_dataset(full_train_df, tokenizer)
    print(f"Tokenized dataset size: {len(tokenized_full_train_dataset)}")

    # 3. Trainer Setup and Training
    os.makedirs(output_dir, exist_ok=True)
    final_training_args = get_final_training_args(output_dir, seed, hf_repo_id)

    trainer = Trainer(
        model=model,
        args=final_training_args,
        train_dataset=tokenized_full_train_dataset, 
        # compute_metrics is not needed since eval_strategy="no"
    )

    trainer.train()
    
    # 4. Save Model Locally AND Push to Hub
    
    final_model_path = os.path.join(output_dir, 'final_model')
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\n✅ Final production model and tokenizer saved locally to: {final_model_path}")
    
    print(f"\n--- Pushing Model and Tokenizer to Hugging Face Hub (Repo: {hf_repo_id}) ---")
    
    try:
        # Pushes model, tokenizer, and TrainingArguments (config)
        trainer.push_to_hub(commit_message=f"Final production model trained on all {len(full_train_df)} samples.")
        print("✅ Successfully pushed final production model to Hugging Face Hub.")
    except Exception as e:
        print(f"❌ Failed to push to Hugging Face Hub. Error: {e}")

    # Return None as there are no evaluation results to return
    return None

# --- 4. Main Execution (ADJUSTED) ---
if __name__ == '__main__':
    
    # Log in to Hugging Face Hub
    if HF_TOKEN:
        print("\n--- Attempting to log in to Hugging Face Hub ---")
        try:
            login(token=HF_TOKEN, add_to_git_credential=True)
            print("Successfully logged in.")
        except Exception as e:
            print(f"Warning: Could not log in to Hugging Face. Error: {e}")
    else:
        print("\n--- HUGGINGFACE_TOKEN not found in .env file. Model will only be saved locally. ---")
    
    
    # 1. Load ALL data files (Train and Test)
    # We load all data using the same preprocessing function
    all_dfs = [
        load_data_and_preprocess(SYNTHETIC_TRAIN_PATH, label_map, keep_metadata=False),
        load_data_and_preprocess(YELP_TRAIN_PATH, label_map, keep_metadata=False),
        load_data_and_preprocess(FINAL_TEST_SYNTHETIC_PATH, label_map, keep_metadata=False),
        load_data_and_preprocess(FINAL_TEST_YELP_PATH, label_map, keep_metadata=False),
    ]
    
    # 2. Combine all DataFrames into one master training set
    full_production_train_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\n--- Total Production Training Samples: {len(full_production_train_df)} ---")
    
    if full_production_train_df.empty:
        raise ValueError("The combined production training DataFrame is empty.")
    
    # 3. Run the final training (NO testing step needed)
    run_production_training(
        full_production_train_df, 
        NEW_TOKENS,
        hf_repo_id=HF_REPO_ID 
    )
    
    # 4. Final output message (metrics logging is removed since there are no test metrics)
    print("\nTraining complete. The model has been saved locally and pushed to the Hugging Face Hub.")
    print(f"Check the directory: **{FINAL_OUTPUT_DIR}**")