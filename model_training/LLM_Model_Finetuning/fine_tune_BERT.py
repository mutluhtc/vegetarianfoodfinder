import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset 
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# --- Dependency Check for Accelerate ---
try:
    import accelerate
except ImportError:
    print("\n[DEPENDENCY WARNING] The 'accelerate' library is missing or outdated.")
    print("Please install it using: pip install accelerate>=0.26.0 or pip install transformers[torch]")
# ----------------------------------------


# --- 1. File Path Definition and Constants ---

OUTPUT_DIR = './results_cv_bert_v7'

# MANDATORY: Use the uploaded file path for the two datasets
SYNTHETIC_CSV_PATH = 'synthetic_train.csv'
YELP_CSV_PATH = 'yelp_sample_train.csv'

# Define label map and its inverse once
label_map = {'Vegetarian': 0, 'Non-Vegetarian': 1}
id_to_label = {0: 'Vegetarian', 1: 'Non-Vegetarian'}

# Define the common new tokens for BERT

#NEW_TOKENS = []

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

# NEW_TOKENS = [
#     "brisket", "sourdough", "buttermilk", "cobbler", "roux", "clam chowder", "jambalaya", "po'boy", "griddle", "slaw",
#     "barbecue", "gravy", "hushpuppies", "pecan", "gumbo", "chili", "coleslaw", "patty", "pulled pork", "cornbread",
#     "hushpuppy", "biscuit", "cheddar", "buffalo", "key lime", "succotash", "apple pie", "chili con carne", "catfish", "meatloaf",
#     "paneer", "masala", "ghee", "korma", "bharta", "tandoor", "vindaloo", "raita", "naan", "biryani",
#     "chapati", "paratha", "aloo", "palak", "tikka", "samosa", "chana", "jalfrezi", "dosa", "curried",
#     "mutter", "rogan josh", "gulab jamun", "chutney", "sambar", "idli", "vada", "thali", "papadum", "kochuri",
#     "wok", "dim sum", "char siu", "bok choy", "hoisin", "mapo tofu", "congee", "szechuan", "xiao long bao", "lo mein",
#     "bao", "shu mai", "peking", "jiaozi", "wonton", "chow mein", "ma po", "kung pao", "chop suey", "scallion",
#     "zongzi", "dan dan", "hot pot", "shaokao", "siu yeh", "dan tat", "fung zao", "jianbing", "char kway teow", "mala",
#     "galangal", "kaffir lime", "lemongrass", "nam pla", "tom kha", "pad see ew", "satay", "sticky rice", "larb", "phanaeng",
#     "chili paste", "fish sauce", "massaman", "green curry", "red curry", "basil", "prik", "pad krapow", "mango sticky rice", "som tam",
#     "khao soi", "pad prik", "tom saap", "hor mok", "pla pao", "kanom krok", "met mamuang", "pad woon sen", "kluay tod", "gang som",
#     "burrata", "prosciutto", "pesto", "ragu", "gnocchi", "bolognese", "osso buco", "focaccia", "polenta", "tiramisu",
#     "mozzarella", "parmesan", "cannoli", "calzone", "bruschetta", "caprese", "lasagna", "risotto", "marinara", "chianti",
#     "arancini", "cacio e pepe", "pancetta", "guanciale", "pecorino", "zabaglione", "panna cotta", "minestrone", "sicilian", "spritz",
#     "hummus", "falafel", "tzatziki", "tagine", "couscous", "tahini", "souvlaki", "pita", "baba ghanoush", "moussaka",
#     "halloumi", "dolmades", "kebabs", "gyro", "feta", "borek", "baklava", "spanakopita", "tabouleh", "kibbeh",
#     "saganaki", "labneh", "za'atar", "basturma", "kibbe", "merguez", "avgolemono", "bourekas", "pita bread", "shish kebab",
#     "sushi", "sashimi", "miso", "dashi", "tempura", "ramen", "udon", "yakitori", "tonkatsu", "teriyaki",
#     "wasabi", "edamame", "gyoza", "shoyu", "nigiri", "mochi", "soba", "katsu", "onigiri", "tamagoyaki",
#     "unagi", "ebi", "tori", "teppanyaki", "okazu", "kaiseki", "ikura", "nori", "tataki", "kombu",
#     "beef", "chicken", "pork", "lamb", "veal", "oxtail", "duck", "turkey", "nuggets", "goose", "fish", "salmon",
#     "meat", "tuna", "cod", "trout", "mackerel", "tilapia", "herring", "sea bass", "anchovy", "sardine", "seafood",
#     "shrimp", "prawn", "shellfish", "crab", "lobster", "mussels", "clams", "oysters", "scallops", "squid", "octopus",
#     "calamari", "crustaceans", "mollusks", "gelatin", "lard", "tallow", "drippings", "bacon", "ham", "manzo",
#     "sausage", "salami", "pepperoni", "chorizo", "chicharron", "meatballs", "kebab", "keema", "mutton", "steak",
#     "asada", "carnitas", "ribs", "ribeye", "chops", "wings", "drumsticks", "offal", "venison", "bison", "rabbit",
#     "quail", "goat", "pheasant", "eel", "swordfish", "halibut", "sole", "beef jerky", "pork rinds", "fish sticks",
#     "crab cakes", "shrimp cocktail", "lobster bisque", "squid ink pasta", "foie gras", "pate", "tripe", "sweetbreads",
#     "escargot", "caviar", "roe", "worcestershire sauce", "oyster sauce", "shrimp paste", "meat extract",
#     "beef bouillon", "chicken bouillon", "pork bouillon", "lamb bouillon", "fish bouillon", "carmine", "cochineal",
#     "Brazil nut", "pumpkin", "capers", "olives", "peppercorn", "homies", "toast", "chayote", "jackfruit",
#     "beyond burger", "brie", "ricotta", "mascarpone", "cream cheese", "queso fresco", "cottage cheese", "chèvre",
#     "havarti", "fontina", "munster", "monterey jack", "provolone", "colby", "edam", "port salut",
#     "pecorino romano", "asiago", "manchego", "grana padano", "cotija", "gruyère", "comté", "danish blue",
#     "camembert", "emmental", "raclette", "queso oaxaca", "panela", "kefalotyri", "american cheese", "pepper jack"
# ]

# --- 2. Data Loading and Splitting into 5 Folds ---

def load_and_split_data_into_folds(csv_filepath, label_map, num_splits=5, random_state=42):
    """Loads a single CSV, preprocesses it, and splits it into N stratified folds (dataframes)."""
    print(f"--- Loading and splitting data from: {csv_filepath} ---")
    df_loaded = pd.read_csv(csv_filepath)
    df_loaded['labels'] = df_loaded['diet'].map(label_map) 
    
    # --- CRITICAL FIX: Handle NaN values by replacing them with empty strings. ---
    # NaN values can break the tokenizer when converting the dataset back from pandas.
    df_loaded['dish_name'] = df_loaded['dish_name'].fillna('').astype(str)
    df_loaded['description'] = df_loaded['description'].fillna('').astype(str)
    
    # Concatenate the text fields
    df_loaded['text'] = df_loaded['dish_name'] + " - " + df_loaded['description']
    # -----------------------------------------------------------------------------
    
    data_df = df_loaded[['text', 'labels']]
    
    # KFold splitting logic
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    
    # Store the fold DataFrames
    fold_dataframes = []
    # kf.split returns (train_indices, test_indices). We only need the test indices to get the 5 parts.
    for _, test_index in kf.split(data_df):
        fold_dataframes.append(data_df.iloc[test_index].reset_index(drop=True))
        
    return fold_dataframes

# Load and split both data files
synthetic_folds = load_and_split_data_into_folds(SYNTHETIC_CSV_PATH, label_map)
yelp_folds = load_and_split_data_into_folds(YELP_CSV_PATH, label_map)

print(f"Total Synthetic Folds (A-E): {len(synthetic_folds)}, Size per fold: ~{len(synthetic_folds[0])}")
print(f"Total Yelp Sample Folds (A-E): {len(yelp_folds)}, Size per fold: ~{len(yelp_folds[0])}")


# --- 3-6. Helper Functions (Tokenization, Metrics, Training Args) ---

# Reusable Tokenization Function
def tokenize_function(examples, tokenizer):
    """Tokenizes the text column for BERT input."""
    # FIX: Explicitly cast to list is maintained to ensure the tokenizer receives a standard batch format
    return tokenizer(list(examples['text']), truncation=True, padding='max_length', max_length=128)

# Reusable Metric Function
def compute_metrics(eval_pred):
    """Computes accuracy and F1 score using sklearn."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary', pos_label=1)

    return {
        'eval_accuracy': accuracy,
        'eval_f1': f1,
    }

# Common Training Arguments
def get_training_args(output_dir, seed=42):
    """Returns a new TrainingArguments instance for a given fold."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=7,
        learning_rate=2e-5,                     
        per_device_train_batch_size=4,          
        per_device_eval_batch_size=4,           
        warmup_steps=100,                       
        weight_decay=0.01,                      
        logging_dir=os.path.join('./logs_cv', os.path.basename(output_dir)),
        logging_steps=10,
        eval_strategy="epoch",                  
        save_strategy="epoch",                  
        load_best_model_at_end=True,            
        metric_for_best_model="eval_f1",        
        seed=seed
    )

# --- 7. Cross-Validation Orchestration ---

def run_cross_validation(synthetic_folds, yelp_folds, id_to_label, new_tokens, num_folds=5, output_dir_base=OUTPUT_DIR):
    
    # Store results for final analysis
    all_results = []
    
    # Ensure CV output directory exists silently
    os.makedirs(output_dir_base, exist_ok=True)
    if not os.path.isdir(os.path.join(output_dir_base, 'fold_1')):
         # Only print if directory was actually created or is fresh
        print(f"Created cross-validation output directory: {output_dir_base}") 

    # Loop through each fold (i is the index of the held-out test set)
    for i in range(num_folds):
        
        print(f"\n=======================================================")
        print(f"--- Starting Cross-Validation Fold {i + 1}/{num_folds} ---")
        print(f"=======================================================")

        # --- Data Setup for Current Fold ---
        # Determine the held-out test sets (A, B, C, D, or E)
        test_synthetic_df = synthetic_folds[i]
        test_yelp_df = yelp_folds[i]
        
        # Determine the training sets (combination of all other 4 folds from both files)
        train_dfs = []
        for j in range(num_folds):
            if i != j:
                # Add Synthetic train parts (A-D or equivalent permutation)
                train_dfs.append(synthetic_folds[j]) 
                # Add Yelp train parts (A-D or equivalent permutation)
                train_dfs.append(yelp_folds[j])
        
        # Concatenate all training dataframes
        train_df = pd.concat(train_dfs, ignore_index=True)

        print(f"Training set size: {len(train_df)}")
        print(f"Test Set 1 (Synthetic) size: {len(test_synthetic_df)}")
        print(f"Test Set 2 (Yelp) size: {len(test_yelp_df)}")


        # --- Model Initialization for Current Fold (Fresh Start) ---
        MODEL_NAME = 'bert-base-uncased'
        NUM_LABELS = 2
        
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
        
        # Re-apply custom tokens and resize embeddings
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))

        # --- Tokenization and Dataset Conversion ---
        
        # Note: Must pass the tokenizer to map due to scope
        tokenized_train_dataset = Dataset.from_pandas(train_df, preserve_index=False).map(
            lambda examples: tokenize_function(examples, tokenizer), batched=True)
        tokenized_synthetic_test = Dataset.from_pandas(test_synthetic_df, preserve_index=False).map(
            lambda examples: tokenize_function(examples, tokenizer), batched=True)
        tokenized_yelp_test = Dataset.from_pandas(test_yelp_df, preserve_index=False).map(
            lambda examples: tokenize_function(examples, tokenizer), batched=True)

        # Final dataset preparation
        tokenized_train_dataset = tokenized_train_dataset.remove_columns(['text']) 
        tokenized_synthetic_test = tokenized_synthetic_test.remove_columns(['text'])
        tokenized_yelp_test = tokenized_yelp_test.remove_columns(['text'])
        
        tokenized_train_dataset.set_format('torch')
        tokenized_synthetic_test.set_format('torch')
        tokenized_yelp_test.set_format('torch')
        
        # --- Trainer Setup and Training ---
        fold_output_dir = os.path.join(output_dir_base, f'fold_{i+1}')
        fold_training_args = get_training_args(fold_output_dir, seed=42 + i)

        trainer = Trainer(
            model=model,
            args=fold_training_args,
            train_dataset=tokenized_train_dataset,
            # Use Synthetic test set for internal evaluation/best model tracking
            eval_dataset=tokenized_synthetic_test, 
            compute_metrics=compute_metrics,
        )

        trainer.train()
        
        # Save the best model for this fold
        trainer.save_model(fold_output_dir)
        print(f"Best model for Fold {i+1} saved to: {fold_output_dir}")

        # --- Dual Testing and Results Storage ---
        
        # 1. Evaluate on Synthetic Held-Out Set
        synthetic_eval_results = trainer.evaluate(tokenized_synthetic_test)
        
        # 2. Evaluate on Yelp Held-Out Set (Crucial comparison)
        yelp_eval_results = trainer.evaluate(tokenized_yelp_test)
        
        # Store comprehensive results
        all_results.append({
            'Fold': i + 1,
            'Synthetic_Held_Out': f'Fold_{i+1}', # A, B, C, D, or E
            'Synth_Accuracy': synthetic_eval_results.get('eval_accuracy'),
            'Synth_F1': synthetic_eval_results.get('eval_f1'),
            'Yelp_Accuracy': yelp_eval_results.get('eval_accuracy'),
            'Yelp_F1': yelp_eval_results.get('eval_f1'),
        })
        
        # --- Export Test Set Predictions (Synthetic and Yelp combined) ---
        
        # 1. Synthetic Predictions
        synthetic_predictions = trainer.predict(tokenized_synthetic_test)
        synthetic_pred_ids = np.argmax(synthetic_predictions.predictions, axis=1)

        synthetic_comparison_df = test_synthetic_df.copy()
        synthetic_comparison_df['predicted_diet'] = synthetic_pred_ids
        synthetic_comparison_df['Source'] = 'Synthetic'
        
        # 2. Yelp Predictions
        yelp_predictions = trainer.predict(tokenized_yelp_test)
        yelp_pred_ids = np.argmax(yelp_predictions.predictions, axis=1)

        yelp_comparison_df = test_yelp_df.copy()
        yelp_comparison_df['predicted_diet'] = yelp_pred_ids
        yelp_comparison_df['Source'] = 'Yelp'

        # Combine, map to text labels, and export
        fold_final_output_df = pd.concat([synthetic_comparison_df, yelp_comparison_df], ignore_index=True)
        fold_final_output_df['Actual Diet'] = fold_final_output_df['labels'].map(id_to_label)
        fold_final_output_df['Predicted Diet'] = fold_final_output_df['predicted_diet'].map(id_to_label)

        final_output_df_clean = fold_final_output_df[['text', 'Actual Diet', 'Predicted Diet', 'Source']]
        final_output_df_clean.columns = ['Dish/Description', 'Actual Diet', 'Predicted Diet', 'Source']
        
        csv_output_path = os.path.join(output_dir_base, f'predictions_fold_{i+1}.csv')
        final_output_df_clean.to_csv(csv_output_path, index=False)
        print(f"Combined predictions for Fold {i+1} exported to: {csv_output_path}")

        # --- DEMO LIMIT: ONLY RUN FIRST FOLD ---
        # NOTE: Remove this line if running the script in a full compute environment 
        # to execute all 5 cross-validation folds.
        #if i == 0:
            #print("\n[DEMO LIMIT] Stopping after Fold 1 for demonstration.")
            #break 
            
    return pd.DataFrame(all_results)

# --- 8. Final Execution ---
if __name__ == '__main__':
    # Run the cross-validation
    results_df = run_cross_validation(
        synthetic_folds, 
        yelp_folds, 
        id_to_label, 
        NEW_TOKENS,
        num_folds=5
    )
    
    print("\n--- Cross-Validation Results Summary (Only Fold 1 shown in this demo run) ---")
    
    # Format results for display
    if not results_df.empty:
        results_df_display = results_df.copy()
        for col in ['Synth_Accuracy', 'Synth_F1', 'Yelp_Accuracy', 'Yelp_F1']:
            results_df_display[col] = results_df_display[col].map('{:.4f}'.format)
        
        # FIX: Catch ImportError if 'tabulate' is missing and fall back to to_string()
        try:
            print(results_df_display.to_markdown(index=False))
        except ImportError:
            print("[Dependency Warning] 'tabulate' module not found. Displaying results using to_string() instead.")
            print(results_df_display.to_string(index=False))
            
    else:
        print("No results were generated.")
