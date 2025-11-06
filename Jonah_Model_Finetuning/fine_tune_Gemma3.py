import pandas as pd
import numpy as np
import os
import re
import shutil # Added for clean directory removal
import time # Added for real-time timing of steps
import gc # Added for memory cleanup
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset 
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
)
from typing import Dict, Any, Union

# --- 1. Custom Trainer for CausalLM (Loss on Label Only) ---
class CausalLMTrainer(Trainer):
    """
    Custom Trainer for Causal LMs that only computes loss over the generated label token.
    This is CRITICAL for efficient and accurate LoRA fine-tuning for classification.
    """
    def __init__(self, *args, tokenizer, label_token_id, **kwargs):
        # We ensure the parent class Trainer init is called correctly
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.label_token_id = label_token_id

    # FIX: Added 'num_items_in_batch=0' to the signature to resolve the TypeError
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=0):
        # 1. Forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        labels = inputs.get('labels')

        # Shift to the left for CausalLM prediction (prediction of token i is based on logits[i-1])
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 2. Create the label mask (only compute loss on the target label token)
        loss_mask = torch.zeros_like(shift_labels, dtype=torch.float) 
        
        for i in range(shift_labels.size(0)):
            # Find the first non-ignored token index, which corresponds to the start of the label
            # This relies on the tokenizer setting the prompt portion to -100
            valid_label_indices = (shift_labels[i] != -100).nonzero(as_tuple=True)[0]
            
            if valid_label_indices.numel() > 0:
                # We only calculate loss on the first token of the predicted label for stability
                target_index = valid_label_indices[0].item() 
                loss_mask[i, target_index] = 1.0

        
        # 3. Calculate Loss 
        # Using CrossEntropyLoss with ignore_index=-100
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # Apply mask to labels
        final_labels = shift_labels.clone()
        final_labels[loss_mask == 0] = -100

        # Loss is calculated only where final_labels != -100
        loss = loss_fct(shift_logits.view(-1, model.config.vocab_size), final_labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


# --- 2. File Path Definition and Constants ---

SYNTHETIC_CSV_PATH = 'synthetic_train.csv'
YELP_CSV_PATH = 'yelp_sample_train.csv'

# Define label map and its inverse for classification tasks
label_map = {'Vegetarian': 0, 'Non-Vegetarian': 1}
id_to_label = {0: 'Vegetarian', 1: 'Non-Vegetarian'}
NUM_LABELS = len(label_map)

# --- CONSTANTS FOR GEMMA 3 270M (Standard LoRA Fine-Tuning) ---
MODEL_NAME = 'google/gemma-3-270m'

# RESTORED SEQUENCE LENGTH
MAX_SEQ_LENGTH = 128 

# PEFT/LoRA Configuration (1. LORA PARAMETERS REDUCED)
LORA_R = 4 # Changed from 8 to 4 (halves trainable parameters)
LORA_ALPHA = 16 # Changed from 32 to 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", 
    "k_proj", 
    "v_proj", 
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
]

# --- 3. Prompting for Causal LM ---
PROMPT_TEMPLATE = "[INST] Classify the following dish as 'Vegetarian' or 'Non-Vegetarian'.\nDish: {text} [/INST] {diet}"
TEST_PROMPT_TEMPLATE = "[INST] Classify the following dish as 'Vegetarian' or 'Non-Vegetarian'.\nDish: {text} [/INST]"


# --- 4. Data Loading and Splitting ---

def load_and_split_data_into_folds(csv_filepath, num_splits=5, random_state=42):
    """Loads a single CSV, preprocesses it, and splits it into N stratified folds (dataframes)."""
    print(f"\n[DIAGNOSTIC] Starting data load and split for: {csv_filepath}")
    start_time = time.time()
    
    try:
        df_loaded = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"[ERROR] Required file not found: {csv_filepath}. Please ensure this file is available.")
        return [pd.DataFrame() for _ in range(num_splits)]

    # CRITICAL FIX: Handle NaN values
    df_loaded['dish_name'] = df_loaded['dish_name'].fillna('').astype(str)
    df_loaded['description'] = df_loaded['description'].fillna('').astype(str)
    
    # Concatenate the text fields
    df_loaded['text'] = df_loaded['dish_name'] + " - " + df_loaded['description']
    
    # Ensure text and integer labels are present
    data_df = df_loaded[['text', 'diet']] 
    
    # KFold splitting logic
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    
    fold_dataframes = []
    # We use 'diet' (string) column for splitting
    for fold_idx, (_, test_index) in enumerate(kf.split(data_df)): 
        fold_dataframes.append(data_df.iloc[test_index].reset_index(drop=True))
        print(f"  -> Created fold {fold_idx + 1}/{num_splits} with size: {len(data_df.iloc[test_index])}")
        
    end_time = time.time()
    print(f"[DIAGNOSTIC] Data load and split complete for {csv_filepath}. Time taken: {end_time - start_time:.2f}s")
    return fold_dataframes

synthetic_folds = load_and_split_data_into_folds(SYNTHETIC_CSV_PATH)
yelp_folds = load_and_split_data_into_folds(YELP_CSV_PATH)

print(f"\nTotal Synthetic Folds: {len(synthetic_folds)}, Size per fold: ~{len(synthetic_folds[0]) if synthetic_folds else 0}")
print(f"Total Yelp Sample Folds: {len(yelp_folds)}, Size per fold: ~{len(yelp_folds[0]) if yelp_folds else 0}")


# --- 5. Tokenization and Prompting ---

def format_and_tokenize(examples, tokenizer):
    """
    Applies the prompt template and tokenizes the input for CausalLM training.
    """
    
    # Tokenize the full prompt (including the target label)
    texts = [PROMPT_TEMPLATE.format(text=t, diet=d) 
             for t, d in zip(examples['text'], examples['diet'])]
    
    tokenized_inputs = tokenizer(
        texts, 
        truncation=True, 
        max_length=MAX_SEQ_LENGTH, 
        padding='max_length',
    )
    
    # --- CRITICAL: Masking labels for LoRA training ---
    labels = []
    for i, t in enumerate(examples['text']):
        # Create a prompt without the label to find the cutoff point
        prompt_end = TEST_PROMPT_TEMPLATE.format(text=t)
        
        prompt_token_ids = tokenizer(
            prompt_end, 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH, 
            padding='max_length',
            return_tensors='np',
        )['input_ids'][0]
        
        full_input_ids = tokenized_inputs['input_ids'][i]
        label_ids = list(full_input_ids)

        prompt_length = min(len(prompt_token_ids), len(label_ids))
        
        for j in range(prompt_length - 1): # Ignore all tokens of the prompt
            label_ids[j] = -100 
            
        labels.append(label_ids)

    # Convert to PyTorch tensors (using list of tensors for DataCollator)
    tokenized_inputs['labels'] = [torch.tensor(l) for l in labels]
    tokenized_inputs['input_ids'] = [torch.tensor(i) for i in tokenized_inputs['input_ids']]
    tokenized_inputs['attention_mask'] = [torch.tensor(a) for a in tokenized_inputs['attention_mask']]
    
    return tokenized_inputs


# --- 6. Metric Function for CausalLM ---

def compute_metrics(p: Any) -> Dict[str, float]:
    """Placeholder function as evaluation is done via decoding in the main loop."""
    return {'accuracy': 0.0, 'f1': 0.0} 


# --- 7. Training Arguments (3. TRAINING ARGUMENTS OPTIMIZED & 2. GRADIENT CHECKPOINTING) ---

def get_training_args(output_dir, seed=42):
    """Returns a new TrainingArguments instance for a given fold, optimized for MPS/CPU."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,           # Changed from 2 to 1
        gradient_accumulation_steps=8,          # Changed from 16 to 8
        warmup_steps=50,                       
        weight_decay=0.01,                      
        logging_dir=os.path.join('./logs_cv', os.path.basename(output_dir)),
        logging_steps=5,
        eval_strategy="no",                     # Changed from "epoch" to "no"
        save_strategy="epoch",                  
        load_best_model_at_end=False,           # Changed from True to False
        metric_for_best_model="eval_loss",      
        seed=seed,
        gradient_checkpointing=True,            # Changed from False to True
        
        # --- NEW OPTIMIZATION FOR DATALOADER ---
        dataloader_num_workers=0,               # Added
        dataloader_pin_memory=False,            # Added
        
        # --- CRITICAL FIXES FOR MPS PERFORMANCE AND CPU RELIEF ---
        # 1. FORCE NATIVE OPTIMIZER: Prevents slow CPU fallback from bitsandbytes failure
        optim="adamw_torch",                    
        fp16=False,
        # 2. Use bfloat16 for modern Mac GPUs (M-series)
        bf16=torch.backends.mps.is_available() or torch.cuda.is_available(), 
    )

# --- 8. Model Setup for Gemma 3 270M Standard LoRA (2. & 4. GRADIENT CHECKPOINTING & MODEL LOADING OPTIMIZED) ---

def setup_causal_lm_model(model_name):
    """Loads Gemma 3 270M and prepares for standard LoRA fine-tuning (no Q-LoRA)."""
    
    print(f"\n[DIAGNOSTIC] Starting Model and Tokenizer setup for {model_name}.")
    start_time = time.time()
    
    # 1. Determine Device and Dtype for Standard LoRA
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] MPS (Mac GPU) is available. Using standard LoRA on MPS.")
        torch_dtype = torch.bfloat16 # Use bfloat16 for M-series chips
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("[INFO] CUDA is available. Using standard LoRA on CUDA.")
        torch_dtype = torch.float16 
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU. Training will be very slow.")
        torch_dtype = torch.float32

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "right" # Use 'right' padding during training

    # 3. Load Model in standard precision
    print(f"[DIAGNOSTIC] Loading base model... (This is the most time-consuming step of setup)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device.type if device.type != 'mps' else None,
        dtype=torch_dtype, 
        trust_remote_code=True,
        use_cache=False, # Changed from True to False (Disables KV cache for memory savings during training)
        low_cpu_mem_usage=True, # Added (Reduces CPU memory footprint during loading)
    )

    # Manually move to MPS if necessary
    if device.type == 'mps':
         model.to(device)
    
    # 2. Enable Gradient Checkpointing for memory savings
    model.gradient_checkpointing_enable() # Added
    
    # 4. Apply Standard LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    print(f"[DIAGNOSTIC] Model is ready with LoRA applied. Trainable parameters: {model.print_trainable_parameters()}")
    
    end_time = time.time()
    print(f"[DIAGNOSTIC] Model setup complete. Time taken: {end_time - start_time:.2f}s")
    
    return model, tokenizer, device

# --- 9. Cross-Validation Orchestration (5. MEMORY CLEANUP) ---

def run_cross_validation(synthetic_folds, yelp_folds, num_folds=5, output_dir_base='./results_cv_gemma_lora'):
    
    all_results = []
    os.makedirs(output_dir_base, exist_ok=True)
    print(f"\n[DIAGNOSTIC] Created cross-validation output directory: {output_dir_base}") 

    # Loop through each fold (i is the index of the held-out test set)
    for i in range(num_folds):
        
        print(f"\n=======================================================")
        print(f"--- Starting Cross-Validation Fold {i + 1}/{num_folds} ---")
        print(f"=======================================================")
        
        # --- Data Setup for Current Fold ---
        test_synthetic_df = synthetic_folds[i]
        test_yelp_df = yelp_folds[i]
        
        train_dfs = []
        for j in range(num_folds):
            if i != j:
                train_dfs.append(synthetic_folds[j]) 
                train_dfs.append(yelp_folds[j])
        
        train_df = pd.concat(train_dfs, ignore_index=True)

        print(f"[DIAGNOSTIC] Training set size for Fold {i+1}: {len(train_df)}")

        # --- Model Initialization for Current Fold (Fresh Start) ---
        model, tokenizer, device = setup_causal_lm_model(MODEL_NAME)
        
        label_token_id = tokenizer.convert_tokens_to_ids("Non-Vegetarian") 

        tokenization_fn = lambda examples: format_and_tokenize(examples, tokenizer)

        # --- Tokenization Start ---
        print(f"\n[DIAGNOSTIC] Starting tokenization (map operation) for ALL datasets (Train/Eval/Test).")
        start_tokenization_time = time.time()
        
        # Convert DataFrames to Hugging Face Datasets
        train_dataset = Dataset.from_pandas(train_df, preserve_index=False).map(
            tokenization_fn, 
            batched=True, 
            remove_columns=['text', 'diet'], 
            load_from_cache_file=False
        )
        print("  -> Train dataset tokenization complete.")
        
        synthetic_test_dataset = Dataset.from_pandas(test_synthetic_df, preserve_index=False).map(
            tokenization_fn, 
            batched=True, 
            remove_columns=['text', 'diet'],
            load_from_cache_file=False
        )
        print("  -> Synthetic test dataset tokenization complete.")

        yelp_test_dataset = Dataset.from_pandas(test_yelp_df, preserve_index=False).map(
            tokenization_fn, 
            batched=True, 
            remove_columns=['text', 'diet'],
            load_from_cache_file=False
        )
        print("  -> Yelp test dataset tokenization complete.")
        
        end_tokenization_time = time.time()
        print(f"[DIAGNOSTIC] Total tokenization time: {end_tokenization_time - start_tokenization_time:.2f}s")


        # --- Trainer Setup and Training ---
        fold_output_dir = os.path.join(output_dir_base, f'fold_{i+1}')
        fold_training_args = get_training_args(fold_output_dir, seed=42 + i)
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )
        
        # Use the CustomTrainer for focused loss calculation
        trainer = CausalLMTrainer(
            model=model,
            args=fold_training_args,
            train_dataset=train_dataset,
            eval_dataset=synthetic_test_dataset, 
            compute_metrics=compute_metrics, 
            tokenizer=tokenizer,
            data_collator=data_collator,
            label_token_id=label_token_id,
        )

        if device.type == 'cpu':
            print("\n[PERFORMANCE WARNING] Standard LoRA on Gemma 3 270M will be slow on CPU. Consider upgrading to a machine with CUDA or MPS.\n")

        print(f"\n[DIAGNOSTIC] Starting **TRAINING** for Fold {i+1} (1 Epoch)...")
        start_training_time = time.time()
        
        # The actual training call
        trainer.train()
        
        end_training_time = time.time()
        print(f"[DIAGNOSTIC] Training for Fold {i+1} complete. Time taken: {end_training_time - start_training_time:.2f}s")

        # Save the LoRA adapter only
        trainer.model.save_pretrained(fold_output_dir)
        tokenizer.save_pretrained(fold_output_dir)
        print(f"[DIAGNOSTIC] LoRA adapter for Fold {i+1} saved to: {fold_output_dir}")

        # --- Dual Testing and Results Storage (Via Decoding) ---
        
        def decode_and_evaluate(model, tokenizer, original_df, source_name, device):
            """Generates predictions by decoding text and evaluates results. (6. EVALUATION BATCH SIZE REDUCED & CLEANUP)"""
            
            print(f"[DIAGNOSTIC] Starting generation/evaluation for {source_name} data...")
            start_eval_time = time.time()
            
            model.to(device)
            model.eval()
            
            # Set padding side to 'left' for efficient CausalLM generation
            tokenizer.padding_side = "left" 
            
            all_predictions = []
            
            # Use small batch size for generation
            batch_size = 2 # Changed from 4 to 2
            total_batches = (len(original_df) + batch_size - 1) // batch_size
            
            for batch_idx, start_idx in enumerate(range(0, len(original_df), batch_size)):
                batch_df = original_df.iloc[start_idx:start_idx + batch_size]
                
                # Print batch start
                if batch_idx % 50 == 0 or batch_idx == total_batches - 1:
                     print(f"  -> Generating batch {batch_idx + 1}/{total_batches}...")

                # 1. Prepare Prompts
                prompts = [TEST_PROMPT_TEMPLATE.format(text=t) for t in batch_df['text']]
                
                # 2. Tokenize Prompts (with left padding)
                inputs = tokenizer(
                    prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=MAX_SEQ_LENGTH
                ).to(device)

                # 3. Generate Prediction
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10, 
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        temperature=1.0, 
                    )

                # 4. Decode output and clean up
                decoded_outputs = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                
                # 5. Extract classification label
                for output in decoded_outputs:
                    match_nonveg = re.match(r'Non-Vegetarian', output, re.IGNORECASE)
                    match_veg = re.match(r'Vegetarian', output, re.IGNORECASE)
                    
                    if match_nonveg:
                        pred = 'Non-Vegetarian'
                    elif match_veg:
                        pred = 'Vegetarian'
                    else:
                        # Default prediction
                        pred = 'Non-Vegetarian' 
                    
                    all_predictions.append(pred)

                # 6. PERIODIC MEMORY CLEANUP
                if (batch_idx + 1) % 20 == 0:
                    if device.type == 'mps':
                        torch.mps.empty_cache()
                    elif device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()

            # 7. Evaluation
            actual_labels = original_df['diet'].tolist()
            predicted_labels_id = [label_map[p] for p in all_predictions]
            actual_labels_id = [label_map[a] for a in actual_labels]

            accuracy = accuracy_score(actual_labels_id, predicted_labels_id)
            f1 = f1_score(actual_labels_id, predicted_labels_id, average='binary', pos_label=1) 

            tokenizer.padding_side = "right" # Reset padding side

            # 8. Create comparison DataFrame
            comparison_df = original_df.copy()
            comparison_df['Actual Diet'] = actual_labels
            comparison_df['Predicted Diet'] = all_predictions
            comparison_df['Source'] = source_name
            comparison_df.rename(columns={'text': 'Dish/Description'}, inplace=True)
            
            end_eval_time = time.time()
            print(f"[DIAGNOSTIC] Generation/Evaluation for {source_name} complete. Time taken: {end_eval_time - start_eval_time:.2f}s")
            
            return comparison_df[['Dish/Description', 'Actual Diet', 'Predicted Diet', 'Source']], accuracy, f1


        # 1. Evaluate on Synthetic Held-Out Set
        print(f"\n--- Evaluating Fold {i+1} on Synthetic Data ({len(test_synthetic_df)} samples) ---")
        synthetic_comparison_df, synth_acc, synth_f1 = decode_and_evaluate(
            model, tokenizer, test_synthetic_df, 'Synthetic', device)
        
        # 2. Evaluate on Yelp Held-Out Set (Crucial comparison)
        print(f"--- Evaluating Fold {i+1} on Yelp Data ({len(test_yelp_df)} samples) ---")
        yelp_comparison_df, yelp_acc, yelp_f1 = decode_and_evaluate(
            model, tokenizer, test_yelp_df, 'Yelp', device)
        
        print(f"\nFold {i+1} Results: Synth Acc={synth_acc:.4f}, Yelp Acc={yelp_acc:.4f}")
        
        # Store comprehensive results
        all_results.append({
            'Fold': i + 1,
            'Synthetic_Held_Out': f'Fold_{i+1}',
            'Synth_Accuracy': synth_acc,
            'Synth_F1': synth_f1,
            'Yelp_Accuracy': yelp_acc,
            'Yelp_F1': yelp_f1,
        })
        
        # --- Export Test Set Predictions ---
        final_output_df_clean = pd.concat([synthetic_comparison_df, yelp_comparison_df], ignore_index=True)
        csv_output_path = os.path.join(output_dir_base, f'predictions_fold_{i+1}.csv')
        final_output_df_clean.to_csv(csv_output_path, index=False)
        print(f"[DIAGNOSTIC] Combined predictions for Fold {i+1} exported to: {csv_output_path}")

        # --- DEMO LIMIT: ONLY RUN FIRST FOLD ---
        if i == 0:
            print("\n[DEMO LIMIT] Stopping after Fold 1 for demonstration.")
            
            # --- 5. MEMORY CLEANUP AT END OF FOLD (NEW) ---
            print(f"[CLEANUP] Performing memory cleanup for Fold {i+1}...")
            del trainer
            del model
            
            # Use appropriate cache clearing function
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect() 

            break 
            
    return pd.DataFrame(all_results)

# --- 10. Final Execution ---
if __name__ == '__main__':
    # Ensure a clean start to avoid issues from the interrupted run
    output_dir_base='./results_cv_gemma_lora'
    if os.path.exists(output_dir_base):
        try:
            print(f"\n[CLEANUP] Removing incomplete directory: {output_dir_base}")
            shutil.rmtree(output_dir_base)
        except OSError as e:
            print(f"[CLEANUP ERROR] Failed to remove directory {output_dir_base}: {e}")

    # Run the cross-validation
    print("\n\n#######################################################")
    print("##### STARTING FINE-TUNING PROCESS (V5.7 - OPTIMIZED) #####")
    print("#######################################################")
    start_total_time = time.time()
    
    results_df = run_cross_validation(
        synthetic_folds, 
        yelp_folds, 
        num_folds=5
    )
    
    end_total_time = time.time()
    print(f"\n[DIAGNOSTIC] TOTAL SCRIPT EXECUTION TIME: {end_total_time - start_total_time:.2f}s")
    
    print("\n--- Cross-Validation Results Summary (Only Fold 1 shown in this demo run) ---")
    
    # Format results for display
    if not results_df.empty:
        results_df_display = results_df.copy()
        for col in ['Synth_Accuracy', 'Synth_F1', 'Yelp_Accuracy', 'Yelp_F1']:
            results_df_display[col] = results_df_display[col].map('{:.4f}'.format)
        
        try:
            # Check if tabulate is installed to safely use to_markdown
            import tabulate
            print(results_df_display.to_markdown(index=False))
        except ImportError:
            # Fallback for environments without tabulate
            print("[Dependency Warning] 'tabulate' module not found. Displaying results using to_string() instead.")
            print(results_df_display.to_string(index=False))
            
    else:
        print("No results were generated.")