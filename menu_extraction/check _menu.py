import pandas as pd
import requests
import time
import json
import os
from typing import Optional, Tuple, Dict, Any
from dotenv import load_dotenv, find_dotenv

# --- Configuration Variables for Dynamic Naming ---
# Change this variable (e.g., to "new_york" or "london") to run analysis 
# for a different city and load/save the corresponding files.
TARGET_CITY = "san_jose" 
# NOTE: The API_KEY will be loaded dynamically using python-dotenv or provided by the environment.
MODEL_NAME = "gemini-2.5-flash-preview-05-20" # Reverted/Optimized Model Name for classification
MAX_RETRIES = 5
ROW_LIMIT = 200 # Maximum number of rows to process per run for verification (set to 20)
# --------------------------------------------------

def setup_config(city: str, api_key: str, model: str) -> Dict[str, Any]:
    """
    Centralizes configuration and dynamically generates file paths and API URL.
    """
    config = {
        "TARGET_CITY": city,
        "INPUT_FILE": f"{city}_restaurants_test_with_rawtext.csv",
        "OUTPUT_FILE": f"{city}_restaurants_with_menu_analysis.csv",
        "API_KEY": api_key,
        "MODEL_NAME": model,
        "API_URL": f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        "MAX_RETRIES": MAX_RETRIES,
        "ROW_LIMIT": ROW_LIMIT
    }
    return config

def call_gemini_api(text_content: str, config: Dict[str, Any]) -> str:
    """
    Calls the Gemini API, implementing exponential backoff and API limit checks.

    Args:
        text_content: The raw text extracted from the website.
        config: Dictionary containing API URL and MAX_RETRIES.

    Returns:
        A classification string ('yes', 'no') or an error status ('API_ERROR', 'API_LIMIT').
    """
    system_instruction = "You are an expert document classifier. Your only output must be 'yes' or 'no'. Do not include any explanations, punctuation, or other text."
    user_query = "Is this a menu with a list of food items? Answer yes or no."

    contents = [{"parts": [{"text": user_query}, {"text": f"Content to analyze: {text_content}"}]}]
    payload = {"contents": contents, "systemInstruction": {"parts": [{"text": system_instruction}]}}

    for attempt in range(config["MAX_RETRIES"]):
        try:
            response = requests.post(
                config["API_URL"],
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=30
            )
            
            # API Limit Check (External Error: 429 or Quota error in 400)
            if response.status_code == 429 or (response.status_code == 400 and "quota" in response.text.lower()):
                 print("\n!!! DAILY QUOTA REACHED OR RATE LIMITED. Returning API_LIMIT. !!!")
                 return "API_LIMIT"
            
            response.raise_for_status() # Raise HTTPError for other bad responses

            result = response.json()
            generated_text = result.get('candidates', [{}])[0]\
                                 .get('content', {})\
                                 .get('parts', [{}])[0]\
                                 .get('text', '')

            return generated_text.strip().lower() if generated_text else "NO_RESPONSE"

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{config['MAX_RETRIES']} failed. Error: {e}")
            if attempt < config["MAX_RETRIES"] - 1:
                wait_time = 2 ** attempt
                print(f"Waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                return "API_ERROR"
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return "GENERAL_ERROR"

    return "API_ERROR"

def load_data(input_filename: str, output_filename: str) -> Optional[pd.DataFrame]:
    """
    Handles file loading, resuming from the output file if it exists (checkpointing),
    and performs pre-analysis data cleaning to avoid unnecessary API calls.
    Includes a step to reset incorrectly labeled 'NO_CONTENT' rows back to 'PENDING'.
    Now also marks rows containing "ERROR" in raw_text as NO_CONTENT.
    """
    file_to_load = output_filename if os.path.exists(output_filename) else input_filename

    if not os.path.exists(file_to_load):
        if not os.path.exists(input_filename):
            print(f"Error: Neither the expected input file '{input_filename}' nor the checkpoint file '{output_filename}' was found.")
            return None
        file_to_load = input_filename

    print(f"Loading data from file: {file_to_load}...")
    try:
        df = pd.read_csv(file_to_load)
        if 'raw_text' not in df.columns:
            print("Error: The CSV must contain a column named 'raw_text'.")
            return None
        
        # Initialize 'is_menu' column if it doesn't exist
        if 'is_menu' not in df.columns:
            df['is_menu'] = 'PENDING'
            
        # --- Pre-analysis Data Quality Check (RE-INTRODUCING ERROR CHECK) ---
        
        # Ensure raw_text column is treated as strings for cleaning/checking
        df['raw_text'] = df['raw_text'].astype(str).fillna('')
        
        # 1. Define masks for content we want to skip (Empty/Missing OR Error Text)
        
        # Mask A: Check for empty string (after strip) or the literal 'nan' string
        empty_text_mask = (df['raw_text'].str.strip() == '') | (df['raw_text'].str.lower() == 'nan')
        
        # Mask B: FIX APPLIED HERE! 
        # Check for the *specific system error pattern* 'ERROR: ' or 'Client Error' (case-insensitive) 
        # This prevents organic text mentions of the word 'error' from triggering a skip.
        # Uses regex r'ERROR:\s' to ensure a space follows 'ERROR:'.
        error_text_mask = df['raw_text'].str.contains(r'ERROR:\s|Client Error', case=False, na=False)
        
        # Combined mask for rows that should be skipped (either empty or containing an error)
        rows_to_be_skipped = empty_text_mask | error_text_mask
        
        # 2. **Recovery Step:** If a row was previously marked NO_CONTENT but now has substantial, non-error content, reset to PENDING.
        # We only reset if the current row ISN'T flagged for skipping (i.e., it has good text).
        rows_to_reset = (df['is_menu'] == 'NO_CONTENT') & (~rows_to_be_skipped)
        df.loc[rows_to_reset, 'is_menu'] = 'PENDING'
        
        # 3. Standard Step: Mark skipped rows as 'NO_CONTENT' if they were 'PENDING' 
        df.loc[rows_to_be_skipped & (df['is_menu'] == 'PENDING'), 'is_menu'] = 'NO_CONTENT'
        
        # Log how many rows were marked/reset
        reset_count = rows_to_reset.sum()
        if reset_count > 0:
            print(f"Pre-analysis check: Reset {reset_count} rows from 'NO_CONTENT' back to 'PENDING' (data recovery).")
            
        marked_count = rows_to_be_skipped.sum()
        if marked_count > 0:
            print(f"Pre-analysis check: Marked {marked_count} rows as 'NO_CONTENT' (due to missing data or ERROR text).")
        
        # --- End Data Quality Check ---
            
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def run_analysis_loop(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, bool]:
    """
    Iterates over pending rows (PENDING, API_ERROR, API_LIMIT), calls the API, 
    and manages the internal row limit.
    
    Returns:
        A tuple containing the updated DataFrame and a boolean indicating if the API limit was hit.
    """
    # The filter now implicitly skips rows marked as 'NO_CONTENT' by load_data
    rows_to_process = df[df['is_menu'].isin(['PENDING', 'API_ERROR', 'API_LIMIT'])]
    total_rows = len(df)
    rows_processed_count = 0
    
    print(f"Rows identified for processing (PENDING/ERROR/LIMIT): {len(rows_to_process)}")
    if config["ROW_LIMIT"] is not None:
        print(f"Processing limit set for this run: {config['ROW_LIMIT']} rows.")

    api_limit_hit = False

    for index in rows_to_process.index:
        # Internal Row Limit Check
        if config["ROW_LIMIT"] is not None and rows_processed_count >= config["ROW_LIMIT"]:
            print(f"\n--- Reached configured maximum limit of {config['ROW_LIMIT']} API calls. Stopping. ---")
            break

        row = df.loc[index]
        
        # Logging before API call
        current_status = row['is_menu']
        status_message = f"Status: {current_status}. Analyzing..." if current_status != 'PENDING' else "Analyzing..."
        print(f"[{index + 1}/{total_rows}] {status_message} Text for '{row['name']}'...")
        
        # API Call
        result = call_gemini_api(row['raw_text'], config)

        # Update DataFrame
        df.at[index, 'is_menu'] = result.upper()
        rows_processed_count += 1
        
        print(f" -> Result: {result.upper()}")

        # Check for external API_LIMIT error
        if result.upper() == 'API_LIMIT':
            print("\n!!! Daily quota reached. Saving progress and exiting. Run again tomorrow. !!!")
            api_limit_hit = True
            break 

        # Polite delay
        time.sleep(2.0)
        
    return df, api_limit_hit


def save_data(df: pd.DataFrame, output_filename: str):
    """
    Saves the processed DataFrame to the specified output file (final checkpoint).
    """
    try:
        df.to_csv(output_filename, index=False)
        print("-" * 50)
        print(f"Processing complete for this run. Results saved to '{output_filename}'")
    except Exception as e:
        print(f"Error saving CSV: {e}")


def main():
    """
    Main function to orchestrate the entire data analysis workflow.
    """
    # Load environment variables from .env file
    load_dotenv(find_dotenv())
    
    # Attempt to load the API Key from the environment variable GOOGLE_API_KEY
    # Falls back to an empty string if not found, allowing the environment to inject it if needed.
    api_key_from_env = os.getenv("GOOGLE_API_KEY")
    final_api_key = api_key_from_env if api_key_from_env else ""
    
    # 1. Setup Configuration
    config = setup_config(TARGET_CITY, final_api_key, MODEL_NAME)

    if not config["API_KEY"]:
        print("Warning: API_KEY is empty. The script relies on the runtime environment to provide the key.")

    # 2. Load Data (Handles Resuming and Pre-filtering)
    # The fix ensures rows that were incorrectly marked NO_CONTENT will be reset to PENDING here.
    df = load_data(config["INPUT_FILE"], config["OUTPUT_FILE"])
    
    if df is None:
        return

    # 3. Run Analysis Loop (Handles API Calls and Internal Limits)
    df_updated, api_limit_hit = run_analysis_loop(df, config)

    # 4. Save Data (Final Checkpoint)
    save_data(df_updated, config["OUTPUT_FILE"])

# --- Main execution block ---
if __name__ == "__main__":
    main()