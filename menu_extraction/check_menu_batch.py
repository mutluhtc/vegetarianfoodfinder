import pandas as pd
import requests
import time
import json
import os
import re
import sys
from typing import Optional, Tuple, Dict, Any, List
# Assuming you use python-dotenv for API key management
from dotenv import load_dotenv, find_dotenv

# --- Configuration Variables ---
TARGET_CITY = "san_jose" 
MODEL_NAME = "gemini-2.5-flash" 
MAX_RETRIES = 5
# CRITICAL FIX: BATCH SIZE REDUCED TO 1 TO PREVENT PAYLOAD SIZE ERRORS
BATCH_SIZE = 75 
# Set to None to process the entire file
ROW_LIMIT = 750
# Set to False for production
DEBUG_PAYLOAD = False
# -----------------------------

def setup_config(city: str, api_key: str, model: str) -> Dict[str, Any]:
    """Centralizes configuration and dynamically generates file paths and API URL."""
    config = {
        "TARGET_CITY": city,
        "INPUT_FILE": f"{city}_restaurants_test_with_rawtext.csv",
        "OUTPUT_FILE": f"{city}_restaurants_with_menu_analysis.csv",
        "API_KEY": api_key,
        "MODEL_NAME": model,
        "API_URL": f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        "MAX_RETRIES": MAX_RETRIES,
        "ROW_LIMIT": ROW_LIMIT,
        "BATCH_SIZE": BATCH_SIZE,
        "DEBUG_PAYLOAD": DEBUG_PAYLOAD
    }
    return config

def call_gemini_batch_api(batch_data: List[pd.Series], config: Dict[str, Any]) -> Tuple[Dict[str, str], str]:
    """
    Calls the Gemini API to classify a batch of text inputs in a single call.
    Uses 'generationConfig' for structured output.
    """
    
    # 1. Prepare JSON Schema for Structured Output
    schema = {
        "type": "array",
        "description": "A list of classification results for the provided restaurants.",
        "items": {
            "type": "object",
            "properties": {
                "index": {"type": "string", "description": "The unique index of the restaurant from the input data (e.g., 1, 2, 3...)"},
                "name": {"type": "string", "description": "The name of the restaurant."},
                "is_menu": {"type": "string", "description": "Classification: 'yes' if it is a food menu, 'no' otherwise."}
            },
            "required": ["index", "name", "is_menu"]
        }
    }
    
    # 2. Prepare the User Query (Input Data) with Aggressive Sanitization
    input_list = []
    
    for row in batch_data:
        
        raw_text_content = str(row['raw_text'])
        
        # --- AGGRESSIVE SANITIZATION STEP ---
        sanitized_text = raw_text_content.encode('ascii', 'ignore').decode('ascii')
        sanitized_text = sanitized_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        sanitized_text = sanitized_text.replace('"', "'") 
        if len(sanitized_text) > 100000: 
             sanitized_text = sanitized_text[:100000]
        # -----------------------------------
        
        input_list.append(f"--- Restaurant Index {row.name} / Name: {row['name']} ---\n{sanitized_text}")
        
    full_text_input = "\n\n".join(input_list)
    
    system_instruction = "You are an expert document classifier. Your task is to analyze the text provided for each restaurant and classify it as 'yes' (it is a food menu) or 'no' (it is not a food menu). You MUST respond ONLY with a single JSON array that conforms to the provided schema. Do not include any explanations, external text, or punctuation outside of the JSON structure."

    user_query = f"""
    Analyze the {len(batch_data)} restaurant contents provided below. For each restaurant, determine if the content contains a food menu.
    
    Respond only with the JSON object.
    
    Content to analyze:
    {full_text_input}
    """

    contents = [{"parts": [{"text": user_query}]}]
    
    # --- FIX APPLIED HERE: RENAMED 'config' to 'generationConfig' ---
    payload = {
        "contents": contents,
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema
        },
        "systemInstruction": {"parts": [{"text": system_instruction}]}
    }
    # -------------------------------------------------------------

    results_map = {}
    status = "API_ERROR"

    payload_str = json.dumps(payload, indent=2 if config["DEBUG_PAYLOAD"] else None)

    for attempt in range(config["MAX_RETRIES"]):
        try:
            if config["DEBUG_PAYLOAD"]:
                 print("\n--- DEBUG: PAYLOAD SENT TO API ---")
                 print(payload_str)
                 print("-----------------------------------\n")

            response = requests.post(
                config["API_URL"],
                headers={'Content-Type': 'application/json'},
                data=payload_str,
                timeout=90 
            )
            
            # --- ENHANCED ERROR TESTING: Check 400 specifically ---
            if response.status_code == 400:
                print("\n!!! 400 BAD REQUEST ERROR !!!")
                error_details = response.text
                print(f"Error Details: {error_details}")
                # If the error is not a quota error, it's a hard payload/key error.
                if "quota" not in error_details.lower():
                    status = "PAYLOAD_ERROR"
                    return {}, status
            # ----------------------------------------------------

            # API Limit Check (429 or quota-related 400)
            if response.status_code == 429 or (response.status_code == 400 and "quota" in response.text.lower()):
                print("\n!!! DAILY QUOTA REACHED OR RATE LIMITED. !!!")
                return {}, "API_LIMIT"
            
            response.raise_for_status()

            # 3. Parse and Validate the JSON Output
            result = response.json()
            generated_text = result.get('candidates', [{}])[0]\
                                 .get('content', {})\
                                 .get('parts', [{}])[0]\
                                 .get('text', '')
            
            if generated_text:
                json_results = json.loads(generated_text)
                
                for item in json_results:
                    original_index = item.get('index') 
                    classification = item.get('is_menu', 'NO_RESPONSE').upper()
                    
                    if original_index is not None:
                        # Index is stored as an integer, so map back to integer index
                        results_map[int(original_index)] = classification
                
                status = "SUCCESS"
                
                if len(results_map) < len(batch_data):
                    status = "PARTIAL_SUCCESS"
                    print(f"Warning: Received {len(results_map)} results, but expected {len(batch_data)}. Some may be missing.")
                
                return results_map, status

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{config['MAX_RETRIES']} failed. Error: {e}")
            if attempt < config["MAX_RETRIES"] - 1:
                wait_time = 2 ** attempt
                print(f"Waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                status = "API_ERROR"
        except json.JSONDecodeError:
            print(f"Attempt {attempt + 1}/{config['MAX_RETRIES']} failed: Invalid JSON response from model.")
            if attempt < config["MAX_RETRIES"] - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                status = "API_ERROR"
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {}, "GENERAL_ERROR"

    return {}, status 

def load_data(input_filename: str, output_filename: str) -> Optional[pd.DataFrame]:
    """Handles file loading, resuming, and pre-analysis data cleaning."""
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
        
        if 'is_menu' not in df.columns:
            df['is_menu'] = 'PENDING'
            
        df['raw_text'] = df['raw_text'].astype(str).fillna('')
        
        # Mask A: Check for empty string or 'nan' string
        empty_text_mask = (df['raw_text'].str.strip() == '') | (df['raw_text'].str.lower() == 'nan')
        
        # Mask B: Check for the specific system error pattern
        error_text_mask = df['raw_text'].str.contains(r'ERROR:\s|Client Error', case=False, na=False)
        
        rows_to_be_skipped = empty_text_mask | error_text_mask
        
        # 2. Recovery Step: Reset rows that were NO_CONTENT but now have good data
        rows_to_reset = (df['is_menu'] == 'NO_CONTENT') & (~rows_to_be_skipped)
        df.loc[rows_to_reset, 'is_menu'] = 'PENDING'
        
        # 3. Standard Step: Mark skipped rows as 'NO_CONTENT' if they were 'PENDING' 
        df.loc[rows_to_be_skipped & (df['is_menu'] == 'PENDING'), 'is_menu'] = 'NO_CONTENT'
        
        reset_count = rows_to_reset.sum()
        if reset_count > 0:
            print(f"Pre-analysis check: Reset {reset_count} rows from 'NO_CONTENT' back to 'PENDING' (data recovery).")
            
        marked_count = rows_to_be_skipped.sum()
        if marked_count > 0:
            print(f"Pre-analysis check: Marked {marked_count} rows as 'NO_CONTENT' (due to missing data or ERROR text).")
            
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def run_analysis_loop(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, bool]:
    """Iterates over pending rows in batches, calling the batch API."""
    rows_to_process = df[df['is_menu'].isin(['PENDING', 'API_ERROR', 'API_LIMIT', 'PAYLOAD_ERROR'])]
    rows_processed_count = 0
    
    print(f"Rows identified for processing (PENDING/ERROR/LIMIT): {len(rows_to_process)}")
    
    # Correctly display the limit based on the config variable type
    limit_display = f"{config['ROW_LIMIT']} rows" if config['ROW_LIMIT'] is not None else "All"
    print(f"Processing limit set for this run: {limit_display} (in batches of {config['BATCH_SIZE']}).")

    api_limit_hit = False
    
    rows_to_process_indices = rows_to_process.index.tolist()
    
    for i in range(0, len(rows_to_process_indices), config["BATCH_SIZE"]):
        
        # Check if we've hit the overall row limit for the run
        if config["ROW_LIMIT"] is not None and rows_processed_count >= config["ROW_LIMIT"]:
            print(f"\n--- Reached configured maximum limit of {config['ROW_LIMIT']} API calls. Stopping. ---")
            break
            
        batch_indices = rows_to_process_indices[i : i + config["BATCH_SIZE"]]
        batch_data = [df.loc[idx] for idx in batch_indices]
        current_batch_size = len(batch_data)
        
        print(f"\n--- Processing Batch {i // config['BATCH_SIZE'] + 1} ({current_batch_size} items) ---")
        
        results_map, result_status = call_gemini_batch_api(batch_data, config)
        
        # Only update the DataFrame if the status is not a critical error
        if result_status in ('SUCCESS', 'PARTIAL_SUCCESS'):
            for index, classification in results_map.items():
                df.at[index, 'is_menu'] = classification
                print(f" -> Index {index}: Result: {classification}") 
                rows_processed_count += 1
        
        # Set the error status on the DataFrame for the current batch if the call failed
        if result_status not in ('SUCCESS', 'PARTIAL_SUCCESS'):
             for index in batch_indices:
                 df.at[index, 'is_menu'] = result_status # Updates PENDING to API_ERROR, PAYLOAD_ERROR, or API_LIMIT
        
        if result_status in ('API_LIMIT', 'PAYLOAD_ERROR', 'GENERAL_ERROR'):
            if result_status == 'API_LIMIT':
                print("\n!!! Daily quota reached. Saving progress and exiting. Run again tomorrow. !!!")
            elif result_status == 'PAYLOAD_ERROR':
                 print("\n!!! Payload Error encountered. Check debug logs. Saving progress and exiting. !!!")
            elif result_status == 'GENERAL_ERROR':
                 print("\n!!! General Error encountered. Saving progress and exiting. !!!")
            
            api_limit_hit = True 
            break 
            
        time.sleep(2.0)
        
    return df, api_limit_hit


def save_data(df: pd.DataFrame, output_filename: str):
    """Saves the processed DataFrame to the specified output file (final checkpoint)."""
    try:
        df.to_csv(output_filename, index=False)
        print("-" * 50)
        print(f"Processing complete for this run. Results saved to '{output_filename}'")
    except Exception as e:
        print(f"Error saving CSV: {e}")


def main():
    """Main function to orchestrate the entire data analysis workflow."""
    # Load environment variables from .env file
    load_dotenv(find_dotenv())
    
    api_key_from_env = os.getenv("GOOGLE_API_KEY")
    final_api_key = api_key_from_env if api_key_from_env else ""
    
    # 1. Setup Configuration
    config = setup_config(TARGET_CITY, final_api_key, MODEL_NAME)

    if not config["API_KEY"]:
        print("Error: API_KEY is empty. Ensure GOOGLE_API_KEY is set in your environment or .env file.")
        sys.exit(1)

    # 2. Load Data (Handles Resuming and Pre-filtering)
    df = load_data(config["INPUT_FILE"], config["OUTPUT_FILE"])
    
    if df is None:
        return

    # 3. Run Analysis Loop (Handles API Calls and Internal Limits)
    df_updated, api_limit_hit = run_analysis_loop(df, config)

    # 4. Save Data (Final Checkpoint)
    save_data(df_updated, config["OUTPUT_FILE"])

if __name__ == "__main__":
    main()