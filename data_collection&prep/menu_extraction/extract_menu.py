import pandas as pd
import requests
import time
import json
import os
import re
import sys
from typing import Optional, Tuple, Dict, Any, List
from dotenv import load_dotenv, find_dotenv

# --- Configuration Variables ---
TARGET_CITY = "san_jose" 
MODEL_NAME = "gemini-2.5-flash" 
MAX_RETRIES = 5
# BATCH SIZE increased for throughput
BATCH_SIZE = 5 
# Set to None to process the entire file
ROW_LIMIT = 500
# Set to False for production
DEBUG_PAYLOAD = False
# -----------------------------

def setup_config(city: str, api_key: str, model: str) -> Dict[str, Any]:
    """Centralizes configuration and dynamically generates file paths and API URL."""
    config = {
        "TARGET_CITY": city,
        "INPUT_FILE": f"{city}_restaurants_with_menu_analysis.csv", 
        "OUTPUT_FILE": f"{city}_restaurants_with_structured_menus.json",
        "API_KEY": api_key,
        "MODEL_NAME": model,
        "API_URL": f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        "MAX_RETRIES": MAX_RETRIES,
        "ROW_LIMIT": ROW_LIMIT,
        "BATCH_SIZE": BATCH_SIZE,
        "DEBUG_PAYLOAD": DEBUG_PAYLOAD
    }
    return config

def parse_llm_menu_text(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parses the pipe-delimited text output from the LLM into a structured list of dictionaries.
    Handles price conversion and errors for a SINGLE menu block.
    """
    structured_menu = []
    
    # Regex to capture the three pipe-separated components
    item_regex = re.compile(r"^(.*?)\s*\|\s*(\S+)\s*\|\s*(.*)$", re.MULTILINE)
    
    # Clean and iterate over lines
    cleaned_text = raw_text.strip()
    
    for line in cleaned_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = item_regex.match(line)
        if not match:
            # Skip lines that don't match the required pipe format
            continue
            
        item_name, price_str, description = match.groups()
        
        # Attempt to clean and convert price to a number (float is safer than int)
        try:
            # Remove currency symbols ($) and strip whitespace
            clean_price_str = price_str.replace('$', '').strip()
            price = float(clean_price_str)
        except ValueError:
            # Skip item if the price cannot be reliably converted to a number
            continue
            
        structured_menu.append({
            "item_name": item_name.strip(),
            "price": price,
            "description": description.strip()
        })
        
    return structured_menu


def call_gemini_batch_extract_menu(batch_data: List[pd.Series], config: Dict[str, Any]) -> Tuple[Dict[str, str], str]:
    """
    Calls the Gemini API to extract raw menu data in a simple text format for a batch.
    Uses index markers to segment the input and parse the output.
    """
    
    # 1. Prepare the User Query (Input Data) with index markers
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
        
        # Use the index as the unique identifier in the prompt
        input_list.append(f"--- RESTAURANT START INDEX {row.name} ---\n{sanitized_text}")
        
    full_text_input = "\n\n".join(input_list)
    
    system_instruction = (
        "You are an expert menu extraction specialist. Your task is to analyze the text provided for multiple restaurants "
        "and extract a list of all menu items you can reliably find for each. "
        "You MUST output the result for each restaurant immediately preceded by the marker: --- MENU START INDEX [index] ---, "
        "where [index] matches the RESTAURANT START INDEX from the input. "
        "The menu items themselves MUST be on a new line, strictly using the format: [ITEM NAME] | [PRICE] | [DESCRIPTION]. "
        "Do not include any headers, preambles, or explanations outside of the required markers."
    )

    user_query = f"""
    Analyze the {len(batch_data)} restaurant menu contents provided below. For each one, extract all menu items and their details.
    
    Strictly follow the output format described in the system instructions.
    
    Content to analyze:
    {full_text_input}
    """

    contents = [{"parts": [{"text": user_query}]}]
    
    # No structured JSON requirements
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_instruction}]}
    }

    results_map = {}
    status = "API_ERROR"
    
    payload_str = json.dumps(payload, indent=2 if config["DEBUG_PAYLOAD"] else None)

    for attempt in range(config["MAX_RETRIES"]):
        try:
            response = requests.post(
                config["API_URL"],
                headers={'Content-Type': 'application/json'},
                data=payload_str,
                timeout=180 
            )
            
            if response.status_code == 400:
                error_details = response.text
                if "quota" not in error_details.lower():
                    status = "PAYLOAD_ERROR"
                    return {}, status

            if response.status_code == 429 or (response.status_code == 400 and "quota" in response.text.lower()):
                status = "API_LIMIT"
                return {}, "API_LIMIT"
            
            response.raise_for_status()

            # 3. Parse the Raw Text Output
            result = response.json()
            generated_text = result.get('candidates', [{}])[0]\
                                 .get('content', {})\
                                 .get('parts', [{}])[0]\
                                 .get('text', '')
            
            if generated_text:
                # Regex to find all blocks starting with the required marker
                menu_block_pattern = re.compile(
                    r"--- MENU START INDEX (\d+) ---\s*\n(.*?)(?=--- MENU START INDEX|\Z)", 
                    re.DOTALL
                )
                
                matches = menu_block_pattern.findall(generated_text)
                
                for index_str, menu_text in matches:
                    original_index = int(index_str)
                    
                    # 4. Use Python to parse and structure the menu list for this block
                    structured_menu_list = parse_llm_menu_text(menu_text)
                    
                    # Store the structured Python list as a JSON string for the DataFrame checkpoint
                    results_map[original_index] = json.dumps(structured_menu_list)
                
                if len(results_map) == len(batch_data):
                    status = "SUCCESS"
                elif len(results_map) > 0:
                    status = "PARTIAL_SUCCESS"
                else:
                    status = "API_ERROR"
                    
                return results_map, status

        except requests.exceptions.RequestException as e:
            if attempt == config["MAX_RETRIES"] - 1:
                status = "API_ERROR"
        except Exception as e:
            status = "GENERAL_ERROR"
            return {}, "GENERAL_ERROR"
            
        if attempt < config["MAX_RETRIES"] - 1:
            wait_time = 2 ** attempt
            time.sleep(wait_time)
            
    return {}, status 

def load_data(input_filename: str) -> Optional[pd.DataFrame]:
    """Handles file loading, filtering for 'is_menu'='YES', and prepares the new column."""
    if not os.path.exists(input_filename):
        print(f"Error: The expected input file '{input_filename}' was not found.")
        return None

    print(f"Loading data from file: {input_filename}...")
    try:
        df = pd.read_csv(input_filename)
        
        if 'raw_text' not in df.columns or 'is_menu' not in df.columns:
            print("Error: The CSV must contain columns 'raw_text' and 'is_menu'.")
            return None
        
        # New column for structured menu data (will hold the JSON string of the menu items)
        # This checks if the column was persisted from a previous run.
        if 'structured_menu_json' not in df.columns:
            print("Initial run detected: Creating 'structured_menu_json' column with PENDING status.")
            df['structured_menu_json'] = 'PENDING'
            
        df['raw_text'] = df['raw_text'].astype(str).fillna('')
        
        # --- Filtering and Cleanup ---
        
        # 1. Filter: Only process rows classified as 'YES'
        df_filtered = df[df['is_menu'] == 'YES'].copy()
        
        # 2. Mark bad raw_text as NO_CONTENT (for 'YES' menus that we intend to process)
        empty_text_mask = (df_filtered['raw_text'].str.strip() == '') | (df_filtered['raw_text'].str.lower() == 'nan')
        error_text_mask = df_filtered['raw_text'].str.contains(r'ERROR:\s|Client Error', case=False, na=False)
        rows_to_be_skipped = empty_text_mask | error_text_mask
        
        # 3. Mark skipped rows as 'NO_CONTENT'
        df_filtered.loc[rows_to_be_skipped & (df_filtered['structured_menu_json'] == 'PENDING'), 'structured_menu_json'] = 'NO_CONTENT'
        
        print(f"Loaded {len(df)} total rows. Filtered down to {len(df_filtered)} rows where 'is_menu' is 'YES'.")
        print(f"Marked {rows_to_be_skipped.sum()} filtered rows as 'NO_CONTENT' due to bad raw_text.")

        return df_filtered
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def run_analysis_loop(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, bool]:
    """Iterates over pending rows in batches, calling the menu extraction API."""
    rows_to_process = df[df['structured_menu_json'].isin(['PENDING', 'API_ERROR', 'API_LIMIT', 'PAYLOAD_ERROR'])]
    rows_processed_count = 0
    
    print(f"Rows identified for processing (PENDING/ERROR/LIMIT): {len(rows_to_process)}")
    
    limit_display = f"{config['ROW_LIMIT']} rows" if config['ROW_LIMIT'] is not None else "All"
    print(f"Processing limit set for this run: {limit_display} (in batches of {config['BATCH_SIZE']}).")

    api_limit_hit = False
    
    rows_to_process_indices = rows_to_process.index.tolist()
    
    for i in range(0, len(rows_to_process_indices), config["BATCH_SIZE"]):
        
        if config["ROW_LIMIT"] is not None and rows_processed_count >= config["ROW_LIMIT"]:
            print(f"\n--- Reached configured maximum limit of {config['ROW_LIMIT']} API calls. Stopping. ---")
            break
            
        batch_indices = rows_to_process_indices[i : i + config["BATCH_SIZE"]]
        batch_data = [df.loc[idx] for idx in batch_indices]
        current_batch_size = len(batch_data)
        
        print(f"\n--- Processing Batch {i // config['BATCH_SIZE'] + 1} ({current_batch_size} items) ---")
        
        results_map, result_status = call_gemini_batch_extract_menu(batch_data, config)
        
        if result_status in ('SUCCESS', 'PARTIAL_SUCCESS'):
            for index, menu_json_str in results_map.items():
                df.at[index, 'structured_menu_json'] = menu_json_str 
                print(f" -> Index {index}: Result: Menu JSON extracted by Python.") 
                rows_processed_count += 1
        
        if result_status not in ('SUCCESS', 'PARTIAL_SUCCESS'):
             for index in batch_indices:
                 df.at[index, 'structured_menu_json'] = result_status 
        
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
    """Saves the processed DataFrame to the specified output file as a JSON array."""
    final_output_list = []
    
    successful_df = df[~df['structured_menu_json'].isin(['PENDING', 'API_ERROR', 'API_LIMIT', 'PAYLOAD_ERROR', 'NO_CONTENT'])]
    
    print("-" * 50)
    print(f"Aggregating {len(successful_df)} successfully processed menus into final JSON structure...")
    
    columns_to_map = {
        'name': 'restaurant_name',
        'city': 'location',
        'rating': 'rating',
        'review_count': 'review_count',
        'categories': 'categories',
        'price': 'price',
        'address': 'address',
        'latitude': 'latitude',
        'longitude': 'longitude',
        'url': 'url',
        'website_url': 'website_url'
    }
    
    valid_cols_to_map = {csv_col: json_key for csv_col, json_key in columns_to_map.items() if csv_col in df.columns}

    for index, row in successful_df.iterrows():
        try:
            entry = {}
            
            # 1. Extract and map the core metadata
            for csv_col, json_key in valid_cols_to_map.items():
                value = row[csv_col]
                entry[json_key] = None if pd.isna(value) else value
            
            # 2. Parse and add the menu list
            menu_list = json.loads(row['structured_menu_json'])
            entry['menu'] = menu_list
            
            final_output_list.append(entry)

        except json.JSONDecodeError:
            print(f"Skipping index {index}: JSONDecodeError on structured_menu_json content.")
        except Exception as e:
            print(f"Skipping index {index} due to unexpected error: {e}")
            
    # Save the final list to a JSON file
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(final_output_list, f, indent=2, ensure_ascii=False)
            
        print(f"Results saved to '{output_filename}'")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error saving JSON: {e}")


def main():
    """Main function to orchestrate the entire structured menu extraction workflow."""
    load_dotenv(find_dotenv())
    
    api_key_from_env = os.getenv("GOOGLE_API_KEY")
    final_api_key = api_key_from_env if api_key_from_env else ""
    
    # 1. Setup Configuration
    config = setup_config(TARGET_CITY, final_api_key, MODEL_NAME)

    if not config["API_KEY"]:
        print("Error: API_KEY is empty. Ensure GOOGLE_API_KEY is set in your environment or .env file.")
        sys.exit(1)

    # 2. Load Data (Filters for is_menu='YES')
    df = load_data(config["INPUT_FILE"])
    
    if df is None:
        return

    # 3. Run Analysis Loop 
    df_updated, api_limit_hit = run_analysis_loop(df, config)

    # --- CRITICAL FIX: Save the updated status column back to the input CSV ---
    try:
        # This overwrites the input file, adding the 'structured_menu_json' column and its state.
        df_updated.to_csv(config["INPUT_FILE"], index=False)
        print(f"\n[CHECKPOINT] Updated status saved to '{config['INPUT_FILE']}' for resumption.")
    except Exception as e:
        print(f"\n[ERROR] Could not save updated status to CSV: {e}")
    # -------------------------------------------------------------------------

    # 4. Save Data (Final JSON Output)
    save_data(df_updated, config["OUTPUT_FILE"])

if __name__ == "__main__":
    main()