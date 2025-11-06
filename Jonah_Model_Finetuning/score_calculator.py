import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import os

FOLDER_PATH = "./final_model_dir" 

def calculate_metrics_by_source(filepath, positive_label='Vegetarian'):
    """
    Loads prediction data from a single CSV file, calculates Accuracy and F1 Score 
    for different data sources ('Synthetic' and 'Yelp') found within that file, 
    and prints the results. Returns a dictionary of calculated metrics for aggregation.

    Args:
        filepath (str): The full path to the CSV file containing predictions.
        positive_label (str): The label to consider as the positive class
                              for F1 score calculation (e.g., 'Vegetarian').
    """
    filename = os.path.basename(filepath) # Extract filename for cleaner printing

    try:
        # Load the CSV file
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return {}
    except pd.errors.EmptyDataError:
        print(f"Error: File is empty at {filepath}")
        return {}
    
    # Ensure necessary columns exist
    required_cols = ['Actual Diet', 'Predicted Diet', 'Source']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV file '{filename}' must contain the columns: {', '.join(required_cols)}")
        return {}

    # Get the unique data sources
    sources = df['Source'].unique()
    results = {} # Dictionary to store results for this file

    print("\n" + "=" * 50)
    print(f"--- Processing File: {filename} ---")
    print(f"Positive Label for F1 Score: '{positive_label}'")
    print("=" * 50)

    for source in sources:
        # Filter the DataFrame for the current source
        df_subset = df[df['Source'] == source].copy()
        
        # Check if subset is empty
        if df_subset.empty:
            print(f"[{filename}] No data found for source: {source}")
            continue
            
        # Extract true labels and predictions
        y_true = df_subset['Actual Diet']
        y_pred = df_subset['Predicted Diet']
        
        # Calculate Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate F1 Score (using binary averaging and specifying the positive label)
        # Handle cases where the positive label might not exist in a subset
        f1 = 0.0
        try:
            f1 = f1_score(y_true, y_pred, pos_label=positive_label)
        except ValueError as e:
            if 'pos_label' in str(e) and positive_label not in y_true.unique():
                # F1 remains 0.0
                print(f"[{filename} - {source}] Warning: Positive label '{positive_label}' missing. F1 set to 0.0.")
            else:
                # If a different error occurs, fall back to weighted average (multiclass)
                f1 = f1_score(y_true, y_pred, average='weighted')
                print(f"[{filename} - {source}] Warning: Could not calculate binary F1. Using 'weighted' average instead.")


        # Store and Output results
        results[source] = {'accuracy': accuracy, 'f1': f1}

        print(f"Metrics for Data Source in {filename}: {source}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print("-" * 50)
        
    return results

# --- Main Execution Block ---

# Define the folder containing the CSV prediction files
# We use '.' to refer to the current directory where the script is run.
# Change this path if your files are in a different folder (e.g., './data/predictions')

# Initialize structure to hold all metrics for averaging
all_metrics = {}

# Helper function to initialize the metric storage structure
def initialize_metric_storage(source):
    if source not in all_metrics:
        all_metrics[source] = {'accuracy': [], 'f1': []}

# Run the calculation for all CSV files in the folder
if os.path.isdir(FOLDER_PATH):
    print(f"Scanning folder: {os.path.abspath(FOLDER_PATH)}")
    
    # Get all files in the directory and filter for CSV files
    csv_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in the folder: {FOLDER_PATH}")
    else:
        for filename in csv_files:
            full_path = os.path.join(FOLDER_PATH, filename)
            # Setting 'Vegetarian' as the positive class for F1
            per_file_metrics = calculate_metrics_by_source(full_path, positive_label='Vegetarian')
            
            # Aggregate metrics
            if per_file_metrics:
                for source, metrics in per_file_metrics.items():
                    initialize_metric_storage(source)
                    all_metrics[source]['accuracy'].append(metrics['accuracy'])
                    all_metrics[source]['f1'].append(metrics['f1'])
                    
        # --- Calculate and print Averages ---
        print("\n" + "=" * 50)
        print("--- SUMMARY: AVERAGED METRICS ACROSS ALL FILES ---")
        print("=" * 50)
        
        if not all_metrics:
             print("No valid metrics were collected for any source.")
        else:
            for source, metrics in all_metrics.items():
                num_files = len(metrics['accuracy'])
                if num_files > 0:
                    avg_accuracy = sum(metrics['accuracy']) / num_files
                    avg_f1 = sum(metrics['f1']) / num_files
                    
                    print(f"Average Metrics for Source: {source} ({num_files} files processed)")
                    print(f"  Average Accuracy: {avg_accuracy:.4f}")
                    print(f"  Average F1 Score: {avg_f1:.4f}")
                    print("-" * 50)
                else:
                     print(f"No metrics collected for source: {source}")

else:
    print(f"Error: The path '{FOLDER_PATH}' is not a valid directory.")
