import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import os
import json
from tqdm import tqdm

# --- 1. Constants (ADJUSTED for Hugging Face) ---
# The Hugging Face repo ID replaces the local path
HF_REPO_ID = 'ytarasov/diet-classifier-bert-v1' 

# The base model name is used for tokenizer initialization
MODEL_NAME = 'bert-base-uncased'
# NOTE: We use HF_REPO_ID now, so FINAL_MODEL_PATH is deprecated but kept for clarity
FINAL_MODEL_PATH = HF_REPO_ID 

# File paths - EDIT THESE AS NEEDED
INPUT_JSON_FILE = "./structured_JSON_data_files/atlanta_restaurants_with_structured_menus.json"
OUTPUT_JSON_FILE = "./final_JSON/atlanta_restaurants_with_vegetarian_labels.json"

# Define label map and its inverse
ID_TO_LABEL = {0: 'Vegetarian', 1: 'Non-Vegetarian'}

# Define the common new tokens for BERT (MUST match the training script)
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

# --- 2. Model and Tokenizer Initialization (ADJUSTED) ---

def load_model_and_tokenizer(hf_repo_id):
    """
    Loads the fine-tuned model and tokenizer from Hugging Face Hub, 
    ensuring custom tokens are included.
    """
    
    print(f"Loading tokenizer from base model: {MODEL_NAME}")
    # Initialize tokenizer with the base model name
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    # Add custom tokens (this is crucial for tokenization to match training)
    # NOTE: The fine-tuned tokenizer pushed to the Hub already has these tokens, 
    # but explicitly adding them here ensures the vocabulary matches for models
    # where the tokenizer wasn't saved with the model path.
    tokenizer.add_tokens(NEW_TOKENS)

    print(f"Loading fine-tuned model from Hugging Face: {hf_repo_id}")
    # Load the fine-tuned model directly from the Hugging Face Hub
    # The `hf_repo_id` is passed directly to from_pretrained()
    model = BertForSequenceClassification.from_pretrained(hf_repo_id)

    return tokenizer, model

# --- 3. Prediction Function (UNCHANGED) ---

def predict_diet_label(dish_input_string, tokenizer, model):
    """
    Takes a raw input string (dish name and description) and predicts the diet label.
    """

    input_text = str(dish_input_string).strip()

    if not input_text:
        return "Error: Input string is empty."

    # Tokenize the input
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=128
    )

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=1).item()

    predicted_label = ID_TO_LABEL.get(predicted_id, "Unknown Label")

    return predicted_label

# --- 4. Process JSON File (UNCHANGED) ---

def process_json_file(input_file_path, output_file_path, tokenizer, model):
    """
    Process the JSON file and add vegetarian labels to all menu items.
    """
    
    print(f"Loading data from: {input_file_path}")
    with open(input_file_path, 'r', encoding='utf-8') as f:
        restaurants_data = json.load(f)
    
    print(f"Found {len(restaurants_data)} restaurants to process")
    
    total_menu_items = sum(len(restaurant.get('menu', [])) for restaurant in restaurants_data)
    print(f"Total menu items to classify: {total_menu_items}")
    
    print("\nProcessing restaurants and menu items...")
    
    for restaurant in tqdm(restaurants_data, desc="Restaurants", unit="restaurant"):
        if 'menu' in restaurant and restaurant['menu']:
            for menu_item in tqdm(restaurant['menu'], desc=f"  {restaurant.get('restaurant_name', 'Unknown')[:20]}...", 
                                 unit="item", leave=False):
                item_name = menu_item.get('item_name', '')
                item_description = menu_item.get('description', '')
                
                input_string = f"{item_name} - {item_description}"
                
                vegetarian_label = predict_diet_label(input_string, tokenizer, model)
                
                menu_item['vegetarian_label'] = vegetarian_label
    
    print("Saving results...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(restaurants_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Processing complete! Output saved to: {output_file_path}")
    
    # Print summary statistics
    total_items = 0
    vegetarian_count = 0
    
    for restaurant in restaurants_data:
        if 'menu' in restaurant and restaurant['menu']:
            for menu_item in restaurant['menu']:
                total_items += 1
                if menu_item.get('vegetarian_label') == 'Vegetarian':
                    vegetarian_count += 1
    
    print(f"\n--- Summary Statistics ---")
    print(f"Total menu items processed: {total_items}")
    print(f"Vegetarian items: {vegetarian_count} ({vegetarian_count/total_items*100:.1f}%)")
    print(f"Non-vegetarian items: {total_items - vegetarian_count} ({(total_items-vegetarian_count)/total_items*100:.1f}%)")

# --- 5. Main Execution ---

if __name__ == '__main__':
    print("=== Restaurant Menu Vegetarian Classifier ===")
    print(f"Input file: {INPUT_JSON_FILE}")
    print(f"Output file: {OUTPUT_JSON_FILE}")
    print()
    
    print("--- Loading Model and Tokenizer from Hugging Face Hub ---")
    
    try:
        # Pass the Hugging Face Repo ID
        tokenizer, model = load_model_and_tokenizer(HF_REPO_ID)
            
        # Process the JSON file
        process_json_file(INPUT_JSON_FILE, OUTPUT_JSON_FILE, tokenizer, model)
        
        # Test a few examples to show the results
        print("\n--- Sample Predictions ---")
        
        # Load the output file to show some examples
        with open(OUTPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        
        # Show a few examples from different restaurants
        sample_count = 0
        print("Displaying sample predictions...")
        for restaurant in processed_data:
            if 'menu' in restaurant and restaurant['menu']:
                restaurant_name = restaurant.get('restaurant_name', 'Unknown')
                print(f"\n🏪 Restaurant: {restaurant_name}")
                
                for i, menu_item in enumerate(restaurant['menu'][:2]): 
                    item_name = menu_item.get('item_name', '')
                    description = menu_item.get('description', '')
                    label = menu_item.get('vegetarian_label', 'Unknown')
                    
                    emoji = "🌱" if label == 'Vegetarian' else "🍖"
                    
                    print(f"  {emoji} {item_name}")
                    if description:
                        print(f"     Description: {description[:80]}..." if len(description) > 80 else f"     Description: {description}")
                    print(f"     Label: {label}")
                    print()
                    
                    sample_count += 1
                    if sample_count >= 6: 
                        break
                if sample_count >= 6:
                    break
        
        print("✅ All tasks completed successfully!")
        
    except Exception as e:
        print(f"\n💥 [CRITICAL ERROR] An unexpected error occurred: {e}")
        print("Please ensure your Hugging Face repository ID is correct and the model is public or you are logged in.")