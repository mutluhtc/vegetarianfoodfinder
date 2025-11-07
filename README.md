# 🥗 VeggieFoodFinder

**An AI-powered restaurant discovery app for vegetarian and vegetarian-friendly dishes**

---

## 🌟 Overview

**VeggieFoodFinder** simplifies the process of finding vegetarian and vegetarian-friendly dining options.  
Instead of manually browsing menus or checking multiple websites, users can simply enter a ZIP code to explore nearby restaurants with clearly labeled vegetarian and non-vegetarian dishes — customizable by cuisine, dietary preference, and distance.

---

## 🚀 Problem & Motivation

Existing platforms like **Yelp** and **Google Maps** offer basic “vegetarian-friendly” filters but fail to display **specific dishes** or detailed menu content.

**VeggieFoodFinder** addresses this gap by combining:
- **Web scraping** for menu data  
- **LLM-based information extraction**  
- **Machine learning classification** for dish labeling  

The result is a unified, structured menu dataset visualized through an **interactive map interface**.

---

## 🧠 How It Works

1. **Input ZIP code** → The system retrieves nearby restaurants using the **Yelp Fusion API**.  
2. **Scraping & Extraction** → Menus (HTML/PDF) are parsed and validated using **LLM-based filters**.  
3. **Data Structuring** → **Gemini 2.5 Flash** extracts dish names, descriptions, and prices into JSON format.  
4. **Classification** → A fine-tuned **BERT** model labels each dish as *vegetarian* or *non-vegetarian*.  
5. **Visualization** → Results are displayed in an interactive interface with filtering options.

---

## 🚀 Overview

This project aims to automate the process of finding vegetarian-friendly restaurants.  
The system:
1. Scrapes restaurant and menu data.
2. Prepares and labels the data for machine learning.
3. Trains a model to classify whether a restaurant offers vegetarian options.
4. Deploys a Streamlit app to visualize and interact with the results.

---

## 🧩 Pipeline Overview

### 1. **Data Collection**
**Folder:** `data_prep/yelp_data_scraping/`

## 🧭 Data Collection & Scraping

### **Step 1: Collecting Restaurant Data**
- Used the **Yelp Fusion API** to gather restaurant details — names, cuisines, ratings, and website links.  
- Focused on **29 major U.S. cities**, collecting data for roughly **1,200 restaurants per city**.  
- The **API’s free-tier rate limits** shaped the overall dataset size and coverage.

### **Step 2: Retrieving Menus**
- Restaurant websites varied widely — some hosted menus as **HTML pages**, others as **PDFs**, and a few only as **images**.  
- Built a **custom scraping system** capable of handling both **HTML and PDF** formats.  
- Extracted **raw text** from each menu, forming the foundation for the next stage: **cleaning and detecting valid menu items**.  

---

### Step 1: **Menu Classification**
**Folder:** `data_prep/menu_extraction/check_menu.py`

**Goal:** Automatically detect whether a scraped document truly represents a restaurant menu.

After collecting raw HTML and PDF text from restaurant websites, many files turned out to be non-menu pages such as contact information, image placeholders, or empty files.  
To ensure high-quality downstream data, we used **Gemini 2.5 Flash** to classify valid menus.

**Model:** Gemini 2.5 Flash  
**System Instruction:**
"You are an expert document classifier. Your only output must be 'yes' or 'no'. 
Do not include any explanations, punctuation, or other text."

### Step 2: **Structured Menu Extraction**
**Folder:** `data_prep/menu_extraction/extract_menu.py`  
**Goal:** Extract dishes, prices, and descriptions from restaurant menus  
**Model:** Gemini 2.5 Flash  
**Process Flow:**  
1. Input: Text output from Phase 1 (validated menus)  
2. Parse restaurant text for structured information  
3. Extract fields in the format → `[Item | Price | Description]`  
4. Store results in JSON format for downstream analysis  

## 🧠 Data Curation and Labeling

In parallel with the menu extraction pipeline, we curated and cleaned **open-source datasets** from **Kaggle** and **Hugging Face** to construct a **synthetic training corpus** for the dish classifier.  

We then:  
- Standardized dish and cuisine categories  
- Merged synthetic data with **manually labeled samples** from real-world **Yelp data**  
- Ensured diversity in cuisines and menu styles for robust model generalization  
- Produced a unified, labeled dataset used for model training and evaluation  

## 🔁 Cross-Validation Strategy 

### Overview
To ensure the classifier generalizes well across synthetic and real-world datasets, we implemented a **5-fold cross-validation** pipeline. Both the **synthetic training corpus** and **Yelp real-world data** were split into **five non-overlapping folds** each.

### Process

1. **Data Splitting**
   - Synthetic data: `Syn_A`, `Syn_B`, `Syn_C`, `Syn_D`, `Syn_E`  
   - Real-world data: `Yelp_A`, `Yelp_B`, `Yelp_C`, `Yelp_D`, `Yelp_E`

2. **Training and Validation Loop**
   - For each iteration (e.g., **Fold E**):
     - Combine the **other four folds** from both datasets to form the **Training Set**:  
       `(Syn_A + Yelp_A) + (Syn_B + Yelp_B) + (Syn_C + Yelp_C) + (Syn_D + Yelp_D)`
     - Use the remaining fold (`Syn_E` and `Yelp_E`) for validation.

3. **Validation Tests**
   - **Test 1:** Validate on `Synthetic_E` (to assess overfitting on synthetic data)
   - **Test 2:** Validate on `Yelp_E` (to assess performance on real-world data)

4. **Performance Averaging**
   - Repeat for all 5 permutations.  
   - Compute the **average accuracy and F1 score** across all folds.  
   - Analyze the performance gap between synthetic and real-world evaluations to measure generalization strength.

### Outcome
This approach provided a **balanced and reliable measure of model performance**, confirming that the classifier maintained consistent accuracy across both curated and real-world datasets.

## 🧠 Model Training and Evaluation

### Overview
Model training was conducted in two phases:  
1. **Baseline Model Development** — classical machine learning approaches  
2. **LLM Fine-Tuning** — transformer-based model optimization for contextual understanding  

All models were evaluated using the 5-fold cross-validation pipeline described earlier.

---

### Phase 1: 🧩 Baseline Model Development
**Folder:** `model_training/baseline_models/`  

**Goal:** Establish a baseline for text-based dish classification using traditional machine learning models.

**Models Tested:**
- XGBoost  
- Decision Tree  
- Random Forest  
- Logistic Regression  
- Naive Bayes  
- K-Nearest Neighbors  
- Support Vector Machine (SVM)

**Text Representations:**
- **TF-IDF (Term Frequency–Inverse Document Frequency)**
- **Word Embeddings**

**Performance Results:**

| Dataset | Technique | F1-Score |
|----------|------------|----------|
| Synthetic Data | TF-IDF | **0.95** |
| Synthetic Data | Word Embeddings | **0.70** |
| Real-World Yelp Data | TF-IDF | **0.90** |
| Real-World Yelp Data | Word Embeddings | **0.70** |

**Outcome:**  
The TF-IDF–based XGBoost model performed best among baseline classifiers, achieving high precision on synthetic data but showing reduced generalization to real-world Yelp data.  

---

### Phase 2: 🤖 LLM Model Fine-Tuning
**Folder:** `model_training/LLM_Model_Finetuning/`  

**Goal:** Improve contextual understanding of menu text using transformer-based language models.

**Models Tested:**
- BERT  
- mBERT (Multilingual BERT)  
- DeBERTa  
- Gemma-3 

**Key Customizations:**
- Fine-tuned with an **enhanced vocabulary** of ~210 **cuisine-specific tokens**  
  *(e.g., “paneer,” “brisket,” “gnocchi”)*  
- Multiple **hyperparameter tuning experiments** (epochs, learning rate) were tested — none yielded major improvements beyond the base configuration.

**Performance Results:**

| Model | Dataset | F1-Score |
|--------|----------|----------|
| **BERT** | Synthetic | **0.98** |
| **BERT** | Real-World Yelp | **0.94** |
| **DeBERTa** | Synthetic | **0.98** |
| **DeBERTa** | Real-World Yelp | **0.94** |
| **Gemma-3** | Overall | **0.40** |
| **mBERT** | Similar to BERT (no significant gain) | — |

**Model Selection:**  
BERT was chosen as the **final production model** because it provides nearly identical performance to DeBERTa while being more computationally efficient and faster to deploy.

**Final Evaluation:**  
Cross-validation and final testing confirmed BERT’s stable performance across both datasets — achieving an average **0.94 F1-score** on real-world Yelp menu data.

---

### Summary
| Model Family | Best Model | Synthetic F1 | Real Yelp F1 | Notes |
|---------------|-------------|--------------|--------------|-------|
| Traditional ML | XGBoost | 0.95 | 0.90 | Strong baseline |
| Transformer (LLM) | BERT | 0.98 | 0.94 | Selected final model |
| Transformer (LLM) | DeBERTa | 0.98 | 0.94 | Similar but heavier |
| Transformer (LLM) | Gemma-3 | 0.40 | 0.40 | Poor performance |

---

**Conclusion:**  
Fine-tuned BERT provided the best balance between accuracy, model size, and generalization across synthetic and real-world restaurant data.  

## 🚀 Deployment

### Streamlit Web Application
**Folder:** `Streamlit_deployment/`  
**Live Demo:** [🌱 VeggieFoodFinder Streamlit App](https://vegetarianfoodfinder.streamlit.app/)

### Overview
The trained **BERT classification model** and extracted restaurant data are integrated into an interactive **Streamlit web application** that allows users to explore vegetarian options across multiple U.S. cities.

### Features
- 🔍 **ZIP Code Search:** Enter a ZIP code to find nearby restaurants.  
- 🥗 **Dish Classification:** View vegetarian and non-vegetarian dishes with clear labels.  
- 📍 **Interactive Map:** Visualize restaurant locations and menu diversity.  
- 🍱 **Custom Filters:** Filter results by cuisine, dietary preference, and distance.  
- ⚙️ **Dynamic Data Loading:** Automatically retrieves the latest available menu data for selected areas.
