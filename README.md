# 🥗 VeggieFoodFinder  
**An AI-powered restaurant discovery app for vegetarian and vegetarian-friendly dishes**

## 📚 Table of Contents
- [Overview](#-overview)
- [Problem & Motivation](#-problem--motivation)
- [System at a Glance (How It Works)](#-system-at-a-glance-how-it-works)
- [Pipeline Overview](#-pipeline-overview)
  - [Data Collection & Scraping](#1-data-collection--scraping)
  - [Menu Validation & Structured Extraction](#2-menu-validation--structured-extraction)
  - [Data Curation & Labeling](#3-data-curation--labeling)
  - [Cross-Validation Strategy](#4--cross-validation-strategy)
  - [Model Training & Evaluation](#5-model-training--evaluation)
- [Deployment](#-deployment)

---

## 🌟 Overview
**VeggieFoodFinder** helps users find vegetarian and vegetarian-friendly dining options by extracting dish-level menu information rather than relying on coarse “vegetarian-friendly” tags.  
Users enter a ZIP code to explore nearby restaurants with dishes labeled as vegetarian vs. non-vegetarian, with filters for cuisine, dietary preference, and distance.

---

## 🚀 Problem & Motivation
Existing platforms like **Yelp** and **Google Maps** offer basic “vegetarian-friendly” filters but typically do not surface **specific dishes** or detailed menu content.

VeggieFoodFinder addresses this gap by combining:
- **Web scraping** for menu data  
- **LLM-based information extraction**  
- **Machine learning classification** for dish labeling  

The result is a unified, structured menu dataset visualized through an **interactive interface**.

---

## 🧠 System at a Glance (How It Works)
1. **Input ZIP code** → Retrieve nearby restaurants via the **Yelp Fusion API**  
2. **Scrape menus** → Collect menus from restaurant websites (HTML / PDF)  
3. **Validate menus** → Filter valid menus using LLM-based classification  
4. **Extract structured items** → Use **Gemini 2.5 Flash** to extract dishes into JSON  
5. **Classify dishes** → Fine-tuned **BERT** labels dishes as vegetarian or non-vegetarian  
6. **Deploy & visualize** → Results displayed via a **Streamlit** application  

---

## 🧩 Pipeline Overview

### 1) Data Collection & Scraping  
**Folder:** `data_prep/yelp_data_scraping/`

#### Step 1: Collect Restaurant Data
- Used the **Yelp Fusion API** to gather restaurant metadata (names, cuisines, ratings, website links).  
- Focused on **29 major U.S. cities**, collecting roughly **1,200 restaurants per city**.  
- Dataset size and coverage were shaped by the API’s free-tier rate limits.

#### Step 2: Retrieve Menus
- Restaurant menus appeared in multiple formats: **HTML**, **PDF**, and occasionally **images**.  
- Built a custom scraping system capable of handling **HTML and PDF** formats.  
- Extracted **raw text** from menus for downstream validation and item extraction.

---

### 2) Menu Validation & Structured Extraction  
**Folder:** `data_prep/menu_extraction/`

#### Step 1: Menu Classification (Valid Menu vs. Non-Menu)
**Script:** `check_menu.py`  
**Goal:** Automatically detect whether a scraped document truly represents a restaurant menu.

Many scraped files corresponded to non-menu pages (contact pages, placeholders, empty files).  
To ensure data quality, **Gemini 2.5 Flash** was used to classify valid menus.

**Model:** Gemini 2.5 Flash  
**System Instruction:**
"You are an expert document classifier. Your only output must be 'yes' or 'no'. 
Do not include any explanations, punctuation, or other text."

#### Step 2: Structured Menu Extraction
**Script:** `extract_menu.py`  
**Goal:** Extract dishes, prices, and descriptions from validated menus.

**Model:** Gemini 2.5 Flash  
**Process Flow:**
1. Input: Text output from menu validation  
2. Parse menu text for structured information  
3. Extract fields → `[Item | Price | Description]`  
4. Store results in JSON format for downstream analysis and modeling  

### 3) Data Curation & Labeling

In parallel with the menu extraction pipeline, we curated and cleaned **open-source datasets** from **Kaggle** and **Hugging Face** to construct a **synthetic training corpus** for the dish classifier.  

Steps included: 
- Standardized dish and cuisine categories  
- Merged synthetic data with **manually labeled samples** from real-world **Yelp data**  
- Ensured diversity in cuisines and menu styles for robust model generalization  
- Produced a unified, labeled dataset used for model training and evaluation  

## 4) Cross-Validation Strategy 

### Overview
To evaluate generalization across synthetic and real-world data, a **5-fold cross-validation** pipeline was implemented.  
Both datasets were split into **five non-overlapping folds**.

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
This approach provided a balanced estimate of model performance and confirmed stable generalization to real-world Yelp data.


## 5) Model Training & Evaluation

### Overview
Model training was conducted in two phases:  
Step 1. **Baseline Model Development** — classical machine learning approaches  
Step 2. **LLM Fine-Tuning** — transformer-based model optimization for contextual understanding  

All models were evaluated using the 5-fold cross-validation pipeline described earlier.

---

### Step 1: 🧩 Baseline Model Development
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

### Step 2: 🤖 LLM Model Fine-Tuning
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
BERT was selected as the final production model due to similar performance to DeBERTa with lower computational cost.

**Final Evaluation:**  
Fine-tuned BERT provided the best balance between accuracy, efficiency, and real-world generalization.

---

## Deployment

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
