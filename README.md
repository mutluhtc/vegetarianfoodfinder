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

