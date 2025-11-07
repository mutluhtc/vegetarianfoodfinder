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
- **Automated web scraping** for menu data  
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

## 📊 Data & Model Development

- **Data Collection:** ~1,200 restaurants per city across **29 U.S. cities** using Yelp API.  
- **Menu Parsing:** Supports both **HTML** and **PDF** menus; image-based menus to be added.  
- **Validation:** Invalid or empty menus automatically filtered out via an **LLM pipeline**.  
- **Model Training:**  
  - Datasets: Kaggle + Hugging Face + real Yelp menus  
  - Benchmarked: XGBoost, BERT, FLAN-T5, Gemma-3 270M, mBERT  
  - **Best Model:** Fine-tuned **BERT** (Accuracy: 94%, F1: 0.94)

---

## 💼 Impact

### For Users
- Instantly discover verified vegetarian dishes nearby  
- Filter by cuisine, distance, or dietary preference  
- Save time and avoid menu guesswork  

### For Businesses
- Showcase inclusive menus to attract dietary-specific audiences  
- Integrate with restaurant discovery and reservation apps  

---

## ⚙️ Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Data Collection** | Yelp Fusion API, Requests, BeautifulSoup, PDFPlumber |
| **Data Processing** | Pandas, JSON, Regex |
| **LLM Extraction** | Gemini 2.5 Flash |
| **Modeling** | BERT, Scikit-learn, XGBoost, PyTorch |
| **Deployment (Future)** | Streamlit / Flask (prototype), Google Maps API |
| **Storage** | CSV / JSON |

---

## 🔮 Future Work

- Expand coverage beyond 29 cities  
- Add **OCR and multimodal training** for image-based menus  
- Improve classification of seafood and conditional dishes  
- Integrate with external restaurant and reservation platforms  
- Optimize for scalability and UI performance  

---

## 🧩 Repository Structure



---

## 📈 Results

| Model | Accuracy | F1 Score |
|--------|-----------|----------|
| XGBoost | 0.88 | 0.87 |
| FLAN-T5 | 0.91 | 0.91 |
| mBERT | 0.92 | 0.92 |
| **Fine-tuned BERT** | **0.94** | **0.94** |

---

## 🤝 Contributors

**Team VeggieFoodFinder**  
- 
- 
-
-

---

## 🧾 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 💡 Acknowledgments

- [Yelp Fusion API](https://www.yelp.com/developers/documentation/v3) for restaurant metadata  
- [Gemini 2.5 Flash](https://ai.google.dev) for structured text extraction  
- Open-source datasets from Kaggle and Hugging Face  
