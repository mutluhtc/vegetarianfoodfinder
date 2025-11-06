# ===========================================
# train_baseline_models.py
# Baseline model training on menu datasets
# ===========================================

import os
import re
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

# Classifier models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# For word embeddings
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

# ===========================================
# Configuration
# ===========================================


synthetic_path = "your_synthetic_train_data_path"
yelp_path = "your_yelp_train_data_path"

# ===========================================
# Load datasets
# ===========================================
synthetic_train_df = pd.read_csv(synthetic_path)
yelp_train_df = pd.read_csv(yelp_path)

# ===========================================
# Preprocessing
# ===========================================
def clean_text(text):
    return re.sub(r'[^A-Za-z\s]', '', str(text).lower())

X_synthetic_text = synthetic_train_df['dish_name'] + ': ' + synthetic_train_df['cuisine'] + ': ' + synthetic_train_df['description'].fillna('')
X_yelp_text = yelp_train_df['dish_name'] + ': ' + yelp_train_df['cuisine'] + ': ' + yelp_train_df['description'].fillna('')

X_synthetic_text = X_synthetic_text.apply(clean_text)
X_yelp_text = X_yelp_text.apply(clean_text)

y_synthetic_labels = synthetic_train_df['diet']
y_yelp_labels = yelp_train_df['diet']

# ===========================================
# Helper Functions
# ===========================================
def mean_calculator(values):
    return 100 * sum(values) / len(values)

# ===========================================
# Define Models
# ===========================================
models_dict_1 = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(solver='liblinear', class_weight='balanced'),
    'XGBoost Classifier': XGBClassifier(n_estimators=100),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced'),
    'KNN': KNeighborsClassifier()
}

# ===========================================
# TF-IDF Cross-Validation
# ===========================================
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.57, stop_words='english')
le = LabelEncoder()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

score_dict = {}

for model_name, model in models_dict_1.items():
    synthetic_acc_scores, synthetic_f1_scores = [], []
    yelp_acc_scores, yelp_f1_scores = [], []

    for (train_synth_idx, test_synth_idx), (train_yelp_idx, test_yelp_idx) in zip(kf.split(X_synthetic_text), kf.split(X_yelp_text)):
        # Split data
        X_synthetic_train, X_synthetic_test = X_synthetic_text.iloc[train_synth_idx], X_synthetic_text.iloc[test_synth_idx]
        y_synthetic_train, y_synthetic_test = y_synthetic_labels.iloc[train_synth_idx], y_synthetic_labels.iloc[test_synth_idx]

        X_yelp_train, X_yelp_test = X_yelp_text.iloc[train_yelp_idx], X_yelp_text.iloc[test_yelp_idx]
        y_yelp_train, y_yelp_test = y_yelp_labels.iloc[train_yelp_idx], y_yelp_labels.iloc[test_yelp_idx]

        # Combine train sets
        X_train_combined = pd.concat([X_synthetic_train, X_yelp_train], axis=0)
        y_train_combined = pd.concat([y_synthetic_train, y_yelp_train], axis=0)

        # TF-IDF encoding
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_combined)
        X_test_tfidf_synthetic = tfidf_vectorizer.transform(X_synthetic_test)
        X_test_tfidf_yelp = tfidf_vectorizer.transform(X_yelp_test)

        # Label encode
        y_train_encoded = le.fit_transform(y_train_combined)
        y_test_encoded_synthetic = le.transform(y_synthetic_test)
        y_test_encoded_yelp = le.transform(y_yelp_test)

        # Train model
        model.fit(X_train_tfidf, y_train_encoded)

        # Predict + score
        y_pred_synthetic = model.predict(X_test_tfidf_synthetic)
        synthetic_acc_scores.append(accuracy_score(y_pred_synthetic, y_test_encoded_synthetic))
        synthetic_f1_scores.append(f1_score(y_pred_synthetic, y_test_encoded_synthetic))

        y_pred_yelp = model.predict(X_test_tfidf_yelp)
        yelp_acc_scores.append(accuracy_score(y_pred_yelp, y_test_encoded_yelp))
        yelp_f1_scores.append(f1_score(y_pred_yelp, y_test_encoded_yelp))

    score_dict[model_name] = {
        "Avg. Accuracy Synthetic": mean_calculator(synthetic_acc_scores),
        "Avg. F1-Score Synthetic": mean_calculator(synthetic_f1_scores),
        "Avg. Accuracy Yelp": mean_calculator(yelp_acc_scores),
        "Avg. F1-Score Yelp": mean_calculator(yelp_f1_scores)
    }

# ===========================================
# Results
# ===========================================
results_df = pd.DataFrame(score_dict).T.sort_values(by=['Avg. F1-Score Synthetic', 'Avg. F1-Score Yelp'], ascending=False)
print("\n===== TF-IDF Results =====")
print(results_df)

# ===========================================
# WORD EMBEDDING SECTION
# ===========================================
print("\n===== Starting Word Embedding Evaluation =====")

models_dict_2 = {
    'Naive Bayes': Pipeline([('scaler', MinMaxScaler()), ('clf', MultinomialNB())]),
    'Logistic Regression': LogisticRegression(solver='liblinear', class_weight='balanced'),
    'XGBoost Classifier': XGBClassifier(n_estimators=80),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=80, max_depth=5, class_weight='balanced'),
    'KNN': KNeighborsClassifier()
}

def get_document_embedding(tokens, word_vectors, vector_size):
    embeddings = [word_vectors[word] for word in tokens if word in word_vectors]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(vector_size)

# Combine datasets
df = pd.concat([synthetic_train_df, yelp_train_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
X = df['dish_name'] + ' ' + df['cuisine'] + ' ' + df['description'].fillna('')
X = X.apply(clean_text)
y = df['diet']

tokenized_corpus = [word_tokenize(sentence) for sentence in X]
word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
document_embeddings = np.array([get_document_embedding(s, word2vec_model.wv, 100) for s in tokenized_corpus])

le = LabelEncoder()
y_encoded = le.fit_transform(y)

def cross_val_evaluations(model, X_data, y_data, cv=5):
    scores = cross_val_score(model, X_data, y_data, cv=cv, scoring="f1")
    acc = cross_val_score(model, X_data, y_data, cv=cv, scoring="accuracy")
    return {'average f1-score': scores.mean() * 100, 'average accuracy': acc.mean() * 100}

wordembed_results = []
for model_name, model in models_dict_2.items():
    scores = cross_val_evaluations(model, document_embeddings, y_encoded)
    wordembed_results.append([model_name, scores['average f1-score'], scores['average accuracy']])

wordembed_df = pd.DataFrame(wordembed_results, columns=['Model', 'Average F1-score', 'Average Accuracy']).sort_values(by='Average F1-score', ascending=False)
print("\n===== Word Embedding Results =====")
print(wordembed_df)

# ===========================================
# END
# ===========================================
print("\nTraining and evaluation complete.")
