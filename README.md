# Amazon-Alexa-Reviews---Sentiment-Analysis


Classify Amazon Alexa product reviews as **positive (1)** or **negative (0)** using classic NLP + ML.  
Includes EDA, text preprocessing, Bag-of-Words features, model training (Random Forest / XGBoost / Decision Tree), cross-validation, and confusion-matrix evaluation.

---

## Overview
- **Goal:** Predict sentiment from review text (verified_reviews) and learn what drives positive/negative feedback.
- **Data size:** ~3,150 rows, 5 columns (rating, date, variation, verified_reviews, feedback).
- **Target:** feedback (1 = positive, 0 = negative).

**Key outcomes**
- Strong baselines with tree-based models.
- Clear class imbalance (~92% positive / ~8% negative) handled with proper evaluation (confusion matrix, CV).
- Saved artifacts (CountVectorizer, MinMaxScaler, trained model) for reuse.

---

## Project Flow

1. **EDA**
   - Rating & feedback distributions (bar/pie).
   - Mean rating by variation.
   - Review length analysis.
   - Global / class-specific word clouds.

2. **Text Preprocessing**
   - Keep letters, lowercase, remove stopwords, **Porter stemming**.
   - Build corpus (cleaned reviews).

3. **Vectorization**
   - CountVectorizer(max_features=2500) → Bag-of-Words matrix X.

4. **Train / Test Split & Scaling**
   - 70/30 split.
   - MinMaxScaler on X_train, transform X_test.

5. **Modeling**
   - **RandomForest**, **XGBoost**, **DecisionTree**.
   - 10-fold **Cross-Validation**.
   - **GridSearchCV** on RandomForest (depth, estimators, min_samples_split).

6. **Evaluation**
   - Accuracy (train/test), confusion matrix.
   - CV mean accuracy & variance for stability.

7. **Persistence**
   - Save vectorizer, scaler, and best model to Models/ via pickle.

---

## Results (illustrative from notebook)
- **Random Forest**: Test accuracy ≈ **94%**, stable across folds (low variance).  
- **XGBoost**: Similar performance to RF.  
- **Decision Tree**: Higher train accuracy but lower test accuracy (overfit).  
- **Insights:** Positive reviews dominate; variations like *Charcoal Fabric* trend higher; negative clouds show terms like “problem/poor/returned”.

> Note: Exact numbers depend on random seed and environment; see notebook outputs.

---

## Tech Stack
- **Python**, **Jupyter Notebook**
- **pandas, numpy, matplotlib, seaborn**
- **nltk** (stopwords, PorterStemmer)
- **scikit-learn** (CountVectorizer, models, CV, GridSearch)
- **xgboost**
- **wordcloud**
- **pickle / joblib** for persistence


