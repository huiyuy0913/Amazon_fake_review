# Fake Review Detection on Amazon

This project analyzes **Amazon product reviews** to detect **fake reviews** using **machine learning, text analysis, and web scraping**. It extracts product and review data, processes text for similarity analysis, and trains models to classify fake vs. genuine reviews.

---

## 🚀 **Features**
- ✅ **Text Processing & Similarity Analysis**: Uses **TF-IDF vectorization** and **cosine similarity** to analyze review text patterns.
- ✅ **Machine Learning for Fake Review Detection**: Trains **Logistic Regression, Random Forest, XGBoost, and SVM** models to classify reviews.
- ✅ **Web Scraping Pipeline**: Extracts **Amazon product data** using APIs from **Oxylabs & Rainforest**.
- ✅ **Large-Scale Data Processing**: Handles **millions of reviews** using **parallel computing** (`pandarallel`) for efficiency.

---

## 🔹 **Codes**
- ✅ [text_analysis_TF_IDF.py](https://github.com/huiyuy0913/Amazon_fake_review/blob/main/text_analysis_TF_IDF.py) computes TF-IDF & cosine similarity for review text.
- ✅ [request_Amazon_API_oxylabs.py](https://github.com/huiyuy0913/Amazon_fake_review/blob/main/request_Amazon_API_oxylabs.py) [request_Amazon_API_oxylabs_second_try.py](https://github.com/huiyuy0913/Amazon_fake_review/blob/main/request_Amazon_API_oxylabs_second_try.py) [request_Amazon_API_rainforest.py](https://github.com/huiyuy0913/Amazon_fake_review/blob/main/request_Amazon_API_rainforest.py)
  scrape product data using Oxylabs & Rainforest APIs
- ✅ [my_code_train_model.py](https://github.com/huiyuy0913/Amazon_fake_review/blob/main/my_code_train_model.py) trains ML models (Random Forest, XGBoost, SVM)
