# 🎬 Movie Recommendation System (Collaborative Filtering)

This project implements a **collaborative filtering–based recommendation system**
using **matrix factorization (SVD)** on the **MovieLens** dataset.

The goal is to predict user preferences and generate **top-N movie recommendations**
to improve user engagement and content discovery.

The project is **educational in origin** and **engineering-oriented in presentation**,
with a clear ML pipeline and production-ready thinking.

---

## 📌 Problem Statement

- **What:** Recommend movies to users based on their past ratings  
- **Who:** Users of a platform that collects explicit movie ratings  
- **Why:** Increase user engagement and retention by providing personalized content recommendations  

---

## 📊 Dataset

We use the **MovieLens (ml-latest-small)** dataset, which contains user–movie interactions
in the form of explicit ratings.

**Main fields:**
- `userId` — user identifier  
- `movieId` — movie identifier  
- `rating` — explicit rating (0.5–5.0)

Dataset source:  
https://grouplens.org/datasets/movielens/

---

## 🧠 Chosen Approach

### Collaborative Filtering with SVD

The main model is **Singular Value Decomposition (SVD)** implemented via the
`surprise` library.

**Why SVD:**
- Well-established baseline for recommender systems  
- Efficient and interpretable  
- Handles sparse user–item interaction matrices well  
- Suitable as a foundation for production systems  

GridSearchCV is used to tune SVD hyperparameters using **cross-validated RMSE**,
whereas the baseline RMSE is obtained from a single train/test split.

---

## 🔄 ML Pipeline

### 1. Data Loading
- Load MovieLens ratings
- Select relevant columns (`userId`, `movieId`, `rating`)

### 2. Data Preparation
- Convert data into a `surprise.Dataset` using `Reader`
- Split data into train/test for baseline evaluation

### 3. Model Training
- Train a baseline SVD model
- Perform hyperparameter tuning with `GridSearchCV`

### 4. Evaluation
- **Metric:** RMSE (Root Mean Squared Error)

**Results:**
- Baseline RMSE (single split): **0.8820**
- Best CV RMSE: **0.8757**

**Best SVD parameters:**
```python
{
  'n_epochs': 20,
  'lr_all': 0.005,
  'n_factors': 100,
  'reg_all': 0.1
}
```

### 5. Final Model

- Retrain the SVD model using the best hyperparameters on the **full dataset**
- This final model is used for inference and recommendation generation

### 6. Inference (Top-N Recommendations)

- Predict ratings for all candidate movies for a given user
- Rank movies by predicted rating
- Return top-N recommendations

---

## 🔍 Example Output (Inference)

| movie_id | predicted_rating |
|---------:|------------------:|
| 318  | 5.00 |
| 3451 | 4.99 |
| 1204 | 4.99 |
| 2959 | 4.98 |
| 750  | 4.98 |

> **Note:** Filtering already-rated movies is not applied in this demo  
> and is listed as a potential improvement.

---

## 📁 Project Structure

```text
recsys-portfolio/
│
├── 02_SVD_Collaborative_Filtering.ipynb
│   Main production-oriented SVD pipeline
│
├── Appendix_Algorithm_Comparison.ipynb
│   Exploratory comparison of SVD, SVD++, and NMF
│
├── Appendix_User_Based_CF_From_Scratch.ipynb
│   From-scratch implementation of user-based collaborative filtering
│   (educational appendix)
│
├── ml-latest-small/
│   MovieLens dataset files
│
└── README.md
```

---

## ▶️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/GennadiiBukh/recsys-portfolio.git
cd recsys-portfolio
```

2. Install dependencies:
```bash
pip install numpy pandas scikit-surprise
```

3. Run notebooks:
- `02_SVD_Collaborative_Filtering.ipynb` — main pipeline  
- Appendix notebooks are optional

---

## 🏭 Production Notes & Next Steps

This project focuses on a solid ML foundation.
Possible production improvements include:

- Filtering out already-rated items during inference  
- Wrapping inference logic into a reusable service function  
- Exposing recommendations via a **FastAPI** endpoint  
- Persisting trained models (e.g. with `joblib`)  
- Incremental retraining with new user interactions  
- Monitoring prediction quality and data drift  

---

## 🧩 Appendix Notebooks

Additional notebooks are included to demonstrate:

- Algorithm comparison (SVD, SVD++, NMF)
- A from-scratch implementation of user-based collaborative filtering

These notebooks are **educational appendices** and are **not part of the production pipeline**.

---

## ✅ Summary

This project demonstrates a complete and reproducible
**collaborative filtering recommendation pipeline**, featuring:

- Clear problem framing  
- Sound ML methodology  
- Interpretable evaluation  
- Practical inference example  
- Production-oriented thinking  

It is designed for **ML Engineer interviews** and portfolio presentation.
