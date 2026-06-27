<div align="center">

# Credit Card Fraud Detection

**Ensemble ML on Highly Imbalanced Transaction Data**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-189ABE?style=flat-square)](https://xgboost.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-86%25%20Recall-yellow?style=flat-square)](https://catboost.ai)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

*284,807 transactions. 492 frauds. 0.17% fraud rate. How do you find the needle?*

</div>

---

## The problem

Credit card fraud detection is one of the hardest classification problems in production ML — not because the model is complex, but because the data isn't. With a fraud rate of 0.17%, a model that predicts "not fraud" every single time achieves 99.83% accuracy. That model is useless.

This project tackles **class imbalance head-on**: comparing undersampling, SMOTE oversampling, and ensemble methods to maximize recall on the minority class without destroying precision.

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data) |
| Total transactions | 284,807 |
| Fraudulent | 492 (0.17%) |
| Features | 31 (V1–V28 PCA-transformed + Time + Amount + Class) |
| Subset used | 30,000 transactions (stratified sample) |

The features V1–V28 are PCA-transformed for privacy. Time and Amount are the only raw features. This makes feature engineering limited — model selection and imbalance handling matter more than feature work.

---

## Approach

```
Raw data (284,807 rows, 0.17% fraud)
      │
      ▼
Stratified 30k sample
      │
      ├── Undersampled dataset  (balanced 1:1)
      └── SMOTE oversampled     (synthetic minority upsampling)
                │
                ▼
        5 classifiers evaluated:
        ├── Logistic Regression  (baseline)
        ├── Random Forest
        ├── XGBoost
        ├── LightGBM
        └── CatBoost             ← winner
                │
                ▼
        Evaluation: Recall · Precision · F1 · AUC-ROC
        Priority metric: RECALL (missing fraud = financial loss)
```

---

## Results

| Model | Dataset | Recall | Precision | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | SMOTE | 0.72 | 0.68 | 0.70 | 0.86 |
| Random Forest | SMOTE | 0.79 | 0.81 | 0.80 | 0.91 |
| XGBoost | SMOTE | 0.82 | 0.79 | 0.80 | 0.93 |
| LightGBM | SMOTE | 0.83 | 0.80 | 0.81 | 0.94 |
| **CatBoost** | **SMOTE** | **0.86** | **0.83** | **0.84** | **0.96** |

**CatBoost selected** — 86% recall means 86 out of every 100 actual frauds are caught. For a financial institution processing millions of transactions, that difference directly translates to prevented losses.

> Why recall over accuracy? In fraud detection, a false negative (missed fraud) costs the bank money and damages customer trust. A false positive (flagged legitimate transaction) is an inconvenience. Recall is the right optimization target.

---

## Why SMOTE over undersampling?

Undersampling throws away 99.8% of the legitimate transactions to balance the dataset. SMOTE (Synthetic Minority Oversampling Technique) generates synthetic fraud examples by interpolating between existing fraud cases. Both were tested — SMOTE consistently produced 4–7% higher recall across all models by preserving the full information content of the majority class.

---

## Tech stack

```
Data pipeline     Pandas · NumPy
Visualization     Matplotlib · Seaborn
Imbalance         imbalanced-learn (SMOTE)
Models            scikit-learn · XGBoost · LightGBM · CatBoost
Evaluation        classification_report · confusion_matrix · ROC-AUC
```

---

## Project structure

```
Credit-Card-Transactions-Fraud-Detection-Project/
├── ADS505_Group4_Final.ipynb         # Full pipeline notebook
├── ADS505_FinalProject_Group4Code.pdf # Project report
├── fraud.csv                          # Dataset (30k sample)
├── requirements.txt
├── tests/                             # Unit tests
└── .github/workflows/                 # CI pipeline
```

---

## Key takeaways

- **Model choice matters less than imbalance strategy.** A tuned Logistic Regression on SMOTE data outperforms an untuned XGBoost on raw imbalanced data.
- **CatBoost handles categorical features natively** — critical for real transaction data with merchant categories and card types.
- **Threshold tuning is underrated.** Shifting the decision threshold from 0.5 to 0.3 boosted recall by ~6% with acceptable precision drop.

---

## Author

**Archana Suresh Patil** — ML Engineer & Data Scientist  
MS Data Science · University of San Diego · GPA 3.9  
📬 apatil@sandiego.edu · [LinkedIn](https://linkedin.com/in/archana-suresh-patil-792213245) · [GitHub](https://github.com/ArchanaChetan07)
