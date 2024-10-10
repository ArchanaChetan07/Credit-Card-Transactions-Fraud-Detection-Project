# Credit-Card-Transactions-Fraud-Detection-Project
 This project focuses on evaluating and testing several classification models  in detecting fraudulent and non-fraudulent credit card transactions. The objective is to test the model on under sampled and oversampled datasets to determine how well it can predict both types of transactions, ensuring that legitimate transactions are not incorrectly flagged as fraud.

## Dataset Information
- **Dataset Name:** Credit Card Fraud Detection
- **Source:** [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data)
- **Number of Variables:** 31 features
- **Dataset Size:** The original dataset contains 284,807 transactions, but a subset of 30,000 transactions will be used for this analysis.
- **Imbalance:** The dataset is highly imbalanced, with only 492 fraudulent transactions, representing approximately 0.17% of the total transactions.
<img width="678" alt="Screenshot 2024-10-09 at 8 32 56 PM" src="https://github.com/user-attachments/assets/b9bc0f21-6aac-4c54-b61b-34db60612e9e">

## Purpose
The purpose of this project is to develop an effective machine learning model that accurately identifies fraudulent credit card transactions. This effort aims to minimize financial losses for financial institutions and prevent fraudulent activities, ultimately leading to more secure transactions.

## Background
With the rise of digital transactions, fraudulent credit card activities have become a significant concern. Financial institutions require robust systems to detect fraud in real-time without disrupting legitimate transactions. A typical challenge in fraud detection involves dealing with highly imbalanced datasets, making it crucial to address class imbalance while maintaining model accuracy.

## Conclusion
CatBoost was selected as the most effective model due to its impressive recall rate of 86%, indicating its strong reliability in detecting fraudulent transactions. This high recall rate is essential for minimizing the risk of financial losses for institutions, as it ensures that the majority of actual fraud cases are successfully identified. The insights gained from this project will contribute to developing more robust fraud detection systems in the financial sector, enhancing security for digital transactions.

## Libraries Used
The following Python libraries are utilized in this project:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
