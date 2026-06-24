import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


class TestDataProcessing:

    def test_dataframe_loads(self):
        df = pd.DataFrame({
            "amount": [100.0, 250.5, 3000.0, 0.99],
            "is_fraud": [0, 0, 1, 0]
        })
        assert not df.empty
        assert "is_fraud" in df.columns

    def test_class_imbalance_detected(self):
        labels = pd.Series([0]*950 + [1]*50)
        fraud_rate = labels.mean()
        assert fraud_rate < 0.2, "Dataset should be imbalanced"

    def test_no_missing_values_after_clean(self):
        df = pd.DataFrame({"amount": [1.0, None, 3.0], "is_fraud": [0, 1, 0]})
        df_clean = df.dropna()
        assert df_clean.isnull().sum().sum() == 0

    def test_feature_scaling(self):
        X = np.array([[100], [200], [300], [400]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert abs(X_scaled.mean()) < 1e-10
        assert abs(X_scaled.std() - 1.0) < 1e-10


class TestFraudDetectionModel:

    def test_model_trains_successfully(self):
        np.random.seed(42)
        X = np.random.rand(200, 5)
        y = (X[:, 0] > 0.7).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_model_auc_above_baseline(self):
        np.random.seed(42)
        X = np.random.rand(300, 8)
        y = (X[:, 0] + X[:, 2] > 1.2).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        assert auc > 0.5, f"AUC {auc:.3f} is below random baseline"

    def test_prediction_output_is_binary(self):
        np.random.seed(0)
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        model = RandomForestClassifier(n_estimators=5, random_state=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})
