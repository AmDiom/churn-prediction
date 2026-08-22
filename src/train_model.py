"""
Model training script for the Telco Customer Churn dataset.
Trains and compares multiple classification models.
"""

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"


def load_processed_data():
    """Load the preprocessed train/test data."""
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Standardize features (mean=0, std=1). Required for Logistic Regression."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression model (interpretable baseline)."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train a Random Forest model (captures non-linear relationships)."""
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_with_cross_validation(model, X_train, y_train, model_name):
    """Run 5-fold cross-validation and print average F1 score."""
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
    print(f"{model_name} - Cross-validation F1 scores: {scores}")
    print(f"{model_name} - Mean F1: {scores.mean():.4f} (+/- {scores.std():.4f})")


def main():
    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed_data()

    print("Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("\nTraining Logistic Regression...")
    log_reg = train_logistic_regression(X_train_scaled, y_train)
    evaluate_with_cross_validation(log_reg, X_train_scaled, y_train, "Logistic Regression")

    print("\nTraining Random Forest...")
    # Random Forest does not require feature scaling
    rf = train_random_forest(X_train, y_train)
    evaluate_with_cross_validation(rf, X_train, y_train, "Random Forest")

    print("\nSaving models...")
    joblib.dump(log_reg, f"{MODELS_DIR}/logistic_regression.pkl")
    joblib.dump(rf, f"{MODELS_DIR}/random_forest.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")

    print("Done. Models saved in models/")


if __name__ == "__main__":
    main()