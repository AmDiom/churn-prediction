"""
Model evaluation script for the Telco Customer Churn dataset.
Evaluates the chosen model (Logistic Regression) on the held-out test set.
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
FIGURES_DIR = "reports/figures"


def load_test_data():
    """Load the held-out test set."""
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").values.ravel()
    return X_test, y_test


def load_model_and_scaler():
    """Load the trained Logistic Regression model and its scaler."""
    model = joblib.load(f"{MODELS_DIR}/logistic_regression.pkl")
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    return model, scaler


def plot_confusion_matrix(y_test, y_pred):
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.title("Confusion Matrix - Logistic Regression")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/confusion_matrix.png", dpi=150)
    plt.show()

    return cm


def plot_roc_curve(y_test, y_proba):
    """Plot and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})", color="#3498db")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/roc_curve.png", dpi=150)
    plt.show()

    return auc


def show_feature_importance(model, feature_names):
    """Show and save the most influential features (Logistic Regression coefficients)."""
    coefficients = pd.Series(model.coef_[0], index=feature_names)
    top_features = coefficients.sort_values(key=abs, ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in top_features.values]
    plt.barh(top_features.index[::-1], top_features.values[::-1], color=colors[::-1])
    plt.title("Top 15 Features Influencing Churn Prediction")
    plt.xlabel("Coefficient (impact on churn probability)")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/feature_importance.png", dpi=150)
    plt.show()

    return top_features

def explain_with_shap(model, X_test, feature_names, sample_size=200):
    """Generate SHAP explanations for a sample of the test set."""
    import shap

    # Using a sample for speed (SHAP can be slow on large datasets)
    X_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)

    explainer = shap.LinearExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot: shows the impact and direction of each feature across many predictions
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.show()

    return explainer, shap_values


def main():
    print("Loading test data and model...")
    X_test, y_test = load_test_data()
    model, scaler = load_model_and_scaler()

    print("Scaling test features...")
    X_test_scaled = scaler.transform(X_test)

    print("Generating predictions...")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    print("Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)

    print("Plotting ROC curve...")
    auc = plot_roc_curve(y_test, y_proba)
    print(f"AUC score: {auc:.4f}")

    print("Analyzing feature importance...")
    top_features = show_feature_importance(model, X_test.columns)
    print("\nTop features:")
    print(top_features)

    print("Generating SHAP explanations...")
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    explain_with_shap(model, X_test_scaled_df, X_test.columns)

    print("\nDone. Figures saved in reports/figures/")

if __name__ == "__main__":
    main()
