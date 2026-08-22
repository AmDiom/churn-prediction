"""
Data preprocessing script for the Telco Customer Churn dataset.
Cleans raw data, encodes categorical variables, and splits into train/test sets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "data/raw/telco_churn.csv"
PROCESSED_DIR = "data/processed"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV file into a DataFrame."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix data quality issues identified during EDA."""
    df = df.copy()

    # TotalCharges is stored as text with some blank values
    # (new customers with tenure=0 have no charges yet)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # customerID is a unique identifier, not a useful feature
    df = df.drop(columns=["customerID"])

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables into numeric format for modeling."""
    df = df.copy()

    # Binary Yes/No columns -> 1/0
    binary_cols = ["Partner", "Dependents", "PhoneService",
                    "PaperlessBilling", "Churn"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Gender -> 1/0
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    # Multi-category columns -> one-hot encoding
    categorical_cols = [
        "InternetService", "Contract", "PaymentMethod", "MultipleLines",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


def split_data(df: pd.DataFrame, target_col: str = "Churn", test_size: float = 0.2):
    """Split features and target into train and test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def main():
    print("Loading raw data...")
    df = load_data(RAW_DATA_PATH)

    print("Cleaning data...")
    df = clean_data(df)

    print("Encoding features...")
    df = encode_features(df)

    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(df)

    print(f"Train set: {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} rows")

    print("Saving processed data...")
    X_train.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DIR}/y_test.csv", index=False)

    print("Done. Processed files saved in data/processed/")


if __name__ == "__main__":
    main()