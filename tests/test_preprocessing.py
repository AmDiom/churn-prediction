"""
Unit tests for the data preprocessing functions.
"""

import sys
import os
import pandas as pd
import pytest

# Allow imports from the src/ folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_preprocessing import clean_data, encode_features


@pytest.fixture
def sample_raw_data():
    """A small sample DataFrame mimicking the raw dataset structure."""
    return pd.DataFrame({
        "customerID": ["1001-ABC", "1002-DEF"],
        "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "tenure": [0, 24],
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["No", "No phone service"],
        "InternetService": ["Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["No", "No"],
        "StreamingMovies": ["No", "No"],
        "Contract": ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [70.5, 45.0],
        "TotalCharges": [" ", "1080.0"],  # blank value, like real dataset
        "Churn": ["Yes", "No"],
    })


def test_clean_data_removes_customer_id(sample_raw_data):
    """customerID should be dropped since it's not a useful feature."""
    cleaned = clean_data(sample_raw_data)
    assert "customerID" not in cleaned.columns


def test_clean_data_fixes_blank_total_charges(sample_raw_data):
    """Blank TotalCharges values should become 0, not stay as text."""
    cleaned = clean_data(sample_raw_data)
    assert cleaned["TotalCharges"].dtype == float
    assert cleaned.loc[0, "TotalCharges"] == 0.0
    assert cleaned.loc[1, "TotalCharges"] == 1080.0


def test_clean_data_no_missing_values(sample_raw_data):
    """After cleaning, there should be no missing values in TotalCharges."""
    cleaned = clean_data(sample_raw_data)
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_encode_features_converts_binary_columns(sample_raw_data):
    """Yes/No columns should become 1/0 integers."""
    cleaned = clean_data(sample_raw_data)
    encoded = encode_features(cleaned)
    assert set(encoded["Churn"].unique()).issubset({0, 1})
    assert set(encoded["Partner"].unique()).issubset({0, 1})


def test_encode_features_creates_dummy_columns(sample_raw_data):
    """Multi-category columns should be one-hot encoded."""
    cleaned = clean_data(sample_raw_data)
    encoded = encode_features(cleaned)
    # Contract had 2 categories in the sample -> should create dummy columns
    assert any(col.startswith("Contract_") for col in encoded.columns)
    assert "Contract" not in encoded.columns


def test_encode_features_all_numeric(sample_raw_data):
    """After encoding, all columns should be numeric (no text/object columns left)."""
    cleaned = clean_data(sample_raw_data)
    encoded = encode_features(cleaned)
    non_numeric = encoded.select_dtypes(include="object").columns.tolist()
    assert non_numeric == []