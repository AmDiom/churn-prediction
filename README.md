# Customer Churn Prediction

A machine learning project predicting customer churn for a telecom company, 
built with Python and scikit-learn. The final model (Logistic Regression) 
achieves an AUC of 0.84 and identifies the key drivers of customer attrition, 
providing actionable insights for retention strategies.

## Business Context

Customer churn (when a customer stops using a company's services) is one                                 of the most costly problems for subscription-based businesses. Acquiring a 
new customer typically costs significantly more than retaining an existing 
one, making churn prediction a high-value use case for machine learning.

This project aims to answer three questions:
- Which customers are most likely to churn?
- What factors drive that decision?
- How can these insights inform retention strategies?

By identifying at-risk customers before they leave, a company can proactively 
target them with retention offers, potentially saving significant revenue.

## Dataset

The project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) 
from Kaggle (originally published by IBM).

- **7,043 customers**, 21 features
- **Target variable:** `Churn` (Yes/No): 26.5% of customers churned
- **Features include:** demographics (gender, senior citizen status), 
  account information (tenure, contract type, payment method), and 
  subscribed services (internet, phone, streaming, tech support)

  ## Methodology

1. **Exploratory Data Analysis (EDA)**: understand data quality, class 
   distribution, and relationships between features and churn
2. **Data Preprocessing**: clean missing values, encode categorical 
   variables, split into train/test sets (80/20, stratified)
3. **Model Training**: train and compare three classifiers (Logistic 
   Regression, Random Forest, XGBoost) using 5-fold cross-validation
4. **Model Evaluation**: assess the best model on the held-out test set 
   using precision, recall, F1-score, and ROC-AUC
5. **Interpretability**: analyze feature importance and SHAP values to 
   explain individual predictions

   ## Key EDA Insights

- **Tenure is the strongest predictor of retention**: customers with low 
  tenure (0-5 months) churn at a much higher rate than long-tenured 
  customers (60+ months)
- **Contract type has a major impact**: month-to-month customers churn 
  15x more often than two-year contract customers (42.7% vs 2.8%)
- **Fiber optic customers churn the most (41.9%)**, despite paying more 
  on average ($91.50/month vs $58.10 for DSL), suggesting price 
  sensitivity or service quality concerns
- **Class imbalance**: only 26.5% of customers churned, which shaped the 
  choice of evaluation metrics and modeling strategy (see below)

![Churn by tenure](reports/figures/tenure_vs_churn.png)

## Modeling Results

Three models were trained and compared using 5-fold cross-validation (F1-score):

| Model | Mean F1-Score |
|---|---|
| **Logistic Regression** | **0.632** |
| Random Forest | 0.604 |
| XGBoost | 0.574 |

Logistic Regression was selected as the final model. Despite its simplicity, 
it outperformed more complex tree-based models, likely because the 
relationships between features and churn are largely linear (as confirmed 
by the EDA), and XGBoost was used without hyperparameter tuning. This result 
highlights that model complexity does not always translate into better 
performance, especially on relatively small, well-understood datasets.

### Performance on the Test Set

| Metric | No Churn | Churn |
|---|---|---|
| Precision | 0.90 | 0.51 |
| Recall | 0.72 | 0.79 |
| F1-Score | 0.80 | 0.62 |

**ROC-AUC: 0.841**

The model correctly identifies 79% of customers who actually churned 
(recall), which is critical in a business context where failing to detect 
a churning customer (false negative) is more costly than a false alarm.

![Confusion Matrix](reports/figures/confusion_matrix.png)
![ROC Curve](reports/figures/roc_curve.png)

## Model Interpretability

### Feature Importance

The Logistic Regression coefficients confirm the patterns found during EDA: 
`tenure` and `Contract_Two year` are the strongest factors reducing churn 
risk, while `InternetService_Fiber optic` and `TotalCharges` increase it.

![Feature Importance](reports/figures/feature_importance.png)

### SHAP Analysis

To go beyond global feature importance, SHAP (SHapley Additive exPlanations) 
values were used to understand how each feature impacts individual 
predictions. This confirms the same top drivers while showing the direction 
and magnitude of their effect on a per-customer basis, useful for explaining 
individual retention risk scores to business stakeholders.

![SHAP Summary](reports/figures/shap_summary.png)

## Project Structure

```
churn-prediction/
├── data/
│   ├── raw/                  # Original dataset (not tracked in git)
│   └── processed/            # Cleaned, encoded train/test sets
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory data analysis
├── src/
│   ├── data_preprocessing.py # Data cleaning and encoding pipeline
│   ├── train_model.py        # Model training and comparison
│   └── evaluate.py           # Model evaluation and interpretability
├── models/                   # Trained models (not tracked in git)
├── reports/
│   └── figures/              # Generated plots and visualizations
├── tests/
│   └── test_preprocessing.py # Unit tests for preprocessing functions
├── requirements.txt
└── README.md
```

## How to Reproduce

1. Clone the repository:
```bash
git clone https://github.com/AmDiom/churn-prediction.git
cd churn-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) 
   and place it as `data/raw/telco_churn.csv`

5. Run the pipeline:
```bash
python src/data_preprocessing.py
python src/train_model.py
python src/evaluate.py
```

6. Run the tests:
```bash
pytest tests/ -v
```

## Limitations and Future Improvements

- **No hyperparameter tuning**: models were trained with default or 
  reasonable manual parameters. A systematic search (GridSearchCV, 
  Optuna) could improve performance, particularly for XGBoost
- **Static dataset**: the model is trained on a snapshot of customer data. 
  In production, it would need to be retrained periodically as customer 
  behavior evolves
- **No cost-sensitive threshold tuning**: the default 0.5 classification 
  threshold was used. Since false negatives (missed churners) are more 
  costly than false positives, adjusting the decision threshold could 
  improve business outcomes
- **Class imbalance handling**: only `class_weight`/`scale_pos_weight` 
  was used. Techniques like SMOTE could be explored for comparison
- **Deployment**: the model is not yet served via an API. A FastAPI 
  endpoint would allow real-time churn risk scoring.

  ## Technologies Used

- **Language:** Python 3.14
- **Data manipulation:** pandas, numpy
- **Machine Learning:** scikit-learn, XGBoost
- **Visualization:** matplotlib, seaborn
- **Interpretability:** SHAP
- **Testing:** pytest
- **Environment:** Jupyter Notebook, VS Code