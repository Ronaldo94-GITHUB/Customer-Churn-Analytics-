# Customer Churn Intelligence — v1.0

## Status

ACTIVE

## Platform

Customer Intelligence & Risk ML Platform

## Release

Customer Churn Intelligence v1.0

## Data

Dataset:
IBM Telco Customer Churn

Customers:
7,043

Held-out evaluation customers:
1,409

Observed churn rate:
26.54%

## Data Science Lifecycle

- Data ingestion
- Data quality validation
- Data cleaning
- Exploratory Data Analysis
- Statistical analysis
- Feature engineering
- SQL analytics
- Machine Learning
- Model evaluation
- SHAP explainability
- Risk segmentation
- Retention intelligence
- Business Intelligence

## Machine Learning

Models evaluated:

- Logistic Regression
- Random Forest
- XGBoost

All models were evaluated on the same stratified held-out test split.

Baseline Logistic Regression:

- Accuracy: 0.8055
- Precision: 0.6572
- Recall: 0.5588
- F1: 0.6040
- ROC-AUC: 0.8421

XGBoost is also used for SHAP explainability and operational risk intelligence.

## Risk Intelligence

Operational segmentation:

- LOW: 874 customers
- MEDIUM: 336 customers
- HIGH: 199 customers

Retention priorities:

- MONITOR: 874
- ENGAGE: 336
- URGENT: 199

HIGH risk observed churn rate on held-out evaluation data:
73.87%

Risk thresholds are operational baselines and were not optimized as part of v1.0.

## Explainability

Global and customer-level SHAP analysis is supported.

SHAP describes model contribution and does not establish causality.

## Statistical Analysis

Implemented:

- Chi-Square
- Cramer's V
- Mann-Whitney U
- Rank-Biserial effect size

Statistical associations are exploratory and do not establish causality.

## Analytics

Implemented:

- Pandas analytics
- SQLite / SQL analytics
- Customer-level intelligence dataset
- Aggregated business reports

## Dashboard

Streamlit dashboard includes:

- Executive KPIs
- Model comparison
- SHAP explainability
- Risk intelligence
- Business analytics
- Statistical analysis
- Data Science stack

## Power BI

Power BI package includes:

- 9 analytical datasets
- 11 documented DAX measures
- 4-page dashboard specification

Pages:

1. Executive Overview
2. Customer Risk Intelligence
3. Customer Profile & Churn Drivers
4. ML & Explainability

## Quality Gate

Release validation:

- Ruff: PASS
- Pytest: 153 PASS
- Python compile: PASS
- Runtime audit: PASS
- Streamlit visual validation: PASS
- Power BI package audit: PASS

## Platform Status After Release

- Fraud Detection: ACTIVE
- Customer Churn: ACTIVE
- Credit Risk: PLANNED
- Customer Lifetime Value: PLANNED
- Next Best Offer: PLANNED

Total modules: 5

Active modules: 2

Planned modules: 3

## Release Status

CUSTOMER_CHURN_V1.0=RELEASE_READY
