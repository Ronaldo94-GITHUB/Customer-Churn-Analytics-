# Customer Churn — Power BI Dashboard

## Page 1 — Executive Overview

### KPI Cards
- Total Customers
- Churned Customers
- Churn Rate %
- High Risk Customers
- Urgent Retention
- Average Churn Probability

### Visuals
1. Churn Rate by Contract
   - Axis: Contract
   - Value: Churn Rate %

2. Customer Risk Distribution
   - Axis: risk_level
   - Value: Total Customers

3. Retention Priority
   - Axis: retention_priority
   - Value: Total Customers

4. Churn Probability by Risk Level
   - Axis: risk_level
   - Value: Average Churn Probability

### Slicers
- Contract
- InternetService
- PaymentMethod
- risk_level
- retention_priority


## Page 2 — Customer Risk Intelligence

### KPI Cards
- High Risk Customers
- High Risk %
- High Risk Actual Churn
- High Risk Actual Churn Rate %

### Customer Risk Table
Columns:
- customer_id
- churn_probability
- risk_level
- retention_priority
- tenure
- MonthlyCharges
- Contract
- InternetService
- TechSupport
- actual_churn

Default business focus:
- HIGH risk
- URGENT retention priority

### Visuals
1. Risk Level vs Observed Churn Rate
2. Monthly Charges by Risk Level
3. Tenure by Risk Level
4. Contract Distribution by Risk Level


## Page 3 — Customer Profile & Churn Drivers

### Visuals
1. Churn by Contract
2. Churn by Internet Service
3. Churn by Payment Method
4. Churn by Tenure Segment
5. Average Monthly Charges
6. Average Tenure

### Statistical Evidence
Use churn_statistical_analysis.csv.

Display:
- Feature
- Test
- P-value
- Effect Size
- Effect Metric

Important:
Statistical associations do not imply causality.


## Page 4 — ML & Explainability

### Model Comparison
Models:
- Logistic Regression
- Random Forest
- XGBoost

Metrics:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

### Explainability
Display global SHAP feature importance for XGBoost.

SHAP represents model contribution magnitude and does not imply causality.

### Evaluation Scope
Model metrics and actual_churn refer to the stratified held-out test dataset.

actual_churn is evaluation information only and is never an input feature.


## Design

Theme:
- Dark professional interface
- Black / graphite background
- Gold accents
- White / light-gray text

Product:
Customer Intelligence & Risk ML Platform

Module:
Customer Churn Intelligence

---

# Credit Risk Intelligence

## Scope

Historical credit-risk analytics module for portfolio,
educational, and model-governance demonstration.

Positive model class:

- 0 = GOOD risk
- 1 = BAD risk

The model output is described as Bad Risk Probability.

## Page 1 - Credit Risk Overview

KPIs:

- Evaluated Customers
- Observed BAD Risk
- Observed BAD Rate
- Average Bad Risk Probability
- HIGH Risk Customers
- Priority Review Customers

Evaluation dataset:

- Customers: 200
- Observed BAD Risk: 60
- Observed BAD Rate: 30.00%
- LOW Risk: 118
- MEDIUM Risk: 51
- HIGH Risk: 31
- Priority Review: 31

Risk segmentation:

- LOW: probability < 0.30
- MEDIUM: 0.30 <= probability < 0.60
- HIGH: probability >= 0.60

These thresholds are analytical baselines and are not
credit approval or denial rules.

## Page 2 - Model Performance and Explainability

Models:

- Logistic Regression
- Random Forest
- XGBoost

Metrics:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Average Precision
- Brier Score

Average Precision summarizes precision-recall performance.

Lower Brier Score indicates lower probabilistic prediction
error.

Explainability:

- Global XGBoost SHAP
- Top 10 mean absolute SHAP features

Mean absolute SHAP represents model influence magnitude,
not direction or causality.

## Page 3 - Risk Intelligence

Segments:

- LOW
- MEDIUM
- HIGH

Analytical priorities:

- LOW -> MONITOR
- MEDIUM -> REVIEW
- HIGH -> PRIORITY_REVIEW

Display:

- Customer count
- Average Bad Risk Probability
- Observed BAD count
- Observed BAD rate
- Average amount
- Average duration

## Page 4 - Business and Statistical Analytics

SQL analytics:

- Risk Segment
- Duration
- Purpose
- Credit History

Statistical analytics:

- Feature
- Statistical test
- P-value
- Effect size
- Effect-size metric

Statistical associations do not establish causality.

## Governance and Evaluation Scope

The South German Credit dataset is historical.

Important limitations:

- Historical population and monetary context
- BAD cases were intentionally oversampled
- Results do not establish current lending performance
- Model scores are not current real-world probabilities
  of default
- Evaluation data has been repeatedly inspected during
  development and is not a pristine production test set
- Governance-sensitive attributes require explicit review
  before any real-world decision use

This module is intended for portfolio, educational,
analytical, and model-governance demonstration.

## Design

Theme:

- Dark professional interface
- Black / graphite background
- Gold accents
- White / light-gray text

Product:
Customer Intelligence & Risk ML Platform

Module:
Credit Risk Intelligence
