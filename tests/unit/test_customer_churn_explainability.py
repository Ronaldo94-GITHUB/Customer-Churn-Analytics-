import pandas as pd
import pytest

from customer_churn.models.explainability import (
    ChurnShapExplainer,
)
from customer_churn.models.logistic import (
    ChurnLogisticRegressionBuilder,
)
from customer_churn.models.xgboost_model import (
    ChurnXGBoostBuilder,
)


def _sample_features():
    return pd.DataFrame(
        {
            "gender": ["Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 1],
            "Partner": ["Yes", "No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "No"],
            "tenure": [1, 12, 48, 72],
            "PhoneService": ["Yes", "Yes", "Yes", "Yes"],
            "MultipleLines": ["No", "Yes", "No", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
            "OnlineSecurity": ["No", "No", "Yes", "No internet service"],
            "OnlineBackup": ["Yes", "No", "Yes", "No internet service"],
            "DeviceProtection": ["No", "Yes", "Yes", "No internet service"],
            "TechSupport": ["No", "No", "Yes", "No internet service"],
            "StreamingTV": ["No", "Yes", "Yes", "No internet service"],
            "StreamingMovies": ["No", "Yes", "Yes", "No internet service"],
            "Contract": [
                "Month-to-month",
                "Month-to-month",
                "One year",
                "Two year",
            ],
            "PaperlessBilling": ["Yes", "Yes", "No", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Electronic check",
                "Credit card (automatic)",
                "Bank transfer (automatic)",
            ],
            "MonthlyCharges": [29.85, 89.10, 64.50, 24.90],
            "TotalCharges": [29.85, 1069.20, 3096.00, 1792.80],
        }
    )


def test_shap_explainer_returns_one_row_per_customer():
    features = _sample_features()
    target = [0, 1, 0, 0]

    pipeline = ChurnXGBoostBuilder(
        n_estimators=10,
    ).build()
    pipeline.fit(features, target)

    result = ChurnShapExplainer().explain(
        pipeline,
        features,
    )

    assert result.values.shape[0] == len(features)


def test_shap_values_match_transformed_feature_count():
    features = _sample_features()
    target = [0, 1, 0, 0]

    pipeline = ChurnXGBoostBuilder(
        n_estimators=10,
    ).build()
    pipeline.fit(features, target)

    result = ChurnShapExplainer().explain(
        pipeline,
        features,
    )

    assert result.values.shape[1] == len(result.feature_names)


def test_shap_feature_names_match_transformed_dataframe():
    features = _sample_features()
    target = [0, 1, 0, 0]

    pipeline = ChurnXGBoostBuilder(
        n_estimators=10,
    ).build()
    pipeline.fit(features, target)

    result = ChurnShapExplainer().explain(
        pipeline,
        features,
    )

    assert tuple(result.transformed_features.columns) == result.feature_names


def test_shap_transformed_dataframe_preserves_rows():
    features = _sample_features()
    target = [0, 1, 0, 0]

    pipeline = ChurnXGBoostBuilder(
        n_estimators=10,
    ).build()
    pipeline.fit(features, target)

    result = ChurnShapExplainer().explain(
        pipeline,
        features,
    )

    assert len(result.transformed_features) == len(features)


def test_shap_rejects_non_xgboost_pipeline():
    features = _sample_features()
    target = [0, 1, 0, 0]

    pipeline = ChurnLogisticRegressionBuilder().build()
    pipeline.fit(features, target)

    with pytest.raises(
        TypeError,
        match="XGBClassifier",
    ):
        ChurnShapExplainer().explain(
            pipeline,
            features,
        )