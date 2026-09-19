import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from customer_churn.models.logistic import (
    ChurnLogisticRegressionBuilder,
)


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SeniorCitizen": [0, 0, 1, 1, 0, 1],
            "tenure": [2, 60, 5, 48, 12, 70],
            "MonthlyCharges": [90.0, 40.0, 85.0, 55.0, 75.0, 45.0],
            "TotalCharges": [180.0, 2400.0, 425.0, 2640.0, 900.0, 3150.0],
            "gender": ["Male", "Female", "Female", "Male", "Male", "Female"],
            "Partner": ["No", "Yes", "No", "Yes", "No", "Yes"],
            "Dependents": ["No", "Yes", "No", "Yes", "No", "Yes"],
            "PhoneService": ["Yes"] * 6,
            "MultipleLines": ["No", "Yes", "No", "Yes", "No", "Yes"],
            "InternetService": ["Fiber optic", "DSL", "Fiber optic", "DSL", "DSL", "No"],
            "OnlineSecurity": ["No", "Yes", "No", "Yes", "No", "No internet service"],
            "OnlineBackup": ["No", "Yes", "No", "Yes", "Yes", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No", "Yes", "No", "No internet service"],
            "TechSupport": ["No", "Yes", "No", "Yes", "No", "No internet service"],
            "StreamingTV": ["Yes", "No", "Yes", "No", "Yes", "No internet service"],
            "StreamingMovies": ["Yes", "No", "Yes", "No", "Yes", "No internet service"],
            "Contract": [
                "Month-to-month",
                "Two year",
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Two year",
            ],
            "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Credit card (automatic)",
                "Electronic check",
                "Bank transfer (automatic)",
                "Mailed check",
                "Credit card (automatic)",
            ],
        }
    )


def _target() -> pd.Series:
    return pd.Series([1, 0, 1, 0, 1, 0])


def test_builder_returns_pipeline():
    pipeline = ChurnLogisticRegressionBuilder().build()

    assert isinstance(pipeline, Pipeline)


def test_pipeline_contains_preprocessor_and_model():
    pipeline = ChurnLogisticRegressionBuilder().build()

    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_pipeline_uses_logistic_regression():
    pipeline = ChurnLogisticRegressionBuilder().build()

    assert isinstance(
        pipeline.named_steps["model"],
        LogisticRegression,
    )


def test_pipeline_can_fit_and_predict():
    pipeline = ChurnLogisticRegressionBuilder().build()

    pipeline.fit(_features(), _target())
    predictions = pipeline.predict(_features())

    assert len(predictions) == 6
    assert set(predictions).issubset({0, 1})


def test_pipeline_produces_probabilities():
    pipeline = ChurnLogisticRegressionBuilder().build()

    pipeline.fit(_features(), _target())
    probabilities = pipeline.predict_proba(_features())[:, 1]

    assert len(probabilities) == 6
    assert ((probabilities >= 0) & (probabilities <= 1)).all()