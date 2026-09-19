import pandas as pd

from customer_churn.analytics.eda import (
    ChurnEDA,
)


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Churn": ["Yes", "Yes", "No", "No"],
            "tenure": [2, 10, 30, 70],
            "MonthlyCharges": [100.0, 80.0, 50.0, 40.0],
            "TotalCharges": [200.0, 800.0, 1500.0, 2800.0],
        }
    )


def test_numerical_analysis_groups_by_churn():
    result = ChurnEDA().numerical_by_churn(_sample())

    assert "Yes" in result.index
    assert "No" in result.index


def test_numerical_analysis_tenure_mean():
    result = ChurnEDA().numerical_by_churn(_sample())

    assert result.loc["Yes", ("tenure", "mean")] == 6.0
    assert result.loc["No", ("tenure", "mean")] == 50.0


def test_numerical_analysis_monthly_charges():
    result = ChurnEDA().numerical_by_churn(_sample())

    assert result.loc["Yes", ("MonthlyCharges", "mean")] == 90.0
    assert result.loc["No", ("MonthlyCharges", "mean")] == 45.0


def test_tenure_segments_preserve_customers():
    result = ChurnEDA().tenure_segments(_sample())

    assert int(result["customers"].sum()) == 4


def test_tenure_segments_identify_early_customers():
    result = ChurnEDA().tenure_segments(_sample())

    early = result[
        result["tenure_segment"] == "0-12 months"
    ].iloc[0]

    assert early["customers"] == 2
    assert early["churned"] == 2
    assert early["churn_rate"] == 1.0