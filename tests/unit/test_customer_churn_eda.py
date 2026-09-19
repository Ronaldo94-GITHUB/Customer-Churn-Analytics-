import pandas as pd

from customer_churn.analytics.eda import (
    ChurnEDA,
)


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Churn": ["Yes", "No", "Yes", "No"],
            "tenure": [2, 20, 5, 40],
            "MonthlyCharges": [90.0, 50.0, 100.0, 40.0],
            "Contract": [
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Two year",
            ],
        }
    )


def test_overview_customer_counts():
    overview = ChurnEDA().overview(_sample())

    assert overview.customers == 4
    assert overview.churned_customers == 2
    assert overview.retained_customers == 2


def test_overview_churn_rate():
    overview = ChurnEDA().overview(_sample())

    assert overview.churn_rate == 0.5


def test_overview_averages():
    overview = ChurnEDA().overview(_sample())

    assert overview.average_tenure == 16.75
    assert overview.average_monthly_charges == 70.0


def test_overview_empty_dataframe():
    dataframe = _sample().iloc[0:0]

    overview = ChurnEDA().overview(dataframe)

    assert overview.customers == 0
    assert overview.churn_rate == 0.0


def test_churn_by_category():
    result = ChurnEDA().churn_by_category(
        _sample(),
        "Contract",
    )

    month = result[
        result["Contract"] == "Month-to-month"
    ].iloc[0]

    assert month["customers"] == 2
    assert month["churned"] == 2
    assert month["churn_rate"] == 1.0


def test_churn_by_category_orders_highest_risk_first():
    result = ChurnEDA().churn_by_category(
        _sample(),
        "Contract",
    )

    assert result.iloc[0]["Contract"] == "Month-to-month"