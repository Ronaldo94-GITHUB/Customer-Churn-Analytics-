import pandas as pd
import pytest

from customer_churn.analytics.sql_analytics import (
    ChurnSQLAnalytics,
)


def test_load_and_query_dataframe():
    data = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "Churn": ["No", "Yes", "No"],
        }
    )

    analytics = ChurnSQLAnalytics()

    try:
        analytics.load_dataframe(data)

        result = analytics.query(
            "SELECT COUNT(*) AS total FROM customer_churn"
        )

        assert int(result.loc[0, "total"]) == 3
    finally:
        analytics.close()


def test_query_can_filter_churned_customers():
    data = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "Churn": ["No", "Yes", "Yes"],
        }
    )

    analytics = ChurnSQLAnalytics()

    try:
        analytics.load_dataframe(data)

        result = analytics.query(
            """
            SELECT customer_id
            FROM customer_churn
            WHERE Churn = 'Yes'
            ORDER BY customer_id
            """
        )

        assert result["customer_id"].tolist() == ["C002", "C003"]
    finally:
        analytics.close()


def test_query_can_group_customers():
    data = pd.DataFrame(
        {
            "Contract": [
                "Month-to-month",
                "Month-to-month",
                "One year",
            ],
            "Churn": ["Yes", "No", "No"],
        }
    )

    analytics = ChurnSQLAnalytics()

    try:
        analytics.load_dataframe(data)

        result = analytics.query(
            """
            SELECT Contract, COUNT(*) AS customers
            FROM customer_churn
            GROUP BY Contract
            ORDER BY customers DESC
            """
        )

        assert result.loc[0, "Contract"] == "Month-to-month"
        assert int(result.loc[0, "customers"]) == 2
    finally:
        analytics.close()


def test_load_replaces_existing_table():
    analytics = ChurnSQLAnalytics()

    try:
        analytics.load_dataframe(
            pd.DataFrame({"value": [1, 2, 3]})
        )

        analytics.load_dataframe(
            pd.DataFrame({"value": [10, 20]})
        )

        result = analytics.query(
            "SELECT COUNT(*) AS total FROM customer_churn"
        )

        assert int(result.loc[0, "total"]) == 2
    finally:
        analytics.close()


def test_empty_dataframe_is_rejected():
    analytics = ChurnSQLAnalytics()

    try:
        with pytest.raises(ValueError):
            analytics.load_dataframe(pd.DataFrame())
    finally:
        analytics.close()


def test_empty_sql_query_is_rejected():
    analytics = ChurnSQLAnalytics()

    try:
        with pytest.raises(ValueError):
            analytics.query("   ")
    finally:
        analytics.close()