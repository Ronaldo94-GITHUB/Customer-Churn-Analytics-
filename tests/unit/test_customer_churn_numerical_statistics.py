import pandas as pd
import pytest

from customer_churn.analytics.statistics import (
    ChurnStatisticalAnalyzer,
)


def test_numerical_association_returns_expected_metadata():
    data = pd.DataFrame(
        {
            "tenure": [1, 2, 3, 10, 20, 30],
            "Churn": ["Yes", "Yes", "Yes", "No", "No", "No"],
        }
    )

    result = ChurnStatisticalAnalyzer().numerical_association(
        data,
        "tenure",
    )

    assert result.feature == "tenure"
    assert result.group_yes_size == 3
    assert result.group_no_size == 3
    assert result.group_yes_median == pytest.approx(2.0)
    assert result.group_no_median == pytest.approx(20.0)


def test_numerical_association_returns_valid_statistics():
    data = pd.DataFrame(
        {
            "value": [1, 2, 3, 4, 5, 6],
            "Churn": ["Yes", "Yes", "Yes", "No", "No", "No"],
        }
    )

    result = ChurnStatisticalAnalyzer().numerical_association(
        data,
        "value",
    )

    assert result.u_statistic >= 0.0
    assert 0.0 <= result.p_value <= 1.0
    assert -1.0 <= result.rank_biserial <= 1.0


def test_lower_values_for_churn_group_produce_negative_effect():
    data = pd.DataFrame(
        {
            "value": [1, 2, 3, 10, 20, 30],
            "Churn": ["Yes", "Yes", "Yes", "No", "No", "No"],
        }
    )

    result = ChurnStatisticalAnalyzer().numerical_association(
        data,
        "value",
    )

    assert result.rank_biserial < 0.0


def test_higher_values_for_churn_group_produce_positive_effect():
    data = pd.DataFrame(
        {
            "value": [10, 20, 30, 1, 2, 3],
            "Churn": ["Yes", "Yes", "Yes", "No", "No", "No"],
        }
    )

    result = ChurnStatisticalAnalyzer().numerical_association(
        data,
        "value",
    )

    assert result.rank_biserial > 0.0


def test_missing_feature_raises_key_error():
    data = pd.DataFrame(
        {
            "Churn": ["Yes", "No"],
        }
    )

    with pytest.raises(
        KeyError,
        match="Feature not found",
    ):
        ChurnStatisticalAnalyzer().numerical_association(
            data,
            "tenure",
        )


def test_missing_target_raises_key_error():
    data = pd.DataFrame(
        {
            "tenure": [1, 2],
        }
    )

    with pytest.raises(
        KeyError,
        match="Target not found",
    ):
        ChurnStatisticalAnalyzer().numerical_association(
            data,
            "tenure",
        )


def test_empty_churn_group_raises_value_error():
    data = pd.DataFrame(
        {
            "tenure": [1, 2, 3],
            "Churn": ["No", "No", "No"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Both churn groups",
    ):
        ChurnStatisticalAnalyzer().numerical_association(
            data,
            "tenure",
        )