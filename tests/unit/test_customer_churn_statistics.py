import pandas as pd
import pytest

from customer_churn.analytics.statistics import (
    ChurnStatisticalAnalyzer,
)


def test_categorical_association_returns_expected_metadata():
    data = pd.DataFrame(
        {
            "Contract": [
                "Monthly",
                "Monthly",
                "Yearly",
                "Yearly",
            ],
            "Churn": [
                "Yes",
                "Yes",
                "No",
                "No",
            ],
        }
    )

    result = ChurnStatisticalAnalyzer().categorical_association(
        data,
        "Contract",
    )

    assert result.feature == "Contract"
    assert result.sample_size == 4
    assert result.degrees_of_freedom == 1


def test_categorical_association_returns_valid_statistics():
    data = pd.DataFrame(
        {
            "Feature": [
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
            ],
            "Churn": [
                "Yes",
                "Yes",
                "No",
                "No",
                "No",
                "Yes",
            ],
        }
    )

    result = ChurnStatisticalAnalyzer().categorical_association(
        data,
        "Feature",
    )

    assert result.chi2 >= 0.0
    assert 0.0 <= result.p_value <= 1.0
    assert 0.0 <= result.cramers_v <= 1.0


def test_perfect_association_has_high_cramers_v():
    data = pd.DataFrame(
        {
            "Feature": (
                ["A"] * 50
                + ["B"] * 50
            ),
            "Churn": (
                ["Yes"] * 50
                + ["No"] * 50
            ),
        }
    )

    result = ChurnStatisticalAnalyzer().categorical_association(
        data,
        "Feature",
    )

    assert result.cramers_v > 0.90


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
        ChurnStatisticalAnalyzer().categorical_association(
            data,
            "Contract",
        )


def test_missing_target_raises_key_error():
    data = pd.DataFrame(
        {
            "Contract": ["Monthly", "Yearly"],
        }
    )

    with pytest.raises(
        KeyError,
        match="Target not found",
    ):
        ChurnStatisticalAnalyzer().categorical_association(
            data,
            "Contract",
        )


def test_insufficient_categories_raises_value_error():
    data = pd.DataFrame(
        {
            "Contract": [
                "Monthly",
                "Monthly",
            ],
            "Churn": [
                "Yes",
                "No",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="at least two categories",
    ):
        ChurnStatisticalAnalyzer().categorical_association(
            data,
            "Contract",
        )