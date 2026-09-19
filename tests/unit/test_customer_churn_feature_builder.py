import pandas as pd
import pytest

from customer_churn.features.builder import (
    ChurnFeatureBuilder,
)


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["A", "B", "C"],
            "tenure": [5, 20, 40],
            "MonthlyCharges": [80.0, 50.0, 60.0],
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
            ],
            "Churn": ["Yes", "No", "No"],
        }
    )


def test_builder_separates_customer_ids():
    result = ChurnFeatureBuilder().build(_sample())

    assert result.customer_ids.tolist() == ["A", "B", "C"]
    assert "customerID" not in result.features.columns


def test_builder_removes_target_from_features():
    result = ChurnFeatureBuilder().build(_sample())

    assert "Churn" not in result.features.columns


def test_builder_encodes_target():
    result = ChurnFeatureBuilder().build(_sample())

    assert result.target.tolist() == [1, 0, 0]


def test_builder_preserves_feature_rows():
    result = ChurnFeatureBuilder().build(_sample())

    assert len(result.features) == 3
    assert len(result.target) == 3
    assert len(result.customer_ids) == 3


def test_builder_rejects_unknown_target():
    dataframe = _sample()
    dataframe.loc[0, "Churn"] = "Unknown"

    with pytest.raises(
        ValueError,
        match="unsupported",
    ):
        ChurnFeatureBuilder().build(dataframe)