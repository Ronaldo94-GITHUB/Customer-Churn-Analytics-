import pandas as pd
import pytest

from customer_churn.data.cleaner import (
    ChurnDataCleaner,
    ChurnDataCleaningError,
)


def test_cleaner_converts_total_charges_to_numeric():
    dataframe = pd.DataFrame(
        {
            "tenure": [10, 20],
            "TotalCharges": ["100.50", "250.75"],
        }
    )

    cleaned = ChurnDataCleaner().clean(dataframe)

    assert cleaned["TotalCharges"].dtype.kind == "f"
    assert cleaned["TotalCharges"].tolist() == [100.50, 250.75]


def test_cleaner_fills_empty_total_charges_for_zero_tenure():
    dataframe = pd.DataFrame(
        {
            "tenure": [0],
            "TotalCharges": [" "],
        }
    )

    cleaned = ChurnDataCleaner().clean(dataframe)

    assert cleaned.loc[0, "TotalCharges"] == 0.0


def test_cleaner_preserves_rows():
    dataframe = pd.DataFrame(
        {
            "tenure": [0, 12, 24],
            "TotalCharges": [" ", "100.00", "500.00"],
        }
    )

    cleaned = ChurnDataCleaner().clean(dataframe)

    assert len(cleaned) == 3


def test_cleaner_does_not_mutate_original_dataframe():
    dataframe = pd.DataFrame(
        {
            "tenure": [0],
            "TotalCharges": [" "],
        }
    )

    ChurnDataCleaner().clean(dataframe)

    assert dataframe.loc[0, "TotalCharges"] == " "


def test_cleaner_rejects_unresolved_total_charges():
    dataframe = pd.DataFrame(
        {
            "tenure": [5],
            "TotalCharges": ["invalid"],
        }
    )

    with pytest.raises(
        ChurnDataCleaningError,
        match="1 TotalCharges",
    ):
        ChurnDataCleaner().clean(dataframe)