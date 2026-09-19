import pandas as pd

from customer_churn.data.quality import (
    ChurnDataQualityValidator,
)


def _dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["A", "B", "C"],
            "TotalCharges": ["100.00", "200.00", "300.00"],
            "Churn": ["No", "Yes", "No"],
        }
    )


def test_quality_report_counts_rows_and_columns():
    report = ChurnDataQualityValidator().analyze(_dataframe())

    assert report.rows == 3
    assert report.columns == 3


def test_quality_report_detects_duplicate_customer_ids():
    dataframe = _dataframe()
    dataframe.loc[2, "customerID"] = "A"

    report = ChurnDataQualityValidator().analyze(dataframe)

    assert report.duplicate_customer_ids == 1
    assert report.is_valid is False


def test_quality_report_detects_invalid_total_charges():
    dataframe = _dataframe()
    dataframe.loc[1, "TotalCharges"] = " "

    report = ChurnDataQualityValidator().analyze(dataframe)

    assert report.invalid_total_charges == 1


def test_quality_report_detects_invalid_target():
    dataframe = _dataframe()
    dataframe.loc[1, "Churn"] = "Unknown"

    report = ChurnDataQualityValidator().analyze(dataframe)

    assert report.invalid_target_values == 1
    assert report.is_valid is False


def test_quality_report_counts_target_distribution():
    report = ChurnDataQualityValidator().analyze(_dataframe())

    assert report.churn_yes == 1
    assert report.churn_no == 2


def test_quality_report_valid_dataset():
    report = ChurnDataQualityValidator().analyze(_dataframe())

    assert report.duplicate_customer_ids == 0
    assert report.invalid_target_values == 0
    assert report.is_valid is True