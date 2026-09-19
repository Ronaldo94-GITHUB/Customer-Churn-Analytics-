import pandas as pd
import pytest

from customer_churn.data.contract import (
    DEFAULT_CHURN_CONTRACT,
)
from customer_churn.data.loader import (
    ChurnDatasetContractError,
    ChurnDatasetLoader,
    ChurnDatasetNotFoundError,
)


def _valid_dataframe() -> pd.DataFrame:
    data = {
        column: [0]
        for column in DEFAULT_CHURN_CONTRACT.required_columns
    }

    return pd.DataFrame(data)


def test_loader_reads_valid_csv(tmp_path):
    dataframe = _valid_dataframe()
    path = tmp_path / "churn.csv"
    dataframe.to_csv(path, index=False)

    loaded = ChurnDatasetLoader().load(path)

    assert len(loaded) == 1
    assert tuple(loaded.columns) == tuple(dataframe.columns)


def test_loader_rejects_missing_file(tmp_path):
    path = tmp_path / "missing.csv"

    with pytest.raises(ChurnDatasetNotFoundError):
        ChurnDatasetLoader().load(path)


def test_loader_rejects_missing_required_column(tmp_path):
    dataframe = _valid_dataframe().drop(columns=["Churn"])
    path = tmp_path / "churn.csv"
    dataframe.to_csv(path, index=False)

    with pytest.raises(
        ChurnDatasetContractError,
        match="Churn",
    ):
        ChurnDatasetLoader().load(path)


def test_validate_contract_accepts_required_columns():
    dataframe = _valid_dataframe()

    ChurnDatasetLoader().validate_contract(dataframe)


def test_validate_contract_reports_multiple_missing_columns():
    dataframe = _valid_dataframe().drop(
        columns=["Churn", "MonthlyCharges"]
    )

    with pytest.raises(ChurnDatasetContractError) as exc:
        ChurnDatasetLoader().validate_contract(dataframe)

    message = str(exc.value)

    assert "Churn" in message
    assert "MonthlyCharges" in message