from pathlib import Path

import pandas as pd

from .contract import DEFAULT_CHURN_CONTRACT, ChurnDatasetContract


class ChurnDatasetError(Exception):
    """Base exception for customer churn dataset errors."""


class ChurnDatasetNotFoundError(ChurnDatasetError):
    """Raised when the customer churn dataset cannot be found."""


class ChurnDatasetContractError(ChurnDatasetError):
    """Raised when the dataset does not satisfy the expected contract."""


class ChurnDatasetLoader:
    """Load and validate raw customer churn datasets."""

    def __init__(
        self,
        contract: ChurnDatasetContract = DEFAULT_CHURN_CONTRACT,
    ) -> None:
        self.contract = contract

    def load(self, path: str | Path) -> pd.DataFrame:
        dataset_path = Path(path)

        if not dataset_path.is_file():
            raise ChurnDatasetNotFoundError(
                f"Customer churn dataset not found: {dataset_path}"
            )

        dataframe = pd.read_csv(dataset_path)

        self.validate_contract(dataframe)

        return dataframe

    def validate_contract(self, dataframe: pd.DataFrame) -> None:
        actual_columns = set(dataframe.columns)
        required_columns = set(self.contract.required_columns)

        missing_columns = sorted(required_columns - actual_columns)

        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ChurnDatasetContractError(
                f"Missing required columns: {missing}"
            )