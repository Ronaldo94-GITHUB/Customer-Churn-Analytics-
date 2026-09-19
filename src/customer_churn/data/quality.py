from dataclasses import dataclass

import pandas as pd

from .contract import DEFAULT_CHURN_CONTRACT, ChurnDatasetContract


@dataclass(frozen=True)
class ChurnDataQualityReport:
    rows: int
    columns: int
    duplicate_rows: int
    duplicate_customer_ids: int
    missing_values: int
    invalid_total_charges: int
    invalid_target_values: int
    churn_yes: int
    churn_no: int

    @property
    def is_valid(self) -> bool:
        return (
            self.duplicate_customer_ids == 0
            and self.invalid_target_values == 0
        )


class ChurnDataQualityValidator:
    """Evaluate data quality before customer churn preprocessing."""

    def __init__(
        self,
        contract: ChurnDatasetContract = DEFAULT_CHURN_CONTRACT,
    ) -> None:
        self.contract = contract

    def analyze(
        self,
        dataframe: pd.DataFrame,
    ) -> ChurnDataQualityReport:
        customer_id = self.contract.customer_id
        target = self.contract.target

        duplicate_rows = int(dataframe.duplicated().sum())

        duplicate_customer_ids = int(
            dataframe[customer_id].duplicated().sum()
        )

        missing_values = int(
            dataframe.isna().sum().sum()
        )

        total_charges = pd.to_numeric(
            dataframe["TotalCharges"],
            errors="coerce",
        )

        invalid_total_charges = int(
            (
                total_charges.isna()
                & dataframe["TotalCharges"].notna()
            ).sum()
        )

        valid_targets = {"Yes", "No"}

        invalid_target_values = int(
            (~dataframe[target].isin(valid_targets)).sum()
        )

        target_counts = dataframe[target].value_counts()

        return ChurnDataQualityReport(
            rows=len(dataframe),
            columns=len(dataframe.columns),
            duplicate_rows=duplicate_rows,
            duplicate_customer_ids=duplicate_customer_ids,
            missing_values=missing_values,
            invalid_total_charges=invalid_total_charges,
            invalid_target_values=invalid_target_values,
            churn_yes=int(target_counts.get("Yes", 0)),
            churn_no=int(target_counts.get("No", 0)),
        )