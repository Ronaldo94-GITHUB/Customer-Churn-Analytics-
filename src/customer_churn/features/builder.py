from dataclasses import dataclass

import pandas as pd

from customer_churn.data.contract import (
    DEFAULT_CHURN_CONTRACT,
    ChurnDatasetContract,
)


@dataclass(frozen=True)
class ChurnMLDataset:
    features: pd.DataFrame
    target: pd.Series
    customer_ids: pd.Series


class ChurnFeatureBuilder:
    """Build the supervised ML dataset for customer churn."""

    def __init__(
        self,
        contract: ChurnDatasetContract = DEFAULT_CHURN_CONTRACT,
    ) -> None:
        self.contract = contract

    def build(
        self,
        dataframe: pd.DataFrame,
    ) -> ChurnMLDataset:
        target_column = self.contract.target
        customer_id_column = self.contract.customer_id

        target = dataframe[target_column].map(
            {
                "No": 0,
                "Yes": 1,
            }
        )

        if target.isna().any():
            raise ValueError(
                "Churn target contains unsupported values."
            )

        target = target.astype(int)

        customer_ids = dataframe[
            customer_id_column
        ].copy()

        features = dataframe.drop(
            columns=[
                customer_id_column,
                target_column,
            ]
        ).copy()

        return ChurnMLDataset(
            features=features,
            target=target,
            customer_ids=customer_ids,
        )