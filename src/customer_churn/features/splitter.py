from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from customer_churn.features.builder import (
    ChurnMLDataset,
)


@dataclass(frozen=True)
class ChurnTrainTestSplit:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    customer_ids_train: pd.Series
    customer_ids_test: pd.Series


class ChurnDatasetSplitter:
    """Create a reproducible stratified train/test split."""

    def __init__(
        self,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state

    def split(
        self,
        dataset: ChurnMLDataset,
    ) -> ChurnTrainTestSplit:
        (
            x_train,
            x_test,
            y_train,
            y_test,
            customer_ids_train,
            customer_ids_test,
        ) = train_test_split(
            dataset.features,
            dataset.target,
            dataset.customer_ids,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=dataset.target,
        )

        return ChurnTrainTestSplit(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            customer_ids_train=customer_ids_train,
            customer_ids_test=customer_ids_test,
        )