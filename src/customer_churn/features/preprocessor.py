from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from customer_churn.data.contract import (
    DEFAULT_CHURN_CONTRACT,
    ChurnDatasetContract,
)


class ChurnPreprocessorBuilder:
    """Build preprocessing for customer churn ML features."""

    def __init__(
        self,
        contract: ChurnDatasetContract = DEFAULT_CHURN_CONTRACT,
    ) -> None:
        self.contract = contract

    def build(self) -> ColumnTransformer:
        numerical_columns = list(
            self.contract.numeric_columns
        )

        categorical_columns = list(
            self.contract.categorical_columns
        )

        return ColumnTransformer(
            transformers=[
                (
                    "numerical",
                    StandardScaler(),
                    numerical_columns,
                ),
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                    categorical_columns,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )