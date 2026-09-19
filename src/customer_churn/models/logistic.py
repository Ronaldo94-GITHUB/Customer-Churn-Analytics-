from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from customer_churn.features.preprocessor import (
    ChurnPreprocessorBuilder,
)


class ChurnLogisticRegressionBuilder:
    """Build the baseline Logistic Regression churn model."""

    def __init__(
        self,
        random_state: int = 42,
        max_iter: int = 1000,
    ) -> None:
        self.random_state = random_state
        self.max_iter = max_iter

    def build(self) -> Pipeline:
        preprocessor = ChurnPreprocessorBuilder().build()

        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=self.max_iter,
        )

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )