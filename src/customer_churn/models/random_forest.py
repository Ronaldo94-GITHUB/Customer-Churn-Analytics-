from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from customer_churn.features.preprocessor import (
    ChurnPreprocessorBuilder,
)


class ChurnRandomForestBuilder:
    """Build the Random Forest churn model."""

    def __init__(
        self,
        n_estimators: int = 300,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state

    def build(self) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "preprocessor",
                    ChurnPreprocessorBuilder().build(),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )