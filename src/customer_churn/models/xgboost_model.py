from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from customer_churn.features.preprocessor import (
    ChurnPreprocessorBuilder,
)


class ChurnXGBoostBuilder:
    """Build the XGBoost customer churn model."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def build(self) -> Pipeline:
        model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
        )

        return Pipeline(
            steps=[
                (
                    "preprocessor",
                    ChurnPreprocessorBuilder().build(),
                ),
                ("model", model),
            ]
        )