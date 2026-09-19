from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ChurnModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


class ChurnModelEvaluator:
    """Evaluate binary customer churn predictions."""

    def evaluate(
        self,
        y_true,
        predictions,
        probabilities,
    ) -> ChurnModelMetrics:
        matrix = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        )

        tn, fp, fn, tp = matrix.ravel()

        return ChurnModelMetrics(
            accuracy=float(
                accuracy_score(y_true, predictions)
            ),
            precision=float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            recall=float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            f1=float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            roc_auc=float(
                roc_auc_score(
                    y_true,
                    np.asarray(probabilities),
                )
            ),
            true_negative=int(tn),
            false_positive=int(fp),
            false_negative=int(fn),
            true_positive=int(tp),
        )