import pytest

from customer_churn.models.evaluation import (
    ChurnModelEvaluator,
    ChurnModelMetrics,
)


def _evaluation():
    y_true = [0, 0, 1, 1]
    predictions = [0, 1, 0, 1]
    probabilities = [0.1, 0.8, 0.4, 0.9]

    return ChurnModelEvaluator().evaluate(
        y_true,
        predictions,
        probabilities,
    )


def test_evaluator_returns_metrics_object():
    result = _evaluation()

    assert isinstance(result, ChurnModelMetrics)


def test_evaluator_calculates_confusion_matrix():
    result = _evaluation()

    assert result.true_negative == 1
    assert result.false_positive == 1
    assert result.false_negative == 1
    assert result.true_positive == 1


def test_evaluator_calculates_classification_metrics():
    result = _evaluation()

    assert result.accuracy == pytest.approx(0.5)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)


def test_evaluator_calculates_roc_auc():
    result = _evaluation()

    assert result.roc_auc == pytest.approx(0.75)


def test_evaluator_handles_zero_positive_predictions():
    result = ChurnModelEvaluator().evaluate(
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0.1, 0.2, 0.3, 0.4],
    )

    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0