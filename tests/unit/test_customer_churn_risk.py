import pytest

from customer_churn.analytics.risk import (
    ChurnRiskLevel,
    ChurnRiskSegmenter,
)


def test_probability_below_medium_is_low():
    result = ChurnRiskSegmenter().segment(0.29)

    assert result.level == ChurnRiskLevel.LOW
    assert result.probability == pytest.approx(0.29)


def test_medium_threshold_is_medium():
    result = ChurnRiskSegmenter().segment(0.30)

    assert result.level == ChurnRiskLevel.MEDIUM


def test_probability_between_thresholds_is_medium():
    result = ChurnRiskSegmenter().segment(0.45)

    assert result.level == ChurnRiskLevel.MEDIUM


def test_high_threshold_is_high():
    result = ChurnRiskSegmenter().segment(0.60)

    assert result.level == ChurnRiskLevel.HIGH


def test_probability_above_high_is_high():
    result = ChurnRiskSegmenter().segment(0.90)

    assert result.level == ChurnRiskLevel.HIGH


@pytest.mark.parametrize(
    "probability",
    [-0.01, 1.01],
)
def test_invalid_probability_is_rejected(probability):
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        ChurnRiskSegmenter().segment(probability)


@pytest.mark.parametrize(
    ("medium_threshold", "high_threshold"),
    [
        (0.60, 0.30),
        (0.50, 0.50),
        (-0.10, 0.60),
        (0.30, 1.10),
    ],
)
def test_invalid_thresholds_are_rejected(
    medium_threshold,
    high_threshold,
):
    with pytest.raises(
        ValueError,
        match="0 <= medium < high <= 1",
    ):
        ChurnRiskSegmenter(
            medium_threshold=medium_threshold,
            high_threshold=high_threshold,
        )