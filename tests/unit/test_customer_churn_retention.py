import pytest

from customer_churn.analytics.retention import (
    ChurnRetentionAdvisor,
    RetentionPriority,
)
from customer_churn.analytics.risk import (
    ChurnRiskAssessment,
    ChurnRiskLevel,
)


@pytest.mark.parametrize(
    ("risk_level", "expected_priority"),
    [
        (ChurnRiskLevel.LOW, RetentionPriority.MONITOR),
        (ChurnRiskLevel.MEDIUM, RetentionPriority.ENGAGE),
        (ChurnRiskLevel.HIGH, RetentionPriority.URGENT),
    ],
)
def test_retention_priority_matches_risk(
    risk_level,
    expected_priority,
):
    risk = ChurnRiskAssessment(
        probability=0.50,
        level=risk_level,
    )

    result = ChurnRetentionAdvisor().assess(risk)

    assert result.priority == expected_priority


def test_retention_assessment_preserves_probability():
    risk = ChurnRiskAssessment(
        probability=0.73,
        level=ChurnRiskLevel.HIGH,
    )

    result = ChurnRetentionAdvisor().assess(risk)

    assert result.probability == pytest.approx(0.73)


def test_retention_assessment_preserves_risk_level():
    risk = ChurnRiskAssessment(
        probability=0.45,
        level=ChurnRiskLevel.MEDIUM,
    )

    result = ChurnRetentionAdvisor().assess(risk)

    assert result.risk_level == ChurnRiskLevel.MEDIUM