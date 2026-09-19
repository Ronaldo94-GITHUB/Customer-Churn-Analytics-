from dataclasses import dataclass
from enum import Enum

from customer_churn.analytics.risk import (
    ChurnRiskAssessment,
    ChurnRiskLevel,
)


class RetentionPriority(str, Enum):
    MONITOR = "MONITOR"
    ENGAGE = "ENGAGE"
    URGENT = "URGENT"


@dataclass(frozen=True)
class RetentionAssessment:
    probability: float
    risk_level: ChurnRiskLevel
    priority: RetentionPriority


class ChurnRetentionAdvisor:
    """Translate churn risk into an operational retention priority."""

    def assess(
        self,
        risk: ChurnRiskAssessment,
    ) -> RetentionAssessment:
        if risk.level == ChurnRiskLevel.HIGH:
            priority = RetentionPriority.URGENT
        elif risk.level == ChurnRiskLevel.MEDIUM:
            priority = RetentionPriority.ENGAGE
        else:
            priority = RetentionPriority.MONITOR

        return RetentionAssessment(
            probability=risk.probability,
            risk_level=risk.level,
            priority=priority,
        )