import pytest

from customer_churn.analytics.customer_profile import (
    CustomerRetentionProfiler,
)
from customer_churn.analytics.retention import (
    RetentionPriority,
)
from customer_churn.analytics.risk import (
    ChurnRiskLevel,
)


@pytest.mark.parametrize(
    ("probability", "risk_level", "priority"),
    [
        (0.10, ChurnRiskLevel.LOW, RetentionPriority.MONITOR),
        (0.45, ChurnRiskLevel.MEDIUM, RetentionPriority.ENGAGE),
        (0.75, ChurnRiskLevel.HIGH, RetentionPriority.URGENT),
    ],
)
def test_profile_maps_probability_to_operational_status(
    probability,
    risk_level,
    priority,
):
    profile = CustomerRetentionProfiler().build(
        customer_id="CUSTOMER-001",
        churn_probability=probability,
    )

    assert profile.risk_level == risk_level
    assert profile.retention_priority == priority


def test_profile_preserves_customer_id():
    profile = CustomerRetentionProfiler().build(
        customer_id="CUSTOMER-123",
        churn_probability=0.25,
    )

    assert profile.customer_id == "CUSTOMER-123"


def test_profile_preserves_probability():
    profile = CustomerRetentionProfiler().build(
        customer_id="CUSTOMER-123",
        churn_probability=0.73,
    )

    assert profile.churn_probability == pytest.approx(0.73)


@pytest.mark.parametrize(
    "customer_id",
    ["", " ", "   "],
)
def test_profile_rejects_empty_customer_id(customer_id):
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        CustomerRetentionProfiler().build(
            customer_id=customer_id,
            churn_probability=0.50,
        )