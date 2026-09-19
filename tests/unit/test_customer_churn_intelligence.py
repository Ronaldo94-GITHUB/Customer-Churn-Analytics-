import pandas as pd
import pytest

from customer_churn.analytics.intelligence import (
    ChurnIntelligenceBuilder,
)


def test_intelligence_builder_creates_expected_columns():
    result = ChurnIntelligenceBuilder().build(
        customer_ids=["C001"],
        probabilities=[0.75],
    )

    assert list(result.columns) == [
        "customer_id",
        "churn_probability",
        "risk_level",
        "retention_priority",
    ]


def test_intelligence_builder_maps_all_risk_levels():
    result = ChurnIntelligenceBuilder().build(
        customer_ids=["C001", "C002", "C003"],
        probabilities=[0.10, 0.45, 0.75],
    )

    assert result["risk_level"].tolist() == [
        "LOW",
        "MEDIUM",
        "HIGH",
    ]

    assert result["retention_priority"].tolist() == [
        "MONITOR",
        "ENGAGE",
        "URGENT",
    ]


def test_intelligence_builder_preserves_customer_ids():
    result = ChurnIntelligenceBuilder().build(
        customer_ids=["C001", "C002"],
        probabilities=[0.20, 0.80],
    )

    assert result["customer_id"].tolist() == [
        "C001",
        "C002",
    ]


def test_intelligence_builder_preserves_probabilities():
    result = ChurnIntelligenceBuilder().build(
        customer_ids=["C001", "C002"],
        probabilities=[0.25, 0.73],
    )

    assert result["churn_probability"].tolist() == pytest.approx(
        [0.25, 0.73]
    )


def test_intelligence_builder_returns_dataframe():
    result = ChurnIntelligenceBuilder().build(
        customer_ids=["C001"],
        probabilities=[0.50],
    )

    assert isinstance(result, pd.DataFrame)


def test_intelligence_builder_rejects_different_lengths():
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        ChurnIntelligenceBuilder().build(
            customer_ids=["C001", "C002"],
            probabilities=[0.50],
        )


def test_intelligence_builder_handles_empty_input():
    result = ChurnIntelligenceBuilder().build(
        customer_ids=[],
        probabilities=[],
    )

    assert result.empty
    assert list(result.columns) == [
        "customer_id",
        "churn_probability",
        "risk_level",
        "retention_priority",
    ]