import numpy as np
import pandas as pd
import pytest

from customer_churn.models.explainability import (
    ChurnShapExplainer,
    ChurnShapExplanation,
)


def build_explanation():
    transformed = pd.DataFrame(
        [
            [1.0, 10.0, 0.0],
            [2.0, 20.0, 1.0],
        ],
        columns=["feature_a", "feature_b", "feature_c"],
    )

    values = np.array(
        [
            [0.10, -0.80, 0.30],
            [-0.20, 0.40, 0.90],
        ]
    )

    return ChurnShapExplanation(
        values=values,
        feature_names=(
            "feature_a",
            "feature_b",
            "feature_c",
        ),
        transformed_features=transformed,
    )


def test_customer_factors_returns_top_absolute_shap_values():
    explanation = build_explanation()

    factors = ChurnShapExplainer().customer_factors(
        explanation,
        row_position=0,
        top_n=2,
    )

    assert len(factors) == 2
    assert factors[0].feature == "feature_b"
    assert factors[0].shap_value == pytest.approx(-0.80)
    assert factors[1].feature == "feature_c"
    assert factors[1].shap_value == pytest.approx(0.30)


def test_customer_factors_preserves_feature_value():
    explanation = build_explanation()

    factors = ChurnShapExplainer().customer_factors(
        explanation,
        row_position=1,
        top_n=1,
    )

    assert factors[0].feature == "feature_c"
    assert factors[0].feature_value == pytest.approx(1.0)


def test_customer_factors_preserves_shap_sign():
    explanation = build_explanation()

    factors = ChurnShapExplainer().customer_factors(
        explanation,
        row_position=0,
        top_n=1,
    )

    assert factors[0].shap_value < 0


def test_customer_factors_rejects_invalid_row():
    explanation = build_explanation()

    with pytest.raises(
        IndexError,
        match="out of range",
    ):
        ChurnShapExplainer().customer_factors(
            explanation,
            row_position=99,
        )


def test_customer_factors_rejects_invalid_top_n():
    explanation = build_explanation()

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        ChurnShapExplainer().customer_factors(
            explanation,
            row_position=0,
            top_n=0,
        )