import numpy as np
import pandas as pd
import pytest

from customer_churn.models.explainability import (
    ChurnShapExplainer,
    ChurnShapExplanation,
    ChurnShapFeatureImportance,
)


def _explanation():
    return ChurnShapExplanation(
        values=np.array(
            [
                [1.0, -3.0, 0.5],
                [-1.0, 1.0, -0.5],
            ]
        ),
        feature_names=("tenure", "Contract_Month-to-month", "MonthlyCharges"),
        transformed_features=pd.DataFrame(
            {
                "tenure": [0.1, -0.2],
                "Contract_Month-to-month": [1.0, 0.0],
                "MonthlyCharges": [0.3, -0.4],
            }
        ),
    )


def test_global_importance_returns_feature_objects():
    ranking = ChurnShapExplainer().global_importance(
        _explanation()
    )

    assert all(
        isinstance(item, ChurnShapFeatureImportance)
        for item in ranking
    )


def test_global_importance_contains_all_features():
    explanation = _explanation()

    ranking = ChurnShapExplainer().global_importance(
        explanation
    )

    assert len(ranking) == len(explanation.feature_names)


def test_global_importance_orders_features_descending():
    ranking = ChurnShapExplainer().global_importance(
        _explanation()
    )

    values = [
        item.mean_absolute_shap
        for item in ranking
    ]

    assert values == sorted(values, reverse=True)


def test_global_importance_identifies_expected_top_feature():
    ranking = ChurnShapExplainer().global_importance(
        _explanation()
    )

    assert ranking[0].feature == "Contract_Month-to-month"
    assert ranking[0].mean_absolute_shap == pytest.approx(2.0)


def test_global_importance_calculates_mean_absolute_values():
    ranking = ChurnShapExplainer().global_importance(
        _explanation()
    )

    by_feature = {
        item.feature: item.mean_absolute_shap
        for item in ranking
    }

    assert by_feature["tenure"] == pytest.approx(1.0)
    assert by_feature["MonthlyCharges"] == pytest.approx(0.5)