from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


@dataclass(frozen=True)
class ChurnShapExplanation:
    values: np.ndarray
    feature_names: tuple[str, ...]
    transformed_features: pd.DataFrame


@dataclass(frozen=True)
class ChurnShapFeatureImportance:
    feature: str
    mean_absolute_shap: float


@dataclass(frozen=True)
class ChurnShapFactor:
    feature: str
    shap_value: float
    feature_value: float


class ChurnShapExplainer:
    """Generate SHAP explanations for a fitted XGBoost churn pipeline."""

    def explain(
        self,
        pipeline: Pipeline,
        features: pd.DataFrame,
    ) -> ChurnShapExplanation:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        if not isinstance(model, XGBClassifier):
            raise TypeError(
                "SHAP explainer requires an XGBClassifier model."
            )

        transformed = preprocessor.transform(features)
        feature_names = tuple(
            preprocessor.get_feature_names_out()
        )

        transformed_df = pd.DataFrame(
            transformed,
            columns=feature_names,
            index=features.index,
        )

        explainer = shap.TreeExplainer(model)
        shap_values = np.asarray(
            explainer.shap_values(transformed_df)
        )

        return ChurnShapExplanation(
            values=shap_values,
            feature_names=feature_names,
            transformed_features=transformed_df,
        )

    def customer_factors(
        self,
        explanation: ChurnShapExplanation,
        row_position: int,
        top_n: int = 5,
    ) -> tuple[ChurnShapFactor, ...]:
        if not 0 <= row_position < len(
            explanation.transformed_features
        ):
            raise IndexError(
                "Customer row position is out of range."
            )

        if top_n < 1:
            raise ValueError(
                "top_n must be greater than zero."
            )

        row_values = explanation.values[row_position]
        feature_values = explanation.transformed_features.iloc[
            row_position
        ].to_numpy()

        factors = [
            ChurnShapFactor(
                feature=feature,
                shap_value=float(shap_value),
                feature_value=float(feature_value),
            )
            for feature, shap_value, feature_value in zip(
                explanation.feature_names,
                row_values,
                feature_values,
                strict=True,
            )
        ]

        factors.sort(
            key=lambda item: abs(item.shap_value),
            reverse=True,
        )

        return tuple(factors[:top_n])
    def global_importance(
        self,
        explanation: ChurnShapExplanation,
    ) -> tuple[ChurnShapFeatureImportance, ...]:
        importance = np.abs(explanation.values).mean(axis=0)

        ranking = [
            ChurnShapFeatureImportance(
                feature=feature,
                mean_absolute_shap=float(value),
            )
            for feature, value in zip(
                explanation.feature_names,
                importance,
                strict=True,
            )
        ]

        return tuple(
            sorted(
                ranking,
                key=lambda item: item.mean_absolute_shap,
                reverse=True,
            )
        )