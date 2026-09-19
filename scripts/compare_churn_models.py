import pandas as pd

from customer_churn.features.builder import (
    ChurnFeatureBuilder,
)
from customer_churn.features.splitter import (
    ChurnDatasetSplitter,
)
from customer_churn.models.evaluation import (
    ChurnModelEvaluator,
)
from customer_churn.models.logistic import (
    ChurnLogisticRegressionBuilder,
)
from customer_churn.models.random_forest import (
    ChurnRandomForestBuilder,
)
from customer_churn.models.xgboost_model import (
    ChurnXGBoostBuilder,
)

df = pd.read_csv(
    "data/processed/telco_customer_churn_clean.csv"
)

dataset = ChurnFeatureBuilder().build(df)
split = ChurnDatasetSplitter().split(dataset)

models = {
    "Logistic Regression": ChurnLogisticRegressionBuilder().build(),
    "Random Forest": ChurnRandomForestBuilder().build(),
    "XGBoost": ChurnXGBoostBuilder().build(),
}

evaluator = ChurnModelEvaluator()

print()
print("=" * 88)
print("CUSTOMER CHURN - MODEL COMPARISON")
print("=" * 88)

for name, model in models.items():
    model.fit(split.x_train, split.y_train)

    predictions = model.predict(split.x_test)
    probabilities = model.predict_proba(split.x_test)[:, 1]

    metrics = evaluator.evaluate(
        split.y_test,
        predictions,
        probabilities,
    )

    print()
    print(name)
    print("-" * 50)
    print(f"Accuracy : {metrics.accuracy:.4f}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall   : {metrics.recall:.4f}")
    print(f"F1       : {metrics.f1:.4f}")
    print(f"ROC-AUC  : {metrics.roc_auc:.4f}")
    print(
        "Confusion:",
        f"TN={metrics.true_negative}",
        f"FP={metrics.false_positive}",
        f"FN={metrics.false_negative}",
        f"TP={metrics.true_positive}",
    )

print()
print("TRAIN_ROWS=", len(split.x_train))
print("TEST_ROWS=", len(split.x_test))
print("CHURN_MODEL_COMPARISON=PASS")
