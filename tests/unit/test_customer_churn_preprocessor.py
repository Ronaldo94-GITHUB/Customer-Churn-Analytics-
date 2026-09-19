import pandas as pd

from customer_churn.data.contract import (
    DEFAULT_CHURN_CONTRACT,
)
from customer_churn.features.preprocessor import (
    ChurnPreprocessorBuilder,
)


def _features() -> pd.DataFrame:
    contract = DEFAULT_CHURN_CONTRACT

    data = {}

    for column in contract.numeric_columns:
        data[column] = [1.0, 2.0, 3.0]

    for column in contract.categorical_columns:
        data[column] = ["A", "B", "A"]

    return pd.DataFrame(data)


def test_preprocessor_transforms_all_rows():
    features = _features()

    transformed = ChurnPreprocessorBuilder().build().fit_transform(
        features
    )

    assert transformed.shape[0] == 3


def test_preprocessor_expands_feature_space():
    features = _features()

    transformed = ChurnPreprocessorBuilder().build().fit_transform(
        features
    )

    assert transformed.shape[1] > len(features.columns)


def test_preprocessor_handles_unknown_categories():
    train = _features()
    test = _features().iloc[[0]].copy()

    categorical = DEFAULT_CHURN_CONTRACT.categorical_columns

    for column in categorical:
        test.loc[test.index[0], column] = "UNSEEN_CATEGORY"

    preprocessor = ChurnPreprocessorBuilder().build()
    preprocessor.fit(train)

    transformed = preprocessor.transform(test)

    assert transformed.shape[0] == 1


def test_preprocessor_returns_feature_names():
    features = _features()

    preprocessor = ChurnPreprocessorBuilder().build()
    preprocessor.fit(features)

    names = preprocessor.get_feature_names_out()

    assert "tenure" in names
    assert any(
        name.startswith("Contract_")
        for name in names
    )


def test_preprocessor_drops_unconfigured_columns():
    features = _features()
    features["unused_column"] = ["X", "Y", "Z"]

    preprocessor = ChurnPreprocessorBuilder().build()

    transformed = preprocessor.fit_transform(features)

    assert transformed.shape[1] < (
        len(features.columns) * 3
    )