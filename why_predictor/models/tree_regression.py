"""Linear Regression model"""
import logging
from typing import Any, Dict, Tuple

import pandas as pd  # type: ignore
from sklearn.tree import DecisionTreeRegressor  # type: ignore

from ..errors import ErrorType

logger = logging.getLogger("logger")


def generate_model(
    train_features: pd.DataFrame,
    train_output: pd.DataFrame,
) -> Any:
    """Generate Decision Tree Regression model"""
    model = DecisionTreeRegressor()
    tree_model = model.fit(train_features, train_output)
    return tree_model


def fit(
    train_features: pd.DataFrame,
    train_output: pd.DataFrame,
    test_features: pd.DataFrame,
    test_output: pd.DataFrame,
    error: ErrorType,
) -> Tuple[Dict[str, Any], Any]:
    """Fit Decision Tree Regression Model"""
    logger.debug("Calculating Decision Tree regression...")
    tree_model = generate_model(train_features, train_output.iloc[:, 1])
    predictions = pd.DataFrame(tree_model.predict(test_features))
    for i in range(1, test_output.shape[1]):
        features = pd.concat([test_features.iloc[:, i:], predictions], axis=1)
        features = features.set_axis(
            [f"col{i}" for i in range(1, features.shape[1] + 1)], axis=1
        )
        predictions = pd.concat(
            [predictions, pd.Series(tree_model.predict(features))], axis=1
        )
    predictions = predictions.set_axis(test_output.columns, axis=1)
    # We cannot measure the Accuracy this way
    # logger.debug(
    #     "Accuracy: %r", linear_model.score(test_features, test_output)
    # )
    error_metric = error.value(test_output, predictions, train_features)
    logger.info("%s decision tree regression:\n%r", error.name, error_metric)
    return {}, error_metric
