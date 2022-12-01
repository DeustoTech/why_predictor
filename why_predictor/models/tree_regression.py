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
):
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
    tree_model = generate_model(train_features, train_output)
    predictions = tree_model.predict(test_features)
    logger.debug("Accuracy: %r", tree_model.score(test_features, test_output))
    error_metric = error.value(test_output, predictions)
    logger.info("%s decision tree regression:\n%r", error.name, error_metric)
    return {}, error_metric
