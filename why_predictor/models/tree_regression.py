"""Linear Regression model"""
import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd  # type: ignore
from sklearn.tree import DecisionTreeRegressor  # type: ignore

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
) -> Tuple[Dict[str, Any], Any]:
    """Fit Decision Tree Regression Model"""
    logger.debug("Calculating Decision Tree regression...")
    tree_model = generate_model(train_features, train_output)
    predictions = tree_model.predict(test_features)
    logger.debug("Accuracy: %r", tree_model.score(test_features, test_output))
    mape = np.absolute((test_output - predictions) / test_output).sum() / len(
        test_output
    )
    logger.info("MAPE decision tree regression:\n%r", mape)
    return {}, mape
