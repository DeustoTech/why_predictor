"""Linear Regression model"""
import logging
from typing import Any, Dict, Tuple

import pandas as pd  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore

from ..errors import ErrorType

logger = logging.getLogger("logger")


def generate_model(
    train_features: pd.DataFrame,
    train_output: pd.DataFrame,
):
    """Generate Random Forest Regression model"""
    model = RandomForestRegressor()
    random_forest_model = model.fit(train_features, train_output)
    return random_forest_model


def fit(
    train_features: pd.DataFrame,
    train_output: pd.DataFrame,
    test_features: pd.DataFrame,
    test_output: pd.DataFrame,
    error: ErrorType,
) -> Tuple[Dict[str, Any], Any]:
    """Fit Random Forest Regression Model"""
    logger.debug("Calculating Random Forest regression...")
    random_forest_model = generate_model(train_features, train_output)
    predictions = random_forest_model.predict(test_features)
    logger.debug(
        "Accuracy: %r", random_forest_model.score(test_features, test_output)
    )
    error_metric = error.value(test_output, predictions)
    logger.info("%s Random Forest regression:\n%r", error.name, error_metric)
    return {}, error_metric
