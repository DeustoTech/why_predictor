"""Linear Regression model"""
import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore

logger = logging.getLogger("logger")


def generate_model(
    train_features: pd.DataFrame,
    train_output: pd.DataFrame,
):
    """Generate Linear Regression model"""
    model = LinearRegression()
    linear_model = model.fit(train_features, train_output)
    return linear_model


def fit(
    train_features: pd.DataFrame,
    train_output: pd.DataFrame,
    test_features: pd.DataFrame,
    test_output: pd.DataFrame,
) -> Tuple[Dict[str, Any], Any]:
    """Fit Linear Regression Model"""
    logger.debug("Calculating linear regression...")
    linear_model = generate_model(train_features, train_output)
    predictions = linear_model.predict(test_features)
    logger.debug(
        "Accuracy: %r", linear_model.score(test_features, test_output)
    )
    mape = np.absolute((test_output - predictions) / test_output).sum() / len(
        test_output
    )
    logger.info("MAPE linear regression:\n%r", mape)
    return {}, mape
