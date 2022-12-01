"""Linear Regression model"""
import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd  # type: ignore
from sklearn.neighbors import KNeighborsRegressor  # type: ignore

logger = logging.getLogger("logger")


def generate_model(
    train_features: pd.DataFrame,
    train_output: pd.DataFrame,
):
    """Generate k-Nearest Neighbors Regression model"""
    model = KNeighborsRegressor()
    knn_model = model.fit(train_features, train_output)
    return knn_model


def fit(
    train_features: pd.DataFrame,
    train_output: pd.DataFrame,
    test_features: pd.DataFrame,
    test_output: pd.DataFrame,
) -> Tuple[Dict[str, Any], Any]:
    """Fit k-Nearest Neighbors Regression Model"""
    logger.debug("Calculating KNN regression...")
    knn_model = generate_model(train_features, train_output)
    predictions = knn_model.predict(test_features)
    logger.debug("Accuracy: %r", knn_model.score(test_features, test_output))
    mape = np.absolute((test_output - predictions) / test_output).sum() / len(
        test_output
    )
    logger.info("MAPE KNN regression:\n%r", mape)
    return {}, mape
