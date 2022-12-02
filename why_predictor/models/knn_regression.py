"""Linear Regression model"""
import logging
from typing import Any

from sklearn.neighbors import KNeighborsRegressor  # type: ignore

from .abstract_model import ChainedModel

logger = logging.getLogger("logger")


class KNNRegressionModel(ChainedModel):
    """KNN Regression Class"""

    name = "KNN Regression"

    def generate_model(self) -> Any:
        model = KNeighborsRegressor()
        knn_model = model.fit(
            self.train_features, self.train_output.iloc[:, 1]
        )
        return knn_model
