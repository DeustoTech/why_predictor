"""Linear Regression model"""
import logging
from typing import Any

from sklearn.neighbors import KNeighborsRegressor  # type: ignore

from .abstract_model import MultioutputModel

logger = logging.getLogger("logger")


class MultioutputKNNRegressionModel(MultioutputModel):
    """Multioutput KNN Regression Class"""

    name = "Multioutput KNN Regression"

    def generate_model(self) -> Any:
        model = KNeighborsRegressor()
        multi_knn_model = model.fit(self.train_features, self.train_output)
        return multi_knn_model
