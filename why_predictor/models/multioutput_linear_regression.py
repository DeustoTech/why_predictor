"""Linear Regression model"""
import logging
from typing import Any

from sklearn.linear_model import LinearRegression  # type: ignore

from .abstract_model import MultioutputModel

logger = logging.getLogger("logger")


class MultioutputLinearRegressionModel(MultioutputModel):
    """Multioutput Linear Regression Class"""

    name = "Multioutput Linear Regression"

    def generate_model(self) -> Any:
        model = LinearRegression()
        multi_linear_model = model.fit(self.train_features, self.train_output)
        return multi_linear_model
