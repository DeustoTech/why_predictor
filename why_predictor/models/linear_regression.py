"""Linear Regression model"""
import logging
from typing import Any

from sklearn.linear_model import LinearRegression  # type: ignore

from .abstract_model import ChainedModel

logger = logging.getLogger("logger")


class LinearRegressionModel(ChainedModel):
    """Linear Regression Class"""

    name = "Linear Regression"

    def generate_model(self) -> Any:
        model = LinearRegression()
        linear_model = model.fit(
            self.train_features, self.train_output.iloc[:, 1]
        )
        return linear_model
