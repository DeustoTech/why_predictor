"""Random Forest Regression model"""
import logging
from typing import Any

from sklearn.ensemble import RandomForestRegressor  # type: ignore

from .abstract_model import ChainedModel

logger = logging.getLogger("logger")


class RandomForestRegressionModel(ChainedModel):
    """Random Forest Regression Class"""

    name = "Random Forest Regression"

    def generate_model(self) -> Any:
        model = RandomForestRegressor()
        random_forest_model = model.fit(
            self.train_features, self.train_output.iloc[:, 1]
        )
        return random_forest_model
