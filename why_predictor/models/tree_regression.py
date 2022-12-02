"""Decission Tree Regression model"""
import logging
from typing import Any

from sklearn.tree import DecisionTreeRegressor  # type: ignore

from .abstract_model import ChainedModel

logger = logging.getLogger("logger")


class DecissionTreeRegressionModel(ChainedModel):
    """Decission Tree Regression Class"""

    name = "Decission Tree Regression"

    def generate_model(self) -> Any:
        model = DecisionTreeRegressor()
        dt_model = model.fit(self.train_features, self.train_output.iloc[:, 1])
        return dt_model
