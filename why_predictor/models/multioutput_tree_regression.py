"""Multioutput Decission Tree Regression model"""
import logging
from typing import Any

from sklearn.tree import DecisionTreeRegressor  # type: ignore

from .abstract_model import MultioutputModel

logger = logging.getLogger("logger")


class MultioutputDecissionTreeRegressionModel(MultioutputModel):
    """Decission Tree Regression Class"""

    name = "Multioutput Decission Tree Regression"

    def generate_model(self) -> Any:
        model = DecisionTreeRegressor()
        multi_dt_model = model.fit(self.train_features, self.train_output)
        return multi_dt_model
