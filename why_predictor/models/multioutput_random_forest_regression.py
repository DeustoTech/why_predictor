"""Random Forest Regression model"""
import logging
from typing import Any

from sklearn.ensemble import RandomForestRegressor  # type: ignore

from .abstract_model import MultioutputModel

logger = logging.getLogger("logger")


class MultioutputRandomForestRegressionModel(MultioutputModel):
    """Multioutput Random Forest Regression Class"""

    name = "Multioutput Random Forest Regression"

    def generate_model(self) -> Any:
        model = RandomForestRegressor()
        multi_rf_linear_model = model.fit(
            self.train_features, self.train_output
        )
        return multi_rf_linear_model
