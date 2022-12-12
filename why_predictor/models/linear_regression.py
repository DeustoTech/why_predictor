"""Linear Regression model"""
import logging
from typing import Any, Dict

from sklearn.linear_model import LinearRegression  # type: ignore

from .abstract_model import BasicModel, ChainedModel, MultioutputModel

logger = logging.getLogger("logger")


class LinearRegressionModel(BasicModel):
    """Linear Regression Class"""

    params: Dict[str, Any] = {}

    def generate_hyperparams(self) -> None:
        """Generate hyperparams"""
        self.generate_hyperparams_objects([{}])


class LinearRegressor(LinearRegressionModel, ChainedModel):
    """Chained Linear Regressor"""

    name = "Linear Regression"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        # We train with only the column for the first hour
        model = LinearRegression(**hyper_params, n_jobs=-1)
        linear_model = model.fit(
            self.train_features.drop("timeseries", axis=1),
            self.train_output.iloc[:, 1],
        )
        return linear_model


class MultioutputLinearRegressor(LinearRegressionModel, MultioutputModel):
    """Multioutput Linear Regressor"""

    name = "Multioutput Linear Regression"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        model = LinearRegression(**hyper_params, n_jobs=-1)
        multi_linear_model = model.fit(
            self.train_features.drop("timeseries", axis=1),
            self.train_output.drop("timeseries", axis=1),
        )
        return multi_linear_model
