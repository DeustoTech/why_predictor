"""Stochastic Gradient Descent Regression model"""
import logging

import pandas as pd  # type: ignore
from sklearn.linear_model import SGDRegressor  # type: ignore
from sklearn.multioutput import (  # type: ignore
    MultiOutputRegressor,
    RegressorChain,
)

from .abstract_model import NUM_HEADERS, MultioutputModel, ShiftedModel

logger = logging.getLogger("logger")


class ShiftedStochasticGradientDescentRegressor(ShiftedModel):
    """Shifted Stochastic Gradient Descent Regressor"""

    name = "Shifted Stochastic Gradient Descent Regressor"
    short_name = "SHIFT_SGD"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        # We train with only the column for the first hour
        shifted_sgd_model = SGDRegressor(**self.hyperparams)
        self._model = shifted_sgd_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.iloc[:, NUM_HEADERS],
        )


class ChainedStochasticGradientDescentRegressor(MultioutputModel):
    """Chained Stochastic Gradient Descent Regressor"""

    name = "Chained Stochastic Gradient Descent Regressor"
    short_name = "CHAIN_SGD"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        # We train with only the column for the first hour
        chained_sgd_model = RegressorChain(SGDRegressor(**self.hyperparams))
        self._model = chained_sgd_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.drop(["dataset", "timeseries"], axis=1),
        )


class MultioutputSGDRegressor(MultioutputModel):
    """Multioutput SGD Regressor"""

    name = "Multioutput SGD Regression"
    short_name = "MULTI_SGD"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        multi_sgd_model = MultiOutputRegressor(
            SGDRegressor(**self.hyperparams)
        )
        self._model = multi_sgd_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.drop(["dataset", "timeseries"], axis=1),
        )
