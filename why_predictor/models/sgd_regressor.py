"""Stochastic Gradient Descent Regression model"""
import logging
from typing import Any, Dict, List, Literal, Optional, TypedDict, cast

import pandas as pd  # type: ignore
from sklearn.linear_model import SGDRegressor  # type: ignore
from sklearn.multioutput import (  # type: ignore
    MultiOutputRegressor,
    RegressorChain,
)

from ..errors import ErrorType
from .abstract_model import BasicModel, ChainedModel, MultioutputModel
from .utils import generate_hyperparams_from_keys, sanitize_params

logger = logging.getLogger("logger")


SGDHyperParamKeys = Literal[
    "alpha",
]


class SGDHyperParams(TypedDict):
    """SGD HyperParams type"""

    alpha: List[float]


class SGDRegressionModel(BasicModel):
    """SGD Regression Class"""

    params: SGDHyperParams = {
        "alpha": [0.0001, 0.001, 0.01],
    }

    def __init__(
        self,
        train_features: pd.DataFrame,
        train_output: pd.DataFrame,
        error_type: ErrorType,
        params: Optional[SGDHyperParams] = None,
    ):
        self.__params = sanitize_params(params) if params else self.params
        super().__init__(train_features, train_output, error_type)

    def generate_hyperparams(self) -> None:
        """Generate hyperparams"""
        keys: List[SGDHyperParamKeys] = cast(
            List[SGDHyperParamKeys], list(self.__params.keys())
        )
        hyperparams = generate_hyperparams_from_keys(self.__params, {}, keys)
        self.generate_hyperparams_objects(hyperparams)


class ShiftedStochasticGradientDescentRegressor(
    SGDRegressionModel, ChainedModel
):
    """Shifted Stochastic Gradient Descent Regressor"""

    name = "Shifted Stochastic Gradient Descent Regressor"
    short_name = "SHIFT_SGD"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        # We train with only the column for the first hour
        model = SGDRegressor(**hyper_params)
        sgd_model = model.fit(
            self.train_features.drop("timeseries", axis=1),
            self.train_output.iloc[:, 1],
        )
        return sgd_model


class ChainedStochasticGradientDescentRegressor(
    SGDRegressionModel, MultioutputModel
):
    """Chained Stochastic Gradient Descent Regressor"""

    name = "Chained Stochastic Gradient Descent Regressor"
    short_name = "CHAIN_SGD"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        # We train with only the column for the first hour
        model = RegressorChain(SGDRegressor(**hyper_params))
        chained_sgd_model = model.fit(
            self.train_features.drop("timeseries", axis=1),
            self.train_output.drop("timeseries", axis=1),
        )
        return chained_sgd_model


class MultioutputSGDRegressor(SGDRegressionModel, MultioutputModel):
    """Multioutput SGD Regressor"""

    name = "Multioutput SGD Regression"
    short_name = "MULTI_SGD"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        model = MultiOutputRegressor(SGDRegressor(**hyper_params))
        multi_sgd_model = model.fit(
            self.train_features.drop("timeseries", axis=1),
            self.train_output.drop("timeseries", axis=1),
        )
        return multi_sgd_model
