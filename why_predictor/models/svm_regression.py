"""Support Vector Regression model"""
import logging
from typing import Any, Dict, List, Literal, Optional, TypedDict, cast

import pandas as pd  # type: ignore
from sklearn.multioutput import (  # type: ignore
    MultiOutputRegressor,
    RegressorChain,
)
from sklearn.svm import LinearSVR  # type: ignore

from ..errors import ErrorType
from .abstract_model import BasicModel, MultioutputModel, ShiftedModel
from .utils import generate_hyperparams_from_keys, sanitize_params

logger = logging.getLogger("logger")


SVRHyperParamKeys = Literal[
    "epsilon",
    "tol",
    "C",
    "loss",
]


class SVRHyperParams(TypedDict):
    """SVR HyperParams type"""

    epsilon: List[float]
    tol: List[float]
    C: List[float]
    loss: List[str]


class SVMRegressionModel(BasicModel):
    """SVM Regression Class"""

    params: SVRHyperParams = {
        "epsilon": [0.0],
        "tol": [1e-4],
        "C": [1.0],
        # "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
        "loss": ["epsilon_insensitive"],
    }

    def __init__(
        self,
        train_features: pd.DataFrame,
        train_output: pd.DataFrame,
        error_type: ErrorType,
        params: Optional[SVRHyperParams] = None,
    ):
        self.__params = sanitize_params(params) if params else self.params
        super().__init__(train_features, train_output, error_type)

    def generate_hyperparams(self) -> None:
        """Generate hyperparams"""
        keys: List[SVRHyperParamKeys] = cast(
            List[SVRHyperParamKeys], list(self.__params.keys())
        )
        hyperparams = generate_hyperparams_from_keys(self.__params, {}, keys)
        self.generate_hyperparams_objects(hyperparams)


class ShiftedSupportVectorRegressor(SVMRegressionModel, ShiftedModel):
    """Shifted Support Vector Regressor"""

    name = "Shifted Support Vector Regression"
    short_name = "SHIFT_SVR"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        # We train with only the column for the first hour
        model = LinearSVR(**hyper_params, max_iter=10000)
        svr_model = model.fit(
            self.train_features.drop("timeseries", axis=1),
            self.train_output.iloc[:, 1],
        )
        return svr_model


class ChainedSupportVectorRegressor(SVMRegressionModel, MultioutputModel):
    """Chained Support Vector Regressor"""

    name = "Chained Support Vector Regression"
    short_name = "CHAIN_SVR"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        # We train with only the column for the first hour
        model = RegressorChain(LinearSVR(**hyper_params, max_iter=10000))
        chained_svr_model = model.fit(
            self.train_features.drop("timeseries", axis=1),
            self.train_output.drop("timeseries", axis=1),
        )
        return chained_svr_model


class MultioutputSVMRegressor(SVMRegressionModel, MultioutputModel):
    """Multioutput Support Vector Regressor"""

    name = "Multioutput Support Vector Regression"
    short_name = "MULTI_SVR"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        model = MultiOutputRegressor(LinearSVR(**hyper_params, max_iter=10000))
        multi_svr_model = model.fit(
            self.train_features.drop("timeseries", axis=1),
            self.train_output.drop("timeseries", axis=1),
        )
        return multi_svr_model
