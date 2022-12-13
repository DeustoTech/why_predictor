"""Multi-Layer Perceptron Regression model"""
import logging
from typing import Any, Dict, List, Literal, Optional, TypedDict, cast

import pandas as pd  # type: ignore
from sklearn.neural_network import MLPRegressor  # type: ignore

from ..errors import ErrorType
from .abstract_model import BasicModel, ChainedModel, MultioutputModel
from .utils import generate_hyperparams_from_keys, sanitize_params

logger = logging.getLogger("logger")


MLPHyperParamKeys = Literal[
    "activation",
]


class MLPHyperParams(TypedDict):
    """MLP HyperParams type"""

    activation: List[str]


class MLPRegressionModel(BasicModel):
    """MLP Regression Class"""

    params: MLPHyperParams = {
        "activation": ["tanh"],
    }

    def __init__(
        self,
        train_features: pd.DataFrame,
        train_output: pd.DataFrame,
        error_type: ErrorType,
        params: Optional[MLPHyperParams] = None,
    ):
        self.__params = sanitize_params(params) if params else self.params
        super().__init__(train_features, train_output, error_type)

    def generate_hyperparams(self) -> None:
        """Generate hyperparams"""
        keys: List[MLPHyperParamKeys] = cast(
            List[MLPHyperParamKeys], list(self.__params.keys())
        )
        hyperparams = generate_hyperparams_from_keys(self.__params, {}, keys)
        self.generate_hyperparams_objects(hyperparams)


class MultiLayerPerceptronRegressor(MLPRegressionModel, ChainedModel):
    """Multi-layer Preceptron Regressor"""

    name = "Stochastic Gradient Descent Regressor"
    short_name = "MLP"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        # We train with only the column for the first hour
        model = MLPRegressor(**hyper_params)
        sgd_model = model.fit(
            self.train_features.drop("timeseries", axis=1),
            self.train_output.iloc[:, 1],
        )
        return sgd_model


class MultioutputMLPRegressor(MLPRegressionModel, MultioutputModel):
    """Multioutput KNN Regressor"""

    name = "Multioutput KNN Regression"
    short_name = "Multi_KNN"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        model = MLPRegressor(**hyper_params)
        multi_knn_model = model.fit(
            self.train_features.drop("timeseries", axis=1),
            self.train_output.drop("timeseries", axis=1),
        )
        return multi_knn_model
