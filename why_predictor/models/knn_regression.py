"""Linear Regression model"""
import logging
from typing import Any, Dict, List, Literal, Optional, TypedDict, cast

import pandas as pd  # type: ignore
from sklearn.neighbors import KNeighborsRegressor  # type: ignore

from ..errors import ErrorType
from .abstract_model import BasicModel, ChainedModel, MultioutputModel
from .utils import generate_hyperparams_from_keys, sanitize_params

logger = logging.getLogger("logger")


KNNHyperParamKeys = Literal[
    "n_neighbors",
    "weights",
    # "algorithm",
    # "leaf_size",
    # "p"
]


class KNNHyperParams(TypedDict):
    """KNN HyperParams type"""

    n_neighbors: List[int]
    weights: List[str]
    # algorithm: List[str]
    # leaf_size: Dict[str, List[int]]
    # p: List[int]


class KNNRegressionModel(BasicModel):
    """KNN Regression Class"""

    params: KNNHyperParams = {
        "n_neighbors": [5, 10, 15],  # TO fit
        "weights": ["distance"],  # Fixed
        # "algorithm": ["ball_tree", "kd_tree", "brute"],
        # "leaf_size": {
        #     "ball_tree": [15, 30, 45],
        #     "kd_tree": [30],
        #     "brute": [30],
        # },
        # "p": [1, 2],
    }

    def __init__(
        self,
        train_features: pd.DataFrame,
        train_output: pd.DataFrame,
        error_type: ErrorType,
        params: Optional[KNNHyperParams] = None,
    ):
        self.__params = sanitize_params(params) if params else self.params
        super().__init__(train_features, train_output, error_type)

    def generate_hyperparams(self) -> None:
        """Generate hyperparams"""
        keys: List[KNNHyperParamKeys] = cast(
            List[KNNHyperParamKeys], list(self.__params.keys())
        )
        hyperparams = generate_hyperparams_from_keys(self.__params, {}, keys)
        self.generate_hyperparams_objects(hyperparams)


class KNNRegressor(KNNRegressionModel, ChainedModel):
    """Chained KNN Regressor"""

    name = "KNN Regression"
    short_name = "KNN"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        # We train with only the column for the first hour
        model = KNeighborsRegressor(**hyper_params, n_jobs=-1)
        knn_model = model.fit(
            self.train_features.drop("timeseries", axis=1),
            self.train_output.iloc[:, 1],
        )
        return knn_model


class MultioutputKNNRegressor(KNNRegressionModel, MultioutputModel):
    """Multioutput KNN Regressor"""

    name = "Multioutput KNN Regression"
    short_name = "Multi_KNN"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        model = KNeighborsRegressor(**hyper_params, n_jobs=-1)
        multi_knn_model = model.fit(
            self.train_features.drop("timeseries", axis=1),
            self.train_output.drop("timeseries", axis=1),
        )
        return multi_knn_model
