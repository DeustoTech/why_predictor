"""Random Forest Regression model"""
import logging
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union, cast

import pandas as pd  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore

from ..errors import ErrorType
from .abstract_model import BasicModel, ChainedModel, MultioutputModel
from .utils import generate_hyperparams_from_keys, sanitize_params

logger = logging.getLogger("logger")


RFHyperParamKeys = Literal[
    "n_estimators",
    "max_depth",
    # "criterion",
    # "min_samples_split",
    # "min_samples_leaf",
    # "min_weight_fraction_leaf",
    # "max_features",
    # "max_leaf_nodes",
    # "min_impurity_decrease",
    # "bootstrap",
    # "oob_score",
    # "ccp_alpha",
]


class RFHyperParams(TypedDict):
    """Random Forest HyperParams Type"""

    n_estimators: List[int]
    max_depth: List[Union[None, int]]
    # criterion: List[str]
    # min_samples_split: List[int]
    # min_samples_leaf: List[float]
    # min_weight_fraction_leaf: List[Union[int, float]]
    # max_features: List[Union[int, float, str]]
    # max_leaf_nodes: List[Union[None, int]]
    # min_impurity_decrease: List[float]
    # bootstrap: List[bool]
    # oob_score: Dict[str, List[bool]]
    # ccp_alpha: List[float]


class RandomForestRegressionModel(BasicModel):
    """Random Forest Regression Class"""

    params: RFHyperParams = {
        "n_estimators": [100, 150, 200],
        "max_depth": [None, 10, 15],
        # "criterion": [
        #     "squared_error",
        #     "friedman_mse",
        #     "absolute_error",
        #     "poisson",
        # ],
        # "min_samples_split": [2, 3, 4, 5],
        # "min_samples_leaf": [1, 2, 3, 4, 5],
        # "min_weight_fraction_leaf": [0.0, 0.5],
        # "max_features": [1.0, "sqrt", "log2"],
        # "max_leaf_nodes": [None, 5, 10],
        # "min_impurity_decrease": [0.0, 0.5, 1, 1.5],
        # "bootstrap": [True, False],
        # "ccp_alpha": [0.0, 0.5, 1, 1.5],
    }

    def __init__(
        self,
        train_features: pd.DataFrame,
        train_output: pd.DataFrame,
        error_type: ErrorType,
        params: Optional[RFHyperParams] = None,
    ):
        self.__params = sanitize_params(params) if params else self.params
        super().__init__(train_features, train_output, error_type)

    def generate_hyperparams(self) -> None:
        """Generate hyperparams"""
        keys: List[RFHyperParamKeys] = cast(
            List[RFHyperParamKeys], list(self.__params.keys())
        )
        hyperparams = generate_hyperparams_from_keys(self.__params, {}, keys)
        self.generate_hyperparams_objects(hyperparams)


class RFRegressor(RandomForestRegressionModel, ChainedModel):
    """Random Forest Regressor"""

    name = "Random Forest Regression"
    short_name = "RF"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        model = RandomForestRegressor(**hyper_params, n_jobs=-1)
        random_forest_model = model.fit(
            self.train_features.drop("timeseries", axis=1),
            self.train_output.iloc[:, 1],
        )
        return random_forest_model


class MultioutputRFRegressor(RandomForestRegressionModel, MultioutputModel):
    """Multioutput Random Forest Regressor"""

    name = "Multioutput Random Forest Regression"
    short_name = "Multi_RF"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        model = RandomForestRegressor(**hyper_params, n_jobs=-1)
        multi_rf_linear_model = model.fit(
            self.train_features.drop("timeseries", axis=1),
            self.train_output.drop("timeseries", axis=1),
        )
        return multi_rf_linear_model
