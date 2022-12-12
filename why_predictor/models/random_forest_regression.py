"""Random Forest Regression model"""
import logging
from typing import Any, Dict, List, Literal, TypedDict, Union, cast

from sklearn.ensemble import RandomForestRegressor  # type: ignore

from .abstract_model import BasicModel, ChainedModel, MultioutputModel

logger = logging.getLogger("logger")


RFHyperParamKeys = Literal[
    "n_estimators",
    "criterion",
    "min_samples_split",
    "min_samples_leaf",
    "min_weight_fraction_leaf",
    "max_features",
    "max_leaf_nodes",
    "min_impurity_decrease",
    "bootstrap",
    "oob_score",
    "ccp_alpha",
]


class RFHyperParams(TypedDict):
    """Random Forest HyperParams Type"""

    n_estimators: List[int]
    criterion: List[str]
    min_samples_split: List[int]
    min_samples_leaf: List[float]
    min_weight_fraction_leaf: List[Union[int, float]]
    max_features: List[Union[int, float, str]]
    max_leaf_nodes: List[Union[None, int]]
    min_impurity_decrease: List[float]
    bootstrap: List[bool]
    oob_score: Dict[str, List[bool]]
    ccp_alpha: List[float]


class RandomForestRegressionModel(BasicModel):
    """Random Forest Regression Class"""

    params: RFHyperParams = {
        # "n_estimators": [50, 100, 150, 200],
        "n_estimators": [100, 150],
        "criterion": [
            "squared_error",
            "friedman_mse",
            "absolute_error",
            "poisson",
        ],
        # "min_samples_split": [2, 3, 4, 5],
        "min_samples_split": [2],
        # "min_samples_leaf": [1, 2, 3, 4, 5],
        "min_samples_leaf": [1],
        # "min_weight_fraction_leaf": [0.0, 0.5],
        "min_weight_fraction_leaf": [0.0],
        # "max_features": [1.0, "sqrt", "log2"],
        "max_features": [1.0],
        # "max_leaf_nodes": [None, 5, 10],
        "max_leaf_nodes": [None],
        # "min_impurity_decrease": [0.0, 0.5, 1, 1.5],
        "min_impurity_decrease": [0.0],
        # "bootstrap": [True, False],
        "bootstrap": [True],
        "oob_score": {"True": [False, True], "False": [False]},
        # "ccp_alpha": [0.0, 0.5, 1, 1.5],
        "ccp_alpha": [0.0],
    }

    def generate_hyperparams(self) -> None:
        """Generate hyperparams"""
        keys: List[RFHyperParamKeys] = cast(
            List[RFHyperParamKeys], list(self.params.keys())
        )
        hyperparams = self.__generate_hyperparams({}, keys)
        self.generate_hyperparams_objects(hyperparams)

    def __generate_hyperparams(
        self,
        current_set: Dict[str, Any],
        hyperparams: List[RFHyperParamKeys],
    ) -> List[Dict[str, Any]]:
        if hyperparams:
            hyperparam = hyperparams.pop(0)
            hyperparam_sets = []
            values = self.params[hyperparam]
            if hyperparam == "oob_score":
                values = self.params[hyperparam][str(current_set["bootstrap"])]
            for value in values:
                my_set = current_set.copy()
                my_set[hyperparam] = value
                hyperparam_sets.extend(
                    self.__generate_hyperparams(my_set, hyperparams[:])
                )
            return hyperparam_sets
        return [current_set]


class RFRegressor(RandomForestRegressionModel, ChainedModel):
    """Random Forest Regressor"""

    name = "Random Forest Regression"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        model = RandomForestRegressor(**hyper_params, n_jobs=-1)
        random_forest_model = model.fit(
            self.train_features, self.train_output.iloc[:, 1]
        )
        return random_forest_model


class MultioutputRFRegressor(RandomForestRegressionModel, MultioutputModel):
    """Multioutput Random Forest Regressor"""

    name = "Multioutput Random Forest Regression"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        model = RandomForestRegressor(**hyper_params, n_jobs=-1)
        multi_rf_linear_model = model.fit(
            self.train_features, self.train_output
        )
        return multi_rf_linear_model
