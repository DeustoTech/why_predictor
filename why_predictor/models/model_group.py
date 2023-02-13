"""Linear Regression model"""
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd  # type: ignore

from ..errors import ErrorType
from .abstract_model import NUM_HEADERS, BasicModel
from .models import Models
from .params import (
    DTHyperParamKeys,
    DTHyperParams,
    KNNHyperParamKeys,
    KNNHyperParams,
    MLPHyperParamKeys,
    MLPHyperParams,
    RFHyperParamKeys,
    RFHyperParams,
    SGDHyperParamKeys,
    SGDHyperParams,
    SVRHyperParamKeys,
    SVRHyperParams,
)
from .utils import generate_hyperparams_from_keys

logger = logging.getLogger("logger")


class BasicModelGroup(ABC):
    """Models must define these methods and properties"""

    def __init__(
        self,
        name: str,
        datasets: Tuple[pd.DataFrame, pd.DataFrame],
        error_type: ErrorType,
        base_path: str,
    ):
        self.name = name
        features, output = datasets
        logger.debug("Generating %s...", self.name)
        self.error_type = error_type
        self.hyper_params: Dict[str, BasicModel] = {}
        self.base_path = base_path
        self.datasets = self.__get_datasets(output)
        self.generate_hyperparams()
        self.median_value = np.nanmean(features.iloc[:, NUM_HEADERS:])
        self.num_features_predictions = [
            datasets[0].shape[1] - NUM_HEADERS,
            datasets[1].shape[1] - NUM_HEADERS,
        ]
        self.__generate_paths()
        self.__train_models(features, output)

    def __generate_paths(self) -> None:
        for check_path in [
            os.path.join(self.base_path, "models"),
            os.path.join(self.base_path, "hyperparameters"),
            os.path.join(self.base_path, "errors"),
            os.path.join(self.base_path, "sum_errors"),
            os.path.join(self.base_path, "post-hoc"),
        ]:
            if not os.path.exists(check_path):
                os.makedirs(check_path)

    @abstractmethod
    def generate_hyperparams(self) -> None:
        """Generate hyperparams"""

    def generate_hyperparams_objects(self, hyperparams: List[Any]) -> None:
        """Generate HyperParams Object"""
        logger.debug(
            "Generating hyper params for %s (%d)...",
            self.name,
            len(hyperparams),
        )
        for hyperparam_set in hyperparams:
            model = Models[self.name].value(hyperparam_set, self.base_path)
            self.hyper_params[model.paramsname] = model

    def __train_models(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        for model in self.hyper_params.values():
            model.generate_model(features, output)
            model.save_model()
            model.clear_model()

    def __get_datasets(self, output: pd.DataFrame) -> List[Tuple[str, str]]:
        if self.base_path.startswith("fforma"):
            return [("fforma", "dataset")]
        return [x[0] for x in output.groupby(["dataset", "timeseries"])]

    def fit(
        self, test_features: pd.DataFrame, test_output: pd.DataFrame
    ) -> None:
        """Generate predictions"""
        logger.debug("Calculating errors %s...", self.name)
        median_values: List[Tuple[float, BasicModel]] = []
        for model in self.hyper_params.values():
            median_error = (
                model.calculate_errors(
                    (test_features, test_output),
                    self.error_type,
                    self.median_value,
                )
                .stack()
                .median()
            )
            logger.info(
                "%s %s: %r", self.error_type.name, self.name, median_error
            )
            median_values.append((median_error, model))
        median_values.sort(key=lambda x: x[0])
        hyperparams_path = os.path.join(self.base_path, "hyperparameters")
        filename = os.path.join(hyperparams_path, f"{self.name}.json")
        with open(filename, "w", encoding="utf8") as f_hyper:
            median_error, model = median_values[0]
            f_hyper.write(f"{median_error}|{json.dumps(model.hyperparams)}")

    def fit2(self) -> None:
        """Generate predictions"""
        logger.debug("Calculating errors %s...", self.name)
        median_values: List[Tuple[float, BasicModel]] = []
        for model in self.hyper_params.values():
            median_error = (
                model.calculate_errors2(
                    self.datasets, self.error_type, self.median_value
                )
                .stack()
                .median()
            )
            logger.info(
                "%s %s: %r", self.error_type.name, self.name, median_error
            )
            median_values.append((median_error, model))
        median_values.sort(key=lambda x: x[0])
        hyperparams_path = os.path.join(self.base_path, "hyperparameters")
        filename = os.path.join(hyperparams_path, f"{self.name}.json")
        with open(filename, "w", encoding="utf8") as f_hyper:
            median_error, model = median_values[0]
            f_hyper.write(f"{median_error}|{json.dumps(model.hyperparams)}")


class KNNRegressionModelGroup(BasicModelGroup):
    """KNN Regression Group Class"""

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

    def generate_hyperparams(self) -> None:
        """Generate hyperparams"""
        keys: List[KNNHyperParamKeys] = cast(
            List[KNNHyperParamKeys], list(self.params.keys())
        )
        hyperparams = generate_hyperparams_from_keys(self.params, {}, keys)
        self.generate_hyperparams_objects(hyperparams)


class LinearRegressionModelGroup(BasicModelGroup):
    """Linear Regression Group Class"""

    params: Dict[str, Any] = {}

    def generate_hyperparams(self) -> None:
        """Generate hyperparams"""
        self.generate_hyperparams_objects([{}])


class MLPRegressionModelGroup(BasicModelGroup):
    """MLP Regression Group Class"""

    params: MLPHyperParams = {
        "hidden_layer_sizes": [(100,), (150,), (200,)],
        "activation": ["tanh"],
    }

    def generate_hyperparams(self) -> None:
        """Generate hyperparams"""
        keys: List[MLPHyperParamKeys] = cast(
            List[MLPHyperParamKeys], list(self.params.keys())
        )
        hyperparams = generate_hyperparams_from_keys(self.params, {}, keys)
        self.generate_hyperparams_objects(hyperparams)


class RandomForestRegressionModelGroup(BasicModelGroup):
    """Random Forest Regression Group Class"""

    params: RFHyperParams = {
        "n_estimators": [100, 150, 200],
        "max_depth": [10, 15],
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

    def generate_hyperparams(self) -> None:
        """Generate hyperparams"""
        keys: List[RFHyperParamKeys] = cast(
            List[RFHyperParamKeys], list(self.params.keys())
        )
        hyperparams = generate_hyperparams_from_keys(self.params, {}, keys)
        self.generate_hyperparams_objects(hyperparams)


class SGDRegressionModelGroup(BasicModelGroup):
    """SGD Regression Group Class"""

    params: SGDHyperParams = {
        "alpha": [0.0001, 0.001, 0.01],
    }

    def generate_hyperparams(self) -> None:
        """Generate hyperparams"""
        keys: List[SGDHyperParamKeys] = cast(
            List[SGDHyperParamKeys], list(self.params.keys())
        )
        hyperparams = generate_hyperparams_from_keys(self.params, {}, keys)
        self.generate_hyperparams_objects(hyperparams)


class SVMRegressionModelGroup(BasicModelGroup):
    """SVM Regression Group Class"""

    params: SVRHyperParams = {
        "epsilon": [0.0],
        "tol": [1e-4],
        "C": [1.0],
        # "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
        "loss": ["epsilon_insensitive"],
    }

    def generate_hyperparams(self) -> None:
        """Generate hyperparams"""
        keys: List[SVRHyperParamKeys] = cast(
            List[SVRHyperParamKeys], list(self.params.keys())
        )
        hyperparams = generate_hyperparams_from_keys(self.params, {}, keys)
        self.generate_hyperparams_objects(hyperparams)


class DecissionTreeRegressionModelGroup(BasicModelGroup):
    """Decission Tree Regression Group Class"""

    params: DTHyperParams = {
        "max_depth": [None, 10, 15],
        # "criterion": [
        #     "squared_error",
        #     "absolute_error",
        #     "friedman_mse",
        #     "poisson",
        # ],
        # "splitter": ["best", "random"],
        # "min_samples_split": [2, 3, 4, 5],
        # "min_samples_leaf": [1, 2, 3, 4, 5],
        # "min_weight_fraction_leaf": [0.0, 0.5],
        # "max_features": [1.0, "sqrt", "log2"],
        # "max_leaf_nodes": [None, 5, 10],
        # "min_impurity_decrease": [0.0, 0.5, 1, 1.5],
        # "ccp_alpha": [0.0, 0.5, 1, 1.5],
        # "ccp_alpha": [0.0, 0.5],
    }

    def generate_hyperparams(self) -> None:
        """Generate hyperparams"""
        keys: List[DTHyperParamKeys] = cast(
            List[DTHyperParamKeys], list(self.params.keys())
        )
        hyperparams = generate_hyperparams_from_keys(self.params, {}, keys)
        self.generate_hyperparams_objects(hyperparams)
