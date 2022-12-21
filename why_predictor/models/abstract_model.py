"""Linear Regression model"""
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd  # type: ignore
import scikit_posthocs as sp  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from scipy.stats import friedmanchisquare  # type: ignore

from ..errors import ErrorType

logger = logging.getLogger("logger")


class HyperParams(TypedDict):
    """HyperParams type"""

    name: str
    params: Dict[str, Any]
    model: Any
    errors: pd.DataFrame
    median: float


class BasicModel(ABC):
    """Models must define these methods and properties"""

    name: str
    short_name: str

    def __init__(
        self,
        train_features: pd.DataFrame,
        train_output: pd.DataFrame,
        error_type: ErrorType,
    ):
        logger.debug("Generating %s...", self.name)
        self.train_features = train_features
        self.train_output = train_output
        self.error_type = error_type
        self.hyper_params: Dict[str, HyperParams] = {}
        self.predictions: pd.DataFrame
        self.test_output: pd.DataFrame
        self.fitted: HyperParams
        self.generate_hyperparams()

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
            name = json.dumps(hyperparam_set)
            self.hyper_params[name] = {
                "name": name,
                "params": hyperparam_set,
                "model": self.generate_model(hyperparam_set),
                "errors": pd.DataFrame,
                "median": 0.0,
            }

    def __friendman_with_post_hoc(self, base_path: str) -> None:
        # Sanity check
        if len(self.hyper_params) < 3:
            return
        # Create post-hoc directory
        base_path = os.path.join(base_path, "post-hoc")
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        # Generate dataframe
        aux = pd.DataFrame()
        for model in self.hyper_params.values():
            aux = pd.concat(
                [aux, model["errors"].iloc[:, 1:].mean(axis=1)], axis=1
            )
        aux.columns = [x["name"] for x in self.hyper_params.values()]
        # Calculate friedmanchisquare
        f_test = friedmanchisquare(*[aux[k] for k in aux.columns])
        logger.debug("Friedmanchisquare for %s: %r", self.name, f_test)
        # Calculate post-hoc if p_value < 0.05
        if f_test.pvalue < 0.05:
            post_hoc = sp.posthoc_nemenyi_friedman(aux)
            heatmap_args = {
                "linewidths": 0.25,
                "linecolor": "0.5",
                "clip_on": False,
                "square": True,
                "cbar_ax_bbox": [0.80, 0.35, 0.04, 0.3],
            }
            sp.sign_plot(post_hoc, **heatmap_args)
            filename = os.path.join(base_path, f"{self.short_name}.png")
            plt.savefig(filename)

    @abstractmethod
    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""

    @abstractmethod
    def calculate_errors(
        self,
        hyperparams_set: str,
        test_features: pd.DataFrame,
        test_output: pd.DataFrame,
    ) -> None:
        """Calculate errors for a hyper param set"""

    def fit(
        self,
        test_features: pd.DataFrame,
        test_output: pd.DataFrame,
        base_path: Optional[str] = None,
    ) -> None:
        """Generate predictions"""
        logger.debug("Calculating %s...", self.name)
        for hyperparams_set in self.hyper_params:
            self.calculate_errors(hyperparams_set, test_features, test_output)
        self.fitted = sorted(
            self.hyper_params.values(), key=lambda x: x["median"]
        )[0]
        logger.debug(
            "Best hyper params set: '%s' => %f",
            self.fitted["name"],
            self.fitted["median"],
        )
        # Save errors and hyperparameters if base_path is set
        if base_path:
            self.save_errors(base_path)
            self.save_best_hyperparameters(base_path)

    def save_errors(self, base_path: str) -> None:
        """Save errors"""
        # Create directory
        error_path = os.path.join(base_path, "errors")
        if not os.path.exists(error_path):
            os.makedirs(error_path)
        # Save hyperparam errors
        for hyperparams in self.hyper_params.values():
            name = f"{self.short_name}_{self.error_type}_{hyperparams['name']}"
            filename = os.path.join(error_path, name)
            hyperparams["errors"].to_csv(filename, index=False)
        # friendman with post-hoc
        self.__friendman_with_post_hoc(base_path)

    def save_best_hyperparameters(self, base_path: str) -> None:
        """Save Best Hyper-parameters set"""
        base_path = os.path.join(base_path, "hyperparameters")
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        filename = os.path.join(base_path, f"{self.short_name}.json")
        with open(filename, "w", encoding="utf8") as fhyper:
            fhyper.write(f"{self.fitted['median']}|{self.fitted['name']}")


class ShiftedModel(BasicModel):
    """Shifted Basic Model"""

    def calculate_errors(
        self,
        hyperparams_set: str,
        test_features: pd.DataFrame,
        test_output: pd.DataFrame,
    ) -> None:
        """Calculate errors for a hyper param set"""
        hyperparams = self.hyper_params[hyperparams_set]
        logger.debug("Hyper params set: %s", hyperparams["name"])
        predictions = pd.DataFrame(
            hyperparams["model"].predict(
                test_features.drop("timeseries", axis=1)
            )
        )
        for i in range(2, test_output.shape[1]):
            # We generate a new features vector, removing first columns and
            # adding the already predicted values as features
            # 1 2 [3 4 5 6 7 ... 70 71 72 P1 P2]
            features = pd.concat(
                [test_features.iloc[:, i:], predictions], axis=1
            )
            features = features.set_axis(
                [f"col{i}" for i in range(1, features.shape[1] + 1)], axis=1
            )
            predictions = pd.concat(
                [
                    predictions,
                    pd.Series(hyperparams["model"].predict(features)),
                ],
                axis=1,
            )
        predictions.insert(0, "timeseries", test_features["timeseries"])
        self.predictions = predictions.set_axis(test_output.columns, axis=1)
        self.test_output = test_output
        # Calculate errors
        error_metric = self.error_type.value(
            test_output, self.predictions, self.train_features
        )
        hyperparams["median"] = error_metric.stack().median()
        error_metric.insert(0, "timeseries", test_features["timeseries"])
        hyperparams["errors"] = error_metric
        logger.info(
            "%s %s: %r", self.error_type.name, self.name, hyperparams["median"]
        )
        logger.debug("Error dataframe\n:%r", error_metric)


class MultioutputModel(BasicModel):
    """Multioutput Basic Model"""

    def calculate_errors(
        self,
        hyperparams_set: str,
        test_features: pd.DataFrame,
        test_output: pd.DataFrame,
    ) -> None:
        """Calculate errors for a hyper param set"""
        hyperparams = self.hyper_params[hyperparams_set]
        logger.debug("Hyper params set: %s", hyperparams["name"])
        predictions = pd.DataFrame(
            hyperparams["model"].predict(
                test_features.drop("timeseries", axis=1)
            )
        )
        predictions.insert(0, "timeseries", test_features["timeseries"])
        self.predictions = predictions.set_axis(test_output.columns, axis=1)
        self.test_output = test_output
        logger.debug(
            "Accuracy: %r",
            hyperparams["model"].score(
                test_features.drop("timeseries", axis=1),
                test_output.drop("timeseries", axis=1),
            ),
        )
        # Calculate errors
        error_metric = self.error_type.value(
            test_output, self.predictions, self.train_features
        )
        hyperparams["median"] = error_metric.stack().median()
        error_metric.insert(0, "timeseries", test_features["timeseries"])
        hyperparams["errors"] = error_metric
        logger.info("%s %s: %r", self.error_type.name, self.name, error_metric)
