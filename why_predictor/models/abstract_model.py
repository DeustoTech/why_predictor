"""Linear Regression model"""
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import joblib  # type: ignore
import pandas as pd  # type: ignore

from .. import panda_utils as pdu
from ..errors import ErrorType

logger = logging.getLogger("logger")

NUM_HEADERS = 2


class BasicModel(ABC):
    """Models must define these methods and properties"""

    name: str
    short_name: str

    def __init__(self, hyperparams: Any, base_path: str):
        self.hyperparams = hyperparams
        self.paramsname = f"{self.short_name}_{json.dumps(hyperparams)}"
        self._model: Optional[Any] = None
        self.base_path = base_path
        self.path = os.path.join(base_path, "models", self.paramsname)

    @property
    def model(self) -> Any:
        """Return model"""
        if self._model is None:
            self.load_model()
        return self._model

    @abstractmethod
    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""

    def save_model(self) -> None:
        """Save model"""
        logger.debug("Saving model %s...", self.paramsname)
        assert self._model is not None
        joblib.dump(self._model, self.path)
        logger.debug("Model %s saved.", self.paramsname)

    def clear_model(self) -> None:
        """Clear model"""
        self._model = None

    def load_model(self) -> None:
        """load model from file"""
        logger.debug("Loading model %s...", self.paramsname)
        self._model = joblib.load(self.path)
        logger.debug("Model %s loaded.", self.paramsname)

    @abstractmethod
    def make_predictions(
        self, test_features: pd.DataFrame, test_output: pd.DataFrame
    ) -> pd.DataFrame:
        """Make predictions"""

    def calculate_errors(
        self,
        datasets: Tuple[pd.DataFrame, pd.DataFrame],
        error: ErrorType,
        median_value: float,
    ) -> pd.DataFrame:
        """Calculate Errors"""
        logger.debug(
            "Calculating errors for %s with params %s",
            error.name,
            self.paramsname,
        )
        error_filename = os.path.join(
            self.base_path, "errors", f"{error.name}_{self.paramsname}.csv.gz"
        )
        test_features, test_output = datasets
        predictions = self.make_predictions(test_features, test_output)
        error_values = error.value(
            test_output.iloc[:, NUM_HEADERS:],
            predictions,
            median_value,
        )
        error_values.to_csv(error_filename, index=False, header=False)
        pd.concat(
            [test_output[["dataset", "timeseries"]], error_values.sum(axis=1)],
            ignore_index=True,
            axis=1,
        ).rename(
            columns={0: "dataset", 1: "timeseries", 2: self.paramsname}
        ).to_csv(
            os.path.join(
                self.base_path, "sum_errors", f"{self.paramsname}.csv.gz"
            ),
            index=False,
        )
        return error_values

    def calculate_errors2(
        self,
        datasets: List[Tuple[str, str]],
        error: ErrorType,
        median_value: float,
    ) -> pd.DataFrame:
        """Calculate Errors"""
        base_path = self.base_path
        error_filename = os.path.join(
            base_path, "errors", f"{error.name}_{self.paramsname}.csv.gz"
        )
        if os.path.exists(error_filename):
            os.remove(error_filename)
        columns_df = pd.DataFrame()
        for dataset, timeseries in datasets:
            test_features = pdu.read_csv(
                f"{base_path}/test/{dataset}/features/{timeseries}.csv.gz"
            )
            test_output = pdu.read_csv(
                f"{base_path}/test/{dataset}/output/{timeseries}.csv.gz"
            )
            predictions = self.make_predictions(test_features, test_output)
            error.value(
                test_output.iloc[:, NUM_HEADERS:],
                predictions,
                median_value,
            ).to_csv(error_filename, mode="a", index=False, header=False)
            columns_df = pd.concat(
                [columns_df, test_output[["dataset", "timeseries"]]],
                ignore_index=True,
            )
            del test_features
            del test_output
            del predictions
        self.clear_model()
        errors = pdu.read_csv(error_filename, header=None)
        # Save sumatory of error
        pd.concat(
            [columns_df, errors.sum(axis=1)],
            ignore_index=True,
            axis=1,
        ).rename(
            columns={0: "dataset", 1: "timeseries", 2: self.paramsname}
        ).to_csv(
            os.path.join(base_path, "sum_errors", f"{self.paramsname}.csv.gz"),
            index=False,
        )
        return errors

    def calculate_timeseries_error(
        self,
        data: Tuple[str, str],
        error: ErrorType,
        median_value: float,
        keep_model: bool = True,
    ) -> float:
        """Calculate timeseries's error for this model"""
        dataset, timeseries = data
        test_features = pdu.read_csv(
            f"{self.base_path}/test/{dataset}/features/{timeseries}.csv.gz"
        )
        test_output = pdu.read_csv(
            f"{self.base_path}/test/{dataset}/output/{timeseries}.csv.gz"
        )
        predictions = self.make_predictions(test_features, test_output)
        if not keep_model:
            self.clear_model()
        error_value: float = (
            error.value(
                test_output.iloc[:, NUM_HEADERS:], predictions, median_value
            )
            .stack()
            .median()
        )
        return error_value


class ShiftedModel(BasicModel):
    """Shifted Basic Model"""

    def make_predictions(
        self, test_features: pd.DataFrame, test_output: pd.DataFrame
    ) -> pd.DataFrame:
        """Make predictions"""
        predictions = pdu.DataFrame(
            self.model.predict(
                test_features.drop(["dataset", "timeseries"], axis=1)
            )
        )
        for i in range(NUM_HEADERS + 1, test_output.shape[1]):
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
                    pd.Series(self.model.predict(features)),
                ],
                axis=1,
            )
        # predictions.insert(0, "timeseries", test_features["timeseries"])
        predictions.set_axis(test_output.columns[2:], axis=1, inplace=True)
        return predictions


class MultioutputModel(BasicModel):
    """Multioutput Basic Model"""

    def make_predictions(
        self, test_features: pd.DataFrame, test_output: pd.DataFrame
    ) -> pd.DataFrame:
        """Make predictions"""
        predictions = pdu.DataFrame(
            self.model.predict(
                test_features.drop(["dataset", "timeseries"], axis=1)
            )
        )
        return pdu.concat(
            [
                test_features[["dataset", "timeseries"]],
                predictions,
            ],
            axis=1,
        ).set_axis(test_output.columns, axis=1)
