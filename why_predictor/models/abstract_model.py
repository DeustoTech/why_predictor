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
        joblib.dump(self._model, self.path, compress=True)
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
        # Define full errors & predictions CSV file names
        error_filename = os.path.join(
            self.base_path,
            "errors",
            "raw",
            f"{error.name}_{self.paramsname}.csv.gz",
        )
        predict_filename = os.path.join(
            self.base_path,
            "predictions",
            f"{error.name}_{self.paramsname}.csv.gz",
        )
        # Obtain feat and out datasets
        test_features, test_output = datasets
        # Generate predictions
        predictions = self.make_predictions(test_features, test_output)
        predictions.to_csv(predict_filename, index=False, header=False)
        # Clear model to save memory
        self.clear_model()
        # Generate error values
        error_values = error.value(
            test_output.iloc[:, NUM_HEADERS:],
            predictions.iloc[:, NUM_HEADERS:],
            median_value,
        )
        # Save error values to CSV (errors/raw)
        error_values.to_csv(error_filename, index=False, header=False)
        # Concat 'dataset' & 'timeseries' columns to a sumatory of the errors
        # per time series and save it to CSV file (errors/sum)
        pdu.concat(
            [test_output[["dataset", "timeseries"]], error_values.sum(axis=1)],
            ignore_index=True,
            axis=1,
        ).rename(
            columns={0: "dataset", 1: "timeseries", 2: self.paramsname}
        ).to_csv(
            os.path.join(
                self.base_path, "errors", "sum", f"{self.paramsname}.csv.gz"
            ),
            index=False,
        )
        # Return errors
        return pdu.downcast(error_values)

    def calculate_timeseries_errors_dataframe(
        self,
        datasets: Tuple[pd.DataFrame, pd.DataFrame],
        error: ErrorType,
        median_value: float,
    ) -> pd.DataFrame:
        """Calculate every timeseries's error for this model (FFORMA Column)"""
        errors = pdu.concat(
            [
                datasets[0].iloc[:, :2],  # dataset and timeseries columns
                self.calculate_errors(datasets, error, median_value),
            ],
            axis=1,
        )
        # Calculate error per timeseries
        results = []
        for dts, values in errors.groupby(["dataset", "timeseries"]):
            results.append((*dts, values.iloc[:, 2:].stack().median()))
        return pdu.DataFrame(
            results, columns=["dataset", "timeseries", self.short_name]
        ).set_index(["dataset", "timeseries"])

    def calculate_errors_per_file(
        self,
        datasets: List[Tuple[str, str]],
        error: ErrorType,
        median_value: float,
    ) -> pd.DataFrame:
        """Calculate Errors per individual file"""
        base_path = self.base_path
        # Define full errors CSV file name
        error_filename = os.path.join(
            base_path,
            "errors",
            "raw",
            f"{error.name}_{self.paramsname}.csv.gz",
        )
        predict_filename = os.path.join(
            self.base_path,
            "predictions",
            f"{error.name}_{self.paramsname}.csv.gz",
        )
        # Sanity check, remove previous error file if it exists
        # (it shouldn't exist)
        if os.path.exists(error_filename):
            os.remove(error_filename)
        # Initialize 'columns' dataframe for timeseries
        columns_df = pd.DataFrame()
        # for each pair of 'dataset' & 'timeseries'
        for dataset, timeseries in datasets:
            # Load features
            test_features = pdu.read_csv(
                f"{base_path}/test/{dataset}/features/{timeseries}.csv.gz"
            )
            # Load output
            test_output = pdu.read_csv(
                f"{base_path}/test/{dataset}/output/{timeseries}.csv.gz"
            )
            # Generate predictions
            predictions = self.make_predictions(test_features, test_output)
            predictions.to_csv(
                predict_filename, mode="a", index=False, header=False
            )
            # Generate errors and append them to CSV (errors/raw)
            error.value(
                test_output.iloc[:, NUM_HEADERS:],
                predictions.iloc[:, NUM_HEADERS:],
                median_value,
            ).to_csv(error_filename, mode="a", index=False, header=False)
            # Append processed 'dataset' & 'timeseries' to columns dataframe
            columns_df = pd.concat(
                [columns_df, test_output[["dataset", "timeseries"]]],
                ignore_index=True,
            )
            # Clear used variables (just in case)
            del test_features
            del test_output
            del predictions
        # Clear model to save memory
        self.clear_model()
        # Load generated error file
        errors = pdu.read_csv(error_filename, header=None)
        # Save sumatory of error
        pd.concat(
            [columns_df, errors.sum(axis=1)],
            ignore_index=True,
            axis=1,
        ).rename(
            columns={0: "dataset", 1: "timeseries", 2: self.paramsname}
        ).to_csv(
            os.path.join(base_path, "errors/sum", f"{self.paramsname}.csv.gz"),
            index=False,
        )
        return errors

    def calculate_timeseries_error(  # TODO update or delete
        self,
        data: Tuple[str, str],
        error: ErrorType,
        median_value: float,
        keep_model: bool = True,
    ) -> float:
        """Calculate timeseries's error for this model"""
        dataset, timeseries = data
        test_features = pdu.read_csv(
            os.path.join(
                self.base_path,
                f"datasets/test/{dataset}/features/{timeseries}.csv.gz",
            )
        )
        test_output = pdu.read_csv(
            os.path.join(
                self.base_path,
                f"datasets/test/{dataset}/output/{timeseries}.csv.gz",
            )
        )
        predictions = self.make_predictions(test_features, test_output)
        if not keep_model:
            self.clear_model()
        error_value: float = (
            error.value(
                test_output.iloc[:, NUM_HEADERS:],
                predictions.iloc[:, NUM_HEADERS:],
                median_value,
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
        predictions.columns = test_output.columns[NUM_HEADERS:3]
        for i in range(NUM_HEADERS + 1, test_output.shape[1]):
            # We generate a new features vector, removing first columns and
            # adding the already predicted values as features
            # 1 2 [3 4 5 6 7 ... 70 71 72 P1 P2]
            features = pdu.concat(
                [test_features.iloc[:, i:], predictions], axis=1
            )
            features.columns = [
                f"col{i}" for i in range(1, features.shape[1] + 1)
            ]
            predictions = pdu.concat(
                [
                    predictions,
                    pd.Series(self.model.predict(features)),
                ],
                axis=1,
            )
            predictions.columns = test_output.columns[NUM_HEADERS : i + 1]
        return pdu.concat(
            [test_features[["dataset", "timeseries"]], predictions],
            axis=1,
        ).set_axis(test_output.columns, axis=1)


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
