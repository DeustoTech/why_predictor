"""Linear Regression model"""
import logging

import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.multioutput import RegressorChain  # type: ignore

from .abstract_model import NUM_HEADERS, MultioutputModel, ShiftedModel

logger = logging.getLogger("logger")


class ShiftedLinearRegressor(ShiftedModel):
    """Shifted Linear Regressor"""

    name = "Shifted Linear Regression"
    short_name = "SHIFT_LR"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        # We train with only the column for the first hour
        shifted_lr_model = LinearRegression(**self.hyperparams, n_jobs=-1)
        self._model = shifted_lr_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.iloc[:, NUM_HEADERS],
        )


class ChainedLinearRegressor(MultioutputModel):
    """Chained Linear Regressor"""

    name = "Chained Linear Regression"
    short_name = "CHAIN_LR"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        chained_lr_model = RegressorChain(
            LinearRegression(**self.hyperparams, n_jobs=-1)
        )
        self._model = chained_lr_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.drop(["dataset", "timeseries"], axis=1),
        )


class MultioutputLinearRegressor(MultioutputModel):
    """Multioutput Linear Regressor"""

    name = "Multioutput Linear Regression"
    short_name = "MULTI_LR"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        multi_lr_model = LinearRegression(**self.hyperparams, n_jobs=-1)
        self._model = multi_lr_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.drop(["dataset", "timeseries"], axis=1),
        )
