"""Random Forest Regression model"""
import logging

import pandas as pd  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.multioutput import RegressorChain  # type: ignore

from .. import config
from .abstract_model import NUM_HEADERS, MultioutputModel, ShiftedModel

logger = logging.getLogger("logger")


class ShiftedRFRegressor(ShiftedModel):
    """Shifted Random Forest Regressor"""

    name = "Shifted Random Forest Regression"
    short_name = "SHIFT_RF"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        logger.debug(
            "Training %s model (%s)...", self.short_name, self.hyperparams
        )
        shifted_rf_model = RandomForestRegressor(
            **self.hyperparams, n_jobs=config.NJOBS
        )
        self._model = shifted_rf_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.iloc[:, NUM_HEADERS],
        )
        logger.debug("%s model trained.", self.short_name)


class ChainedRFRegressor(MultioutputModel):
    """Chained Random Forest Regressor"""

    name = "Chained Random Forest Regression"
    short_name = "CHAIN_RF"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        logger.debug(
            "Training %s model (%s)...", self.short_name, self.hyperparams
        )
        chained_rf_model = RegressorChain(
            RandomForestRegressor(**self.hyperparams, n_jobs=config.NJOBS)
        )
        self._model = chained_rf_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.drop(["dataset", "timeseries"], axis=1),
        )
        logger.debug("%s model trained.", self.short_name)


class MultioutputRFRegressor(MultioutputModel):
    """Multioutput Random Forest Regressor"""

    name = "Multioutput Random Forest Regression"
    short_name = "MULTI_RF"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        logger.debug(
            "Training %s model (%s)...", self.short_name, self.hyperparams
        )
        multi_rf_model = RandomForestRegressor(
            **self.hyperparams, n_jobs=config.NJOBS
        )
        self._model = multi_rf_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.drop(["dataset", "timeseries"], axis=1),
        )
        logger.debug("%s model trained.", self.short_name)
