"""Errors"""
from enum import Enum
from functools import partial
from typing import Any

import numpy as np
import pandas as pd  # type: ignore

NUM_HEADERS = 2


def calculate_mape2(
    output: pd.DataFrame, predictions: pd.DataFrame, mean: float
) -> Any:
    """Calculate MAPE2"""
    output2 = output[output != 0].fillna(mean)
    # return np.absolute((output - predictions) / output2).sum() / len(output)
    return np.absolute((output - predictions) / output2)


def calculate_mape(
    output: pd.DataFrame, predictions: pd.DataFrame, _: float = 0.0
) -> Any:
    """Calculate MAPE"""
    # return np.absolute((output - predictions) / output).sum() / len(output)
    return np.absolute((output - predictions) / output)


def calculate_mae(
    output: pd.DataFrame, predictions: pd.DataFrame, _: float = 0.0
) -> Any:
    """Calculate MAE"""
    # return np.absolute((output - predictions)).sum() / len(output)
    return np.absolute((output - predictions))


def calculate_rmse(
    output: pd.DataFrame, predictions: pd.DataFrame, _: float = 0.0
) -> Any:
    """Calculate RMSE"""
    # return ((output - predictions).pow(2).sum() / len(output)).pow(1.0 / 2)
    return pd.DataFrame(
        ((output - predictions).pow(2).sum() / len(output)).pow(1.0 / 2)
    )


def calculate_smape(
    output: pd.DataFrame, predictions: pd.DataFrame, _: float = 0.0
) -> Any:
    """Calculate SMAPE"""
    # return (
    #     np.absolute(predictions - output) / ((output + predictions) / 2)
    # ).sum() / len(output)
    return np.absolute(predictions - output) / ((output + predictions) / 2)


class ErrorType(Enum):
    """Enum with error calculators"""

    MAPE2 = partial(calculate_mape2)
    MAPE = partial(calculate_mape)
    MAE = partial(calculate_mae)
    RMSE = partial(calculate_rmse)
    SMAPE = partial(calculate_smape)
