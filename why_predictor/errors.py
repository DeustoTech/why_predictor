"""Errors"""
import math
from enum import Enum
from functools import partial
from typing import Any
import numpy as np
import pandas as pd  # type: ignore


def calculate_mape(output: pd.DataFrame, predictions: pd.DataFrame) -> Any:
    """Calculate MAPE"""
    return np.absolute((output - predictions) / output).sum() / len(output)


def calculate_mae(output: pd.DataFrame, predictions: pd.DataFrame) -> Any:
    """Calculate MAE"""
    return np.absolute((output - predictions)).sum() / len(output)


def calculate_rmse(output: pd.DataFrame, predictions: pd.DataFrame) -> Any:
    """Calculate RMSE"""
    return math.sqrt((output - predictions).pow(2).sum() / len(output))


def calculate_smape(output: pd.DataFrame, predictions: pd.DataFrame) -> Any:
    """Calculate SMAPE"""
    return (
        np.absolute(predictions - output) / ((output + predictions) / 2)
    ).sum() / len(output)


class ErrorType(Enum):
    """Enum with error calculators"""
    MAPE = partial(calculate_mape)
    MAE = partial(calculate_mae)
    RMSE = partial(calculate_rmse)
    SMAPE = partial(calculate_smape)
