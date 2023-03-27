"""FFORM* utils module"""
import logging
import os
from typing import Dict, List, Tuple

import pandas as pd  # type: ignore

from .. import config, loading
from ..errors import ErrorType
from ..models import BasicModel

logger = logging.getLogger("logger")


def retrieve_models_and_datasets(
    series: Dict[str, List[str]], base_path: str
) -> Tuple[BasicModel, Dict[str, BasicModel], pd.DataFrame, pd.DataFrame]:
    """Obtain models and test datasets (features and output)"""
    phase2_path = os.path.join(base_path, "phase2")
    phase1_path = os.path.join(phase2_path, "phase1")
    # Load or generate models
    logger.debug("Retrieving predicting models...")
    models_dict = loading.models.load_or_train_best_models(
        loading.datasets.get_train_datasets(
            series,
            config.TRAINING_PERCENTAGE_FFORMA_EVALUATION,
            (config.NUM_FEATURES, config.NUM_PREDICTIONS),
            config.TRAIN_TEST_RATIO_FFORMA_EVALUATION,
            phase1_path,
        ),
        phase1_path,
    )
    logger.debug("Predicting models retrieved.")
    # Loading test datasets
    test_feat, test_out = loading.datasets.load_test_datasets(
        phase1_path, config.NUM_FEATURES, config.NUM_PREDICTIONS
    )
    # Load best FFORMA model
    logger.debug("Retrieving FFORMA model...")
    fforma_model = loading.models.load_or_train_best_trained_model(
        loading.fforma.generate_fforma_datasets(
            models_dict,
            (test_feat, test_out),
            config.TRAIN_TEST_RATIO_FFORMA_EVALUATION,
            phase2_path,
        ),
        phase2_path,
    )
    # Return
    return fforma_model, models_dict, test_feat, test_out


def generate_fforma_predictions(
    fforma_model: BasicModel, phase2_path: str
) -> pd.DataFrame:
    """Predict the error of the models"""
    fform_predictions = fforma_model.make_predictions(
        *loading.fforma.load_fforma_test_datasets(phase2_path)
    ).set_index(["dataset", "timeseries"], inplace=True)
    return fform_predictions


def calculate_fform_output(
    models_dict: Dict[str, BasicModel],
    test_feat: pd.DataFrame,
    test_out: pd.DataFrame,
    fform_predictions: pd.DataFrame,
    median_value: float,
) -> pd.DataFrame:
    """Calculate FFORM* output
    Best prediction model per series
    """
    # Generate model predictions
    sum_dtf = None
    for model in models_dict.values():
        dtf = model.calculate_timeseries_errors_dataframe(
            (test_feat, test_out),
            ErrorType[config.ERROR_TYPE_FFORMA_EVALUATION],
            median_value,
        )
        # Update with FFORMS
        for didx in fform_predictions.index:
            value = fform_predictions.loc[didx][model.short_name]
            print(didx, value)
            dtf.loc[
                (dtf["dataset"] == didx[0]) & (dtf["timeseries"] == didx[1]),
                dtf.columns.str.startswith("col"),
            ] *= value
        # Sum dataframes
        if sum_dtf is not None:
            sum_dtf += dtf.iloc[:, 2:]
        else:
            sum_dtf = dtf.iloc[:, 2:]
    return sum_dtf


def calculate_fform_error(
    test_output: pd.DataFrame,
    fform_output: pd.DataFrame,
    median_value: float,
    base_path: str,
) -> float:
    """Calculate FFORM error"""
    errors = ErrorType[config.ERROR_TYPE_FFORMA_EVALUATION].value(
        test_output.iloc[:, 2:],
        fform_output,
        median_value,
    )
    errors.to_csv(
        os.path.join(base_path, "evaluation_errors.csv.gz"),
        index=False,
    )
    value: float = errors.stack().median()
    return value
