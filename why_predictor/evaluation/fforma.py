"""FFORMA Evaluation module"""
import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd  # type: ignore

from .. import config, loading
from ..errors import ErrorType
from ..models import BasicModel

logger = logging.getLogger("logger")


def evaluate_fforma(series: Dict[str, List[str]], base_path: str) -> None:
    """Evaluate FFORMA"""
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
    median_value = np.nanmean(test_feat.iloc[:, 2:])
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
    logger.debug("FFORMA model retrieved.")
    # Generate FFORMA predictions
    fforma_predictions = generate_fforma_predictions(fforma_model, base_path)
    # Calculate FFORMA output
    fforma_output = calculate_fforma_output(
        models_dict, test_feat, test_out, fforma_predictions, median_value
    )
    # Evaluate
    final_error = calculate_fforma_error(
        test_out, fforma_output, median_value, base_path
    )
    logger.info(
        "Final ERROR (%s): %r",
        config.ERROR_TYPE_FFORMA_EVALUATION,
        final_error,
    )


def generate_fforma_predictions(
    fforma_model: BasicModel, base_path: str
) -> pd.DataFrame:
    """Predict the error of the models"""
    phase2_path = os.path.join(base_path, "phase2")
    logger.debug("Generating FFORMA predictions.")
    fforma_predictions = fforma_model.make_predictions(
        *loading.fforma.load_fforma_test_datasets(phase2_path)
    ).set_index(["dataset", "timeseries"], inplace=True)
    fforma_predictions["sum"] = fforma_predictions.sum(axis=1)
    fforma_predictions["inv_sum"] = 1 / fforma_predictions["sum"]
    fforma_predictions.to_csv(
        os.path.join(base_path, "fforma_errors_predictions.csv.gz"),
        index=False,
    )
    logger.debug("FFORMA predictions generated.")
    return fforma_predictions


def calculate_fforma_output(
    models_dict: Dict[str, BasicModel],
    test_feat: pd.DataFrame,
    test_out: pd.DataFrame,
    fforma_predictions: pd.DataFrame,
    median_value: float,
) -> pd.DataFrame:
    """Calculate FFORMA output
    Sum(prediction model(n) * Error FFORMA model (n))
    -------------------------------------------------
    Sum(Error FFORMA model (n)))
    """
    # Generate model predictions
    sum_dtf = None
    for model in models_dict.values():
        dtf = model.calculate_timeseries_errors_dataframe(
            (test_feat, test_out),
            ErrorType[config.ERROR_TYPE_FFORMA_EVALUATION],
            median_value,
        )
        # Update with FFORMA
        for didx in fforma_predictions.index:
            value = fforma_predictions.loc[didx][model.short_name]
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
    # Final result
    sum_dtf /= fforma_predictions["inv_sum"]
    return sum_dtf


def calculate_fforma_error(
    test_output: pd.DataFrame,
    fforma_output: pd.DataFrame,
    median_value: float,
    base_path: str,
) -> float:
    """Calculate FFORMA error"""
    errors = ErrorType[config.ERROR_TYPE_FFORMA_EVALUATION].value(
        test_output.iloc[:, 2:],
        fforma_output,
        median_value,
    )
    errors.to_csv(
        os.path.join(base_path, "evaluation_errors.csv.gz"),
        index=False,
    )
    value: float = errors.stack().median()
    return value
