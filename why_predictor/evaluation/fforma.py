"""FFORMA Evaluation module"""
import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd  # type: ignore

from .. import config
from ..models import BasicModel
from . import utils

logger = logging.getLogger("logger")


def evaluate_fforma(series: Dict[str, List[str]], base_path: str) -> None:
    """Evaluate FFORMA"""
    # Retrieve models and datasets
    (
        fforms_model,
        models,
        test_feat,
        test_out,
    ) = utils.retrieve_models_and_datasets(series, base_path)
    median_value = np.nanmean(test_feat.iloc[:, 2:])
    logger.debug("FFORMA model retrieved.")
    # Generate FFORMA predictions
    fforma_predictions = generate_fforma_predictions(fforms_model, base_path)
    # Calculate FFORMA output
    fforma_output = calculate_fforma_output(
        models, test_feat, test_out, fforma_predictions, median_value
    )
    # Evaluate
    final_error = utils.calculate_fform_error(
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
    fforma_predictions = utils.generate_fforma_predictions(
        fforma_model, phase2_path
    )
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
    sum_dtf = utils.calculate_fform_output(
        models_dict, test_feat, test_out, fforma_predictions, median_value
    )
    # Final result
    sum_dtf /= fforma_predictions["inv_sum"]
    return sum_dtf
