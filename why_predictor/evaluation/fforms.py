"""FFORMS Evaluation module"""
import logging
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd  # type: ignore

from .. import config
from ..models import BasicModel
from . import utils

logger = logging.getLogger("logger")


def evaluate_fforms(series: Dict[str, List[str]], base_path: str) -> None:
    """Evaluate FFORMS"""
    # Retrieve models and datasets
    (
        fforma_model,
        models_dict,
        test_feat,
        test_out,
    ) = utils.retrieve_models_and_datasets(series, base_path)
    median_value = np.nanmean(test_feat.iloc[:, 2:])
    logger.debug("FFORMA (FFORMS) model retrieved.")
    # Generate FFORMS predictions
    fforms_predictions = generate_fforms_predictions(fforma_model, base_path)
    # Calculate FFORMS output
    fforms_output = utils.calculate_fform_output(
        models_dict, test_feat, test_out, fforms_predictions, median_value
    )
    # Evaluate
    final_error = utils.calculate_fform_error(
        test_out, fforms_output, median_value, base_path
    )
    logger.info(
        "Final ERROR (%s): %r",
        config.ERROR_TYPE_FFORMA_EVALUATION,
        final_error,
    )


def _create_mask(row: Any) -> Any:
    min_val = row.min()
    return row.eq(min_val).astype(int)


def generate_fforms_predictions(
    fforma_model: BasicModel, base_path: str
) -> pd.DataFrame:
    """Predict the error of the models"""
    phase2_path = os.path.join(base_path, "phase2")
    logger.debug("Generating FFORMS predictions.")
    fforms_predictions = utils.generate_fforma_predictions(
        fforma_model, phase2_path
    )
    fforms_predictions = fforms_predictions.apply(_create_mask, axis=1)
    fforms_predictions.to_csv(
        os.path.join(base_path, "fforms_errors_predictions.csv.gz"),
        index=False,
    )
    logger.debug("FFORMS predictions generated.")
    return fforms_predictions
