"""FFORMA related functions"""
import logging
import os
from typing import Dict, List

from .. import config, loading
from ..errors import ErrorType
from . import models

logger = logging.getLogger("logger")


def train_fforma(series: Dict[str, List[str]]) -> None:
    """Train FFORMA"""
    fforma_path = config.FFORMA_PATH
    phase1_path = os.path.join(fforma_path, "phase1")
    # Load or generate models
    logger.debug("Retrieving predicting models...")
    models_dict = loading.models.load_or_train_best_models(
        loading.datasets.get_train_datasets(
            series,
            config.TRAINING_PERCENTAGE_FFORMA,
            (config.NUM_FEATURES, config.NUM_PREDICTIONS),
            config.TRAIN_TEST_RATIO_MODELS,
            phase1_path,
        ),
        phase1_path,
    )
    logger.debug("Predicting models retrieved.")
    # Train FFORMA models
    models.train_to_fit_hyperparameters(
        config.MODELS_FFORMA,
        loading.fforma.generate_fforma_datasets(
            models_dict,
            loading.datasets.load_test_datasets(
                phase1_path, config.NUM_FEATURES, config.NUM_PREDICTIONS
            ),
            config.TRAIN_TEST_RATIO_FFORMA,
            fforma_path,
        ),
        ErrorType[config.ERROR_TYPE_FFORMA],
        fforma_path,
    )
