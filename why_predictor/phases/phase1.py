"""Phase 1: Training models"""
import logging
import os
import shutil
from typing import Dict, List

import pandas as pd  # type: ignore

from .. import config, training

logger = logging.getLogger("logger")


def execute_phase_1(series: Dict[str, List[str]]) -> None:
    """Execute Phase 1: Training models"""
    logger.info("* Phase 1: Training models...")
    delete_previous_execution(config.TRAINING_PATH)
    generate_phase1_tree(config.TRAINING_PATH)
    logger.info("Selecting best hyperparameters...")
    if config.INITIAL_TRAINING_PATH:
        logger.info("- Using initial path %s...", config.INITIAL_TRAINING_PATH)
        training.models.select_hyperparameters_initial_path(series)
    else:
        training.models.select_hyperparameters(series)
    logger.info("Phase 1 finished.")


def exists_previous_execution(training_path: str) -> bool:
    """Check if it exists a previous phase1 execution"""
    return os.path.exists(training_path)


def delete_previous_execution(training_path: str) -> None:
    """Delete previous phase1 execution"""
    if exists_previous_execution(training_path):
        shutil.rmtree(training_path)


def generate_phase1_tree(base_path: str) -> None:
    """Generate directories for phase1"""
    os.makedirs(os.path.join(base_path, "models"))
    os.makedirs(os.path.join(base_path, "hyperparameters"))
    os.makedirs(os.path.join(base_path, "predictions"))
    os.makedirs(os.path.join(base_path, "post-hoc"))
    for folder in ["train", "test"]:
        os.makedirs(os.path.join(base_path, "datasets", folder))
    for folder in ["raw", "sum"]:
        os.makedirs(os.path.join(base_path, "errors", folder))
    _initialize_datasets(
        config.NUM_FEATURES, config.NUM_PREDICTIONS, base_path
    )


def _initialize_datasets(
    num_features: int,
    num_predictions: int,
    root_path: str,
) -> None:
    """Initialize datasets"""
    for test_train in ["test", "train"]:
        for feat_output in ["features", "output"]:
            if feat_output == "features":
                cols = [f"col{i}" for i in range(1, num_features + 1)]
            else:
                cols = [
                    f"col{i}"
                    for i in range(
                        num_features + 1,
                        num_features + num_predictions + 1,
                    )
                ]
            logger.debug("Initializing %s dataset...", test_train)
            pd.DataFrame(columns=["dataset", "timeseries", *cols]).to_csv(
                os.path.join(root_path, f"{test_train}_{feat_output}.csv.gz"),
                compression={
                    "method": "gzip",
                    "compresslevel": 1,
                    "mtime": 1,
                },
                index=False,
            )
