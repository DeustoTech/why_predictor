"""Phase 2: Training FFORMA"""
import logging
import os
import shutil
from typing import Dict, List

from .. import config, training
from . import phase1, utils

logger = logging.getLogger("logger")


def execute_phase_2(series: Dict[str, List[str]]) -> None:
    """Execute Phase 2: Training FFORMA"""
    check_phase1_is_generated(config.TRAINING_PATH, series)
    logger.info("* Phase 2: Training FFORMA...")
    delete_previous_execution(config.FFORMA_PATH)
    generate_phase2_tree(config.FFORMA_PATH)
    move_models_to_phase2(config.FFORMA_PATH, config.TRAINING_PATH)
    logger.info("Generating FFORMA...")
    training.fforma.train_fforma(series)
    logger.info("Phase 2 finished.")


def exists_previous_execution(fforma_path: str) -> bool:
    """Check if it exists a previous phase2 execution"""
    return os.path.exists(fforma_path)


def delete_previous_execution(fforma_path: str) -> None:
    """Delete previous phase2 execution"""
    if os.path.exists(fforma_path):
        shutil.rmtree(fforma_path)


def generate_phase2_tree(fforma_path: str) -> None:
    """Generate directories for phase2"""
    os.makedirs(os.path.join(fforma_path, "models"))
    os.makedirs(os.path.join(fforma_path, "hyperparameters"))
    os.makedirs(os.path.join(fforma_path, "predictions"))
    os.makedirs(os.path.join(fforma_path, "post-hoc"))
    for folder in ["train", "test"]:
        os.makedirs(os.path.join(fforma_path, "datasets", folder))
    for folder in ["raw", "sum"]:
        os.makedirs(os.path.join(fforma_path, "errors", folder))
    internal_phase1_path = os.path.join(fforma_path, "phase1")
    phase1.generate_phase1_tree(internal_phase1_path)


def check_phase1_is_generated(
    training_path: str, series: Dict[str, List[str]]
) -> None:
    """Move model from phase1"""
    # First, check if there is a previous phase1 execution
    if not phase1.exists_previous_execution(training_path):
        # If there is not, generate a phase1 execution
        logger.info("Phase 1 is not generated. Proceeding to create it first")
        phase1.execute_phase_1(series)


def move_models_to_phase2(fforma_path: str, training_path: str) -> None:
    """Move models to phase2 folder"""
    dest_path = os.path.join(fforma_path, "phase1")
    utils.copy_models(training_path, dest_path)
