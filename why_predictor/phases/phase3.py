"""Phase 3: Evaluating FFORMA"""
import logging
import os
import shutil
from typing import Dict, List

from .. import config, evaluation
from . import phase2, utils

logger = logging.getLogger("logger")


def execute_phase_3(series: Dict[str, List[str]]) -> None:
    """Execute Phase 3: Evaluating FFORMA"""
    check_phase2_is_generated(config.FFORMA_PATH, series)
    logger.info("* Phase 3: Evaluating FFORMA...")
    delete_previous_execution(config.EVALUATION_PATH)
    generate_phase3_tree(config.EVALUATION_PATH)
    move_models_to_phase3(
        config.EVALUATION_PATH, config.FFORMA_PATH, config.TRAINING_PATH
    )
    if config.USE_FFORMS:
        logger.info("Starting FFORMS evaluation process...")
        evaluation.fforms.evaluate_fforms(series, config.EVALUATION_PATH)
    else:
        logger.info("Starting FFORMA evaluation process...")
        evaluation.fforma.evaluate_fforma(series, config.EVALUATION_PATH)
    logger.info("Phase 3 finished.")


def delete_previous_execution(fforma_eval_path: str) -> None:
    """Delete previous phase3 execution"""
    if os.path.exists(fforma_eval_path):
        shutil.rmtree(fforma_eval_path)


def generate_phase3_tree(fforma_eval_path: str) -> None:
    """Generate directories for phase3"""
    internal_phase2_path = os.path.join(fforma_eval_path, "phase2")
    phase2.generate_phase2_tree(internal_phase2_path)


def check_phase2_is_generated(
    fforma_path: str, series: Dict[str, List[str]]
) -> None:
    """Move model from phase1"""
    # First, check if there is a previous phase1 execution
    if not phase2.exists_previous_execution(fforma_path):
        # If there is not, generate a phase1 execution
        logger.info("Phase 2 is not generated. Proceeding to create it first")
        phase2.execute_phase_2(series)


def move_models_to_phase3(
    fforma_eval_path: str, fforma_path: str, training_path: str
) -> None:
    """Move models to phase3 folder"""
    fforma_dest_path = os.path.join(fforma_eval_path, "phase2")
    utils.copy_models(fforma_path, fforma_dest_path)
    # Copy phase1 predicting models to phase2
    phase2.move_models_to_phase2(fforma_dest_path, training_path)
