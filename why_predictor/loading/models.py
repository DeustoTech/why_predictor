"""Model loading related functions"""
import glob
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd  # type: ignore

from ..models import BasicModel, Models

logger = logging.getLogger("logger")


def load_best_trained_models(base_path: str) -> Dict[str, BasicModel]:
    """Load best trained models"""
    dict_models: Dict[str, BasicModel] = {}
    hyperparameter_path = os.path.join(base_path, "hyperparameters")
    for filename in glob.glob(os.path.join(hyperparameter_path, "*")):
        name = os.path.splitext(os.path.split(filename)[-1])[0]
        _, hyperparams = load_error_and_hyperparameters(filename)
        dict_models[name] = Models[name].value(hyperparams, base_path)
    return dict_models


def load_or_train_best_models(
    train_datasets: Tuple[pd.DataFrame, pd.DataFrame], base_path: str
) -> Dict[str, BasicModel]:
    """Load models with best hyperparameter set and train them if train
    datasets are provide"""
    logger.debug("Loading best trained models...")
    dict_models = load_best_trained_models(base_path)
    logger.debug("Models loaded.")
    # If we pass datasets to train, we train the models
    if len(train_datasets[0]):
        logger.debug("Re-training models...")
        for model in dict_models.values():
            model.generate_model(*train_datasets)
            model.save_model()
        logger.debug("Models re-trained.")
    return dict_models


def load_or_train_best_trained_model(
    train_datasets: Tuple[pd.DataFrame, pd.DataFrame], base_path: str
) -> BasicModel:
    """Load best trained model with its best hyperparameter set and re-train it
    if train datasets are provided"""
    # Find best model
    logger.debug("Finding best model...")
    _models: List[Tuple[float, str, Dict[str, Any]]] = []
    hyperparameter_path = os.path.join(base_path, "hyperparameters")
    for filename in glob.glob(os.path.join(hyperparameter_path, "*")):
        name = os.path.splitext(os.path.split(filename)[-1])[0]
        error_value, hyperparameters = load_error_and_hyperparameters(filename)
        _models.append((error_value, name, hyperparameters))
    _models.sort(key=lambda x: x[0])
    error_value, model_name, hyperparameters = _models[0]
    logger.debug(
        "Best model: %s => %f.6 %r", model_name, error_value, hyperparameters
    )
    # Create model object
    model: BasicModel = Models[model_name].value(hyperparameters, base_path)
    if len(train_datasets[0]):
        logger.debug("Retraining model...")
        model.generate_model(*train_datasets)
        model.save_model()
        logger.debug("Model retrained.")
    return model


def load_error_and_hyperparameters(
    filename: str,
) -> Tuple[float, Dict[str, Any]]:
    """load error and hyperparameters"""
    error: float = 0.0
    hyperparameters: Dict[str, Any]
    try:
        with open(filename, encoding="utf8") as fhyper:
            error_text, hyper_text = fhyper.read().split("|")
            error = float(error_text)
            hyperparameters = json.loads(hyper_text)
    except FileNotFoundError:
        logger.error("File '%s' not found. Aborting...", filename)
        sys.exit(1)
    return (error, hyperparameters)
