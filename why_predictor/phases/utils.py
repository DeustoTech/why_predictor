"""Utils for phases module"""
import glob
import json
import os
import shutil

from .. import loading


def copy_models(origin_path: str, new_path: str) -> None:
    """Copy models from one phase to another"""
    hyperparams_path = os.path.join(origin_path, "hyperparameters")
    new_models_path = os.path.join(new_path, "phase2", "models")
    for model_path in glob.glob(os.path.join(hyperparams_path, "*")):
        filename = os.path.split(model_path)[-1]
        name = os.path.splitext(filename)[0]
        _, hyperparams = loading.models.load_error_and_hyperparameters(
            model_path
        )
        model_filename = f"{name}_{json.dumps(hyperparams)}"
        original_path = os.path.join(
            origin_path, "models", model_filename
        )
        # Copy model
        shutil.copy(original_path, new_models_path)
        # Copy hyperparameters file
        shutil.copy(
            model_path, os.path.join(new_path, "phase2", "hyperparameters")
        )
