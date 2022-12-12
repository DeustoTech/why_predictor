""" WHY predictor
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

import argcomplete  # type: ignore
from dotenv import load_dotenv

from .errors import ErrorType
from .load_sets import (
    find_csv_files,
    load_files,
    select_training_set,
    split_dataset_in_train_and_test,
)
from .models import BasicModel, Models

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)7s:  %(message)s",
)
logger = logging.getLogger("logger")
load_dotenv(dotenv_path=Path("./config.env"))


def generate_parser() -> argparse.ArgumentParser:
    """Generate parser for command line arguments"""
    parser = argparse.ArgumentParser(
        prog="python -m why-predictor",
        description="WHY Predictor",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--base-path-dataset",
        dest="dataset_basepath",
        default=os.getenv("DATASET_BASEPATH"),
        help="base path where dataset are stored",
    )
    parser.add_argument(
        "--dataset-dir-name",
        dest="dataset_dir_name",
        default=os.getenv("DATASET_DIRNAME"),
        help="exact name of the directory containing the CSV files",
    )
    parser.add_argument(
        "--percentage-csv-files-for-training",
        dest="training_percentage",
        type=float,
        default=os.getenv("TRAINING_PERCENTAGE"),
        help="Percentage of the CSV files that will be used for training",
    )
    parser.add_argument(
        "--train-test-ratio",
        dest="train_test_ratio",
        type=float,
        default=os.getenv("TRAIN_TEST_RATIO"),
        help="ratio of samples used for training "
        + "(1 - this value will be used for testing)",
    )
    models = os.getenv("MODELS")
    parser.add_argument(
        "--use-models",
        dest="models",
        choices=[f"{e.name}" for e in Models],
        nargs="+",
        type=str.upper,
        default=json.loads(models) if models else "LR",
        help="Select what models to use ["
        + ", ".join([f"{e.name} ({e.value.name})" for e in Models])
        + "]",
    )
    parser.add_argument(
        "--error-type",
        dest="error_type",
        choices=["MAPE", "MAE", "RMSE", "SMAPE"],
        type=str.upper,
        default=os.getenv("ERROR_TYPE"),
        help="metric to calculate the error",
    )
    parser.add_argument(
        "--window-num-features",
        dest="num_features",
        type=int,
        default=os.getenv("NUM_FEATURES"),
        help="num of hours used as features",
    )
    parser.add_argument(
        "--window-num-predictions",
        dest="num_predictions",
        type=int,
        default=os.getenv("NUM_PREDICTIONS"),
        help="num of hours used as predictions",
    )
    argcomplete.autocomplete(parser)
    return parser


def select_hyperparameters(args: argparse.Namespace) -> None:
    """Execute"""
    basepath = args.dataset_basepath
    dirname = args.dataset_dir_name
    series = find_csv_files(basepath, dirname)
    # Select training set
    training_set, _ = select_training_set(series, args.training_percentage)
    # Load training set
    data = load_files(training_set, args.num_features, args.num_predictions)
    # Train and test datasets
    (
        train_features,
        train_output,
        test_features,
        test_output,
    ) = split_dataset_in_train_and_test(
        data, args.train_test_ratio, args.num_features
    )
    # Calculate models
    models_dict = {}
    for model_name in args.models:
        models_dict[model_name] = Models[model_name].value(
            train_features, train_output, ErrorType[args.error_type]
        )
        models_dict[model_name].fit(
            test_features,
            test_output,
        )
    # Create errors and hyperparameters directory
    if not os.path.exists("errors"):
        os.makedirs("errors")
    if not os.path.exists("hyperparameters"):
        os.makedirs("hyperparameters")
    # Save errors and hyperparameters
    save_errors_and_hyperparameters(models_dict, args.error_type)


def save_errors_and_hyperparameters(
    models_dict: Dict[str, BasicModel], error_name: str
):
    """Save errors as CSV files and export best hyperparameter set as a JSON
    file"""
    for model_name, model in models_dict.items():
        for hyperparams in model.hyper_params.values():
            name = f"{model_name}_{error_name}_{hyperparams['name']}"
            # Export errors to CSV
            hyperparams["errors"].to_csv(f"errors/{name}.csv", index=False)
            # Exports hyperparams to file
            filename = f"hyperparameters/{model_name}.json"
            with open(filename, "w", encoding="utf8") as fhyper:
                fhyper.write(hyperparams["name"])


def main() -> None:
    """Main"""
    parser = generate_parser()

    # Parse args
    args = parser.parse_args()

    # Set logger
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.debug("Args: %r", args)
    select_hyperparameters(args)


if __name__ == "__main__":
    main()
