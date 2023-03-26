""" WHY predictor
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import argcomplete
from dotenv import load_dotenv

from . import config, loading, phases
from .models import Models

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)7s:  %(message)s",
)
logger = logging.getLogger("logger")
load_dotenv(dotenv_path=Path("./config.env"))


class CustomFormatter(argparse.HelpFormatter):
    """Custom formatter to break lines in Help text preceded by #"""

    def _split_lines(self, text: str, width: int) -> Any:
        # pylint: disable=protected-access
        if text.startswith("#"):
            return text[1:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def generate_parser() -> argparse.ArgumentParser:
    """Generate parser for command line arguments"""
    parser = argparse.ArgumentParser(
        prog="python -m why_predictor",
        description="WHY Predictor",
        formatter_class=CustomFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-m",
        "--mode",
        dest="mode",
        choices=[
            "generate-csvs",
            "generate-hyperparams",
            "generate-fforma",
            "evaluate-fforma",
            "full",
        ],
        default="full",
        help="Select the operation mode, by default it will run in full mode"
        + " that includes both generate-errors and generate fforma. "
        + "generate-errors: will only train the models to generate the error "
        + "files, while generate-fforma will assume the hyperparameters are "
        + "already set, so it will generate the FFORMA model.",
    )
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
    parser.add_argument(
        "--save-datasets",
        dest="save_datasets",
        type=bool,
        default=os.getenv("SAVE_DATASETS", "False") == "True",
        help="save generated rolling-window datasets to disk",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=os.getenv("NJOBS"),
        help="Number of CPUs to use. When negative values are provided, -1 "
        + "means all CPUs, -2: means all CPUs but one, -3: means all CPUs but "
        + "two...",
    )
    generate_errors = parser.add_argument_group("Model training")
    models = os.getenv("MODELS_TRAINING")
    generate_errors.add_argument(
        "--use-models-training",
        dest="models_training",
        choices=[f"{e.name}" for e in Models],
        nargs="+",
        type=str.upper,
        default=json.loads(models) if models else "SHIFT_LR",
        help="#Select what models to use:\n    "
        + "\n    ".join([f"{e.name} ({e.value.name})" for e in Models]),
    )
    generate_errors.add_argument(
        "--error-type-training",
        dest="error_type_models",
        choices=["MAPE", "MAE", "RMSE", "SMAPE"],
        type=str.upper,
        default=os.getenv("ERROR_TYPE_MODELS"),
        help="metric to calculate the error",
    )
    generate_errors.add_argument(
        "--percentage-csv-files-for-training-hyperparameters",
        dest="training_percentage_hyperparams",
        type=float,
        default=os.getenv("TRAINING_PERCENTAGE_MODELS"),
        help="Percentage of the CSV files that will be used for training",
    )
    exclusive = generate_errors.add_mutually_exclusive_group()
    exclusive.add_argument(
        "--train-test-ratio-hyperparameters",
        dest="train_test_ratio_hyperparams",
        type=float,
        default=os.getenv("TRAIN_TEST_RATIO_MODELS"),
        help="ratioof samples used for training "
        + "(1 - this value will be used for testing)",
    )
    exclusive.add_argument(
        "--initial-training-path",
        dest="initial_training_path",
        default=os.getenv("INITIAL_TRAINING_PATH"),
        help="path to a folder where datasets will be used just for training"
        + " of models in phase1 (if this mode if used)",
    )
    generate_fforma = parser.add_argument_group("FFORMA training")
    models_fforma = os.getenv("MODELS_FFORMA")
    generate_fforma.add_argument(
        "--use-models-fforma",
        dest="models_fforma",
        choices=[f"{e.name}" for e in Models],
        nargs="+",
        type=str.upper,
        default=json.loads(models_fforma) if models_fforma else "LR",
        help="#Select what models to use:\n    "
        + "\n    ".join(
            [
                f"{e.name} ({e.value.name})"
                for e in Models
                if e.name.startswith("MULTI")
            ]
        ),
    )
    generate_fforma.add_argument(
        "--error-type-fforma",
        dest="error_type_fforma",
        choices=["MAPE", "MAE", "RMSE", "SMAPE"],
        type=str.upper,
        default=os.getenv("ERROR_TYPE_FFORMA"),
        help="metric to calculate the error",
    )
    generate_fforma.add_argument(
        "--percentage-csv-files-for-training-fforma",
        dest="training_percentage_fforma",
        type=float,
        default=os.getenv("TRAINING_PERCENTAGE_FFORMA"),
        help="Percentage of the CSV files that will be used for training",
    )
    generate_fforma.add_argument(
        "--train-test-ratio-fforma",
        dest="train_test_ratio_fforma",
        type=float,
        default=os.getenv("TRAIN_TEST_RATIO_FFORMA"),
        help="ratio of samples used for training "
        + "(1 - this value will be used for testing)",
    )
    argcomplete.autocomplete(parser)
    evaluate_fforma = parser.add_argument_group("FFORMA evaluation")
    evaluate_fforma.add_argument(
        "--percentage-csv-files-for-fforma-eval",
        dest="training_percentage_fforma_eval",
        type=float,
        default=os.getenv("TRAINING_PERCENTAGE_FFORMA_EVALUATION"),
        help="Percentage of the CSV files that will be used for evaluation",
    )
    evaluate_fforma.add_argument(
        "--train-test-ratio-fforma-eval",
        dest="train_test_ratio_fforma_eval",
        type=float,
        default=os.getenv("TRAIN_TEST_RATIO_FFORMA_EVALUATION"),
        help="ratio of samples used for evaluation"
        + "(1 - this value will be used for evaluation)",
    )
    evaluate_fforma.add_argument(
        "--error-type-fforma-eval",
        dest="error_type_fforma_eval",
        choices=["MAPE", "MAE", "RMSE", "SMAPE"],
        type=str.upper,
        default=os.getenv("ERROR_TYPE_FFORMA_EVALUATION"),
        help="metric to calculate the error when evaluating "
        + "the final output of FFORMA",
    )
    return parser


def process_args(args: argparse.Namespace) -> None:
    """process args and save them in config"""
    config.DATASET_BASEPATH = args.dataset_basepath
    config.DATASET_DIRNAME = args.dataset_dir_name
    config.NUM_FEATURES = args.num_features
    config.NUM_PREDICTIONS = args.num_predictions
    config.SAVE_DATASETS = args.save_datasets
    config.INITIAL_TRAINING_PATH = args.initial_training_path
    config.NJOBS = args.njobs
    # Hyperparams
    config.MODELS_TRAINING = args.models_training
    config.ERROR_TYPE_MODELS = args.error_type_models
    config.TRAINING_PERCENTAGE_MODELS = args.training_percentage_hyperparams
    config.TRAIN_TEST_RATIO_MODELS = args.train_test_ratio_hyperparams
    # FFORMA
    config.MODELS_FFORMA = args.models_fforma
    config.ERROR_TYPE_FFORMA = args.error_type_fforma
    config.TRAINING_PERCENTAGE_FFORMA = args.training_percentage_fforma
    config.TRAIN_TEST_RATIO_FFORMA = args.train_test_ratio_fforma
    # Final evaluation
    config.TRAINING_PERCENTAGE_FFORMA_EVALUATION = (
        args.training_percentage_fforma_eval
    )
    config.TRAIN_TEST_RATIO_FFORMA_EVALUATION = (
        args.train_test_ratio_fforma_eval
    )
    config.ERROR_TYPE_FFORMA_EVALUATION = args.error_type_fforma_eval


def main() -> None:
    """Main"""
    parser = generate_parser()

    # Parse args
    args = parser.parse_args()
    process_args(args)

    # Set logger
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    logger.debug("Args: %r", args)

    # Obtain available timeseries
    series = loading.raw.retrieve_csv_series_dict(
        args.dataset_basepath, args.dataset_dir_name
    )

    # Execute select_hyperparameters if mode is: generate-hyperparams or full
    if args.mode in ["generate-hyperparams", "full"]:
        print()
        phases.execute_phase_1(series)
    # Execute select_fforma if mode is:
    # generate-hypeparams, generate-fforma or full
    if args.mode in ["generate-fforma", "full"]:
        print()
        phases.execute_phase_2(series)
    # Execute final_fforma_prediction if mode is: evaluate-fforma or full
    if args.mode in ["evaluate-fforma", "full"]:
        print()
        phases.execute_phase_3(series)
    # Generate CSVs
    if args.mode == "generate-csvs":
        print()
        logger.info("* Generating CSVs...")
        loading.datasets.process_and_save(
            series, args.num_features, args.num_predictions
        )


if __name__ == "__main__":
    main()
