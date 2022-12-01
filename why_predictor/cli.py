""" WHY predictor
"""
import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .load_sets import (
    find_csv_files,
    load_files,
    select_training_set,
    split_dataset_in_train_and_test,
)
from .models import linear_regression

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
    )
    parser.add_argument(
        "--dataset-dir-name",
        dest="dataset_dir_name",
        default=os.getenv("DATASET_DIRNAME"),
    )
    parser.add_argument(
        "--percentage-training",
        dest="training_percentage",
        type=float,
        default=os.getenv("TRAINING_PERCENTAGE"),
    )
    return parser


def execute(args: argparse.Namespace) -> None:
    """Execute"""
    basepath = args.dataset_basepath
    dirname = args.dataset_dir_name
    series = find_csv_files(basepath, dirname)
    # Select training set
    training_set, _ = select_training_set(series, args.training_percentage)
    # Load training set
    data = load_files(training_set)
    # Train and test datasets
    (
        train_features,
        train_output,
        test_features,
        test_output,
    ) = split_dataset_in_train_and_test(data)
    # Linear regression
    linear_regression.generate_model(
        train_features, train_output, test_features, test_output
    )


def main() -> None:
    """Main"""
    parser = generate_parser()

    # Parse args
    args = parser.parse_args()

    # Set logger
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    print(args)
    logger.debug("Args: %r", args)
    execute(args)


if __name__ == "__main__":
    main()
