""" WHY predictor
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Union

import argcomplete  # type: ignore
import pandas as pd  # type: ignore
from dotenv import load_dotenv

from .errors import ErrorType
from .load_sets import find_csv_files, get_train_and_test_datasets
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
        "-m",
        "--mode",
        dest="mode",
        choices=["generate-errors", "generate-fforma", "full"],
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
    generate_errors = parser.add_argument_group("Generate Errors")
    generate_errors.add_argument(
        "--percentage-csv-files-for-training-hyperparameters",
        dest="training_percentage_hyperparams",
        type=float,
        default=os.getenv("TRAINING_PERCENTAGE_HYPERPARAMETERS"),
        help="Percentage of the CSV files that will be used for training",
    )
    generate_errors.add_argument(
        "--train-test-ratio-hyperparameters",
        dest="train_test_ratio_hyperparams",
        type=float,
        default=os.getenv("TRAIN_TEST_RATIO_HYPERPARAMETERS"),
        help="ratio of samples used for training "
        + "(1 - this value will be used for testing)",
    )
    generate_errors = parser.add_argument_group("Generate FFORMA")
    generate_errors.add_argument(
        "--percentage-csv-files-for-training-fforma",
        dest="training_percentage_fforma",
        type=float,
        default=os.getenv("TRAINING_PERCENTAGE_FFORMA"),
        help="Percentage of the CSV files that will be used for training",
    )
    generate_errors.add_argument(
        "--train-test-ratio-fforma",
        dest="train_test_ratio_fforma",
        type=float,
        default=os.getenv("TRAIN_TEST_RATIO_FFORMA"),
        help="ratio of samples used for training "
        + "(1 - this value will be used for testing)",
    )
    argcomplete.autocomplete(parser)
    return parser


def select_hyperparameters(
    series: Dict[str, List[str]], args: argparse.Namespace
) -> None:
    """Execute"""
    (
        train_features,
        train_output,
        test_features,
        test_output,
    ) = get_train_and_test_datasets(
        series,
        args.training_percentage_hyperparams,
        args.num_features,
        args.num_predictions,
        args.train_test_ratio_hyperparams,
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
    _save_errors_and_hyperparameters(models_dict, args.error_type)


def _save_errors_and_hyperparameters(
    models_dict: Dict[str, BasicModel], error_name: str
) -> None:
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


def generate_fforma(
    series: Dict[str, List[str]], args: argparse.Namespace
) -> pd.DataFrame:
    """Generate FFORMA ensemble"""
    (
        train_features,
        train_output,
        test_features,
        test_output,
    ) = get_train_and_test_datasets(
        series,
        args.training_percentage_fforma,
        args.num_features,
        args.num_predictions,
        args.train_test_ratio_fforma,
    )
    # Calculate models
    models_dict = _load_models_dict(
        args, train_features, train_output, test_features, test_output
    )
    # Generate FFORMA dataset
    timeseries_dict = _generate_timeseries_dict(models_dict)
    column_names: List[Union[str, float]] = ["timeseries"]
    column_names.extend(["c1", "c2", "c3"])  # TODO to be decided
    column_names.extend([m.short_name for m in models_dict.values()])
    df_fforma = pd.DataFrame(columns=column_names)
    for timeseries_id, timeseries in timeseries_dict.items():
        row_data: List[Union[str, float]] = [timeseries_id, "X", "X", "X"]
        row_data.extend(timeseries.values())
        df_fforma = pd.concat(
            [pd.DataFrame([row_data], columns=df_fforma.columns), df_fforma],
            ignore_index=True,
        )
    df_fforma.to_csv("fforma.csv", index=False)
    return df_fforma


def _load_models_dict(
    args: argparse.Namespace,
    train_features: pd.DataFrame,
    train_output: pd.DataFrame,
    test_features: pd.DataFrame,
    test_output: pd.DataFrame,
) -> Dict[str, BasicModel]:
    models_dict = {}
    for model_name in args.models:
        filename = f"hyperparameters/{model_name}.json"
        try:
            with open(filename, encoding="utf8") as fhyper:
                hyperparameters = json.loads(fhyper.read())
        except FileNotFoundError:
            logger.error("File '%s' not found. Aborting...", filename)
            sys.exit(1)
        models_dict[model_name] = Models[model_name].value(
            train_features,
            train_output,
            ErrorType[args.error_type],
            hyperparameters,
        )
        models_dict[model_name].fit(
            test_features,
            test_output,
        )
    return models_dict


def _generate_timeseries_dict(
    models_dict: Dict[str, BasicModel]
) -> Dict[str, Dict[str, float]]:
    timeseries_dict: Dict[str, Dict[str, float]] = {}
    for model in models_dict.values():
        for timeseries, error_metric in model.fitted["errors"].groupby(
            "timeseries"
        ):
            if timeseries not in timeseries_dict:
                timeseries_dict[timeseries] = {}
            timeseries_dict[timeseries][model.short_name] = (
                error_metric.drop("timeseries", axis=1).stack().median()
            )
    return timeseries_dict


def main() -> None:
    """Main"""
    parser = generate_parser()

    # Parse args
    args = parser.parse_args()

    # Set logger
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.debug("Args: %r", args)
    series = find_csv_files(args.dataset_basepath, args.dataset_dir_name)
    # Execute select_hyperparameters if mode is [generate-errors or full]
    if args.mode != "generate-fforma":
        logger.info("* Selecting best hyperparameters...")
        select_hyperparameters(series, args)
    # Execute select_fforma if mode is [generate-fforma or full]
    if args.mode != "generate-errors":
        logger.info("* Generating FFORMA...")
        generate_fforma(series, args)


if __name__ == "__main__":
    main()
