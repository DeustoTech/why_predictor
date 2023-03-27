"""Test for Linear regression"""
import json
import os
import shutil
import unittest
from typing import Any, Dict

import numpy as np
import pandas as pd  # type: ignore
from pandas.testing import assert_frame_equal  # type: ignore

from why_predictor.errors import ErrorType
from why_predictor.models import linear_regression as lr

NAME = "Chained Linear Regression"
SHORT_NAME = "CHAIN_LR"


class TestChainedLinearRegressorModel(unittest.TestCase):
    """Tests for ChainedLinearRegressor (Model generated)"""

    def setUp(self) -> None:
        num_features = 72
        os.makedirs("tests/results/models")
        os.makedirs("tests/results/hyperparameters")
        os.makedirs("tests/results/predictions")
        os.makedirs("tests/results/errors/sum")
        os.makedirs("tests/results/errors/raw")
        dtf = pd.read_csv("tests/data/rolling.csv.gz")
        self.feat = dtf.iloc[:, : num_features + 2]
        self.out = dtf.drop(dtf.iloc[:, 2 : num_features + 2], axis=1)
        self.hyperparams: Dict[str, Any] = {}
        self.model = lr.ChainedLinearRegressor(
            self.hyperparams, "tests/results"
        )
        # We train the model with first 50 samples
        self.model.generate_model(self.feat.iloc[:50], self.out.iloc[:50])

    def tearDown(self) -> None:
        shutil.rmtree("tests/results")

    def test_clear_model(self) -> None:
        """test .clear_model"""
        self.model.clear_model()
        with self.assertRaises(FileNotFoundError):
            print(self.model.model is not None)

    def test_save_model(self) -> None:
        """test .save_model"""
        model_path = (
            "tests/results/models/"
            + f"{SHORT_NAME}_{json.dumps(self.hyperparams)}"
        )
        self.model.save_model()
        self.assertTrue(os.path.exists(model_path))

    def test_load_model_error(self) -> None:
        """Test .load_model"""
        self.model.save_model()
        self.model.clear_model()
        self.assertIsNotNone(self.model.model)

    def test_make_predictions_error0(self) -> None:
        """Test .make_predictions"""
        # Init
        expected_df = pd.DataFrame(
            {
                "dataset": ["mydataset", "mydataset"],
                "timeseries": ["mytimeseries", "mytimeseries"],
                "col73": pd.Series([300, 200], dtype="uint16"),
                "col74": pd.Series([200.0, 200.0], dtype="uint8"),
                "col75": pd.Series([200.0, 200.0], dtype="uint8"),
                "col76": pd.Series([200.0, 200.0], dtype="uint8"),
                "col77": pd.Series([200.0, 200.0], dtype="uint8"),
                "col78": pd.Series([200.0, 250.0], dtype="uint8"),
            }
        )
        # Execute
        dtf = self.model.make_predictions(
            self.feat.iloc[:2], self.out.iloc[:2]
        )
        # Evaluate
        assert_frame_equal(dtf, expected_df)

    def test_calculate_errors_error0(self) -> None:
        """test calculate_errors"""
        # Init
        median_value = np.nanmean(self.feat.iloc[:, 2:])
        expected_df = pd.DataFrame(
            {
                "col73": pd.Series([0, 0], dtype="uint8"),
                "col74": pd.Series([0, 0], dtype="uint8"),
                "col75": pd.Series([0, 0], dtype="uint8"),
                "col76": pd.Series([0, 0], dtype="uint8"),
                "col77": pd.Series([0, 0], dtype="uint8"),
                "col78": pd.Series([0, 0], dtype="uint8"),
            }
        )
        # Execute
        errors = self.model.calculate_errors(
            (self.feat.iloc[:2], self.out[:2]), ErrorType.MAPE2, median_value
        )
        # Validate
        assert_frame_equal(errors, expected_df)
        self.assertTrue(
            os.path.exists(
                "tests/results/errors/raw/"
                + f"{ErrorType.MAPE2.name}_{SHORT_NAME}_"
                + f"{json.dumps(self.hyperparams)}.csv.gz"
            )
        )
        self.assertTrue(
            os.path.exists(
                "tests/results/errors/sum/"
                + f"{SHORT_NAME}_{json.dumps(self.hyperparams)}.csv.gz"
            )
        )

    def test_make_predictions(self) -> None:
        """Test .make_predictions"""
        # Init
        expected_df = pd.DataFrame(
            {
                "dataset": ["mydataset", "mydataset"],
                "timeseries": ["mytimeseries", "mytimeseries"],
                "col73": pd.Series([981.448486, 236.129852], dtype="float32"),
                "col74": pd.Series([752.9654, -242.86504], dtype="float32"),
                "col75": pd.Series([156.44662, -843.453], dtype="float32"),
                "col76": pd.Series([-446.6729, -391.30585], dtype="float32"),
                "col77": pd.Series([-562.9962, 3.8677747], dtype="float32"),
                "col78": pd.Series([163.3786, -118.470825], dtype="float32"),
            }
        )
        # Execute
        index = self.feat.shape[0] - 2
        dtf = self.model.make_predictions(
            self.feat.iloc[index:].reset_index(drop=True),
            self.out[index:].reset_index(drop=True),
        )
        # Evaluate
        assert_frame_equal(dtf, expected_df, atol=18)

    def test_calculate_errors(self) -> None:
        """test calculate_errors"""
        # Init
        median_value = np.nanmean(self.feat.iloc[:, 2:])
        expected_df = pd.DataFrame(
            {
                "col73": [2.9257939453125, 0.7751144263857886],
                "col74": [0.2828901, 1.4047751],
                "col75": [0.7392556, 6.62302],
                "col76": [3.9778194, 2.3043528],
                "col77": [2.8766541, 0.98452896],
                "col78": [0.34648558, 1.3949027],
            }
        )
        # Execute
        index = self.feat.shape[0] - 2
        errors = self.model.calculate_errors(
            (
                self.feat.iloc[index:].reset_index(drop=True),
                self.out[index:].reset_index(drop=True),
            ),
            ErrorType.MAPE2,
            median_value,
        )
        # Validate
        assert_frame_equal(errors, expected_df.astype("float32"), atol=0.09)
        self.assertTrue(
            os.path.exists(
                "tests/results/errors/raw/"
                + f"{ErrorType.MAPE2.name}_{SHORT_NAME}_"
                + f"{json.dumps(self.hyperparams)}.csv.gz"
            )
        )
        self.assertTrue(
            os.path.exists(
                "tests/results/errors/sum/"
                + f"{SHORT_NAME}_{json.dumps(self.hyperparams)}.csv.gz"
            )
        )

    def test_calculate_errors_per_file(self) -> None:
        """test calculate_errors_per_file"""
        # Init
        median_value = np.nanmean(self.feat.iloc[:, 2:])
        index1 = self.feat.shape[0] - 4
        index2 = self.feat.shape[0] - 2
        # - Save features dtf 1
        os.makedirs("tests/results/test/test1/features", exist_ok=True)
        (
            self.feat.iloc[index1:index2]
            .replace("mydataset", "test1")
            .replace("mytimeseries", "series1")
            .to_csv(
                "tests/results/test/test1/features/series1.csv.gz",
                index=False,
            )
        )
        # - Save output dtf 1
        os.makedirs("tests/results/test/test1/output", exist_ok=True)
        (
            self.out.iloc[index1:index2]
            .replace("mydataset", "test1")
            .replace("mytimeseries", "series1")
            .to_csv(
                "tests/results/test/test1/output/series1.csv.gz",
                index=False,
            )
        )
        # - Save features dtf 2
        os.makedirs("tests/results/test/test2/features", exist_ok=True)
        (
            self.feat.iloc[index2:]
            .replace("mydataset", "test2")
            .replace("mytimeseries", "series2")
            .to_csv(
                "tests/results/test/test2/features/series2.csv.gz",
                index=False,
            )
        )
        # - Save output dtf 2
        os.makedirs("tests/results/test/test2/output", exist_ok=True)
        (
            self.out.iloc[index2:]
            .replace("mydataset", "test2")
            .replace("mytimeseries", "series2")
            .to_csv(
                "tests/results/test/test2/output/series2.csv.gz",
                index=False,
            )
        )
        # * Expected errors
        expected_errors = pd.DataFrame(
            {
                0: [0.18911141, 5.423783, 2.925794, 0.7751144],
                1: [4.480224, 5.555337, 0.2828901, 1.4047751],
                2: [5.09518, 0.2504291, 0.7392556, 6.62302],
                3: [0.102073684, 0.10299123, 3.9778194, 2.3043528],
                4: [0.019848429, 4.3737288, 2.8766541, 0.98452896],
                5: [4.6326957, 2.622863, 0.34648558, 1.3949027],
            }
        )
        # Execute
        errors = self.model.calculate_errors_per_file(
            [("test1", "series1"), ("test2", "series2")],
            ErrorType.MAPE2,
            median_value,
        )
        # Validate
        assert_frame_equal(
            errors, expected_errors.astype("float32"), atol=0.09
        )
        self.assertTrue(
            os.path.exists(
                "tests/results/errors/raw/"
                + f"{ErrorType.MAPE2.name}_{SHORT_NAME}_"
                + f"{json.dumps(self.hyperparams)}.csv.gz"
            )
        )
        self.assertTrue(
            os.path.exists(
                "tests/results/errors/sum/"
                + f"{SHORT_NAME}_{json.dumps(self.hyperparams)}.csv.gz"
            )
        )


class TestChainedLinearRegressorBasic(unittest.TestCase):
    """Tests for ChainedLinearRegressor (Basic info)"""

    def setUp(self) -> None:
        num_features = 72
        os.makedirs("tests/results/models")
        os.makedirs("tests/results/hyperparameters")
        os.makedirs("tests/results/predictions")
        os.makedirs("tests/results/errors/sum")
        os.makedirs("tests/results/errors/raw")
        dtf = pd.read_csv("tests/data/rolling.csv.gz")
        self.feat = dtf.iloc[:, : num_features + 2]
        self.out = dtf.drop(dtf.iloc[:, 2 : num_features + 2], axis=1)
        self.hyperparams: Dict[str, Any] = {}
        self.model = lr.ChainedLinearRegressor(
            self.hyperparams, "tests/results"
        )

    def tearDown(self) -> None:
        shutil.rmtree("tests/results")

    def test_generate_model(self) -> None:
        """Test .generate_model"""
        self.model.generate_model(self.feat, self.out)
        self.assertIsNotNone(self.model.model)
