"""Test for Stochastic Gradient Descent regression"""
import json
import math
import os
import shutil
import unittest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from why_predictor.errors import ErrorType
from why_predictor.models import sgd_regressor as sgd

NAME = "Chained Stochastic Gradient Descent Regressor"
SHORT_NAME = "CHAIN_SGD"


class TestChainedStochasticGradientDescentRegressorBasic(unittest.TestCase):
    """Tests for ChainedStochasticGradientDescentRegressor (Basic info)"""

    def setUp(self):
        num_features = 72
        os.makedirs("tests/results/models")
        os.makedirs("tests/results/hyperparameters")
        os.makedirs("tests/results/predictions")
        os.makedirs("tests/results/errors/sum")
        os.makedirs("tests/results/errors/raw")
        dtf = pd.read_csv("tests/data/rolling.csv.gz")
        self.feat = dtf.iloc[:, : num_features + 2]
        self.out = dtf.drop(dtf.iloc[:, 2 : num_features + 2], axis=1)
        self.hyperparams = {"alpha": 0.0001}
        self.model = sgd.ChainedStochasticGradientDescentRegressor(
            self.hyperparams, "tests/results"
        )

    def tearDown(self):
        shutil.rmtree("tests/results")

    def test_hyperparams(self):
        """Test .hyperparams"""
        self.assertEqual(self.model.hyperparams, self.hyperparams)

    def test_paramsname(self):
        """Test .params name"""
        self.assertEqual(
            self.model.paramsname,
            f"{SHORT_NAME}_{json.dumps(self.hyperparams)}",
        )

    def test_name(self):
        """Test .name"""
        self.assertEqual(self.model.name, NAME)

    def test_short_name(self):
        """Test .short_name"""
        self.assertEqual(self.model.short_name, SHORT_NAME)

    def test_path(self):
        """Test .path"""
        self.assertEqual(
            self.model.path,
            "tests/results/models/"
            + f"{SHORT_NAME}_{json.dumps(self.hyperparams)}",
        )

    def test_load_model_error(self):
        """Test .load_model"""
        self.assertRaises(FileNotFoundError, self.model.load_model)

    def test_generate_model(self):
        """Test .generate_model"""
        self.model.generate_model(self.feat, self.out)
        self.assertIsNotNone(self.model.model)


class TestChainedStochasticGradientDescentRegressorModel(unittest.TestCase):
    """Tests for ChainedStochasticGradientDescentRegressor (Model generated)"""

    def setUp(self):
        num_features = 72
        os.makedirs("tests/results/models")
        os.makedirs("tests/results/hyperparameters")
        os.makedirs("tests/results/predictions")
        os.makedirs("tests/results/errors/sum")
        os.makedirs("tests/results/errors/raw")
        dtf = pd.read_csv("tests/data/rolling.csv.gz")
        self.feat = dtf.iloc[:, : num_features + 2]
        self.out = dtf.drop(dtf.iloc[:, 2 : num_features + 2], axis=1)
        self.hyperparams = {"alpha": 0.0001}
        self.model = sgd.ChainedStochasticGradientDescentRegressor(
            self.hyperparams, "tests/results"
        )
        # We train the model with half the samples
        self.model.generate_model(
            self.feat.iloc[: math.ceil(self.feat.shape[0] / 2)],
            self.out.iloc[: math.ceil(self.out.shape[0] / 2)],
        )

    def tearDown(self):
        shutil.rmtree("tests/results")

    def test_clear_model(self):
        """test .clear_model"""
        self.model.clear_model()
        with self.assertRaises(FileNotFoundError):
            print(self.model.model is not None)

    def test_save_model(self):
        """test .save_model"""
        model_path = (
            "tests/results/models/"
            + f"{SHORT_NAME}_{json.dumps(self.hyperparams)}"
        )
        self.model.save_model()
        self.assertTrue(os.path.exists(model_path))

    def test_load_model_error(self):
        """Test .load_model"""
        self.model.save_model()
        self.model.clear_model()
        self.assertIsNotNone(self.model.model)

    def test_make_predictions(self):
        """Test .make_predictions"""
        # Init
        expected_deltas = {
            "col73": [np.inf, np.inf],
            "col74": [np.inf, np.inf],
            "col75": [np.inf, np.inf],
            "col76": [np.inf, np.inf],
            "col77": [np.inf, np.inf],
            "col78": [np.inf, np.inf],
        }
        index = self.feat.shape[0] - 2
        feat = self.feat.iloc[index:].reset_index(drop=True)
        out = self.out[index:].reset_index(drop=True)
        # Execute
        dtf = self.model.make_predictions(feat, out)
        # Evaluate
        for col in dtf.columns[2:]:
            for i, value in enumerate(dtf[col]):
                self.assertAlmostEqual(
                    value, out[col][i], delta=expected_deltas[col][i]
                )

    def test_calculate_errors(self):
        """test calculate_errors"""
        # Init
        median_value = np.nanmean(self.feat.iloc[:, 2:])
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
        for col in errors.columns:
            for value in errors[col]:
                self.assertGreaterEqual(value, 0.0)
                self.assertGreaterEqual(value, 100000000000)
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

    def test_calculate_errors_per_file(self):
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
        # Execute
        errors = self.model.calculate_errors_per_file(
            [("test1", "series1"), ("test2", "series2")],
            ErrorType.MAPE2,
            median_value,
        )
        # Validate
        self.assertEqual(errors.shape, (4, 6))
        for col in errors.columns[2:]:
            self.assertIn(errors[col].dtype, [np.float32, np.uint64])
            for value in errors[col]:
                self.assertGreaterEqual(value, 0.0)
                self.assertGreaterEqual(value, 10000000000)
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

    def test_calculate_timeseries_errors(self):
        """test calculate_timeseries_errors"""
        # Init
        index = self.feat.shape[0] - 2
        median_value = np.nanmean(self.feat.iloc[:, 2:])
        base_path = "tests/results/datasets/test/mydataset"
        for folder in ["features", "output"]:
            os.makedirs(os.path.join(base_path, folder))
        filename = os.path.join(base_path, "features", "mytimeseries.csv.gz")
        self.feat.iloc[index:].to_csv(filename, index=False)
        filename = os.path.join(base_path, "output", "mytimeseries.csv.gz")
        self.out.iloc[index:].to_csv(filename, index=False)
        # Execute
        value = self.model.calculate_timeseries_error(
            ["mydataset", "mytimeseries"], ErrorType.MAPE2, median_value
        )
        # Validate
        self.assertGreaterEqual(value, 0.0)
        self.assertGreaterEqual(value, 100000000000)
