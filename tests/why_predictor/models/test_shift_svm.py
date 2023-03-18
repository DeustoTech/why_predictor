"""Test for Support Vector regression"""
import json
import os
import shutil
import unittest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pandas.testing import assert_frame_equal  # type: ignore

from why_predictor.errors import ErrorType
from why_predictor.models import svm_regression as svm

NAME = "Shifted Support Vector Regression"
SHORT_NAME = "SHIFT_SVR"


class TestShiftedSVMRegressorBasic(unittest.TestCase):
    """Tests for ShiftedSVMRegressor (Basic info)"""

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
        self.hyperparams = {
            "epsilon": 0.0,
            "tol": 1e-4,
            "C": 1.0,
            "loss": "epsilon_insensitive",
        }
        self.model = svm.ShiftedSupportVectorRegressor(
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
        self.model.generate_model(self.feat.iloc[:50], self.out.iloc[:50])
        self.assertIsNotNone(self.model.model)


class TestShiftedSVMRegressorModel(unittest.TestCase):
    """Tests for ShiftedSVMRegressor (Model generated)"""

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
        self.hyperparams = {
            "epsilon": 0.0,
            "tol": 1e-4,
            "C": 1.0,
            "loss": "epsilon_insensitive",
        }
        self.model = svm.ShiftedSupportVectorRegressor(
            self.hyperparams, "tests/results"
        )
        # We train the model with first 50 samples
        self.model.generate_model(self.feat.iloc[:50], self.out.iloc[:50])

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

    def test_make_predictions_error0(self):
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
        for col in dtf.columns[:2]:
            for i, value in enumerate(dtf[col]):
                self.assertAlmostEqual(value, expected_df[col][i], places=7)

    def test_calculate_errors_error0(self):
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
        for col in errors.columns:
            for i, value in enumerate(errors[col]):
                self.assertAlmostEqual(value, expected_df[col][i], places=2)
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

    def test_calculate_timeseries_errors_error0(self):
        """test calculate_errors"""
        # Init
        median_value = np.nanmean(self.feat.iloc[:, 2:])
        base_path = "tests/results/datasets/test/mydataset"
        for folder in ["features", "output"]:
            os.makedirs(os.path.join(base_path, folder))
        filename = os.path.join(base_path, "features", "mytimeseries.csv.gz")
        self.feat.iloc[:2].to_csv(filename, index=False)
        filename = os.path.join(base_path, "output", "mytimeseries.csv.gz")
        self.out.iloc[:2].to_csv(filename, index=False)
        # Execute
        value = self.model.calculate_timeseries_error(
            ["mydataset", "mytimeseries"], ErrorType.MAPE2, median_value
        )
        # Validate
        self.assertAlmostEqual(value, 0.0, places=2)

    def test_make_predictions(self):
        """Test .make_predictions"""
        # Init
        expected_df = pd.DataFrame(
            {
                "dataset": ["mydataset", "mydataset"],
                "timeseries": ["mytimeseries", "mytimeseries"],
                "col73": pd.Series([999, 249], dtype="float32"),
                "col74": pd.Series([665, -19], dtype="float32"),
                "col75": pd.Series([361, -517], dtype="float32"),
                "col76": pd.Series([-358, -152], dtype="float32"),
                "col77": pd.Series([-219, 428], dtype="float32"),
                "col78": pd.Series([362, -124], dtype="float32"),
            }
        )
        # Execute
        index = self.feat.shape[0] - 2
        dtf = self.model.make_predictions(
            self.feat.iloc[index:].reset_index(drop=True),
            self.out[index:].reset_index(drop=True),
        )
        # Evaluate
        assert_frame_equal(dtf, expected_df, check_exact=False, atol=1)
        # for col in dtf.columns[2:]:
        #     for i, value in enumerate(dtf[col]):
        #        self.assertAlmostEqual(math.floor(value), expected_df[col][i])

    def test_calculate_errors(self):
        """test calculate_errors"""
        # Init
        median_value = np.nanmean(self.feat.iloc[:, 2:])
        expected_df = pd.DataFrame(
            {
                "col73": [2.9966474, 0.762248],
                "col74": [0.3667313, 1.0328848],
                "col75": [0.3981051, 4.451276],
                "col76": [3.3887513, 1.5082067],
                "col77": [1.7306051, 0.7145698],
                "col78": [0.44895753, 1.416437],
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
        assert_frame_equal(
            errors,
            expected_df.astype("float32"),
            check_exact=False,
            atol=0.005,
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
        # * Expected errors
        expected_errors = pd.DataFrame(
            {
                0: [0.16485725, 5.5241218, 2.9964144, 0.76230013],
                1: [5.158454, 5.447012, 0.36668643, 1.0328448],
                2: [5.178847, 0.16818406, 0.3981932, 4.4515333],
                3: [0.1414546, 0.006338908, 3.388778, 1.5081289],
                4: [0.013255106, 4.0453396, 1.7307129, 0.7144476],
                5: [3.9673746, 2.0572717, 0.4487477, 1.4163654],
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
            errors, expected_errors.astype("float32"), atol=0.005
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
        self.assertGreater(value, 0.0)
        self.assertLessEqual(value, 1.30)
