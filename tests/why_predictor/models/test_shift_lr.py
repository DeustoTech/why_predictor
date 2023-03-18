"""Test for Linear regression"""
import json
import os
import shutil
import unittest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pandas.testing import assert_frame_equal  # type: ignore

from why_predictor.errors import ErrorType
from why_predictor.models import linear_regression as lr

NAME = "Shifted Linear Regression"
SHORT_NAME = "SHIFT_LR"


class TestShiftedLinearRegressorBasic(unittest.TestCase):
    """Tests for ShiftedLinearRegressor (Basic info)"""

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
        self.hyperparams = {}
        self.model = lr.ShiftedLinearRegressor(
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


class TestShiftedLinearRegressorModel(unittest.TestCase):
    """Tests for ShiftedLinearRegressor (Model generated)"""

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
        self.hyperparams = {}
        self.model = lr.ShiftedLinearRegressor(
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
        assert_frame_equal(dtf, expected_df)

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
        self.assertEqual(value, 0.0)

    def test_make_predictions(self):
        """Test .make_predictions"""
        # Init
        expected_df = pd.DataFrame(
            {
                "dataset": ["mydataset", "mydataset"],
                "timeseries": ["mytimeseries", "mytimeseries"],
                "col73": pd.Series([981.448486, 236.129852], dtype="float32"),
                "col74": pd.Series([640.907958, -38.794788], dtype="float32"),
                "col75": pd.Series([331.659088, -539.009399], dtype="float32"),
                "col76": pd.Series(
                    [-385.909088, -172.548278], dtype="float32"
                ),
                "col77": pd.Series([-242.963989, 412.84536], dtype="float32"),
                "col78": pd.Series([340.854888, -141.747161], dtype="float32"),
            }
        )
        # Execute
        index = self.feat.shape[0] - 2
        dtf = self.model.make_predictions(
            self.feat.iloc[index:].reset_index(drop=True),
            self.out[index:].reset_index(drop=True),
        )
        # Evaluate
        assert_frame_equal(dtf, expected_df)

    def test_calculate_errors(self):
        """test calculate_errors"""
        # Init
        median_value = np.nanmean(self.feat.iloc[:, 2:])
        expected_df = pd.DataFrame(
            {
                "col73": [2.9257939453125, 0.7751144263857886],
                "col74": [0.38961146763392857, 1.0646579806009928],
                "col75": [0.44723485310872396, 4.59339599609375],
                "col76": [3.5727272542317707, 1.5751609293619793],
                "col77": [1.8098799641927084, 0.6513814697265625],
                "col78": [0.3634195556640625, 1.4724905395507812],
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
        assert_frame_equal(errors, expected_df.astype("float32"))
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
                0: [0.18911141, 5.423783, 2.925794, 0.7751144],
                1: [5.0051723, 5.3269787, 0.38961145, 1.064658],
                2: [5.020488, 0.13361944, 0.44723484, 4.593396],
                3: [0.10346098, 0.068815514, 3.572727, 1.575161],
                4: [0.04454132, 4.268916, 1.80988, 0.6513815],
                5: [4.169648, 2.1657588, 0.36341956, 1.4724905],
            }
        )
        # Execute
        errors = self.model.calculate_errors_per_file(
            [("test1", "series1"), ("test2", "series2")],
            ErrorType.MAPE2,
            median_value,
        )
        # Validate
        assert_frame_equal(errors, expected_errors.astype("float32"))
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
        self.assertAlmostEqual(value, 1.268574, places=5)
