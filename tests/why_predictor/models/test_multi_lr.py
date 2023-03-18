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

NAME = "Multioutput Linear Regression"
SHORT_NAME = "MULTI_LR"


class TestMultioutputLinearRegressorBasic(unittest.TestCase):
    """Tests for MultioutputLinearRegressor (Basic info)"""

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
        self.model = lr.MultioutputLinearRegressor(
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


class TestMultioutputLinearRegressorModel(unittest.TestCase):
    """Tests for MultioutputLinearRegressor (Model generated)"""

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
        self.model = lr.MultioutputLinearRegressor(
            self.hyperparams, "tests/results"
        )
        # We train the model with first 50 elemnnts
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
                "col74": pd.Series([761.382385, -246.30949], dtype="float32"),
                "col75": pd.Series([158.485702, -841.70159], dtype="float32"),
                "col76": pd.Series([-438.83923, -388.62310], dtype="float32"),
                "col77": pd.Series([-560.08789, 6.992072], dtype="float32"),
                "col78": pd.Series([170.916748, -117.50131], dtype="float32"),
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
                "col73": [2.925793, 0.775114],
                "col74": [0.274873, 1.410515],
                "col75": [0.735857, 6.611343],
                "col76": [3.925594, 2.295410],
                "col77": [2.866959, 0.972031],
                "col78": [0.316333, 1.391671],
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
                1: [4.484938, 5.5783777, 0.2748739, 1.4105159],
                2: [5.0799184, 0.2564695, 0.7358571, 6.611344],
                3: [0.098511204, 0.09713013, 3.9255948, 2.2954104],
                4: [0.025132243, 4.3583302, 2.8669596, 0.9720317],
                5: [4.628258, 2.629805, 0.316333, 1.3916711],
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
        self.assertEqual(value, 1.4010934829711914)
