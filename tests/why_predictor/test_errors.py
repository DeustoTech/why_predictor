"""Test for module errors"""
import unittest

import numpy as np
import pandas as pd  # type: ignore
from pandas.testing import assert_frame_equal  # type: ignore

from why_predictor import errors


class TestErrorCalculation(unittest.TestCase):
    """Tests for errors module"""

    def setUp(self):
        self.output = pd.DataFrame({"col1": [2, 0, 1], "col2": [1, 1, 1]})
        self.predictions = pd.DataFrame({"col1": [1, 0, 1], "col2": [1, 2, 3]})
        self.unsig_out = pd.DataFrame(
            {
                "col73": pd.Series([800, 200, 250, 1050], dtype="uint16"),
                "col74": pd.Series([200, 250, 1050, 600], dtype="uint16"),
                "col75": pd.Series([250, 1050, 600, 150], dtype="uint16"),
                "col76": pd.Series([1050, 600, 150, 300], dtype="uint16"),
                "col77": pd.Series([600, 150, 300, 250], dtype="uint16"),
                "col78": pd.Series([150, 300, 250, 300], dtype="uint16"),
            }
        )
        self.unsig_pred = pd.DataFrame(
            {
                "col73": pd.Series([200, 1100, 1100, 150], dtype="uint16"),
                "col74": pd.Series([250, 1300, 1300, 200], dtype="uint16"),
                "col75": pd.Series([250, 250, 250, 200], dtype="uint16"),
                "col76": pd.Series([300, 150, 150, 200], dtype="uint16"),
                "col77": pd.Series([300, 450, 450, 200], dtype="uint16"),
                "col78": pd.Series([200, 200, 200, 300], dtype="uint16"),
            }
        )

    def test_calculate_mape2(self):
        """test calculate_mape2"""
        expected = pd.DataFrame(
            {"col1": [0.5, 0.0, 0.0], "col2": [0.0, 1.0, 2.0]}
        )
        mean = 2
        result = errors.calculate_mape2(self.output, self.predictions, mean)
        assert_frame_equal(result, expected)

    def test_calculate_mape2_unsigned(self):
        """test calculate_mape2 (unsigned)"""
        expected = pd.DataFrame(
            {
                "col73": [0.75, 4.5, 3.4, 0.857143],
                "col74": [0.25, 4.2, 0.238095, 0.66666],
                "col75": [0.0, 0.7619048, 0.583333, 0.33333],
                "col76": [0.7142857, 0.75, 0.0, 0.33333],
                "col77": [0.5, 2, 0.5, 0.2],
                "col78": [0.33333, 0.33333, 0.2, 0.0],
            }
        )
        mean = 2
        result = errors.calculate_mape2(self.unsig_out, self.unsig_pred, mean)
        assert_frame_equal(result, expected.astype("float32"), atol=0.00001)

    def test_calculate_mape2_error(self):
        """test calculate_mape2 (different dimensions)"""
        pred = pd.DataFrame(
            {"col1": [0.5, 1, 0], "col2": [0, 1, 2.0], "col3": [0.0, 1, 1]}
        )
        mean = 2
        self.assertRaises(
            ValueError, errors.calculate_mape2, self.output, pred, mean
        )

    def test_calculate_mape(self):
        """test calculate_mape"""
        expected = pd.DataFrame(
            {"col1": [0.5, np.NaN, 0.0], "col2": [0.0, 1.0, 2.0]}
        )
        result = errors.calculate_mape(self.output, self.predictions)
        assert_frame_equal(result, expected)

    def test_calculate_mape_unsigned(self):
        """test calculate_mape (unsigned)"""
        expected = pd.DataFrame(
            {
                "col73": [0.75, 4.5, 3.4, 0.857143],
                "col74": [0.25, 4.2, 0.238095, 0.66666],
                "col75": [0.0, 0.7619048, 0.583333, 0.33333],
                "col76": [0.7142857, 0.75, 0.0, 0.33333],
                "col77": [0.5, 2, 0.5, 0.2],
                "col78": [0.33333, 0.33333, 0.2, 0.0],
            }
        )
        mean = 2
        result = errors.calculate_mape(self.unsig_out, self.unsig_pred, mean)
        assert_frame_equal(result, expected.astype("float32"), atol=0.00001)

    def test_calculate_mape_error(self):
        """test calculate_mape (different dimensions)"""
        pred = pd.DataFrame(
            {"col1": [0.5, 1, 0], "col2": [0, 1, 2.0], "col3": [0.0, 1, 1]}
        )
        self.assertRaises(ValueError, errors.calculate_mape, self.output, pred)

    def test_calculate_mae(self):
        """test calculate_mae"""
        expected = pd.DataFrame({"col1": [1, 0, 0], "col2": [0, 1, 2]})
        result = errors.calculate_mae(self.output, self.predictions)
        assert_frame_equal(result, expected.astype("float32"))

    def test_calculate_mae_unsigned(self):
        """test calculate_mae (unsigned)"""
        expected = pd.DataFrame(
            {
                "col73": [600, 900, 850, 900],
                "col74": [50, 1050, 250, 400],
                "col75": [0, 800, 350, 50],
                "col76": [750, 450, 0, 100],
                "col77": [300, 300, 150, 50],
                "col78": [50, 100, 50, 0],
            }
        )
        mean = 2
        result = errors.calculate_mae(self.unsig_out, self.unsig_pred, mean)
        assert_frame_equal(result, expected.astype("float32"), atol=0.00001)

    def test_calculate_mae_error(self):
        """test calculate_mae (different dimensions)"""
        pred = pd.DataFrame(
            {"col1": [0.5, 1, 0], "col2": [0, 1, 2.0], "col3": [0.0, 1, 1]}
        )
        self.assertRaises(ValueError, errors.calculate_mae, self.output, pred)

    def test_calculate_rmse(self):
        """test calculate_rmse"""
        expected = pd.DataFrame({"col1": [1, 0, 0], "col2": [0, 1, 4]})
        result = errors.calculate_rmse(self.output, self.predictions)
        assert_frame_equal(result, expected.astype("float32"))

    def test_calculate_rmse_unsigned(self):
        """test calculate_rmse (unsigned)"""
        expected = pd.DataFrame(
            {
                "col73": [360000, 810000, 722500, 810000],
                "col74": [2500, 1102500, 62500, 160000],
                "col75": [0, 640000, 122500, 2500],
                "col76": [562500, 202500, 0, 10000],
                "col77": [90000, 90000, 22500, 2500],
                "col78": [2500, 10000, 2500, 0],
            }
        )
        mean = 2
        result = errors.calculate_rmse(self.unsig_out, self.unsig_pred, mean)
        assert_frame_equal(result, expected.astype("float32"), atol=0.00001)

    def test_calculate_rmse_error(self):
        """test calculate_rmse (different dimensions)"""
        pred = pd.DataFrame(
            {"col1": [0.5, 1, 0], "col2": [0, 1, 2.0], "col3": [0.0, 1, 1]}
        )
        self.assertRaises(ValueError, errors.calculate_rmse, self.output, pred)

    def test_calculate_smape(self):
        """test calculate_smape"""
        expected = pd.DataFrame(
            {"col1": [0.6666667, np.NaN, 0.0], "col2": [0.0, 0.666667, 1.0]}
        )
        result = errors.calculate_smape(self.output, self.predictions)
        assert_frame_equal(result, expected)

    def test_calculate_smape_unsigned(self):
        """test calculate_smape (unsigned)"""
        expected = pd.DataFrame(
            {
                "col73": [1.2, 1.3846153846153846, 1.2592592592592593, 1.5],
                "col74": [
                    0.22222222,
                    1.3548387096774193,
                    0.2127659574468085,
                    1.0,
                ],
                "col75": [
                    0.0,
                    1.230769230769,
                    0.823529411764,
                    0.2857142857142,
                ],
                "col76": [1.1111111111111112, 1.2, 0.0, 0.4],
                "col77": [0.6666666666666666, 1.0, 0.4, 0.2222222222222222],
                "col78": [0.2857142857142857, 0.4, 0.2222222222222222, 0.0],
            }
        )
        mean = 2
        result = errors.calculate_smape(self.unsig_out, self.unsig_pred, mean)
        assert_frame_equal(result, expected, atol=0.00001)

    def test_calculate_smape_error(self):
        """test calculate_smape (different dimensions)"""
        prd = pd.DataFrame(
            {"col1": [0.5, 1, 0], "col2": [0, 1, 2.0], "col3": [0.0, 1, 1]}
        )
        self.assertRaises(ValueError, errors.calculate_smape, self.output, prd)

    def test_error_type(self):
        """test ErrorType"""
        expected = pd.DataFrame({"col1": [1, 0, 0], "col2": [0, 1, 2]})
        error_type = errors.ErrorType.MAE
        result = error_type.value(self.output, self.predictions)
        assert_frame_equal(result, expected.astype("float32"))

    def test_error_type_mape2(self):
        """test ErrorType.MAPE2"""
        # Init
        expected = pd.DataFrame(
            {
                "col73": [0.75, 4.5, 3.4, 0.857143],
                "col74": [0.25, 4.2, 0.238095, 0.66666],
                "col75": [0.0, 0.7619048, 0.583333, 0.33333],
                "col76": [0.7142857, 0.75, 0.0, 0.33333],
                "col77": [0.5, 2, 0.5, 0.2],
                "col78": [0.33333, 0.33333, 0.2, 0.0],
            }
        )
        mean = 2
        error_type = errors.ErrorType.MAPE2
        # Execute
        result = error_type.value(self.unsig_out, self.unsig_pred, mean)
        # Validate
        assert_frame_equal(result, expected.astype("float32"), atol=0.00001)
