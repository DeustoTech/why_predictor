"""Test functions for panda_utils module"""
import unittest

import numpy as np
import pandas as pd  # type: ignore
from pandas.testing import assert_frame_equal  # type: ignore

from why_predictor import panda_utils as pdu


class TestPandaUtils(unittest.TestCase):
    """test panda_utils module"""

    def setUp(self) -> None:
        self.dtf = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4.0, 5.1, 6.0],
                "col3": [257, 258, 259],
                "col4": [10.0, 11.0, 12.0],
            }
        )

    def test_downcast(self) -> None:
        """Test downcast function"""
        expected_df = pd.DataFrame(
            {
                "col1": pd.Series([1, 2, 3], dtype="uint8"),
                "col2": pd.Series([4.0, 5.1, 6.0], dtype="float32"),
                "col3": pd.Series([257, 258, 259], dtype="uint16"),
                "col4": pd.Series([10.0, 11.0, 12.0], dtype="uint8"),
            }
        )
        downcasted_df = pdu.downcast(self.dtf)
        assert_frame_equal(downcasted_df, expected_df)

    def test_dataframe(self) -> None:
        """Test DataFrame wrapper"""
        expected_df = pd.DataFrame(
            {
                "col1": pd.Series([1, 2, 3], dtype="uint8"),
                "col2": pd.Series([4.0, 5.1, 6.0], dtype="float32"),
                "col3": pd.Series([257, 258, 259], dtype="uint16"),
                "col4": pd.Series([10, 11, 12], dtype="uint8"),
            }
        )
        wrapped_df = pdu.DataFrame(
            data=self.dtf.to_dict("list"), columns=self.dtf.columns
        )
        assert_frame_equal(wrapped_df, expected_df)

    def test_concat(self) -> None:
        """Test concat wrapper"""
        df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df2 = pd.DataFrame({"A": [7, 8, 9], "B": [10, 11, 12]})
        expected_df = pd.DataFrame(
            {
                "A": pd.Series([1, 2, 3, 7, 8, 9], dtype="uint8"),
                "B": pd.Series([4, 5, 6, 10, 11, 12], dtype="uint8"),
            }
        )
        expected_df.index = [0, 1, 2, 0, 1, 2]
        wrapped_df = pdu.concat([df1, df2])
        assert_frame_equal(wrapped_df, expected_df)

    def test_read_csv(self) -> None:
        """Test read_csv wrapper"""
        dtf = pdu.read_csv("tests/data/maxpower.csv.gz")
        self.assertEqual(dtf.kWh.dtype, np.float32)
        self.assertEqual(dtf.imputed.dtype, np.uint8)
