"""Test for module training.models"""
import json
import os
import unittest

import pandas as pd  # type: ignore

from why_predictor import training


class TestFriedmanTests(unittest.TestCase):
    """Tests for errors module"""

    def setUp(self):
        os.makedirs("tests/results/errors/sum")
        os.makedirs("tests/results/errors/raw")
        os.makedirs("tests/results/post-hoc")

    def test_friedman_test_with_post_hoc_complete(self):
        """Test training.friedman_test_with_post_hoc (complete)"""
        # Init
        values = {
            "TEST_1": [4, 6, 3, 4, 3, 2, 2, 7, 6, 5],
            "TEST_2": [5, 6, 8, 7, 7, 8, 4, 6, 4, 5],
            "TEST_3": [2, 2, 5, 3, 2, 2, 1, 4, 3, 2],
        }
        for test in ["TEST_1", "TEST_2", "TEST_3"]:
            pd.DataFrame(
                {
                    "dataset": ["mydataset"] * 10,
                    "timeseries": ["mytimeseries"] * 10,
                    test: values[test],
                }
            ).to_csv(f"tests/results/errors/sum/{test}.csv.gz", index=False)
        # Execute
        training.models.friedman_test_with_post_hoc(
            list(values.keys()), "tests/results"
        )
        # Evaluate
        self.assertTrue(os.path.exists("tests/results/post-hoc/result.txt"))
        self.assertTrue(os.path.exists("tests/results/post-hoc/legend.csv"))
        self.assertTrue(os.path.exists("tests/results/post-hoc/posthoc.png"))
        with open("tests/results/post-hoc/legend.csv", encoding="utf8") as flg:
            self.assertEqual(
                json.loads(flg.read()),
                {"A": "TEST_1", "B": "TEST_2", "C": "TEST_3"},
            )
        with open("tests/results/post-hoc/result.txt", encoding="utf8") as frs:
            self.assertEqual(
                frs.read().strip(),
                "Friedmanchisquare result: "
                + "FriedmanchisquareResult(statistic=13.351351351351344, "
                + "pvalue=0.0012612201221243594)",
            )
