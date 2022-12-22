"""Pandas utils"""
from typing import Any, Optional

import pandas as pd  # type: ignore


def downcast(dtf: pd.DataFrame) -> pd.DataFrame:
    """Downcast the dataframe"""
    for col in dtf.select_dtypes("number"):
        dtf[col] = pd.to_numeric(dtf[col], downcast="unsigned")
        if dtf[col].dtype == "float":
            dtf[col] = pd.to_numeric(dtf[col], downcast="float")
    return dtf


def DataFrame(  # pylint: disable=W0631
    data: Optional[Any] = None, columns: Optional[Any] = None
) -> pd.DataFrame:
    """DataFrame wrapper"""
    return downcast(pd.DataFrame(data, columns=columns))


def concat(
    objs: Any, axis: int = 0, ignore_index: bool = False
) -> pd.DataFrame:
    """concat wrapper"""
    return downcast(pd.concat(objs, axis=axis, ignore_index=ignore_index))


def read_csv(filename: str, usecols: Optional[Any] = None) -> pd.DataFrame:
    """read_csv wrapper"""
    return downcast(pd.read_csv(filename, usecols=usecols))
