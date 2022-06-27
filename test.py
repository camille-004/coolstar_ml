"""This is a test module."""
import os

import numpy as np
import pandas as pd


def func1(_a: int, _b: int) -> int:
    """This is my first function."""
    return _a + _b


def func2() -> None:
    """This is my second function."""
    print(np.random)
    print(os.path)
    print(pd.io)
