import pandas as pd

from utils import infer_pip_size


def test_infers_standard_pip_for_major_pair():
    close = pd.Series([1.05, 1.10, 1.12])
    assert infer_pip_size(close) == 0.0001


def test_infers_jpy_pip_for_yen_pair():
    close = pd.Series([110.5, 115.2, 108.9])
    assert infer_pip_size(close) == 0.01
