import pandas as pd


def infer_pip_size(close: pd.Series) -> float:
    """0.01 for JPY-quoted pairs (price ~100+), 0.0001 for other majors (price ~1)."""
    return 0.01 if close.median() > 20 else 0.0001
