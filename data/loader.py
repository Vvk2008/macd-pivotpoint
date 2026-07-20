import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close"]
TIMESTAMP_ALIASES = ("timestamp", "datetime", "date", "time")


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """Load an OHLCV CSV into a standardized DataFrame indexed by UTC timestamp.

    Expects one timestamp-like column (timestamp/datetime/date/time) plus
    open, high, low, close (case-insensitive). Volume is optional and
    defaults to 0 if absent.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    ts_col = next((c for c in TIMESTAMP_ALIASES if c in df.columns), None)
    if ts_col is None:
        raise ValueError(
            f"No timestamp column found. Expected one of {TIMESTAMP_ALIASES}, got {list(df.columns)}"
        )

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns {missing}. Got {list(df.columns)}")

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.set_index(ts_col).sort_index()
    df.index.name = "timestamp"

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[~df.index.duplicated(keep="last")]
    return df[["open", "high", "low", "close", "volume"]].astype(float)
