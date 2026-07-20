import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close"]
TIMESTAMP_ALIASES = ("timestamp", "datetime", "date", "time")


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """Load an OHLCV CSV into a standardized DataFrame indexed by UTC timestamp.

    Handles two shapes, auto-detecting delimiter (comma/tab/etc.) either way:
    - A header row naming a timestamp-like column (timestamp/datetime/date/
      time) plus open/high/low/close (case-insensitive), volume optional.
    - A headerless broker export (e.g. MT4/MT5 hourly dumps): 5 columns
      (timestamp, open, high, low, close) or 6 (+ volume), in that order.
    """
    first_cell = str(pd.read_csv(path, sep=None, engine="python", header=None, nrows=1).iloc[0, 0]).strip()
    has_header = first_cell.lower() in TIMESTAMP_ALIASES or not _looks_like_timestamp(first_cell)

    if has_header:
        df = pd.read_csv(path, sep=None, engine="python")
        df.columns = [c.strip().lower() for c in df.columns]
    else:
        df = pd.read_csv(path, sep=None, engine="python", header=None)
        n_cols = df.shape[1]
        if n_cols == 5:
            df.columns = ["timestamp", "open", "high", "low", "close"]
        elif n_cols == 6:
            df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        else:
            raise ValueError(f"Headerless CSV with unexpected column count: {n_cols}")

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


def _looks_like_timestamp(value: str) -> bool:
    try:
        pd.Timestamp(value)
        return True
    except (ValueError, TypeError):
        return False
