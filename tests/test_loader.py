import pytest

from data.loader import load_ohlcv_csv


def test_load_ohlcv_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "Timestamp,Open,High,Low,Close,Volume\n"
        "2024-01-01 00:00:00,1.10,1.11,1.09,1.105,1000\n"
        "2024-01-01 01:00:00,1.105,1.12,1.10,1.115,1200\n"
    )
    df = load_ohlcv_csv(str(csv_path))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df.index.tz is not None


def test_load_ohlcv_csv_missing_column_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("timestamp,open,high,close\n2024-01-01,1.1,1.2,1.15\n")
    with pytest.raises(ValueError):
        load_ohlcv_csv(str(csv_path))


def test_load_ohlcv_csv_defaults_missing_volume(tmp_path):
    csv_path = tmp_path / "novolume.csv"
    csv_path.write_text("date,open,high,low,close\n2024-01-01,1.1,1.2,1.05,1.15\n")
    df = load_ohlcv_csv(str(csv_path))
    assert (df["volume"] == 0.0).all()
