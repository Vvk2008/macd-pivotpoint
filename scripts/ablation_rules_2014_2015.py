"""Same core-rules ablation as scripts/ablation_rules.py, restricted to a
single 2-year window (2014-01-01 to 2016-01-01) instead of the full ~11-year
in-sample period -- checks whether the per-rule verdicts hold up in a much
shorter, single-regime slice rather than being an artifact of averaging over
a decade of mixed market conditions.

Reuses the exact same config list, run_config logic, and R-multiple metric
from ablation_rules.py -- only the date window differs, via pre-populating
that module's df cache before any config is run.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.ablation_rules as ar
from data.loader import load_ohlcv_csv

WINDOW_START = pd.Timestamp("2014-01-01", tz="UTC")
WINDOW_END = pd.Timestamp("2016-01-01", tz="UTC")

for pair in ar.PAIRS:
    full = load_ohlcv_csv(f"data/raw/{pair}.csv")
    ar._df_cache[pair] = full[(full.index >= WINDOW_START) & (full.index < WINDOW_END)]

if __name__ == "__main__":
    sample_pair = ar.PAIRS[0]
    df = ar._df_cache[sample_pair]
    print(f"Window: {WINDOW_START.date()} -> {WINDOW_END.date()}  ({len(df)} bars per pair, e.g. {sample_pair})")
    ar.main()
