"""Data/factor defaults shared with the Daily-Trade adapters.
Strategy and model settings live in weekly.Settings.
"""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
BINANCE_DIR = DATA_DIR / 'binance'
SHORT_HISTORY_PATH = DATA_DIR / 'short_history.parquet'
UNIVERSE_SIZE = 50
MIN_HISTORY_DAYS = 400
START_DATE = '2023-09-21'
BENCHMARK_SYMBOL = 'BTCUSDT'
WARMUP_DAYS = 90
