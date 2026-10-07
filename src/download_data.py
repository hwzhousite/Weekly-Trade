"""Standalone public Binance data downloader; no model dependencies or API key."""
import json
from pathlib import Path
import pandas as pd
import binance_data
import config

LEADERS = ('BTCUSDT', 'ETHUSDT', 'SOLUSDT')


def download(data_dir, start=None, end=None, symbols=None, universe_size=50):
    if universe_size < 5:
        raise ValueError('universe-size must be at least 5 for cross-sectional factors')
    start = start or config.START_DATE
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end) if end else pd.Timestamp.now(tz='UTC').tz_localize(None).normalize()
    if start_date.tzinfo is not None or end_date.tzinfo is not None or start_date >= end_date:
        raise ValueError('Use timezone-free dates with start < end')
    if symbols is None:
        universe = binance_data.top_perpetuals(n=universe_size)
        selected = universe.symbol.tolist()
    else:
        selected = [s.upper() for s in symbols]
    selected = list(dict.fromkeys(selected + list(LEADERS)))
    frames = binance_data.download_universe(symbols=selected, start=str(start_date.date()),
        end=str(end_date.date()), cache_dir=data_dir)
    manifest = {'downloaded_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'start': str(start_date.date()), 'end_exclusive': str(end_date.date()),
        'requested_symbols': selected, 'failed_or_insufficient_history': sorted(set(selected) - set(frames)),
        'symbols': {s: {'bars': len(d), 'first_bar': str(d.index.min().date()),
                        'last_bar': str(d.index.max().date())} for s, d in frames.items()}}
    dest = Path(data_dir)
    dest.mkdir(parents=True, exist_ok=True)
    temp = dest / 'download_manifest.json.tmp'
    temp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    temp.replace(dest / 'download_manifest.json')
    print(f'Downloaded {len(frames)} histories to {dest.resolve()}')
    print('Run: python main.py plan --offline --capital 10000')
    return frames
