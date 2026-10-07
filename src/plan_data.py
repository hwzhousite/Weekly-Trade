"""Online weekly plan data: fresh returned frames only, never stale cache fallback."""
import pandas as pd
import math
import binance_data
from weekly import LEADERS


def refresh_plan_frames(cache_dir, now=None, required_symbols=()):
    universe = binance_data.top_perpetuals()
    symbols = list(dict.fromkeys(universe.symbol.tolist() + list(LEADERS) + list(required_symbols)))
    frames = binance_data.download_universe(symbols=symbols, cache_dir=cache_dir)
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    expected = now.tz_convert('UTC').tz_localize(None).normalize() - pd.Timedelta(days=1)
    fresh = {s: d for s, d in frames.items() if pd.Timestamp(d.index.max()) == expected}
    missing = (set(LEADERS) | set(required_symbols)) - set(fresh)
    if missing:
        raise ValueError(f'Fresh leader history unavailable: {sorted(missing)}; no cached fallback')
    if len(fresh) < 5:
        raise ValueError('Fewer than five fresh eligible histories downloaded')
    print(f'Fresh Binance universe: {len(fresh)} symbols; latest closed UTC bar {expected.date()}')
    return fresh


def current_quotes(symbols):
    """Public futures ticker, informational only; no API key/order permission."""
    rows = []
    for symbol in symbols:
        raw = binance_data._request(f'{binance_data.FAPI}/fapi/v2/ticker/price', {'symbol': symbol})
        price = float(raw['price'])
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f'Invalid current quote for {symbol}')
        rows.append({'Symbol': symbol, 'current_price': price,
                     'quote_time_utc': pd.to_datetime(raw['time'], unit='ms', utc=True).isoformat()})
    return pd.DataFrame(rows)
