"""Weekly directional Top 3; plans never submit orders."""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import binance_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['download','plan','backtest'])
    p.add_argument('--data-dir', type=Path, default=ROOT/'data'/'binance')
    p.add_argument('--output-dir', type=Path, default=ROOT/'output')
    p.add_argument('--capital', type=float, default=10000.)
    p.add_argument('--offline', action='store_true')
    p.add_argument('--market-source', choices=['gpt','quant'], default='gpt')
    p.add_argument('--news-file', type=Path, help='Optional archived timestamped news JSON')
    p.add_argument('--start'); p.add_argument('--end')
    p.add_argument('--symbols', nargs='+')
    p.add_argument('--universe-size', type=int, default=50)
    args = p.parse_args()
    if args.capital <= 0:
        p.error('capital must be positive')
    if args.command == 'download':
        from download_data import download
        download(args.data_dir, args.start, args.end, args.symbols, args.universe_size)
        return
    import directional
    if args.command == 'plan':
        directional.run(args)
        return
    import weekly
    frames = binance_data.load_universe(cache_dir=args.data_dir, min_bars=90)
    panel, market, features, leaders, clean = weekly.prepare(frames, directional.SETTINGS)
    predictions, audits = [], []
    for date in market.index:
        # Skip initial history only; a failed IC selection after warmup is an error.
        train = weekly.mature(panel, date, directional.SETTINGS)
        mt = weekly.mature(market, date, directional.SETTINGS).dropna(subset=['market_return'])
        if len(mt)<26 or train.loc[train.eligible].dropna(subset=['label_return']).Date.nunique()<26:
            continue
        current, audit = directional.predict(panel, market, features, leaders, date)
        direction = 'BULL' if current.market_prob.iloc[0]>=.5 else 'BEAR'
        predictions.append(directional.rank(current, direction))
        audit['signal_date'] = date; audits.append(audit)
    if not predictions:
        raise ValueError('Insufficient mature weekly history')
    predictions = pd.concat(predictions, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir/'weekly_predictions.csv', index=False)
    pd.concat(audits).to_csv(args.output_dir/'factor_ic_history.csv', index=False)
    curve = weekly.simulate(predictions, clean, directional.SETTINGS, mode='directional')
    curve.to_csv(args.output_dir/'backtest_directional_quant.csv', index=False)
    metrics = {'market_source':'quant_only_no_historical_gpt', **weekly.summary(curve)}
    (args.output_dir/'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    try:
        main()
    except (binance_data.BinanceError, ValueError) as exc:
        raise SystemExit(str(exc))
