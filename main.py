"""Run from any cwd. Reads Daily-Trade-compatible cached daily parquet files."""
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
    p.add_argument('command', choices=['download', 'backtest', 'plan'])
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'binance')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'output')
    p.add_argument('--capital', type=float, default=10000.)
    p.add_argument('--as-of', help='Saturday UTC bar date; required for historical plans')
    p.add_argument('--fee-bps', type=float, default=10.)
    p.add_argument('--slippage-bps', type=float, default=5.)
    p.add_argument('--offline', action='store_true', help='Explicitly skip Binance refresh/quotes for cached research plans')
    p.add_argument('--start', default=None, help='Download start date YYYY-MM-DD')
    p.add_argument('--end', default=None, help='Download exclusive end date YYYY-MM-DD')
    p.add_argument('--symbols', nargs='+', help='Download these symbols instead of the current top-volume universe')
    p.add_argument('--universe-size', type=int, default=50)
    args = p.parse_args()
    if args.capital <= 0 or min(args.fee_bps, args.slippage_bps) < 0:
        p.error('capital must be positive and costs nonnegative')
    if args.command == 'download':
        from download_data import download
        download(args.data_dir, args.start, args.end, args.symbols, args.universe_size)
        return
    import weekly
    import plan_data
    settings = weekly.Settings(fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
    now = None
    if args.as_of:
        as_of = pd.Timestamp(args.as_of)
        if as_of.dayofweek != 5:
            p.error('--as-of must be a Saturday UTC bar date')
        now = (as_of + pd.Timedelta(days=1)).tz_localize('UTC')
        if now > pd.Timestamp.now(tz='UTC'):
            p.error('--as-of bar has not closed')
    if args.command == 'plan' and not args.offline:
        frames = plan_data.refresh_plan_frames(args.data_dir)
    else:
        frames = binance_data.load_universe(cache_dir=args.data_dir, min_bars=90)
    panel, market, features, leaders, clean = weekly.prepare(frames, settings, now=now)
    predictions, priors = weekly.score(panel, market, features, leaders, settings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == 'backtest':
        predictions.to_csv(args.output_dir / 'weekly_predictions.csv', index=False)
        priors.to_csv(args.output_dir / 'market_oof.csv', index=False)
        report = {'market': weekly.market_metrics(priors), 'strategies': {}}
        for mode in ('baseline', 'conditional_full', 'conditional'):
            curve = weekly.simulate(predictions, clean, settings, mode)
            curve.to_csv(args.output_dir / f'backtest_{mode}.csv', index=False)
            report['strategies'][mode] = weekly.summary(curve)
        (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    else:
        date = predictions.Date.max()
        expected = pd.Timestamp.now(tz='UTC').tz_localize(None).normalize()
        expected -= pd.Timedelta(days=(expected.dayofweek - 5) % 7)
        if expected + pd.Timedelta(days=1) > pd.Timestamp.now(tz='UTC').tz_localize(None):
            expected -= pd.Timedelta(days=7)
        if args.as_of and date != pd.Timestamp(args.as_of):
            raise ValueError('Requested week unavailable or insufficient training history')
        if not args.as_of and date != expected:
            raise ValueError('Latest weekly signal is stale; refresh cache before generating plan')
        plan = weekly.allocation(predictions.loc[predictions.Date == date], settings)
        ranges = weekly.price_ranges(panel, priors, features, leaders, date, plan.Symbol.tolist(), settings)
        cols = ['Date', 'entry_date', 'exit_date', 'Symbol', 'rank', 'market_prob', 'stance', 'pred_net_7d', 'target_weight']
        plan = plan[cols].copy()
        plan = plan.merge(ranges, on='Symbol', validate='one_to_one')
        if not args.offline and not args.as_of:
            plan = plan.merge(plan_data.current_quotes(plan.Symbol.tolist()), on='Symbol', validate='one_to_one')
        plan['data_mode'] = 'offline_cache' if args.offline else 'binance_refreshed'
        plan['target_notional'] = plan.target_weight * args.capital
        # Actual quantity must use live execution quotes, never the signal close.
        path = args.output_dir / f'plan_{date:%Y-%m-%d}.csv'
        plan.to_csv(path, index=False)
        print(plan.to_string(index=False))
        print(f'Cash weight: {1 - plan.target_weight.sum():.1%}; saved {path}')
        print('Target plan only: reconcile actual positions and use live quotes to determine order quantities.')

if __name__ == '__main__':
    try:
        main()
    except binance_data.BinanceError as exc:
        raise SystemExit(str(exc))
