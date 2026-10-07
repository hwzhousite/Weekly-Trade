"""Monday frozen selection; daily conditional ranges to the SAME Sunday end."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import weekly
import binance_data
import plan_data


def clock(now=None):
    t = pd.Timestamp.now(tz='Asia/Shanghai') if now is None else pd.Timestamp(now)
    if t.tzinfo is None:
        t = t.tz_localize('Asia/Shanghai')
    t = t.tz_convert('Asia/Shanghai')
    if t.hour < 8:
        raise ValueError('Run after 08:00 Beijing time, when the previous UTC bar is closed')
    monday = t.tz_localize(None).normalize() - pd.Timedelta(days=t.dayofweek)
    latest = t.tz_convert('UTC').tz_localize(None).normalize() - pd.Timedelta(days=1)
    return t, monday, latest


def remaining_range(panel, clean, priors, features, leaders, latest, symbols, frozen_prob, settings):
    """Same weekday historical examples -> remaining Tuesday/Sunday etc. extrema."""
    horizon = 7 - ((latest.dayofweek + 1) % 7)
    data = panel.loc[panel.Date.dt.dayofweek == latest.dayofweek].copy()
    data['week_signal'] = data.Date - pd.to_timedelta((data.Date.dt.dayofweek + 1) % 7, unit='D')
    data = data.merge(priors[['Date', 'market_prob']].rename(columns={'Date':'week_signal'}), on='week_signal', validate='many_to_one')
    targets = []
    for symbol, raw in clean.items():
        hi = pd.concat([raw.High.shift(-k) for k in range(1, horizon + 1)], axis=1)
        lo = pd.concat([raw.Low.shift(-k) for k in range(1, horizon + 1)], axis=1)
        targets.append(pd.DataFrame({'Date':raw.index, 'Symbol':symbol,
            'range_hi':np.log(hi.max(axis=1).where(hi.count(axis=1)==horizon) / raw.Close),
            'range_lo':np.log(lo.min(axis=1).where(lo.count(axis=1)==horizon) / raw.Close)}))
    data = data.merge(pd.concat(targets), on=['Date','Symbol'], validate='one_to_one')
    data['range_end'] = data.Date + pd.Timedelta(days=horizon + 1)
    cols = list(dict.fromkeys(features + leaders + ['market_prob']))
    def predict(date, rows):
        train = data.loc[(data.range_end <= date + pd.Timedelta(days=1)) &
                         (data.Date >= date-pd.Timedelta(weeks=settings.train_weeks)) & data.eligible].dropna(subset=['range_hi','range_lo'])
        if train.Date.nunique() < settings.selection_min_weeks:
            return None
        out = rows[['Symbol','Close']].copy()
        for name, alpha in [('lo',.05),('hi',.95)]:
            model = LGBMRegressor(objective='quantile', alpha=alpha, n_estimators=100,
                learning_rate=.03, num_leaves=7, max_depth=3, min_child_samples=30,
                reg_lambda=5., n_jobs=1, random_state=42, verbosity=-1)
            model.fit(train[cols], train['range_'+name])
            out[name] = model.predict(rows[cols])
        return out
    current = data.loc[(data.Date == latest) & data.Symbol.isin(symbols)].copy()
    current['market_prob'] = frozen_prob
    result = predict(latest, current)
    if result is None or set(result.Symbol) != set(symbols):
        raise ValueError('Insufficient current features or mature historical samples for frozen Top 3 ranges')
    residuals = []
    dates = sorted(data.loc[data.range_end <= latest+pd.Timedelta(days=1), 'Date'].unique())[-12:]
    for date in dates:
        rows = data.loc[(data.Date==date) & data.eligible].dropna(subset=['range_hi','range_lo'])
        if rows.empty:
            continue
        pred = predict(pd.Timestamp(date), rows)
        if pred is not None:
            pred = pred.merge(rows[['Symbol','range_hi','range_lo']], on='Symbol')
            pred['Date'] = date
            residuals.append(pred)
    cal = pd.concat(residuals) if residuals else pd.DataFrame()
    n = cal.Date.nunique() if not cal.empty else 0
    a = min(0., float((cal.range_lo-cal.lo).quantile(.05))) if n >= 4 else 0.
    b = max(0., float((cal.range_hi-cal.hi).quantile(.95))) if n >= 4 else 0.
    result['range_reference_close'] = result.Close
    result['week_low_price'] = result.Close * np.exp(np.minimum(result.lo+a, result.hi+b))
    result['week_high_price'] = result.Close * np.exp(np.maximum(result.lo+a, result.hi+b))
    result['range_start_utc'] = latest + pd.Timedelta(days=1)
    result['range_end_exclusive_utc'] = latest + pd.Timedelta(days=horizon+1)
    result['remaining_days'] = horizon
    result['range_calibration'] = 'pooled_oof_empirical' if n>=4 else 'uncalibrated'
    result['range_calibration_weeks'] = n
    result['range_nominal_coverage'] = .90
    return result.drop(columns=['Close','lo','hi'])


def run(command, data_dir, output_dir, capital=10000., offline=False, now=None, fee_bps=10., slippage_bps=5.):
    t, monday, latest = clock(now)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / f'week_{monday:%Y-%m-%d}.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if command == 'plan' and state is None and t.dayofweek != 0:
        raise ValueError('Create the frozen weekly plan on Monday; daily updates require that saved plan')
    if command == 'daily' and state is None:
        raise ValueError('No frozen plan for this week; run plan on Monday first')
    required = [p['Symbol'] for p in state['picks']] if state else []
    frames = binance_data.load_universe(cache_dir=data_dir, min_bars=90) if offline else plan_data.refresh_plan_frames(data_dir, required_symbols=required)
    if any(s not in frames for s in required):
        raise ValueError('Frozen Top 3 missing from current histories; cannot substitute another symbol')
    settings = weekly.Settings(signal_weekday=6, entry_delay=1, fee_bps=fee_bps, slippage_bps=slippage_bps)
    panel, market, features, leaders, clean = weekly.prepare(frames, settings, now=t.tz_convert('UTC'), daily=True)
    if panel.Date.max() != latest:
        raise ValueError('Cache is stale for daily update; latest completed UTC bar required')
    priors = weekly.prior_predictions(market, leaders, settings)
    if state is None:
        sunday = monday-pd.Timedelta(days=1)
        predictions, priors = weekly.score(panel.loc[panel.Date.dt.dayofweek==6], market, features, leaders, settings)
        current = predictions.loc[predictions.Date==sunday]
        if current.empty:
            raise ValueError('Sunday features / mature training history unavailable for Monday selection')
        picks = weekly.allocation(current, settings)
        if len(picks) != 3:
            raise ValueError('Need exactly three eligible recommendations')
        fields = ['Symbol','rank','market_prob','stance','pred_net_7d','target_weight']
        state = {'week_monday':str(monday.date()), 'signal_bar':str(sunday.date()),
                 'created_at':t.isoformat(), 'capital':capital, 'universe':sorted(frames),
                 'picks':picks[fields].to_dict('records')}
    picks = pd.DataFrame(state['picks'])
    bands = remaining_range(panel, clean, priors, features, leaders, latest,
                            picks.Symbol.tolist(), float(picks.market_prob.iloc[0]), settings)
    report = picks.merge(bands, on='Symbol', validate='one_to_one')
    report['target_notional'] = report.target_weight * state['capital']
    report['as_of_bar'] = latest
    report['week_monday'] = monday
    report['generated_at'] = t.isoformat()
    if not offline:
        report = report.merge(plan_data.current_quotes(report.Symbol.tolist()), on='Symbol', validate='one_to_one')
    report['data_mode'] = 'offline_cache' if offline else 'binance_refreshed'
    # Commit the freeze only after data, bands and quotes successfully complete.
    if not state_path.exists():
        tmp = state_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(state, indent=2), encoding='utf-8')
        tmp.replace(state_path)
    dest = root / f'ranges_{monday:%Y-%m-%d}_{t:%Y-%m-%d}.csv'
    report.to_csv(dest, index=False)
    if command == 'plan':
        report.to_csv(root / f'plan_{monday:%Y-%m-%d}.csv', index=False)
    print(report.to_string(index=False))
    print(f'Saved {dest}; weekly names, prior, rank and target weights remain frozen.')
    return report
