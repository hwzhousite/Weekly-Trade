"""Weekly leader prior -> conditional return ranking; no exchange order submission."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from lightgbm import LGBMRegressor
import factors

LEADERS = ('BTCUSDT', 'ETHUSDT', 'SOLUSDT')

@dataclass(frozen=True)
class Settings:
    market_min_weeks: int = 26
    selection_min_weeks: int = 26
    train_weeks: int = 104
    top_n: int = 3
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    bull_threshold: float = .55
    bear_threshold: float = .45
    bear_exposure: float = .25
    neutral_exposure: float = .5
    min_history: int = 90


def prepare(frames, settings=Settings(), now=None):
    """Saturday UTC bar -> Monday open -> next Monday open (seven calendar days).

    Full daily reindex is essential: shift(-2) means calendar days, not rows.
    Eligibility uses only trailing history/liquidity; future missing observations
    invalidate labels, never replace them with zero.
    """
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize('UTC')
    cutoff = now.tz_convert('UTC').tz_localize(None).normalize()
    clean = {}
    for symbol, raw in frames.items():
        d = raw.copy()
        d.index = pd.DatetimeIndex(pd.to_datetime(d.index, utc=True)).tz_localize(None).normalize()
        if d.index.has_duplicates:
            raise ValueError(f'Duplicate daily bars: {symbol}')
        d = d.loc[d.index < cutoff].sort_index()
        if d.empty:
            continue
        d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq='D'))
        if (d[['Open', 'Close']].stack() <= 0).any():
            raise ValueError(f'Nonpositive price: {symbol}')
        clean[symbol] = d
    if any(s not in clean for s in LEADERS):
        raise ValueError('BTCUSDT, ETHUSDT and SOLUSDT histories are required')
    panel = factors.build_panel(clean, warmup_days=0, verbose=False).reset_index()
    excluded = set(factors.LABEL_COLS + factors.CARRY_COLS + ['Date', 'Symbol'])
    feature_cols = [c for c in panel if c not in excluded and pd.api.types.is_numeric_dtype(panel[c])]
    extras = []
    for symbol, d in clean.items():
        e = pd.DataFrame(index=d.index)
        e['eligible'] = (d['Close'].rolling(settings.min_history).count() == settings.min_history) & (d['QuoteVolume'].rolling(30).mean() > 0)
        entry = d['Open'].shift(-2)
        exit_price = d['Open'].shift(-9)
        e['label_return'] = exit_price / entry - 1
        # Daily funding rates are converted to entry-notional cash flows using
        # daily close as settlement-price proxy; intraday funding marks unavailable.
        funding = sum(d['funding_daily'].shift(-k) * d['Close'].shift(-k) / entry for k in range(2, 9))
        e['label_net'] = e['label_return'] - funding
        # Whole Monday-Sunday trading range, relative to the observable signal close.
        highs = pd.concat([d.High.shift(-k) for k in range(2, 9)], axis=1)
        lows = pd.concat([d.Low.shift(-k) for k in range(2, 9)], axis=1)
        e['label_high_7d'] = np.log(highs.max(axis=1).where(highs.count(axis=1) == 7) / d.Close)
        e['label_low_7d'] = np.log(lows.min(axis=1).where(lows.count(axis=1) == 7) / d.Close)
        e['label_end'] = e.index + pd.Timedelta(days=9)
        e['Symbol'] = symbol
        e['Date'] = e.index
        extras.append(e.reset_index(drop=True))
    panel = panel.merge(pd.concat(extras), on=['Date', 'Symbol'], validate='one_to_one')
    panel = panel.loc[panel.Date.dt.dayofweek == 5].copy()
    # Leader state features are entirely historical and deliberately low dimensional.
    states = []
    for symbol in LEADERS:
        d = clean[symbol]
        f = pd.DataFrame(index=d.index)
        for n in (7, 14, 28):
            f[f'{symbol}_ret_{n}'] = d.Close.pct_change(n, fill_method=None)
        f[f'{symbol}_vol_28'] = d.Close.pct_change(fill_method=None).rolling(28).std()
        f[f'{symbol}_drawdown_28'] = d.Close / d.Close.rolling(28).max() - 1
        f[f'{symbol}_volume_ratio'] = d.QuoteVolume.rolling(7).mean() / d.QuoteVolume.rolling(28).mean()
        f[f'{symbol}_funding_7'] = d.funding_daily.rolling(7).sum()
        states.append(f)
    market = pd.concat(states, axis=1).replace([np.inf, -np.inf], np.nan)
    leader_cols = market.columns.tolist()
    market = market.dropna(subset=leader_cols)
    market = market.loc[market.index.dayofweek == 5].copy()
    eligible = panel.loc[panel.eligible]
    # Freeze each date's eligible basket. Require a complete forward basket label.
    grouped = eligible.groupby('Date').label_return
    market['market_return'] = grouped.mean().where(grouped.count() == grouped.size())
    market['label_end'] = market.index + pd.Timedelta(days=9)
    market['up'] = (market.market_return > 0).astype(float).where(market.market_return.notna())
    panel = panel.merge(market[leader_cols], left_on='Date', right_index=True, how='inner')
    return panel.replace([np.inf, -np.inf], np.nan), market, feature_cols, leader_cols, clean


def mature(frame, as_of, settings):
    """At Saturday's signal close only labels ending by Sunday 00 UTC exist."""
    available = pd.Timestamp(as_of) + pd.Timedelta(days=1)
    return frame.loc[(frame.label_end <= available) & (frame.label_end > available - pd.Timedelta(weeks=settings.train_weeks))]


def prior_predictions(market, cols, settings):
    rows = []
    for date, row in market.iterrows():
        train = mature(market, date, settings).dropna(subset=['up'])
        if len(train) < settings.market_min_weeks:
            continue
        model = None
        p = float(train.up.mean())
        if train.up.nunique() == 2:
            model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=.1, max_iter=1000))
            model.fit(train[cols], train.up)
            p = float(model.predict_proba(row[cols].to_frame().T)[0, 1])
        rows.append({'Date': date, 'market_prob': p, 'base_rate': float(train.up.mean()),
                     'market_up': row.up, 'market_train_end': train.label_end.max()})
    return pd.DataFrame(rows)


def score(panel, market, features, leader_cols, settings=Settings()):
    priors = prior_predictions(market, leader_cols, settings)
    if priors.empty:
        raise ValueError('Insufficient mature weeks for market model')
    data = panel.merge(priors, on='Date', validate='many_to_one')
    output = []
    # Every past market_prob was generated using only data available then.
    for date, current in data.groupby('Date', sort=True):
        train = mature(data, date, settings)
        train = train.loc[train.eligible].dropna(subset=['label_net'])
        if train.Date.nunique() < settings.selection_min_weeks:
            continue
        current = current.loc[current.eligible].copy()
        if current.empty:
            continue
        for conditional in (True, False):
            cols = features + (leader_cols + ['market_prob'] if conditional else [])
            cols = list(dict.fromkeys(cols))
            model = LGBMRegressor(n_estimators=150, learning_rate=.03, num_leaves=7,
                                  max_depth=3, min_child_samples=30, reg_lambda=5.,
                                  colsample_bytree=.7, random_state=42, n_jobs=1, verbosity=-1)
            model.fit(train[cols], train.label_net)
            name = 'pred_net_7d' if conditional else 'baseline_pred_net_7d'
            current[name] = model.predict(current[cols])
        current['selection_train_end'] = train.label_end.max()
        current['entry_date'] = date + pd.Timedelta(days=2)
        current['exit_date'] = date + pd.Timedelta(days=9)
        output.append(current)
    if not output:
        raise ValueError('Insufficient OOF market history for conditional selection; need roughly 54+ weeks after warmup')
    return pd.concat(output, ignore_index=True), priors


def allocation(grp, settings=Settings(), mode='conditional'):
    col = 'baseline_pred_net_7d' if mode == 'baseline' else 'pred_net_7d'
    ranked = grp.sort_values([col, 'Symbol'], ascending=[False, True]).head(settings.top_n).copy()
    p = float(grp.market_prob.iloc[0])
    exposure = 1. if mode != 'conditional' or p >= settings.bull_threshold else settings.bear_exposure if p <= settings.bear_threshold else settings.neutral_exposure
    ranked['rank'] = range(1, len(ranked) + 1)
    # Predicted return is funding-net already. Round-trip execution costs remain.
    hurdle = 2 * (settings.fee_bps + settings.slippage_bps) / 1e4
    ranked['target_weight'] = np.where(ranked[col] > hurdle, exposure / settings.top_n, 0.)
    ranked['stance'] = 'BULL' if p >= settings.bull_threshold else 'BEAR' if p <= settings.bear_threshold else 'NEUTRAL'
    return ranked


def simulate(predictions, frames, settings=Settings(), mode='conditional'):
    """Linear USDT perp cash ledger, fixed quantities between Monday opens."""
    fee_rate = (settings.fee_bps + settings.slippage_bps) / 1e4
    plans = {pd.Timestamp(d) + pd.Timedelta(days=2): allocation(g, settings, mode)
             for d, g in predictions.groupby('Date')}
    start = min(plans)
    # Need the exit Monday Open; do not backtest incompletely observed weeks.
    complete = {d: g for d, g in plans.items() if all(d + pd.Timedelta(days=7) in frames[s].index and pd.notna(frames[s].at[d + pd.Timedelta(days=7), 'Open']) for s in g.Symbol)}
    if not complete:
        raise ValueError('No complete out-of-sample holding weeks')
    stop = max(complete) + pd.Timedelta(days=7)
    expected = set(pd.date_range(start, stop - pd.Timedelta(days=7), freq='7D'))
    if set(complete) != expected:
        raise ValueError('Missing weekly decision or complete holding interval; cannot silently extend a holding')
    cash, equity, quantities, rows = 1., 1., {}, []
    last_prices = {}
    def prices(date, column, symbols):
        result = {}
        for s in symbols:
            value = frames[s].at[date, column] if date in frames[s].index else np.nan
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'Missing {column} for held/traded {s} at {date}; provide delisting/settlement data')
            result[s] = float(value)
        return result
    for date in pd.date_range(start, stop, freq='D'):
        previous = equity
        opening = prices(date, 'Open', quantities)
        cash += sum(q * (opening[s] - last_prices[s]) for s, q in quantities.items())
        cost, turnover = 0., 0.
        if date in complete or date == stop:
            plan = complete.get(date)
            weights = {} if plan is None else dict(zip(plan.Symbol, plan.target_weight))
            px = prices(date, 'Open', set(quantities) | {s for s, w in weights.items() if w > 0})
            # Reserve fees while keeping gross exposure <= available equity.
            budget = cash / (1 + 2 * fee_rate)
            desired = {s: budget * w / px[s] for s, w in weights.items() if w > 0}
            turnover = sum(abs(desired.get(s, 0) - quantities.get(s, 0)) * px[s] for s in px)
            cost = turnover * fee_rate
            cash -= cost
            quantities = desired
        if cash <= 0:
            raise ValueError('Nonpositive account equity; liquidation model required')
        if date == stop:
            funding_cost = 0.
            equity = cash
        else:
            closes = prices(date, 'Close', quantities)
            funding_cost = 0.
            for s, q in quantities.items():
                rate = frames[s].at[date, 'funding_daily']
                if not np.isfinite(rate):
                    raise ValueError(f'Missing funding for {s} at {date}')
                funding_cost += q * closes[s] * rate
            cash += sum(q * (closes[s] - opening.get(s, frames[s].at[date, 'Open'])) for s, q in quantities.items()) - funding_cost
            last_prices = closes
            equity = cash
        rows.append(dict(Date=date, equity=equity, net_return=equity / previous - 1,
                         fee=cost, funding_cost=funding_cost, turnover=turnover / previous,
                         n_held=len(quantities)))
    return pd.DataFrame(rows)


def summary(curve):
    r = curve.net_return
    equity = np.r_[1., curve.equity.to_numpy()]
    return {'total_return': equity[-1] - 1, 'max_drawdown': float((equity / np.maximum.accumulate(equity) - 1).min()),
            'sharpe_daily': float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else np.nan,
            'fee_sum_initial_equity': float(curve.fee.sum()), 'daily_turnover': float(curve.turnover.mean())}


def market_metrics(priors):
    d = priors.dropna(subset=['market_up'])
    if d.empty:
        return {}
    return {'weeks': len(d), 'brier': brier_score_loss(d.market_up, d.market_prob),
            'base_brier': brier_score_loss(d.market_up, d.base_rate),
            'auc': roc_auc_score(d.market_up, d.market_prob) if d.market_up.nunique() == 2 else None}


def price_ranges(panel, priors, features, leaders, date, symbols, settings=Settings(), calibration_weeks=12):
    """q05 of weekly minimum / q95 of weekly maximum, signal-close-relative.

    Historical calibration uses rolling OOF predictions and fully mature labels.
    Pooled serially dependent residuals give empirical calibration, not a formal
    independent-sample coverage guarantee. Quotes never change the band anchor.
    """
    date = pd.Timestamp(date)
    data = panel.merge(priors[['Date', 'market_prob']], on='Date', validate='many_to_one')
    cols = list(dict.fromkeys(features + leaders + ['market_prob']))
    targets = [('low', 'label_low_7d', .05), ('high', 'label_high_7d', .95)]
    def predict_at(d, current):
        train = mature(data, d, settings)
        train = train.loc[train.eligible].dropna(subset=['label_low_7d', 'label_high_7d'])
        if train.Date.nunique() < settings.selection_min_weeks:
            return None
        result = current[['Date', 'Symbol', 'Close', 'label_low_7d', 'label_high_7d']].copy()
        for name, target, alpha in targets:
            model = LGBMRegressor(objective='quantile', alpha=alpha,
                n_estimators=100, learning_rate=.03, num_leaves=7, max_depth=3,
                min_child_samples=30, reg_lambda=5., random_state=42, n_jobs=1, verbosity=-1)
            model.fit(train[cols], train[target])
            result[name] = model.predict(current[cols])
        return result
    historical = mature(data, date, settings)
    dates = sorted(historical.Date.unique())[-calibration_weeks:]
    errors = []
    for d in dates:
        current = historical.loc[(historical.Date == d) & historical.eligible].dropna(subset=['label_low_7d', 'label_high_7d'])
        if current.empty:
            continue
        out = predict_at(pd.Timestamp(d), current)
        if out is not None:
            errors.append(out)
    current = data.loc[(data.Date == date) & data.Symbol.isin(symbols)]
    out = predict_at(date, current)
    if out is None or len(out) != len(symbols):
        raise ValueError('Insufficient mature data for weekly price-range heads')
    calibration = pd.concat(errors, ignore_index=True) if errors else pd.DataFrame()
    n_weeks = calibration.Date.nunique() if not calibration.empty else 0
    calibrated = n_weeks >= 4
    low_adjust = high_adjust = 0.
    if calibrated:
        low_adjust = min(0., float((calibration.label_low_7d - calibration.low).quantile(.05)))
        high_adjust = max(0., float((calibration.label_high_7d - calibration.high).quantile(.95)))
    lo, hi = out.low + low_adjust, out.high + high_adjust
    # Sort crossing predictions; positive prices follow from the log target.
    out['week_low_price'] = out.Close * np.exp(np.minimum(lo, hi))
    out['week_high_price'] = out.Close * np.exp(np.maximum(lo, hi))
    out['range_reference_close'] = out.Close
    out['range_nominal_coverage'] = .90
    out['range_calibration'] = 'pooled_oof_empirical' if calibrated else 'uncalibrated'
    out['range_calibration_weeks'] = n_weeks
    out['range_calibration_rows'] = len(calibration)
    return out[['Symbol', 'range_reference_close', 'week_low_price', 'week_high_price',
                'range_nominal_coverage', 'range_calibration', 'range_calibration_weeks', 'range_calibration_rows']]
