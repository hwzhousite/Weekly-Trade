"""Monday-only direction ranking with training-only positive Spearman IC."""
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import weekly
import binance_data
import plan_data

SETTINGS = weekly.Settings(signal_weekday=6, entry_delay=1)


def ic_table(train, columns, target='label_return', cross_sectional=True):
    rows = []
    for col in dict.fromkeys(columns):
        values = []
        groups = train.groupby('Date') if cross_sectional else [(None, train)]
        for _, group in groups:
            pair = group[[col, target]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) >= (3 if cross_sectional else 12) and pair[col].nunique() > 1 and pair[target].nunique() > 1:
                values.append(pair[col].rank().corr(pair[target].rank()))
        ic = float(np.mean(values)) if values else np.nan
        rows.append({'factor':col, 'ic':ic, 'observations':len(values),
                     'method':'mean_weekly_cross_sectional_rank_ic' if cross_sectional else 'weekly_time_series_rank_ic',
                     'selected': bool(np.isfinite(ic) and ic > .015)})
    return pd.DataFrame(rows)


def fit_probability(train, current, cols, target):
    y = train[target]
    if y.nunique() < 2:
        return np.full(len(current), float(y.mean()))
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          LogisticRegression(C=.1, max_iter=1000))
    model.fit(train[cols], y)
    return model.predict_proba(current[cols])[:, 1]


def predict(panel, market, features, leaders, date, settings=SETTINGS):
    date = pd.Timestamp(date)
    mt = weekly.mature(market, date, settings).dropna(subset=['market_return'])
    train = weekly.mature(panel, date, settings)
    train = train.loc[train.eligible].dropna(subset=['label_return']).copy()
    if len(mt) < settings.market_min_weeks or train.Date.nunique() < settings.selection_min_weeks:
        raise ValueError('Insufficient mature weekly training history')
    mi = ic_table(mt.reset_index(names='Date'), leaders, 'market_return', False)
    fi = ic_table(train, features)
    mc = mi.loc[mi.selected, 'factor'].tolist()
    fc = fi.loc[fi.selected, 'factor'].tolist()
    if not mc or not fc:
        raise ValueError('No factors pass signed IC > 0.015 for market or coin ranking; inspect IC report')
    mp = float(fit_probability(mt, market.loc[[date]], mc, 'up')[0])
    current = panel.loc[(panel.Date == date) & panel.eligible].copy()
    train['up'] = (train.label_return > 0).astype(int)
    coin_cols = list(dict.fromkeys(fc + mc))
    current['coin_up_prob'] = fit_probability(train, current, coin_cols, 'up')
    train['down'] = (train.label_return < 0).astype(int)
    current['coin_down_prob'] = fit_probability(train, current, coin_cols, 'down')
    current['market_prob'] = mp
    current['selection_train_end'] = train.label_end.max()
    current['entry_date'] = date + pd.Timedelta(days=1)
    current['exit_date'] = date + pd.Timedelta(days=8)
    fi['head'] = 'coin'; mi['head'] = 'market'
    return current, pd.concat([fi, mi], ignore_index=True)


def rank(current, direction):
    out = current.copy()
    col = 'coin_up_prob' if direction == 'BULL' else 'coin_down_prob'
    out['direction_probability'] = out[col]
    out = out.sort_values([col, 'Symbol'], ascending=[False, True]).head(3).copy()
    if len(out) != 3:
        raise ValueError('Exactly three eligible symbols required')
    out['rank'] = range(1, 4)
    out['stance'] = direction
    out['side'] = 'LONG' if direction == 'BULL' else 'SHORT'
    out['target_weight'] = (1 if direction == 'BULL' else -1) / 3
    return out


def run(args, now=None):
    t = pd.Timestamp.now(tz='Asia/Shanghai') if now is None else pd.Timestamp(now).tz_convert('Asia/Shanghai')
    monday = t.normalize() - pd.Timedelta(days=t.dayofweek)
    cutoff = monday + pd.Timedelta(hours=8, minutes=15)
    root = args.output_dir; root.mkdir(parents=True, exist_ok=True)
    path = root / f'direction_week_{monday:%Y-%m-%d}.json'
    if path.exists():
        state = json.loads(path.read_text())
        if state.get('judgement', {}).get('source') != 'quant':
            raise ValueError('Saved plan uses a removed market source. Move this week JSON to a backup, then create a quantitative plan on Monday.')
        report = pd.DataFrame(state['picks'])
        print(report.to_string(index=False))
        return report
    if t.dayofweek != 0 or t < cutoff:
        raise ValueError('Create plans only on Monday after 08:15 Beijing time; existing plans may be viewed anytime')
    frames = binance_data.load_universe(cache_dir=args.data_dir, min_bars=90) if args.offline else plan_data.refresh_plan_frames(args.data_dir)
    sunday = monday.tz_localize(None) - pd.Timedelta(days=1)
    panel, market, features, leaders, clean = weekly.prepare(frames, SETTINGS, now=cutoff.tz_convert('UTC'))
    if any(sunday not in d.index or pd.isna(d.loc[sunday, 'Close']) for d in clean.values()):
        raise ValueError('All candidate histories must include the completed Sunday bar')
    if sunday not in market.index:
        raise ValueError('Missing latest completed Sunday market data')
    current, audit = predict(panel, market, features, leaders, sunday)
    audit.to_csv(root / f'factor_ic_{monday:%Y-%m-%d}.csv', index=False)
    snapshot = json.loads(market.loc[sunday, leaders].to_json())
    snapshot['quant_market_up_probability'] = float(current.market_prob.iloc[0])
    judgement = {
        'direction':'BULL' if snapshot['quant_market_up_probability'] >= .5 else 'BEAR', 'source':'quant', 'snapshot':snapshot}
    report = rank(current, judgement['direction'])
    cols = ['Date','entry_date','exit_date','Symbol','rank','stance','side','market_prob',
            'coin_up_prob','coin_down_prob','direction_probability','target_weight','selection_train_end']
    report = report[cols].copy()
    report['target_notional'] = report.target_weight * args.capital
    report['decision_cutoff'] = cutoff.isoformat()
    report['market_source'] = 'quant'
    state = {'version':2,'cutoff':cutoff.isoformat(),'judgement':judgement,
             'picks':json.loads(report.to_json(orient='records', date_format='iso'))}
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(state, indent=2)); tmp.replace(path)
    report.to_csv(root / f'plan_{monday:%Y-%m-%d}.csv', index=False)
    print(report.to_string(index=False))
    return report
