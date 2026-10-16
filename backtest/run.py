"""Independent walk-forward diagnostics for the two-stage weekly strategy."""
import argparse
import json
import sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score, confusion_matrix
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import weekly
import directional
import binance_data


def correlation(x, y):
    valid = pd.DataFrame({'x':x, 'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
    return float(valid.x.rank().corr(valid.y.rank())) if len(valid)>=3 and valid.x.nunique()>1 and valid.y.nunique()>1 else np.nan


def classification(rows):
    y=(rows.market_return>0).astype(int); p=rows.market_prob; pred=(p>=.5).astype(int)
    return dict(weeks=len(rows),accuracy=accuracy_score(y,pred),
        balanced_accuracy=balanced_accuracy_score(y,pred), brier=brier_score_loss(y,p),
        training_base_rate_brier=brier_score_loss(y,rows.base_rate),
        log_loss=log_loss(y,p,labels=[0,1]),
        auc=roc_auc_score(y,p) if y.nunique()==2 else None,
        confusion_matrix_labels_bear_bull=confusion_matrix(y,pred,labels=[0,1]).tolist())


def analyze_week(current, market_row):
    bull=float(current.market_prob.iloc[0])>=.5
    sign=1 if bull else -1
    score=current.coin_up_prob if bull else current.coin_down_prob
    ordered=current.assign(direction_probability=score).sort_values(['direction_probability','Symbol'],ascending=[False,True])
    top=ordered.head(3)
    return dict(Date=current.Date.iloc[0],stance='BULL' if bull else 'BEAR',
        market_prob=float(current.market_prob.iloc[0]),market_return=float(market_row.market_return),
        direction_correct=bool(bull==(market_row.market_return>0)),
        rank_ic=correlation(score, sign*current.label_return),
        top3_direction_hit=float((sign*top.label_return>0).mean()),
        top3_signed_price_return=float(sign*top.label_return.mean()),
        universe_signed_price_return=float(sign*current.label_return.mean()),
        selection_excess_price_return=float(sign*(top.label_return.mean()-current.label_return.mean())),
        candidates=len(current))


def factor_observations(current, market_row, audit):
    rows=[]
    for item in audit.itertuples():
        if item.head=='coin':
            value=correlation(current[item.factor],current.label_return)
            rows.append(dict(Date=current.Date.iloc[0],head='coin',factor=item.factor,
                selected=bool(item.selected),train_ic=item.ic,oos_ic=value))
        else:
            rows.append(dict(Date=current.Date.iloc[0],head='market',factor=item.factor,
                selected=bool(item.selected),train_ic=item.ic,oos_ic=np.nan,
                feature_value=market_row[item.factor],market_return=market_row.market_return))
    return rows


def factor_summary(observations):
    rows=[]
    for (head,factor),g in observations.groupby(['head','factor']):
        if head=='coin':
            ic=g.oos_ic.dropna(); mean=ic.mean(); sd=ic.std()
            oos=mean; ir=mean/sd if len(ic)>1 and sd>0 else np.nan
            positive=float((ic>0).mean()) if len(ic) else np.nan
        else:
            oos=correlation(g.feature_value,g.market_return); ir=positive=np.nan
        rows.append(dict(head=head,factor=factor,weeks=len(g),selection_frequency=g.selected.mean(),
            mean_training_ic=g.train_ic.mean(),oos_rank_ic=oos,oos_ic_ir=ir,oos_positive_ic_fraction=positive))
    return pd.DataFrame(rows).sort_values(['head','oos_rank_ic'],ascending=[True,False])


def evaluate(frames, output, start=None, end=None, now=None, settings=directional.SETTINGS):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    panel,market,features,leaders,clean=weekly.prepare(frames,settings,now=now)
    predictions=[]; audits=[]; market_rows=[]; diagnostics=[]; factor_rows=[]; skips=[]
    for date,row in market.iterrows():
        entry=date+pd.Timedelta(days=settings.entry_delay)
        if start and entry<pd.Timestamp(start): continue
        if end and entry>pd.Timestamp(end): continue
        if pd.isna(row.market_return):
            skips.append({'Date':date,'reason':'incomplete_market_outcome'}); continue
        train=weekly.mature(panel,date,settings)
        mt=weekly.mature(market,date,settings).dropna(subset=['market_return'])
        if len(mt)<settings.market_min_weeks or train.loc[train.eligible].dropna(subset=['label_return']).Date.nunique()<settings.selection_min_weeks:
            skips.append({'Date':date,'reason':'initial_training_warmup'}); continue
        current,audit=directional.predict(panel,market,features,leaders,date,settings)
        if len(current)<3 or current.label_return.isna().any():
            raise ValueError(f'Incomplete eligible coin outcomes at {date}; cannot drop missing or delisted coins')
        current['predicted_stance']='BULL' if current.market_prob.iloc[0]>=.5 else 'BEAR'
        predictions.append(current)
        audit['signal_date']=date; audits.append(audit)
        market_rows.append(dict(Date=date,entry_date=entry,market_prob=float(current.market_prob.iloc[0]),
            market_return=row.market_return,base_rate=float(mt.up.mean()),market_train_end=mt.label_end.max()))
        diagnostics.append(analyze_week(current,row))
        factor_rows.extend(factor_observations(current,row,audit))
        print(f'{entry.date()} {current.predicted_stance.iloc[0]} p={current.market_prob.iloc[0]:.3f} coins={len(current)}',flush=True)
    if not predictions: raise ValueError('No complete out-of-sample weeks in requested range')
    data=pd.concat(predictions,ignore_index=True)
    m=pd.DataFrame(market_rows); d=pd.DataFrame(diagnostics); f=pd.DataFrame(factor_rows)
    picks=[]; peer=[]; always_long=[]
    for date,g in data.groupby('Date'):
        sign=1 if g.market_prob.iloc[0]>=.5 else -1
        picks.append(directional.rank(g,'BULL' if sign==1 else 'BEAR'))
        peers=g.copy(); peers['target_weight']=sign/len(g); peer.append(peers)
        always_long.append(directional.rank(g,'BULL'))
    strategies={}; curves={}
    for name,plans in [('combined_top3',picks),('same_direction_universe',peer),('always_long_top3',always_long)]:
        curve=weekly.simulate(pd.concat(plans),clean,settings,mode='directional')
        curves[name]=curve; curve.to_csv(output/f'equity_{name}.csv',index=False)
        strategies[name]=weekly.summary(curve)
    # Week returns use the daily ledger; exit-Monday fees belong to next week except terminal exit.
    curve=curves['combined_top3']; grouped=[]
    for i,entry in enumerate(m.entry_date):
        last=entry+pd.Timedelta(days=7)
        segment=curve.loc[(curve.Date>=entry)&((curve.Date<last)|((i==len(m)-1)&(curve.Date==last)))]
        grouped.append(float((1+segment.net_return).prod()-1))
    d['portfolio_net_week_return']=grouped
    metrics={'market_classifier':classification(m),'strategies':strategies,'ranking':{}}
    for name,g in [('all',d),('predicted_bull',d[d.stance=='BULL']),('predicted_bear',d[d.stance=='BEAR']),
                   ('direction_correct',d[d.direction_correct]),('direction_wrong',d[~d.direction_correct])]:
        metrics['ranking'][name]={'weeks':len(g),**{c:float(g[c].mean()) for c in
            ['rank_ic','top3_direction_hit','selection_excess_price_return','portfolio_net_week_return']}}
    # Reliability bins are evaluated only, never fitted or used to tune the model.
    calibration=m.assign(bin=pd.cut(m.market_prob,np.linspace(0,1,11),include_lowest=True),actual_up=(m.market_return>0).astype(int))
    calibration.groupby('bin',observed=True).agg(weeks=('actual_up','size'),mean_probability=('market_prob','mean'),actual_up_rate=('actual_up','mean')).to_csv(output/'market_calibration.csv')
    m.to_csv(output/'market_predictions.csv',index=False)
    d.to_csv(output/'weekly_diagnostics.csv',index=False)
    data.to_csv(output/'coin_predictions.csv',index=False)
    pd.concat(picks).to_csv(output/'top3_plans.csv',index=False)
    pd.concat(audits).to_csv(output/'factor_training_ic.csv',index=False)
    f.to_csv(output/'factor_oos_weekly.csv',index=False)
    factor_summary(f).to_csv(output/'factor_summary.csv',index=False)
    pd.DataFrame(skips,columns=['Date','reason']).to_csv(output/'excluded_weeks.csv',index=False)
    def safe(value):
        if isinstance(value,dict): return {k:safe(v) for k,v in value.items()}
        if isinstance(value,list): return [safe(v) for v in value]
        if isinstance(value,(float,np.floating)) and not np.isfinite(value): return None
        return value
    metrics=safe(metrics)
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    (output/'run_config.json').write_text(json.dumps({'settings':asdict(settings),'start':start,'end':end,
        'generated_at':pd.Timestamp.now(tz='UTC').isoformat(),'symbols':sorted(frames),'candidate_factors':features,'market_factors':leaders,
        'signal':'Sunday closed UTC bar','execution_proxy':'Monday open','ic_threshold':.015},indent=2))
    report=['# Weekly model evaluation','',f'Out-of-sample weeks: {len(m)}','',
        '## Market classifier','',pd.Series(metrics['market_classifier']).to_string(),'',
        '## Combined portfolio','',pd.Series(strategies['combined_top3']).to_string(),'',
        '## Interpretation','',
        'See weekly_diagnostics.csv for bull/bear and correct/wrong direction splits; factor_summary.csv for training selection frequency and held-out factor IC.',
        'Factor OOS statistics are diagnostics only, not inputs to selection. Market factor IC is one time-series statistic across OOS weeks; coin factor IC is the average of weekly cross-sectional statistics.',
        'Daily equity uses fixed signed quantities and funding/fees. Week returns compound daily ledger returns; Monday rebalance fees are assigned to the new week, terminal exit fees to the last week.',
        'Current-universe survivorship bias remains. Daily Monday-open proxy differs from the live 08:15 decision fill. There is no liquidation simulator. No profitability conclusion follows from synthetic tests.']
    (output/'report.md').write_text('\n'.join(report)+'\n')
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data'/'binance')
    p.add_argument('--output-dir',type=Path,default=ROOT/'backtest'/'output')
    p.add_argument('--start',help='First Monday entry date inclusive')
    p.add_argument('--end',help='Last Monday entry date inclusive')
    p.add_argument('--fee-bps',type=float,default=10.)
    p.add_argument('--slippage-bps',type=float,default=5.)
    args=p.parse_args()
    if min(args.fee_bps,args.slippage_bps)<0: p.error('Costs must be nonnegative')
    if args.start and args.end and pd.Timestamp(args.start)>pd.Timestamp(args.end): p.error('start must precede end')
    settings=weekly.Settings(signal_weekday=6,entry_delay=1,fee_bps=args.fee_bps,slippage_bps=args.slippage_bps)
    result=evaluate(binance_data.load_universe(cache_dir=args.data_dir,min_bars=90),args.output_dir,args.start,args.end,settings=settings)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
