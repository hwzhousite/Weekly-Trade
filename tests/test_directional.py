import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import tempfile
import json
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import directional as d
import weekly

class DirectionalTests(unittest.TestCase):
    def test_signed_ic_and_constant(self):
        data = pd.DataFrame({'Date':np.repeat(pd.date_range('2024-01-07',periods=5,freq='7D'),4),
                             'good':list(range(4))*5,'bad':list(range(3,-1,-1))*5,'constant':1,
                             'label_return':list(range(4))*5})
        result=d.ic_table(data,['good','bad','constant']).set_index('factor')
        self.assertTrue(result.loc['good','selected'])
        self.assertFalse(result.loc['bad','selected'])
        self.assertFalse(result.loc['constant','selected'])
    def test_direction_changes_top_three(self):
        data=pd.DataFrame({'Symbol':list('ABCDE'),'coin_up_prob':[.9,.8,.7,.2,.1]})
        data['coin_down_prob']=1-data.coin_up_prob
        self.assertEqual(d.rank(data,'BULL').Symbol.tolist(),list('ABC'))
        short=d.rank(data,'BEAR')
        self.assertEqual(short.Symbol.tolist(),list('EDC'))
        self.assertAlmostEqual(short.target_weight.sum(),-1.)
    def test_short_pnl_funding_and_no_midweek_trade(self):
        date=pd.Timestamp('2024-01-07')
        raw=pd.DataFrame({'Open':[100,90,80,70,60,50,40,30],
                          'Close':[90,80,70,60,50,40,30,30],'funding_daily':.001},
                         index=pd.date_range('2024-01-08',periods=8))
        pred=pd.DataFrame({'Date':[date],'Symbol':['A'],'target_weight':[-1.]})
        settings=weekly.Settings(signal_weekday=6,entry_delay=1,fee_bps=0,slippage_bps=0)
        curve=weekly.simulate(pred,{'A':raw},settings,'directional')
        self.assertGreater(curve.equity.iloc[-1],1.7)
        self.assertTrue((curve.funding_cost.iloc[:-1]<0).all())
        self.assertTrue((curve.turnover.iloc[1:-1]==0).all())
    def test_future_labels_do_not_change_prediction(self):
        from test_weekly import frames
        panel,market,features,leaders,_=weekly.prepare(frames(700),d.SETTINGS,now='2025-01-01')
        date=market.index[-1]
        old,_=d.predict(panel,market,features,leaders,date)
        changed=panel.copy(); future=changed.label_end>date+pd.Timedelta(days=1)
        changed.loc[future,'label_return']=100
        changed_market=market.copy()
        future=changed_market.label_end>date+pd.Timedelta(days=1)
        changed_market.loc[future,'market_return']=100
        changed_market.loc[future,'up']=1
        new,_=d.predict(changed,changed_market,features,leaders,date)
        pd.testing.assert_frame_equal(old[['Symbol','coin_up_prob','market_prob']],new[['Symbol','coin_up_prob','market_prob']])

    def test_weekly_plan_quant_only_and_freeze(self):
        for probability, stance in ((.8,'BULL'),(.2,'BEAR')):
            with self.subTest(stance=stance), tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                args=SimpleNamespace(output_dir=root,data_dir=root,capital=10000,offline=True)
                sunday=pd.Timestamp('2026-10-11')
                market=pd.DataFrame({'state':[1.]},index=[sunday])
                clean={'A':pd.DataFrame({'Close':[100.]},index=[sunday])}
                current=pd.DataFrame({'Date':[sunday]*3,'entry_date':[sunday+pd.Timedelta(days=1)]*3,
                    'exit_date':[sunday+pd.Timedelta(days=8)]*3,'Symbol':list('ABC'),
                    'market_prob':[probability]*3,'coin_up_prob':[.9,.7,.6],
                    'coin_down_prob':[.1,.3,.4],'selection_train_end':[sunday]*3})
                with patch.object(d.binance_data,'load_universe',return_value={}) as load, \
                     patch.object(d.weekly,'prepare',return_value=(None,market,[],['state'],clean)), \
                     patch.object(d,'predict',return_value=(current,pd.DataFrame({'selected':[True]}))), \
                     patch('urllib.request.urlopen',side_effect=AssertionError('Unexpected API call')):
                    report=d.run(args,now='2026-10-12T10:00:00+08:00')
                    self.assertTrue((report.stance==stance).all())
                    self.assertTrue((report.market_source=='quant').all())
                    self.assertEqual(load.call_count,1)
                    repeated=d.run(args,now='2026-10-13T10:00:00+08:00')
                    self.assertEqual(load.call_count,1)
                    self.assertEqual(report.Symbol.tolist(),repeated.Symbol.tolist())
                    state=json.loads((root/'direction_week_2026-10-12.json').read_text())
                    self.assertEqual(state['judgement']['source'],'quant')
