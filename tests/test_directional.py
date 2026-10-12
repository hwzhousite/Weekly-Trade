import sys
import unittest
from pathlib import Path
from unittest.mock import patch
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
    def test_news_future_rejected_before_request(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'news.json'
            path.write_text(json.dumps([{'published_at':'2026-10-13T00:00Z','retrieved_at':'2026-10-13T00:00Z','url':'https://example.com','text':'news'}]))
            with patch.dict('os.environ',{'OPENAI_API_KEY':'test','OPENAI_MODEL':'test'}):
                with self.assertRaisesRegex(ValueError,'no later'):
                    d.gpt_direction({},pd.Timestamp('2026-10-12T00:15Z'),path)

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
