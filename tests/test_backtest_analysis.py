import importlib.util
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('bt_analysis',ROOT/'backtest'/'run.py')
bt=importlib.util.module_from_spec(spec); spec.loader.exec_module(bt)

class AnalysisTests(unittest.TestCase):
    def test_market_metrics_one_row_per_week(self):
        rows=pd.DataFrame({'market_return':[-.1,.1,-.2,.3],'market_prob':[.1,.9,.2,.8],'base_rate':[.5]*4})
        out=bt.classification(rows)
        self.assertEqual(out['weeks'],4)
        self.assertEqual(out['accuracy'],1)
        self.assertEqual(out['auc'],1)
        self.assertLess(out['brier'],out['training_base_rate_brier'])
    def test_bear_rank_and_signed_excess(self):
        current=pd.DataFrame({'Date':[pd.Timestamp('2024-01-07')]*4,'Symbol':list('ABCD'),
            'market_prob':[.2]*4,'coin_up_prob':[.1,.2,.3,.9],'coin_down_prob':[.9,.8,.7,.1],
            'label_return':[-.4,-.3,-.2,.1]})
        out=bt.analyze_week(current,pd.Series({'market_return':-.2}))
        self.assertAlmostEqual(out['rank_ic'],1)
        self.assertEqual(out['top3_direction_hit'],1)
        self.assertGreater(out['selection_excess_price_return'],0)
    def test_factor_summary_market_and_coin_methods(self):
        rows=[]
        for i in range(4):
            rows.extend([dict(head='coin',factor='a',selected=i<2,train_ic=.1,oos_ic=.2),
                dict(head='market',factor='b',selected=True,train_ic=.1,feature_value=i,market_return=i,oos_ic=np.nan)])
        out=bt.factor_summary(pd.DataFrame(rows)).set_index('factor')
        self.assertEqual(out.loc['a','selection_frequency'],.5)
        self.assertAlmostEqual(out.loc['b','oos_rank_ic'],1.)
    def test_real_pipeline_one_complete_week(self):
        sys.path.insert(0,str(ROOT/'tests'))
        from test_weekly import frames
        settings=bt.weekly.Settings(signal_weekday=6,entry_delay=1,market_min_weeks=8,selection_min_weeks=8)
        # Real factors, mature training, IC selection, models and signed ledger.
        with tempfile.TemporaryDirectory() as folder:
            result=bt.evaluate(frames(450),folder,start='2024-02-12',end='2024-02-12',now='2024-04-01',settings=settings)
            self.assertEqual(result['market_classifier']['weeks'],1)
            root=Path(folder)
            self.assertEqual(len(pd.read_csv(root/'top3_plans.csv')),3)
            curves=[pd.read_csv(root/f'equity_{name}.csv') for name in result['strategies']]
            for curve in curves:
                self.assertEqual(curve.Date.iloc[0],'2024-02-12')
                self.assertEqual(curve.Date.iloc[-1],'2024-02-19')
                self.assertTrue((curve.turnover.iloc[1:-1]==0).all())
            diagnostics=pd.read_csv(root/'weekly_diagnostics.csv')
            self.assertAlmostEqual(diagnostics.portfolio_net_week_return.iloc[0],curves[0].equity.iloc[-1]-1)
            self.assertTrue((root/'factor_summary.csv').exists())
