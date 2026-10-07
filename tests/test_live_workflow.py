import sys
import json
import tempfile
from pathlib import Path
from unittest import TestCase, mock
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import live_workflow as live

class LifecycleTests(TestCase):
    def test_monday_freeze_daily_does_not_reselect(self):
        names=['BTCUSDT','ETHUSDT','SOLUSDT']
        preds=pd.DataFrame({'Date':pd.Timestamp('2025-09-21'),'Symbol':names,
            'pred_net_7d':[.1,.09,.08], 'market_prob':.8})
        panel=pd.DataFrame({'Date':[pd.Timestamp('2025-09-21')], 'Symbol':['BTCUSDT']})
        bands=pd.DataFrame({'Symbol':names,'week_low_price':[1.,2.,3.], 'week_high_price':[4.,5.,6.]})
        with tempfile.TemporaryDirectory() as out, mock.patch.object(live.binance_data,'load_universe', return_value=dict.fromkeys(names)), mock.patch.object(live.weekly,'prepare', return_value=(panel,pd.DataFrame(),[],[],{})) as prepare, mock.patch.object(live.weekly,'prior_predictions',return_value=pd.DataFrame()), mock.patch.object(live.weekly,'score',return_value=(preds,pd.DataFrame())) as score, mock.patch.object(live,'remaining_range',return_value=bands) as ranges:
            monday=live.run('plan','/tmp/cache',out,offline=True,now='2025-09-22T10:00:00+08:00')
            state_path=Path(out)/'week_2025-09-22.json'
            original=state_path.read_bytes()
            panel2=panel.copy(); panel2.Date=pd.Timestamp('2025-09-22')
            prepare.return_value=(panel2,pd.DataFrame(),[],[],{})
            updated=bands.copy(); updated.week_low_price+=.5
            ranges.return_value=updated
            tuesday=live.run('daily','/tmp/cache',out,offline=True,now='2025-09-23T10:00:00+08:00')
            self.assertEqual(score.call_count,1)
            self.assertEqual(state_path.read_bytes(),original)
            self.assertEqual(monday.Symbol.tolist(),tuesday.Symbol.tolist())
            self.assertEqual(monday.market_prob.tolist(),tuesday.market_prob.tolist())
            self.assertFalse(monday.week_low_price.equals(tuesday.week_low_price))
            self.assertTrue((Path(out)/'ranges_2025-09-22_2025-09-23.csv').exists())

    def test_missing_weekly_plan_and_before_close_rejected(self):
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaisesRegex(ValueError,'No frozen plan'):
                live.run('daily','/tmp/cache',out,offline=True,now='2025-09-23T10:00:00+08:00')
            with self.assertRaisesRegex(ValueError,'Monday'):
                live.run('plan','/tmp/cache',out,offline=True,now='2025-09-23T10:00:00+08:00')
        with self.assertRaisesRegex(ValueError,'08:00'):
            live.clock('2025-09-22T07:00:00+08:00')
