import sys
from pathlib import Path
from unittest import TestCase, mock
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import plan_data

class RefreshTests(TestCase):
    def test_refresh_adds_leaders_excludes_stale_and_uses_returned_frames(self):
        index = pd.date_range('2024-01-01', '2024-01-07')
        frames = {s: pd.DataFrame({'Close': 1.}, index=index) for s in plan_data.LEADERS + ('A', 'B', 'OLD')}
        frames['OLD'] = frames['OLD'].iloc[:-1]
        with mock.patch.object(plan_data.binance_data, 'top_perpetuals', return_value=pd.DataFrame({'symbol': ['A','B','OLD']})), mock.patch.object(plan_data.binance_data, 'download_universe', return_value=frames) as download:
            result = plan_data.refresh_plan_frames('/tmp/cache', now='2024-01-08T02:00:00Z')
        self.assertNotIn('OLD', result)
        self.assertTrue(set(plan_data.LEADERS).issubset(download.call_args.kwargs['symbols']))

    def test_missing_leader_aborts(self):
        with mock.patch.object(plan_data.binance_data, 'top_perpetuals', return_value=pd.DataFrame({'symbol':['A']})), mock.patch.object(plan_data.binance_data, 'download_universe', return_value={}):
            with self.assertRaisesRegex(ValueError, 'Fresh leader'):
                plan_data.refresh_plan_frames('/tmp/cache', now='2024-01-08T02:00:00Z')

    def test_quote_endpoint_and_timestamp(self):
        with mock.patch.object(plan_data.binance_data, '_request', return_value={'price':'123.5', 'time':1704672000000}) as request:
            quote = plan_data.current_quotes(['BTCUSDT'])
        self.assertIn('/fapi/v2/ticker/price', request.call_args.args[0])
        self.assertEqual(quote.current_price.iloc[0], 123.5)
