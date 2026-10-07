import json
import sys
import tempfile
import urllib.error
from pathlib import Path
from unittest import TestCase, mock
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import download_data
import binance_data

class DownloadTests(TestCase):
    def test_symbols_leaders_and_manifest(self):
        data = {'BTCUSDT': pd.DataFrame({'Close': [1.]}, index=pd.to_datetime(['2024-01-01']))}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(binance_data, 'download_universe', return_value=data) as api:
            download_data.download(directory, '2023-01-01', '2024-01-02', ['adausdt'])
            args = api.call_args.kwargs
            self.assertEqual(args['symbols'], ['ADAUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
            manifest = json.loads((Path(directory) / 'download_manifest.json').read_text())
            self.assertIn('ADAUSDT', manifest['failed_or_insufficient_history'])
            self.assertEqual(manifest['symbols']['BTCUSDT']['bars'], 1)

    def test_451_fails_without_retries(self):
        error = urllib.error.HTTPError('https://fapi.binance.com', 451, 'Restricted', {}, None)
        with mock.patch('urllib.request.urlopen', side_effect=error) as request:
            with self.assertRaisesRegex(binance_data.BinanceError, 'HTTP 451'):
                binance_data._request('https://fapi.binance.com/fapi/v1/exchangeInfo')
            self.assertEqual(request.call_count, 1)

    def test_invalid_window_does_not_download(self):
        with mock.patch.object(binance_data, 'download_universe') as api:
            with self.assertRaises(ValueError):
                download_data.download('/tmp/cache', '2025-01-01', '2024-01-01')
            api.assert_not_called()
