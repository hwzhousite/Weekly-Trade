import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import weekly


def frames(n=700):
    rng = np.random.default_rng(17)
    out = {}
    index = pd.date_range('2023-01-01', periods=n)
    for symbol in weekly.LEADERS + ('ADAUSDT', 'XRPUSDT', 'BNBUSDT'):
        close = 100 * np.exp(np.cumsum(rng.normal(.0004, .02, n)))
        volume = rng.uniform(500, 1000, n)
        out[symbol] = pd.DataFrame({'Open': close * .999, 'High': close * 1.02,
            'Low': close * .98, 'Close': close, 'Volume': volume,
            'QuoteVolume': volume * close, 'Trades': 100,
            'TakerBuyBase': volume * .5, 'TakerBuyQuote': volume * close * .5,
            'funding_daily': .0001, 'funding_mean': .0001 / 3,
            'SpotClose': close * .999, 'SpotVolume': volume}, index=index)
    return out

class WeeklyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = frames()
        cls.settings = weekly.Settings(market_min_weeks=8, selection_min_weeks=8)
        cls.prepared = weekly.prepare(cls.frames, cls.settings, now='2025-01-01')
        p, m, f, l, _ = cls.prepared
        cls.pred, cls.priors = weekly.score(p, m, f, l, cls.settings)

    def test_training_labels_mature_and_oof(self):
        self.assertTrue((self.priors.market_train_end <= self.priors.Date + pd.Timedelta(days=1)).all())
        self.assertTrue((self.pred.selection_train_end <= self.pred.Date + pd.Timedelta(days=1)).all())
        self.assertTrue((self.pred.entry_date.dt.dayofweek == 0).all())

    def test_future_mutation_does_not_change_history(self):
        date = self.pred.Date.unique()[5]
        changed = {s: d.copy() for s, d in self.frames.items()}
        for d in changed.values():
            mask = d.index > pd.Timestamp(date)
            d.loc[mask, ['Open', 'Close', 'High', 'Low', 'SpotClose']] *= 2
        p, m, f, l, _ = weekly.prepare(changed, self.settings, now='2025-01-01')
        pred, priors = weekly.score(p, m, f, l, self.settings)
        old = self.pred.loc[self.pred.Date <= date, ['Date', 'Symbol', 'pred_net_7d', 'market_prob']].reset_index(drop=True)
        new = pred.loc[pred.Date <= date, old.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(old, new)

    def test_fixed_quantity_and_terminal_fee(self):
        signal = pd.Timestamp('2024-01-06')
        index = pd.date_range('2024-01-08', periods=8)
        data = {'A': pd.DataFrame({'Open': [100, 110, 120, 130, 140, 150, 160, 170],
                                  'Close': [110, 120, 130, 140, 150, 160, 170, 180],
                                  'funding_daily': 0.}, index=index)}
        preds = pd.DataFrame({'Date': [signal], 'Symbol': ['A'], 'market_prob': [.8],
                              'pred_net_7d': [.2], 'baseline_pred_net_7d': [.2]})
        s = weekly.Settings(top_n=1, fee_bps=0, slippage_bps=0)
        curve = weekly.simulate(preds, data, s)
        self.assertAlmostEqual(curve.equity.iloc[-1], 1.7)
        self.assertTrue((curve.turnover.iloc[1:-1] == 0).all())
        self.assertEqual(curve.n_held.iloc[-1], 0)
        paid = weekly.simulate(preds, data, weekly.Settings(top_n=1))
        self.assertGreater(paid.fee.iloc[-1], 0)
        self.assertLess(paid.equity.iloc[-1], 1.7)

    def test_ranking_bear_cash_and_cost_hurdle(self):
        g = pd.DataFrame({'Symbol': ['A', 'B', 'C', 'D'], 'pred_net_7d': [.04, .02, -.01, .03], 'market_prob': .3})
        p = weekly.allocation(g)
        self.assertEqual(p.Symbol.tolist(), ['A', 'D', 'B'])
        self.assertAlmostEqual(p.target_weight.sum(), .25)
        g.pred_net_7d = 0
        self.assertEqual(weekly.allocation(g).target_weight.sum(), 0)

    def test_calendar_gap_not_compressed(self):
        raw = {s: d.copy() for s, d in self.frames.items()}
        saturday = pd.Timestamp('2023-06-03')
        raw['ADAUSDT'] = raw['ADAUSDT'].drop(saturday + pd.Timedelta(days=2))
        p, _, _, _, _ = weekly.prepare(raw, self.settings, now='2025-01-01')
        value = p.loc[(p.Date == saturday) & (p.Symbol == 'ADAUSDT'), 'label_return'].iloc[0]
        self.assertTrue(pd.isna(value))

    def test_missing_held_data_errors(self):
        signal = pd.Timestamp('2024-01-06')
        index = pd.date_range('2024-01-08', periods=8)
        data = {'A': pd.DataFrame({'Open': 100., 'Close': 100., 'funding_daily': 0.}, index=index)}
        data['A'].loc[index[2], 'Close'] = np.nan
        pred = pd.DataFrame({'Date': [signal], 'Symbol': ['A'], 'market_prob': [.8], 'pred_net_7d': [.2]})
        with self.assertRaisesRegex(ValueError, 'Missing Close'):
            weekly.simulate(pred, data)

if __name__ == '__main__':
    unittest.main()
