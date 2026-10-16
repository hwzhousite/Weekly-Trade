# Weekly two-model evaluation

Run from the repository root (or use an absolute path to this script):

```bash
python backtest/run.py
python backtest/run.py --start 2025-01-06 --end 2026-10-05
python backtest/run.py --data-dir data/binance --output-dir backtest/output --fee-bps 10 --slippage-bps 5
```

Reads cached Binance data; download first with `bash download.sh`. No live API calls or orders. Results are written only to the output directory, separate from live weekly plans. The start/end filters apply to Monday entries; older history is retained for training. Train windows and IC selection are recomputed at each decision using only fully realized labels. Incomplete final weeks and initial warmup are listed in `excluded_weeks.csv`. Missing eligible coin outcomes or no passing IC factors cause an error rather than silently excluding unfavorable observations.

Two conceptual models are evaluated:

1. Market direction: regularized logistic classification using BTC/ETH/SOL and market aggregates; target is next-week eligible-universe equal-weight price return > 0.
2. Direction-aware coin ordering: the existing separate coin up/down probability classifiers use coin factors plus global context; the market prediction determines whether to rank upside or downside probabilities. This is not a pairwise learning-to-rank model or a separately trained bull/bear-regime model. This analysis reuses the current production implementation rather than changing the model being measured.

## Files

| File | Meaning |
|---|---|
| `report.md`, `metrics.json` | Summary; market accuracy, balanced accuracy, ROC AUC, Brier, log loss; portfolio return, drawdown, daily Sharpe and costs |
| `market_predictions.csv` | One out-of-sample market probability per week, realized outcome, historical base rate, mature training end |
| `market_calibration.csv` | Probability bins versus realized frequency; evaluation only |
| `coin_predictions.csv` | Every eligible coin's historical out-of-sample up/down probabilities and realized weekly labels |
| `top3_plans.csv` | Actual direction-conditioned Top 3 and signed weights |
| `weekly_diagnostics.csv` | RankIC, Top 3 directional hit rate, signed return versus equal-weight universe, ledger weekly return |
| `equity_combined_top3.csv` | Full strategy: predicted direction + Top 3 |
| `equity_same_direction_universe.csv` | Same predicted direction, all eligible coins equal-weight; isolates selection contribution |
| `equity_always_long_top3.csv` | Always use upside Top 3; practical comparison for the market direction gate, not a pure causal attribution |
| `factor_training_ic.csv` | Training-only IC and selected flag at each historical decision |
| `factor_oos_weekly.csv` | Weekly held-out coin factor IC; held-out market factor values/outcomes |
| `factor_summary.csv` | Selection frequency, average training IC, OOS IC, coin IC mean/std and fraction positive |
| `excluded_weeks.csv`, `run_config.json` | Exclusions, parameters, symbol pool, feature names and run timestamp |

Market classification metrics use one observation per week, never duplicate the market label for every coin. Coin RankIC compares up/down score with direction-signed realized **price return**, within each week. A flat coin return is not a directional hit. Bull/bear and correct/wrong market calls are reported separately in metrics.json.

Factor selection remains **signed training IC > 0.015**. Negative factors are not reversed. OOS factor IC never changes feature selection. `oos_ic_ir` is descriptive mean/std of coin weekly IC, not an annualized statistic or significance test. Market features are constant across coins each week, so their OOS IC is time-series Spearman correlation across held-out weeks. Missing/constant observations yield undefined IC rather than zero.

Net portfolio week returns compound daily ledger returns. Monday rebalance fees belong to the entering week; terminal liquidation fees belong to the last week. Raw signed Top 3/peer returns omit fees/funding and are labeled as price returns. All portfolio baselines use the same intervals and cost settings, fixed signed quantities between weekly decisions, and daily funding accounting.

Limitations: the current downloaded universe can introduce survivorship bias; no historical point-in-time universe reconstruction. Monday open is a daily-data fill proxy earlier than live 08:15 execution. Liquidation is not simulated. Small crypto weekly sample sizes make performance and factor estimates unstable; this output is not proof of profitability. No hyperparameter tuning on these OOS results is performed.

Tests: `python -m unittest discover -s tests -p test_backtest_analysis.py -v`.
