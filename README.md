# Weekly-Trade

Monday-only crypto perpetual direction ranking. No orders are submitted.

## Workflow

1. Monday after 08:15 Asia/Shanghai, download Binance histories (BTC/ETH/SOL and current liquid perpetual universe).
2. Use the completed UTC Sunday bar and earlier history. Freeze the information cutoff at Monday 08:15 Beijing time. Whole-market means the downloaded liquid universe, not every listed cryptocurrency.
3. Fit a regularized market classifier using BTC/ETH/SOL and market aggregates. GPT receives that quantitative snapshot and optional archived news, and chooses BULL or BEAR. Its qualitative judgement is not reported as a calibrated probability.
4. Fit a coin direction classifier with retained asset factors and retained global context. BULL ranks P(7-day price return > 0); BEAR ranks P(7-day price return < 0). Select exactly three: equal +1/3 weights for longs or -1/3 for shorts, gross exposure 1.
5. Save the decision and Top 3 for the week. Re-running plan reads the saved decision without downloading or reselecting. No daily update command in this version. Rebalance next Monday only.

## Run

```bash
pip install -r requirements.txt
export OPENAI_API_KEY='your-key'
export OPENAI_MODEL='your-supported-Responses-model'
python main.py plan --capital 10000
```

Do not commit credentials. Model name must be explicitly configured. GPT failures stop plan creation; there is no silent quantitative fallback. The GPT call uses the Responses API with strict JSON Schema:
https://developers.openai.com/api/docs/guides/structured-outputs

Explicit quantitative-only mode:

```bash
python main.py plan --market-source quant --capital 10000
```

Offline caches skip Binance refresh, not GPT calls. Use `--offline --market-source quant` for fully offline research. Every candidate must have the latest Sunday bar. Binance HTTP 451 or network restrictions remain external limitations.

Optional news input (`--news-file news.json`) is a JSON array, each item containing `published_at`, `retrieved_at` (timezone-aware ISO timestamps), `url`, and `text`. Both timestamps must be <= Monday 08:15. GPT has no live web search and is instructed to use only supplied evidence; without a news file it analyses the quantitative snapshot only. News downloading is not implemented. Historical GPT judgements can contain model-memory bias; do not treat retrospective GPT calls as leakage-free backtests.

## IC selection

Full candidate names are in [FACTOR_CATALOG.md](FACTOR_CATALOG.md). The original calculation library stays intact so retained factors can be recalculated and dependent intermediate series stay available.

- Coin-specific factors: mean of weekly cross-sectional Spearman correlations with the next week's **price return**, at least three valid coins per cross section.
- Global market factors: weekly time-series Spearman correlation with eligible-universe equal-weight price return, at least 12 mature observations.
- Keep only finite **signed IC > 0.015**, strictly. Negative IC factors are removed, not reversed. Constant features are removed.
- Selection and model fitting use only fully realized labels within the trailing 104 weeks, with minimum 26 mature weeks. Future labels never enter selection.
- Selected global context enters the coin classifier; shared market columns cannot have cross-sectional IC and are assessed with market time-series IC.
- Outputs: `output/factor_ic_YYYY-MM-DD.csv`, including every candidate, measured IC, method and selected flag. No passing factors means stop, not fallback to all factors.
- IC is an in-training selection statistic, not evidence of sample-out-of-sample profitability. Probability outputs are regularized classifier estimates, not independently calibrated guarantees.

Separate positive-return and negative-return classifiers target long/short price direction. Fees and funding are accounted for in backtesting but not in the event label. High direction probability does not imply high expected net return.

## Backtest

```bash
python main.py backtest
```

This is **quantitative-only** walk-forward verification, never a GPT strategy backtest. It recalculates IC in every historical training window, ranks probabilities, and uses signed fixed contract quantities, daily funding, entry/exit fees and daily equity. Sunday close -> Monday open -> next Monday open is a daily-data execution proxy; it does not reproduce 08:15 fills. Current-universe histories can introduce survivorship bias. Binance universe discovery occurs when the command runs; it is not an archived 08:15 universe snapshot. No leverage or liquidation simulation; nonpositive equity raises an error. Real historical profitability and live GPT calls remain unverified.

Outputs include `weekly_predictions.csv`, `factor_ic_history.csv`, `backtest_directional_quant.csv`, `metrics.json`. Weekly plans use `direction_week_DATE.json` (version 2), so previous long-only plan state is not reused.

## Download separately

```bash
bash download.sh
bash download.sh --symbols BTCUSDT ETHUSDT SOLUSDT ADAUSDT XRPUSDT BNBUSDT
```

The command is manual; set your own Monday schedule to run it automatically. Trade execution remains manual.
