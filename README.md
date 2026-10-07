# Weekly-Trade：BTC / ETH / SOL 市场先验与周频 Top 3

基于 [Daily-Trade](https://github.com/hwzhousite/Daily-Trade) 的日线数据适配器和因子层，源版本 `666a3c8983bcae86d0bf74b365b6d96de1405676`。这是独立的研究、回测和计划生成项目，不连接交易账户或自动下单。

## 策略

1. 用 BTCUSDT、ETHUSDT、SOLUSDT 的 7/14/28 日趋势、28 日波动/回撤、成交量变化、资金费，预测下一周**信号时点合格币种池等权收益为正**的概率。
2. 第一层采用强正则 Logistic Regression，每周一个样本；第二层采用共享 LightGBM 回归模型，将个币日线因子、三个领涨币状态和市场概率共同输入，预测下一周扣除资金费后的收益。
3. 按预测净收益排序选前三；不是按上涨概率排序，也不会把排名末尾直接做空。
4. 市场概率 >= 0.55：总目标仓位 100%；<= 0.45：25%；其余：50%。这些是待验证的研究默认值，不代表最优参数。等权分配至三槽位。若某币预测收益 <= 往返手续费与滑点，该槽位留现金；不向下补位。
5. 周一调仓，周内固定合约数量，逐日核算价格盈亏和资金费。没有日频补仓、等权恢复或择时门控。

## 时间约定（重要）

数据日期指 UTC 日线的开始日期。为避免使用收盘后尚不可交易的信息，首版使用 **UTC 周六 bar（北京周日 08:00 收盘）**生成信号，**下一个 UTC 周一 Open（北京周一 08:00）**作为成交代理，持有至下周一 Open。可于周日准备计划，周一执行。

模型标签 = `Open[t+9] / Open[t+2] - 1`，扣去持有七天的资金费现金流代理；t 为周六。
标签按自然日完整索引计算，缺日不压缩。训练时只使用 `label_end <= signal_date + 1 day` 的成熟标签。生产预测不会读取未来开盘价。

这不是 08:15 的实际成交回测；若在 08:15 执行，应使用分钟成交数据替换 Open 代理。默认每边手续费 10bps + 滑点 5bps。

## 防止信息泄漏

- 第一层每个历史概率均按当时可见的成熟标签重新训练；不把样本内拟合概率喂给第二层。
- 第二层只在已有历史 OOF 概率且币种标签成熟的周训练。
- 默认第一层至少 26 周，第二层至少 26 个 OOF 周，滚动训练窗口 104 周。加上 90 日资格窗口，通常需要一年多历史才开始有效回测。
- 未收盘 UTC bar 自动剔除；实时计划拒绝陈旧缓存，历史计划需显式 `--as-of`。
- 资金费标签、未来收益、训练截止日期、最终排名均不属于模型输入。

## 安装和运行

```bash
git clone https://github.com/hwzhousite/Weekly-Trade.git
cd Weekly-Trade
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

复用 Daily-Trade 的 Binance 缓存，无需复制或重复下载：

```bash
python main.py backtest --data-dir ../Daily-Trade/data/binance
python main.py plan --data-dir ../Daily-Trade/data/binance --capital 10000
```

`plan` 每次默认联网刷新当前 Binance USDT 永续合约币种池与历史行情，再生成计划。无需先执行 download；API 失败不会自动退回旧缓存。可在能够访问 Binance API 的环境运行：

```bash
python main.py download
python main.py backtest
python main.py plan --capital 10000
```

历史周计划（参数必须是 UTC 周六日期）：

```bash
python main.py plan --offline --data-dir ../Daily-Trade/data/binance --as-of 2026-10-03
```

费用实验：

```bash
python main.py backtest --fee-bps 10 --slippage-bps 10
```

默认输出 `output/`：

| 文件 | 内容 |
|---|---|
| `market_oof.csv` | 每周市场概率、历史基准概率、成熟标签截止时间 |
| `weekly_predictions.csv` | 所有合格币的两组预测、市场先验、真实收益标签 |
| `backtest_baseline.csv` | 原有因子选币，不额外输入周市场概率和新增 leader 状态，满仓上限 |
| `backtest_conditional_full.csv` | 使用条件选币、满仓上限，分离选币贡献 |
| `backtest_conditional.csv` | 条件选币 + 市场概率控制仓位 |
| `metrics.json` | 市场 Brier/AUC 与策略收益、回撤、日净值 Sharpe、成本 |
| `plan_YYYY-MM-DD.csv` | 前三榜单、市场概率、状态、预测净收益、目标权重和金额，以及整周价格区间、当前价格与报价时间 |

三组策略均使用收益成本门槛；“满仓上限”不等于必须满仓。计划即使目标权重为零也保留前三榜单。执行数量必须使用实时价格，并与真实持仓对账；CSV 是目标计划，不是买卖差额订单。模型每次运行按历史重训，尚不持久化生产模型；预测 CSV 保留每次研究的审计信息，请保存输出快照。

## 验证与限制

```bash
python -m unittest discover -s tests -v
```

测试覆盖未来数据扰动不改变历史预测、两层标签成熟条件、自然日缺口、Top 3/看跌现金、固定数量持仓、期末平仓成本、缺失持仓行情报错。

当前验证是合成数据和程序正确性验证，**没有宣称策略具有实盘或历史盈利能力**。需在真实缓存上运行三组对照，确认市场 Brier 优于历史上涨基准、条件选币确有额外贡献，再决定是否采用市场门控。

继承的数据下载器按当前成交量选币，仍有幸存者偏差；本项目用历史连续 90 日价格和过去 30 日流动性做资格筛选，但不能恢复已退市或未被下载的币。缺失 held 行情/资金费报错，不虚构零收益。资金费使用日资金费率和日收盘名义金额作为结算代理，未使用每次结算时点价格；下载适配器对无资金费记录日期默认补零，研究前需核对其数据覆盖。滑点采用固定 bps，未模拟冲击、最小下单数量、保证金清算或交易所中断；出现非正权益即停止。期末统一平仓计费。三个市场阈值不做同样本最优参数扫描。

## 每次周计划在线刷新与整周价格区间

```bash
python main.py plan --capital 10000
```

运行顺序：Binance exchangeInfo/24h 成交量排名 → 当前 Top 50 USDT 永续合约（强制包含 BTC/ETH/SOL）→ 下载日线/资金费/现货基差数据 → 检查最新已收盘日线 → 两层模型 → Top 3 → 周价格区间 → 三个币的实时 futures ticker → 输出 CSV。公共行情接口不需要 API key。只有本次下载成功且包含最新已收盘 UTC bar 的币进入此次模型，不重新加载目录中的旧币种缓存。BTC/ETH/SOL 任一缺失或新鲜币种不足 5 个时停止。可用币仍需满足 400 日下载历史门槛和模型的成熟周样本要求。

首次/每次刷新可能较慢，继承的适配器会分页下载历史并处理限流。网络或地区限制（例如 HTTP 451）会导致计划失败，应在允许访问 Binance API 的运行环境执行。

| 新字段 | 含义 |
|---|---|
| `current_price` / `quote_time_utc` | 当前永续合约价格及交易所报价时间，仅实时在线计划提供 |
| `range_reference_close` | 周六信号 bar 收盘价，区间固定基准 |
| `week_low_price` | 下一 UTC 周一至周日最低价的 q05 预测，经可用历史 OOF 误差向外修正 |
| `week_high_price` | 同一整周最高价的 q95 预测，经可用历史 OOF 误差向外修正 |
| `range_nominal_coverage` | 90% 名义整周覆盖目标，非保证 |
| `range_calibration` | `pooled_oof_empirical` 或历史不足时 `uncalibrated` |
| `range_calibration_weeks` / `range_calibration_rows` | 校准可用的历史周数与币种行数 |
| `data_mode` | `binance_refreshed` 或显式离线的 `offline_cache` |

价格区间是未来整周最高/最低路径边界，不是下周末收盘价区间，也不是自动止损/止盈订单。两个额外 LightGBM 分位数头使用和选币相同的历史因子、leader 状态和 OOF 市场概率。标签为 `log(max(High[t+2:t+9]) / Close[t])`、`log(min(Low[t+2:t+9]) / Close[t])`；七天都完整才有效。输出通过指数转换还原绝对 USDT 价格，实时 ticker 不改动固定的预测基准。

校准使用最多 12 个已成熟历史周的滚动样本外预测，至少 4 个可用周才启用；误差跨币种汇总，只向外扩展，避免凭有限样本收窄区间。同周币种与相邻周相关，因此这是经验校准，不宣称严格的 90% 实际覆盖率。校准样本不是独立测试集。当前测试覆盖程序正确性，真实数据上的区间覆盖仍需评估。

仅研究时显式使用缓存：

```bash
python main.py plan --offline --data-dir ../Daily-Trade/data/binance --capital 10000
```

离线模式不提供实时 ticker；历史计划建议同时指定 `--offline --as-of`，避免用当前币种池解释历史结果。所有计划均使用既有周六信号与周一至下周一持有约定；周中重跑仍报告本交易周，不生成未来尚未具备信号的数据。

## 独立数据下载入口

下载已集成到本仓库，不依赖 Daily-Trade 目录，也不需要模型训练后才能下载：

```bash
git pull
pip install -r requirements.txt
bash download.sh
# 等价：python main.py download
python main.py plan --offline --capital 10000
```

默认获取当前成交量前 50 的 USDT 永续合约，并强制包含 BTC/ETH/SOL；历史不足 400 天的币会跳过。写入 `data/binance/币种.parquet` 和 `download_manifest.json`（成功币种、日期范围、失败/历史不足币种），写入采用临时文件原子替换。

自定义下载：

```bash
bash download.sh --start 2023-09-21 --universe-size 50
bash download.sh --symbols BTCUSDT ETHUSDT SOLUSDT ADAUSDT XRPUSDT BNBUSDT --start 2023-09-21
bash download.sh --data-dir /path/to/cache --start 2023-09-21 --end 2026-10-07
```

`--end` 是不包含该日期的 UTC 边界，未收盘 bar 不下载。指定币种后仍会补入三个 leader；周频因子需要至少 5 个币种，模型还需要足够的成熟周样本。数据包含永续 OHLCV、成交量微观字段、日资金费和现货基差参考。下载命令不会加载 LightGBM，也不自动生成计划。数据文件保存在本地，GitHub 只保存下载代码。每次在线计划仍执行自动刷新；只有显式 `--offline` 才使用这些缓存。

**HTTP 451** 是 Binance API 的网络/地区访问限制；加入下载脚本不能消除该限制。请在 Binance 允许访问的运行环境下载。程序会给出清晰提示，不悄悄切换行情源或假装下载成功。
