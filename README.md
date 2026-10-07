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

或在可访问 Binance API 的环境下载：

```bash
python main.py download
python main.py backtest
python main.py plan --capital 10000
```

历史周计划（参数必须是 UTC 周六日期）：

```bash
python main.py plan --data-dir ../Daily-Trade/data/binance --as-of 2026-10-03
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
| `plan_YYYY-MM-DD.csv` | 前三榜单、市场概率、状态、预测净收益、目标权重和金额 |

三组策略均使用收益成本门槛；“满仓上限”不等于必须满仓。计划即使目标权重为零也保留前三榜单。执行数量必须使用实时价格，并与真实持仓对账；CSV 是目标计划，不是买卖差额订单。模型每次运行按历史重训，尚不持久化生产模型；预测 CSV 保留每次研究的审计信息，请保存输出快照。

## 验证与限制

```bash
python -m unittest discover -s tests -v
```

测试覆盖未来数据扰动不改变历史预测、两层标签成熟条件、自然日缺口、Top 3/看跌现金、固定数量持仓、期末平仓成本、缺失持仓行情报错。

当前验证是合成数据和程序正确性验证，**没有宣称策略具有实盘或历史盈利能力**。需在真实缓存上运行三组对照，确认市场 Brier 优于历史上涨基准、条件选币确有额外贡献，再决定是否采用市场门控。

继承的数据下载器按当前成交量选币，仍有幸存者偏差；本项目用历史连续 90 日价格和过去 30 日流动性做资格筛选，但不能恢复已退市或未被下载的币。缺失 held 行情/资金费报错，不虚构零收益。资金费使用日资金费率和日收盘名义金额作为结算代理，未使用每次结算时点价格；下载适配器对无资金费记录日期默认补零，研究前需核对其数据覆盖。滑点采用固定 bps，未模拟冲击、最小下单数量、保证金清算或交易所中断；出现非正权益即停止。期末统一平仓计费。三个市场阈值不做同样本最优参数扫描。
