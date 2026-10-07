# Weekly-Trade

周一市场判断与 Top 3 固定选币，周内每日更新这三个币的预测价格区间。基于 Daily-Trade 的 Binance 数据适配器与日线因子，源版本 `666a3c8983bcae86d0bf74b365b6d96de1405676`。只生成研究报告和目标计划，不自动下单。

## 四步流程

| 步骤 | 命令/行为 | 保存结果 |
|---|---|---|
| 1. 数据下载 | Binance 当前成交量 Top 50 USDT 永续合约，强制含 BTC/ETH/SOL；获取日线、资金费、现货基差 | `data/binance/*.parquet` |
| 2. 周一市场判断 | 周一北京时间 08:00 后，用刚收盘 UTC 周日数据；BTC/ETH/SOL + 市场趋势、宽度、波动、资金费等聚合特征 | 本周上涨概率和市场状态 |
| 3. 条件选币 | 个币因子 + leader 状态 + 市场样本外概率，预测资金费净收益并排序前三 | `week_周一日期.json`、`plan_周一日期.csv` |
| 4. 每日区间更新 | 更新行情；保持周一概率、排名、币种、目标权重，重新预测这些币本周剩余日期的最高/最低边界 | `ranges_周一日期_更新日期.csv` |

“市场”指下载覆盖的 USDT 永续合约池，不是所有现货/所有交易所。市场层标签是信号时点合格池的等权未来周收益是否为正。Logistic Regression 强正则市场模型与 LightGBM 条件选币模型使用成熟历史标签；历史市场概率由滚动样本外预测生成。默认每层至少 26 个历史周，滚动窗口 104 周；考虑 OOF 与 90 日资格条件，需要一年多历史。下载器还要求至少 400 日行情。

市场概率 >= .55 为 BULL（总仓位上限 100%）、<= .45 为 BEAR（25%），中间 NEUTRAL（50%）。前三等槽位；预测净收益未超过往返手续费/滑点则该槽位留现金。看跌不自动做空。默认每边手续费 10bps、滑点 5bps；阈值为研究默认值，未宣称最优。

## 安装

```bash
git pull
pip install -r requirements.txt
```

## 数据下载

```bash
bash download.sh
# 等价
python main.py download
# 指定币种（仍补入三个 leader）
bash download.sh --symbols BTCUSDT ETHUSDT SOLUSDT ADAUSDT XRPUSDT BNBUSDT
# 指定日期/路径
bash download.sh --start 2023-09-21 --end 2026-10-07 --data-dir /path/to/cache
```

写入 `download_manifest.json` 记录成功、失败、历史不足币种和覆盖日期。`--end` 为不包含该日的 UTC 边界。下载命令不加载 LightGBM，不依赖 Daily-Trade 目录。缓存不提交到 GitHub。

## 周一生成固定计划

**每周一北京时间 08:00 后运行，建议 08:15：**

```bash
python main.py plan --capital 10000
```

默认自动联网下载最新行情并筛选当下币种池，再执行市场判断、选前三和价格区间。第一次成功完成所有数据/报价/预测后才写入本周冻结 JSON；同一周再次运行不会更换概率、前三或目标权重，会更新区间。没有本周计划时，周二至周日不能临时补选本周交易币；下一周必须重新运行 plan。

`--capital` 在首次冻结时保存，周内不因重复运行改变。保存的目标计划不是实际持仓；执行时需对账，并使用实时成交价确定数量。

## 每天更新这三个币的价格区间

每天北京时间 08:00 后运行：

```bash
python main.py daily
```

重新获取当前市场与三个固定标的的行情（即使固定币不再位于当前成交量 Top 50，仍强制下载）。BTC/ETH/SOL 或固定币缺最新日线时停止，绝不以其他币替换。本周冻结 JSON 必须存在；周一前尚未收盘或陈旧缓存均拒绝。

区间预测日期始终截止**本周 UTC 周日结束（下周一 00:00 UTC/北京时间 08:00）**：周一输出未来 7 天；周二输出剩余 6 天；周日输出剩余 1 天。没有已结束时段混入更新区间，也不会每天把终点滚动延长一周。每日区间是剩余期间最高/最低路径边界，不是未来末日收盘区间或自动止损/止盈指令。

两个 LightGBM 分位数头使用最新已收盘日线的个币/市场特征和**冻结的本周市场概率**；历史训练用相同剩余天数、同一 weekday 的成熟标签，以及当时的 OOF 周概率。以更新日收盘为价格基准，q05 最低价、q95 最高价；最多 12 个历史周的 OOF 误差进行只向外扩展的经验校准。少于 4 周标记 uncalibrated。名义整段覆盖目标 90%，跨币/跨周相关，非严格覆盖保证；需真实样本外覆盖评估。

| 字段 | 含义 |
|---|---|
| `Symbol` / `rank` | 周一固定前三与排名 |
| `market_prob` / `stance` | 周一冻结的市场概率/状态 |
| `pred_net_7d` / `target_weight` | 周一收益预测与目标权重，daily 不重新计算 |
| `as_of_bar` / `generated_at` | 此次实际行情截至日期和运行时间 |
| `range_start_utc` / `range_end_exclusive_utc` | 剩余区间起止，终点不包含 |
| `remaining_days` | 本周剩余预测天数 |
| `range_reference_close` | 此次更新的已知收盘基准 |
| `week_low_price` / `week_high_price` | 本周剩余日期预测下限/上限（USDT） |
| `current_price` / `quote_time_utc` | 在线模式实时永续报价与时间 |
| `range_calibration` / `range_calibration_weeks` | 经验校准状态及样本周数 |

离线验证：

```bash
python main.py plan --offline --data-dir ../Daily-Trade/data/binance --capital 10000
python main.py daily --offline --data-dir ../Daily-Trade/data/binance
```

离线仍要求周一计划与当前已收盘数据，且不提供实时 ticker。命令不会自己定时运行；在上述时间调用一次即完成当天更新。

## 历史研究与回测

```bash
python main.py backtest --data-dir ../Daily-Trade/data/binance
python main.py plan --offline --as-of 2026-10-03 --data-dir ../Daily-Trade/data/binance
```

当前 `backtest` 和 `--as-of` 保留此前研究约定：UTC 周六 bar 信号，周一 Open 入场，下周一 Open 出场。历史 `--as-of` 必须为周六，输出历史整周区间；它不创建新的实盘冻结状态。**这与新增周一使用周日数据的在线流程不是同一个成交时间版本**；不能直接把旧回测收益作为新增在线流程验证。周一在线流程在 08:00 后读取周日收盘，实际成交需后续报价，代码不将收盘价当作已实现成交。

三组历史对照：原有因子 Top 3、条件选币满仓上限、条件选币与市场仓位控制。回测固定合约数量、日资金费现金流、真实换手计费和期末平仓，缺失 held 行情报错。输出 weekly_predictions.csv、market_oof.csv、backtest_*.csv、metrics.json。

## 验证与限制

```bash
python -m unittest discover -s tests -v
```

测试包括未来数据扰动、标签成熟、缺日不压缩、固定数量记账、价格区间、下载接口，以及周一保存后 daily 不重新选币/不改变冻结文件但更新区间。

真实在线请求在当前开发环境返回 HTTP 451；完整流程通过合成数据与接口模拟验证，不宣称完成真实 Binance 下载或验证策略盈利。请在 Binance 允许访问的环境运行；失败不退回旧缓存或切换其他行情源。公共行情接口无需 API key。

当前按今天成交量选池存在幸存者偏差；不能恢复未下载或已退市币。下载器无资金费记录日期补零，研究前应核验覆盖；回测按日收盘名义金额近似资金费结算。未模拟盘口冲击、最小数量、清算或交易所中断。每日模型训练的历史币种池可能随当前下载覆盖改变，周一三个币和输出先验则固定。本地冻结文件和每日 CSV 应备份，删除冻结文件后无法还原原计划。

## 周中首次运行 / 补建本周计划

如果周一没有运行、冻结文件丢失或首次安装已到周中，可以显式补建：

```bash
python main.py plan --bootstrap --capital 10000
# 缓存研究模式
python main.py plan --bootstrap --offline --capital 10000
# 之后照常每日更新
python main.py daily
```

补建时仍选用本周一之前的 UTC 周日因子与当时已成熟标签，周二以后的价格和收益不参与本周币种排序；再使用当天最新已收盘行情更新剩余区间。冻结文件和报告含 `reconstructed_midweek=true`，不声称它是周一实际生成或成交的计划。历史币种池快照不可用，因此候选池使用本次下载覆盖（`universe_as_of` 记录时间），可能与真实周一候选池不同。补建不是恢复已丢失的原始计划，也不意味着应追溯执行周一订单。已有冻结文件时不会重选。`daily` 不自动补建；必须先运行上述 plan 命令。
