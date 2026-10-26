# 信号与组合模块：开发规范与实施计划

版本：`signal_spec_v1`；核验日期：**2026-10-25**。

本文约定下一步 `signal&conbination.ipynb` 和自研回测模块的输入、输出及行为。总体架构沿用 [bactest_benchmark.md](<E:/document-prefer/python for finanace/multi-model-bactest/bactest_benchmark.md>)：Qlib 研究基准、A 股账户与撮合适配、成本感知组合以及冻结方案后的前推评价。

## 1. 本模块的交付目标

将六模型已有的滚动预测转成统一信号，并提供基于账户状态生成目标组合的接口：

```text
冻结的滚动预测
    → 日期、版本与分数校验
    → 当时可知的股票资格与横截面排名
    → 目标权重
    → 目标股数及调整意图
    → 交给订单、撮合与账户模块
```

第一版交付四类内容：

1. **统一信号数据**：六模型采用同一主键、时间语义与来源记录。
2. **确定性的组合策略**：先实现等权 Top-K，再实现带换手缓冲的版本。
3. **目标生成接口**：接收真实持仓和资金快照，返回目标及调整原因。
4. **质量与边界报告**：覆盖率、分数异常、规则未知、缺少下个交易日等问题可追溯。

目标权重与目标股数属于交易意图。实际持仓、成交费用和净值由后续账户与撮合模块生成。

## 2. 已有输入的实际情况

项目根目录：`E:\document-prefer\python for finanace\multi-model-bactest`。

本次核验的滚动预测目录为：

```text
E:\document-prefer\python for finanace\multi-model-bactest\data\processed\model_training\rolling\run_20261025T122254Z_28994c61
```

| 项目 | 实际内容 |
| --- | --- |
| 运行状态 | `complete` |
| 绑定的静态训练运行 | `run_20261025T114810Z_ce49dd93` |
| 模型 | `ols`、`lasso`、`random_forest`、`xgboost`、`lightgbm`、`mlp` |
| 预测范围 | 2025 年，四个滚动窗口 |
| 训练窗口 | 最近 **8 个月**，每 3 个月重新拟合；以当前运行的 config 为准 |
| 每窗口拟合上限 | 100000 行 |
| 每模型预测行数 | 1242601；六模型合计 7455606 条模型—股票—日期记录 |
| 单模型索引 | `(datetime, instrument)` |
| 预测文件 | 每个模型的 `<model>_predictions.parquet` |
| 来源与时点 | `metadata.json`、`config.json`、`windows.csv`、每折模型 JSON |

预测文件目前包括：

```text
score, label, label_end_date, model, role, fold,
label_usable, evaluation_label
```

其中索引中的 `datetime` 是**信号日**，不是成交日。现有角色为 `walk_forward_test`。每折模型摘要提供 `available_after`、`max_train_label_end`、来源版本和模型校验值。

读取时先解析 latest 指针，再固定具体运行目录并检查完整性。后续即使 latest 改变，本次信号运行也继续使用已经绑定的版本。

来源校验至少覆盖：六个预测文件的 SHA256 与运行清单一致；`frozen_params_sha256` 与冻结参数一致；因子列表及顺序、模型数据与筛选版本在基础运行和各折摘要中一致；每行的模型名、role、fold 及日期符合对应清单。文件校验采用流式读取，不需要加载模型 pickle 才能导入已有分数。

## 3. 模块边界与调用方式

### 3.1 可以独立完成的部分

信号读取、数据校验、交易日映射、已知信息下的资格检查和每日排名，可以批量生成并保存。它们不需要假定昨天的订单已经成交。

### 3.2 必须接收账户反馈的部分

带换手缓冲的选股、可卖数量约束、目标股数与买卖资金预算，需要知道：

- 此刻实际持有哪些股票，而不是上次希望持有哪些股票。
- 总数量、可卖数量、冻结数量和未完成订单。
- 可用现金、预留资金、账户净值及已知估值价格。
- 已发生的成交、拒单与公司行动。

因此，**不能脱离账户，假设每天目标全部成交，再预生成一整年的“真实持仓路径”。**

在账户模块完成前，可以提供独立日期的组合快照示例；如果保存跨日理想组合，应明确标记 `idealized_preview`，不能冒用成交记录或正式回测净值。

### 3.3 建议接口

```python
load_signal_source(run_dir) -> source_manifest

iter_signal_batches(source_manifest, model, date_range) -> raw_scores

normalize_signals(raw_scores, source_manifest, calendar, config) -> signals

rank_signals(signals, known_market_snapshot, config) -> ranked_signals

build_target_weights(
    ranked_signals, account_snapshot, known_market_snapshot, config
) -> targets, decision_log

estimate_target_shares(
    targets, account_snapshot, known_price_snapshot, rules, config
) -> quantity_intents, validation_log
```

这些是计划实现的函数签名，目前不表示相应 Python 模块已经存在。策略接口借鉴 Qlib 将预测分数转换为目标组合的做法，第一版实现会记录自己的版本和精确定义，不宣称与 Qlib 的默认策略逐笔等价。[^qlib]

## 4. 信号的时间语义

现有标签为：

```text
LABEL0(t) = adjusted_close(t+2) / adjusted_close(t+1) - 1
```

所有 `t+1`、`t+2` 均按市场交易日历计算。第一版采用：

| 字段/环节 | 约定 |
| --- | --- |
| `signal_date` | t 日，对应生成分数所用因子的日期 |
| `signal_available_at` | t 日收盘数据处理完成后；研究模式使用显式登记的处理延迟假设 |
| `decision_at` | 本次目标实际生成的时间；输入快照不得晚于它 |
| `earliest_execution_date` | 下一个市场交易日 |
| `execution_reference` | `next_trading_day_close`，只作为日频模拟参考 |
| `execution_price` | 由后续撮合读取真实未复权行情并处理费用与限制 |

不能把数据文件的实际写入时间当成历史上真实发生的信号发布时间。历史模拟中的可用时间是假设与校验合同，后续实时系统应记录真实到达和计算完成时间。

模型时点至少检查：

```text
model.available_after < signal_date
model.max_train_label_end < 对应滚动窗口起点
窗口起点 <= signal_date <= 窗口终点
所有决策输入的 available_at <= decision_at
earliest_execution_date > signal_date
```

日期字段保存交易日标签，时间戳字段统一使用 `Asia/Shanghai`。窗口切换不得导致同一个模型、股票、信号日重复出现。

现有交易日历截至 2025-12-31。对日历末端尚无下一个交易日的信息：保留原分数，令 `earliest_execution_date` 为空，记录 `awaiting_calendar`，不生成可执行调整。不能机械加一天或虚构下一交易日。

## 5. 交易决策使用的信号表

主键：`(signal_date, instrument, model)`。模型名单固定；同一模型键必须唯一。

| 字段 | 含义与约束 |
| --- | --- |
| `signal_date` | 从原预测索引 datetime 映射，含义不变 |
| `instrument` | 保留现有代码格式，例如 `000001.SZ`；跨引擎映射另存明确对照 |
| `model` | 六个固定模型之一 |
| `score_raw` | 原始预测，不因排序或组合处理覆盖 |
| `score_rank` | 同日、同模型候选集合内的确定性排名，1 为最高 |
| `score_percentile` | 可选，仅表示横截面位置，不能解释为收益概率 |
| `candidate_eligible` | 基于决策时可知信息的新增买入资格 |
| `eligibility_status` | `eligible`、`excluded`、`unknown` 等明确状态 |
| `reason_code` | 缺分数、状态未知、日历不足等机器可读原因 |
| `signal_available_at`、`earliest_execution_date` | 信号与最早成交时点 |
| `fold`、`prediction_role` | 原窗口及角色 |
| `rolling_run`、`base_model_run` | 固定来源 |
| `model_available_after`、`max_train_label_end` | 从模型清单关联，禁止从未来标签推导 |
| `source_hash`、`config_hash` | 输入与当前处理配置的校验值 |

交易信号表采用**允许字段清单**。`label`、`label_end_date`、`label_usable`、`evaluation_label` 和 `supervised_eligible` 等未来监督信息，只能进入独立的事后评价数据集，不能暴露给组合策略。

去掉所有标签列、打乱标签值或改变未来标签可用性后，当前信号排名和目标组合应保持一致。

缺少未来收益标签的最新预测仍然保留；它能否形成交易意图由日历和当时的交易条件决定，不由标签是否存在决定。

## 6. 股票资格与排序规则

### 6.1 资格只依据当时可知的信息

第一版从模型预测覆盖的沪深股票中形成候选，使用当时可核验的上市区间、分数、特征资格及市场状态。

| 条件 | 处理 |
| --- | --- |
| 非有限分数 | 不参与新增买入排名，记录异常；不按标签是否存在筛选 |
| 尚未上市或已明确退市 | 按当时生效区间排除新增买入；已有持仓交给账户处置 |
| 历史 ST、停牌或其他风险状态 | 仅在有当时已知的可靠记录时启用对应规则 |
| 板块权限、交易规则未知 | 严格模式停止相关调整或拒绝新增买入，不能默认为通过 |
| 只知道下一日最终行情才可确认的状态 | 由下一日的执行过程处理，不能反过来修改前一日选股 |
| 已有持仓缺少今日分数 | 保留持仓记录，默认暂停该股票的主动换仓，不前填分数制造排名 |

“不进入新买入候选”不代表“已有持仓自动清零”。两者必须分别记录。历史行业资料不足时，配置明确标记未启用行业中性，不能用当前行业假装完成历史约束。

严格模式与研究模式分别标记。研究模式允许显式登记的状态假设，输出应保留 `research_only`，不得静默降级后继续标记为严格可执行结果。

### 6.2 排名规则

- 同日、同模型独立排序：先按分数降序，再按股票代码升序打破并列。
- 不跨模型直接比较原始分数，也不把原始分数简单相加。
- 不用全样本均值或未来分位数归一化；当日横截面排名仅使用当时已知分数。
- 若整个可用横截面只有一个有效分数值或低于登记的最小覆盖要求，标记为 `uninformative_cross_section` 或 `insufficient_coverage`，默认暂停新增买入并保留已有账户状态。
- 分数都小于零时，排名基线仍按既定规则排序。是否引入绝对收益阈值或择时，是后续独立登记的策略变更。

代码顺序解决普通并列的可复现问题；全横截面恒定时不依赖代码顺序制造选股能力。

若输出 `score_percentile`，采用同集合内的平均并列名次 `r_avg`，当 n>1 时定义为 `(n-r_avg)/(n-1)`，使分数越高越接近 1。n=1 时留空并触发覆盖不足；选取边界的代码排序只用于稳定选择，不把同分股票伪造成不同的预测水平。

## 7. 第一版组合规则

下列参数沿用回测规划中的起始示例，用于开发与基准比较，没有经过本次测试结果优化：

```json
{
  "strategy_version": "topk_buffer_v1",
  "models": ["ols", "lasso", "random_forest", "xgboost", "lightgbm", "mlp"],
  "top_k": 50,
  "max_discretionary_replacements": 5,
  "target_stock_weight": 0.95,
  "single_name_weight_cap": 0.03,
  "rebalance_frequency": "daily",
  "execution_reference": "next_trading_day_close",
  "missing_signal_policy": "hold_existing_no_new_buy",
  "ensemble_enabled": false,
  "status_mode": "strict",
  "preview_enabled": true
}
```

`status_mode=strict` 是可执行目标的要求；缺少必要历史状态时输出明确的不可执行状态。样例预览需主动设置为研究模式并标记结果等级。真实初始资金由账户配置提供；本模块不另外维护一份相互冲突的初始资金。

### 7.1 两个策略分别实现

**A. 等权 Top-K**：作为透明基线，在有信号和资格的候选中取前 K。它不设置主动替换上限，交易仍受账户和撮合约束。

**B. Top-K 换手缓冲**：在 A 的基础上，以真实旧持仓为输入，限制主动轮换数量。其规则固定为：

1. 首次从空仓建仓时，可提出最多 K 个候选，不受“每天替换 5 只”的限制。
2. 正常轮换时，从仍有有效分数、已跌出当前前 K 的旧持仓中，优先选择排名最差且符合已知卖出条件的股票；最多提出 5 个完整退出意图。
3. 对应提出排名最高的未持有合格候选。由执行层确认旧仓实际退出和资金可用后，再决定可执行的新买入数量。
4. 已存在的空缺席位可按规则补足；不能为了每天都凑满 5 个替换而强制交易。
5. 风险或账户强制退出采用独立原因和规则登记，不与主动轮换次数混写。
6. 部分卖出但尚未归零的股票仍占用持仓席位。未完成退出不会自动变成一个可买入的新席位。

这里采用受 Qlib Topk-Dropout 启发的自定义规则，配置名与实现版本固定保存。[^qlib]

### 7.2 权重预算

普通情况下，每个完整席位的起始预算为：

```text
slot_weight = target_stock_weight / top_k
            = 0.95 / 50
            = 0.019
```

若只有 30 个合格席位，默认保留未分配现金，不把每只权重自动扩大为 `0.95/30`。单只权重上限与现金缓冲都属于事前规则。

不可卖、无有效报价或缺分数而保留的旧持仓，应按账户的已知估值保留并占用预算。剩余预算再分配给可调整席位；新买入不能假设这些旧仓已经消失。

若不可调整的持仓已超过股票仓位预算或单只上限，记录 `constraint_breach_existing_position`，暂停会扩大违反程度的新买入；只有可执行的减仓可以缓解约束。不能把名义目标强行写成“约束已经满足”。

首版的“最多替换 5 只”限制股票轮换数量，不等于金额换手限制。若每天将全部留仓重新等权，会额外产生买卖；因此缓冲策略默认保留留仓的实际股数，把新席位和完整退出作为主要调整。全量再等权应另设策略版本。

## 8. 目标组合与目标股数

### 8.1 账户快照

快照至少包括 `account_id`、`as_of`、`cash_available`、`cash_reserved`、`nav`，以及每只股票的总股数、可卖股数、冻结股数、估值价格/时间和未完成订单。

六模型分别对应独立策略账户，同一模型与策略版本下的现金、持仓、未成交订单保持连续。不能将另一个模型的账户快照带入本模型，也不能在季度换模型文件时把持仓清零。

`account_snapshot.as_of <= decision_at`，估值与市场快照也必须满足相同时点约束。旧持仓价格陈旧时可以带标记估值，但不能把前填价格当成新的可成交报价。

### 8.2 目标表

目标表主键建议为 `(decision_at, account_id, strategy_id, instrument)`。记录旧持仓、候选以及受限制而继续持有的股票，不通过“该股票没有一行”隐含清仓。

| 字段 | 用途 |
| --- | --- |
| `strategy_id`、`account_id`、`model` | 独立策略与账户标识 |
| `signal_date`、`decision_at`、`earliest_execution_date` | 决策及执行时间 |
| `current_quantity`、`sellable_quantity` | 从真实账户读取 |
| `current_weight` | 以当前已知估值和 NAV 计算，包含价格漂移 |
| `desired_weight` | 排名策略的理想目标 |
| `constrained_target_weight` | 纳入已知不可调整持仓、资金和预算后的目标 |
| `target_quantity_estimate` | 使用已知参考价计算的股数意图，可为空 |
| `quantity_delta` | 与真实持仓比较，不能以上次理想目标为基准 |
| `reference_price`、`reference_price_as_of` | 定价及股数估算依据 |
| `intent` | `open`、`hold`、`exit`、`reduce`、`defer` 等 |
| `condition` | 如等待旧仓退出、等待可用资金，不满足时不能直接下单 |
| `decision_reason`、`feasible`、`result_level` | 调整原因、可执行性及研究/严格标记 |
| `source_hash`、`config_hash`、`snapshot_hash` | 可复现输入 |

### 8.3 股数转换

```text
目标金额 = constrained_target_weight * 决策时已知 NAV
股数意图 = 目标金额 / 决策时已知的参考价格
随后按板块规则合法化，并复查费用、现金与可卖数量
```

日频首版可用 t 日未复权收盘价估算 t+1 订单数量；不能使用 t+1 最终收盘价预先精确计算股数。成交价格变化造成的资金不足，交给撮合/账户按固定规则缩量或拒单。

整手规则交给统一的历史规则适配器。不能把全部沪深股票统一向下取整成 100 股；例如科创板竞价申报的最低数量和递增单位有区别。具体规则由板块、交易方式和生效日期决定。[^star]

预估费用需要通过费用接口使用同一套配置。信号模块可以预算费用，实际费用只在成交后由账户记账一次。

## 9. 多模型与后续融合

第一版产出六条独立策略信号，使用同一资格政策、组合规则和执行约定比较，不依据 2025 年结果挑一个模型作为唯一正式输出。

后续如增加融合，推荐先登记简单、可解释的横截面排名平均：

```text
同日共同合格股票池内，各模型分别计算排名分位
ensemble_score = 各模型排名分位的等权平均
```

融合需要明确参与模型、共同股票池、缺失处理和权重来源。默认要求全部登记模型有效；缺某模型时记录缺失并暂停对应融合，不用 0 补分数，也不根据当期收益临时换模型。

融合分数仍表示相对排名。若之后进入需要预期收益的优化器，收益校准只能在允许的训练/开发数据上拟合，并在新评价前固定。Qlib 官方文档也区分了仅依赖排序的策略与需要有意义分数尺度的策略。[^qlib]

## 10. 输出目录与质量报告

建议每次生成独立运行：

```text
data/processed/signals/run_<time>_<id>/
    config.json
    source.json
    metadata.json
    signals/model=<model>/part-*.parquet
    rankings/model=<model>/part-*.parquet
    evaluation_labels/                   # 独立可选导出，策略接口不读取
    reports/
        daily_coverage.csv
        score_quality.csv
        exclusions.csv
        calendar_gaps.csv
        source_validation.json
    previews/                            # 可选单日期示例，注明使用的账户快照
```

真实回测过程中调用策略函数产生的目标，保存到该回测运行的 `target_positions.parquet`，与 `orders`、`fills` 和账户快照共同追踪。这样，同一批分数可以用于不同初始资金和组合策略，目标仍来自各自的账户状态。

六模型长表约 745 万行，读取与导出按模型和日期批次执行；不为方便而默认把原始因子、标签、全部模型分数同时拼成一个巨大宽表。比较覆盖范围时区分上游预测范围与本模块筛选结果，差异必须有原因统计。

仅在全部必需产物和校验成功后更新 signals 的 latest 指针。`metadata.json` 至少保存 schema、完成状态、模型名单、时间范围、行数、资格模式、处理假设、来源哈希和结果等级。

## 11. notebook 与 Python 模块的实施顺序

建议把已有的 [signal&conbination.ipynb](<E:/document-prefer/python for finanace/multi-model-bactest/signal&conbination.ipynb>) 作为运行入口，核心逻辑逐步放到以下计划模块：

```text
src/backtest/
    schemas.py       # 信号、快照、目标与原因码
    signals.py       # 来源读取、校验、时间映射、排序
    portfolio.py     # Top-K、缓冲策略和目标生成
    rules.py         # 板块规则查询接口
```

| 顺序 | notebook 内容 | 验收结果 |
| --- | --- | --- |
| 1 | 配置、固定来源、读取 metadata/config/windows | 完成状态和版本正确，不重新训练模型 |
| 2 | 分批读取预测，建立交易字段允许清单 | 去掉未来标签后信号生成不受影响 |
| 3 | 交易日与模型可用时间映射 | 日期不机械加一，末端日历不足可追溯 |
| 4 | 资格状态与每日排序 | 只用当时可知信息，普通并列可复现，恒定分数可识别 |
| 5 | 等权 Top-K 的独立日期预览 | 排名、预算和现金余额可以人工核对 |
| 6 | 输入账户快照的缓冲策略 | 不假定旧目标全部成交，不丢掉受限旧持仓 |
| 7 | 参考价格与规则接口下的股数意图 | 价格时点正确，股数和费用预算可校验 |
| 8 | 分区保存信号与质量报告 | 成功后更新 latest，失败保留上个成功版本 |
| 9 | 给后续 engine 的调用示例 | 定义清楚账户反馈如何进入下一次决策 |

步骤 1—4、8 可以在现有预测基础上完成。步骤 5—7 先使用显式的账户快照做场景验收，账户内核完成后再接入实际成交反馈。

## 12. 必要的边界验收

| 场景 | 预期行为 |
| --- | --- |
| 删除或打乱未来标签 | 分数处理、排名和同一快照下的目标保持不变 |
| 缺少下一市场交易日 | 保留信号，执行日期为空且不可执行 |
| 季度切换 | 来源窗口唯一，模型不提前可用，真实持仓连续 |
| 普通分数并列 | 使用固定规则得到相同排名 |
| 全横截面恒定或覆盖不足 | 不依赖股票代码产生新的选股交易 |
| 旧持仓缺少分数、停牌或不可卖 | 继续占用账户、席位和预算，并留下原因 |
| 一个计划卖单失败或只卖出部分 | 下一决策读取真实数量，不虚增资金或席位 |
| 空仓首次建仓 | 可以提出最多 K 个候选，不受正常轮换数误限 |
| 合格候选少于 K | 未分配预算保留为现金，不自动集中到少数股票 |
| 原有受限持仓已违反仓位上限 | 显示违反状态，不靠改写名义目标掩盖 |
| 跨模型或跨运行重复键 | 拒绝混用，不能静默覆盖 |
| 变更账户、来源或配置 | 生成新的可追溯策略/运行标识 |

测试比较的是接口、规则和时点行为。收益评价在完整账户链路中另行进行。已经生成过的 2025 年静态与滚动结果，保留原版本记录；若据此调整新策略，新版本需要新的未见期验证。

## 参考与核验范围

本文实际检查了现有预测 Parquet 的列、索引、各模型行数，滚动 config、metadata、windows 及首折模型摘要；没有读取各模型测试收益来选择组合参数。项目数据核验日期为 2026-10-25，后续运行以固定目录中的配置和来源为准。

[^qlib]: [Qlib 官方策略文档](https://qlib.readthedocs.io/en/stable/component/strategy.html)，包括预测分数索引、WeightStrategyBase、TopkDropoutStrategy 和分数尺度说明。该文档用于接口与方法参考，本文自定义规则以本文和后续实现版本为准。

[^star]: 上海证券交易所投教：[科创板股票交易规则必知（一）](https://edu.sse.com.cn/tib/qa/c/4866268.shtml)。用于核对不同板块数量规则的差异；实际执行依照对应交易日期、板块和交易方式的有效规则。
