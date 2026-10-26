# A 股多因子、多模型预测与回测研究

基于 Python 和 Jupyter Notebook 的研究项目，串联原始行情、Alpha158 因子、标签与预处理、因子筛选、六模型训练、信号与组合、事件驱动回测，以及策略结果评价和账面损益归因。

第一版范围到第八模块 `result.ipynb`。重点是把预测结果转成可追溯的订单、成交、持仓与净值，并检查收益来源和数据质量。

> **当前状态：研究流程已串通，数据与交易处理仍待修复验证。**
>
> 已发现异常报价、长期陈旧估值，以及超限持仓阻断后续买入等问题。当前回测采用 `research_price_return` 口径，不能将图表中的账面收益认定为已经验证的经济收益或实盘表现。2025 年已经用于查看结果和修改方案，按开发回放呈现。

## 1. 项目概览

| 项目 | 当前设置或能力 |
| --- | --- |
| 研究范围 | 沪深 A 股，按已有股票池和上市信息过滤；历史状态尚不完整 |
| 默认原始数据区间 | 2022-01-01 至 2025-12-31 |
| 数据接口 | Tushare 风格接口与现有本地 CSV；实际服务地址需核验 |
| 因子 | 自行计算的 Alpha158，滚动窗口为 5、10、20、30、60 日 |
| 当前筛选产物 | 40 个因子；数量以每次成功运行的元数据为准 |
| 模型 | OLS、Lasso、Random Forest、XGBoost、LightGBM、MLP |
| 预测任务 | 默认回归未来收益 `LABEL0` |
| 组合基线 | 等权 Top-K、带换手缓冲的 Top-K |
| 回测 | 日频事件循环、独立模型账户、只做多、无杠杆 |
| 结果输出 | 指标表、个股贡献、质量审计、PNG、Markdown 结论和离线 HTML |

Alpha158 和研究协议参考 Qlib 思路，当前运行代码主要基于 pandas、NumPy 和 scikit-learn 等库，不需要先安装 Qlib。自研交易引擎不宣称与 Qlib 或真实市场逐笔成交等价。

## 2. 八个模块与运行顺序

```text
原始数据
  → Alpha158
  → 标签与预处理
  → 因子评价与筛选
  → 六模型训练及滚动预测
  → 信号与目标组合
  → 订单、撮合、账户和净值
  → 结果评价、账面损益归因和可视化
```

| 顺序 | Notebook | 职责 | 主要输出 |
| --- | --- | --- | --- |
| 1 | [data_loader.ipynb](data_loader.ipynb) | 当前为数据接口连接及示例调用，需配合批量下载流程准备原始数据 | `data/raw/` |
| 2 | [feature.ipynb](feature.ipynb) | 输入覆盖检查、异常行隔离、复权研究价格、Alpha158 和辅助标签 | `data/processed/alpha158/` |
| 3 | [labels&process.ipynb](labels%26process.ipynb) | 标签可用性、时间切分、训练期预处理拟合及模型数据输出 | `data/processed/model_dataset/` |
| 4 | [factors_select.ipynb](factors_select.ipynb) | 逐日 IC/RankIC、覆盖率、稳定性、相关性去冗余与训练期筛选 | `data/processed/factor_selection/` |
| 5 | [model.ipynb](model.ipynb) | 六模型训练、时间验证调参、保存加载、独立测试接口及滚动训练 | `data/processed/model_training/` |
| 6 | [signal&conbination.ipynb](signal%26conbination.ipynb) | 接入滚动预测，生成信号、排序和基于真实账户状态的目标持仓 | `data/processed/signals/` |
| 7 | [backtest.ipynb](backtest.ipynb) | 订单规划、成交约束、账户连续性、费用、净值与模型比较 | `data/processed/backtest/` |
| 8 | [result.ipynb](result.ipynb) | 表现复算、月度收益、成交统计、个股成本归因、异常审计及最终报告 | `data/processed/results/` |

文件名中的 `signal&conbination.ipynb` 沿用当前项目命名。部分模块会读取其他 notebook 的函数定义或检查来源文件，随意改名会影响接口和来源校验。

## 3. 目录结构

```text
multi-model-bactest/
├── README.md
├── data_loader.ipynb
├── feature.ipynb
├── labels&process.ipynb
├── factors_select.ipynb
├── model.ipynb
├── signal&conbination.ipynb
├── backtest.ipynb
├── result.ipynb
├── bactest_benchmark.md          # 回测架构、研究依据与评价规范
├── signal.md                    # 信号与组合接口规范
├── data/
│   ├── README.md                # 原始字段、单位、标签及数据产物说明
│   ├── raw/
│   │   ├── basic/               # 股票信息与市场交易日历
│   │   ├── daily/<年份>/        # 未复权日线，文件名 YYYYMMDD.csv
│   │   ├── adj_factor/<年份>/   # 复权因子
│   │   ├── daily_basic/<年份>/  # 每日估值、规模等辅助指标
│   │   └── benchmark/           # 沪深300价格指数等
│   ├── checks/                  # 下载与覆盖检查
│   └── processed/
│       ├── alpha158/
│       ├── model_dataset/
│       ├── factor_selection/
│       ├── model_training/
│       │   └── rolling/
│       ├── signals/
│       ├── backtest/
│       │   ├── pilot/
│       │   └── benchmark/
│       └── results/
└── tmp/                         # 本地临时文件，不作为正式产物
```

大部分正式产物采用 `run_<UTC时间>_<标识>/` 分版本保存，由所属目录的 `latest_run.json` 指向最近成功结果。不同模块的元数据格式存在差异，读取时使用对应模块接口；较后的训练、信号、回测和结果报告会记录明确的运行状态与文件校验。

`latest_run.json` 表示最近成功运行，不表示收益最好。复现具体实验时，应固定运行目录及其配置，而不只依赖 latest 指针。

## 4. 环境配置

建议使用独立的 Conda 环境。已有 `factor_research` 环境时，可直接激活并检查依赖。

```powershell
conda create -n factor_research python=3.11 -y
conda activate factor_research

python -m pip install numpy pandas scipy pyarrow matplotlib scikit-learn xgboost lightgbm cloudpickle threadpoolctl tushare ipykernel jupyterlab

python -m ipykernel install --user --name factor_research --display-name "Python (factor_research)"
```

在项目根目录启动：

```powershell
jupyter lab
```

打开 notebook 后选择 `Python (factor_research)` 内核。项目当前还没有独立的 `requirements.txt`、`environment.yml` 或统一命令行入口，上面的安装命令用于建立环境，并非精确版本锁文件。

### 当前核验环境

以下是 2026-10-26 本地核验的版本快照，不表示只有这些版本才能运行。加载已保存模型时，优先匹配训练环境和产物中记录的版本。

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.17 |
| NumPy | 2.4.6 |
| pandas | 3.1.0 |
| SciPy | 1.17.1 |
| PyArrow | 26.0.0 |
| Matplotlib | 3.11.2 |
| scikit-learn | 1.9.1 |
| XGBoost | 3.2.0 |
| LightGBM | 4.7.0 |
| cloudpickle | 3.1.2 |
| threadpoolctl | 3.7.0 |
| Tushare | 1.4.29 |
| ipykernel / JupyterLab | 7.4.0 / 4.6.4 |

## 5. 数据准备与接口配置

完整字段、单位、文件约定和数据质量说明见 [data/README.md](data/README.md)。

最重要的原始数据约定：

- 个股数据以 `(ts_code, trade_date)` 为联合主键；代码按字符串读取，保留前导零。
- 日期文件名使用 `YYYYMMDD.csv`，放在对应年份目录。
- `daily` 的价格为未复权价格；`vol` 以手计，转为股时乘 100；`amount` 以千元计，转为元时乘 1000。
- `adj_factor` 与日线按股票和日期对齐，不能用未来值补齐。
- 股票信息中的下载状态和当前简称不能代替每日历史 ST、停复牌或退市状态。
- 合并版和分年版基准文件可能同时存在，不能直接拼接后重复计算日期。

当前 `data_loader.ipynb` 包含接口初始化、一个示例查询，以及自定义服务地址配置。它不是完整的全市场批量下载入口；完整下载流程尚未以独立脚本随项目交付。复现前需要自行准备符合上述结构的数据，并核实接口权限、服务地址和数据来源。只有代码仓库不能直接重建全部现有数据。

### 凭据管理

上传仓库前，将 notebook 中写死的真实 Token 改为本地配置，并清除包含凭据的单元格输出。例如把原 Token 赋值替换为：

```python
import os

TOKEN = os.environ["TUSHARE_TOKEN"]
```

这里读取的是当前进程的环境变量，不会自动加载 `.env`。不要将真实 Token 写进 README、notebook 输出或 Git 提交。私有仓库同样不应保存凭据。

## 6. 两种运行方式

### 6.1 从原始数据开始构建

1. 先准备股票池、交易日历、日线、复权因子及基准数据，核对数据字典和覆盖报告。
2. 检查各 notebook 顶部的项目路径。当前代码含本地绝对路径，移动项目后需要统一调整；从项目根目录启动 Jupyter 有助于减少路径混淆。
3. 依次运行 `feature.ipynb`、`labels&process.ipynb`、`factors_select.ipynb`。
4. 在 `model.ipynb` 中训练六模型，并检查验证与调参报告。验证期用于选择参数，不是最终无偏测试。
5. 后续信号依赖滚动预测。需要生成新预测时，在 `model.ipynb` 开启 `RUN_ROLLING=True`，明确滚动区间、历史窗口和重训频率，再运行滚动训练章节。
6. 运行 `signal&conbination.ipynb`，检查来源、每日覆盖率、时间戳与信号导出结果。
7. 运行 `backtest.ipynb`。首次可令 `RUN_PILOT=True`、`RUN_FULL_BENCHMARK=False`，先检查 OLS 首季度试运行；完整六模型比较需主动开启后者。
8. 最后运行 `result.ipynb`，核对质量审计与损益桥，再查看图表和结论。

每个 notebook 内从上到下运行，先定义函数、读取输入，再执行计算。不要只运行最后的导出单元格。

当前保存的 `model.ipynb` 中，静态独立测试和滚动训练开关均为 `False`；当前保存的 `backtest.ipynb` 已将完整比较开关设为 `True`。运行前检查开关，避免无意重训或重复跑全量回测。

### 6.2 只分析已有回测结果

已有成功的 backtest 产物时，直接打开并从头运行 `result.ipynb` 即可。它读取已有账本，不触发模型训练、信号重建或交易回测。

配置示例：

```python
BACKTEST_SOURCE = "auto"       # auto / benchmark / pilot
BACKTEST_RUN_OVERRIDE = None   # 可填某次成功运行的绝对路径
DETAIL_MODEL = None            # 可改为 "lasso"、"lightgbm" 等
```

`auto` 优先使用 benchmark，缺失时才使用 pilot；不会混合不同运行中的模型曲线。需要稳定复现时填写 `BACKTEST_RUN_OVERRIDE`。修改 `DETAIL_MODEL` 只切换明细分析对象，不改变回测结果。

## 7. 标签、时间划分与信息可用性

默认标签使用完整市场交易日历对齐后的复权收盘价：

```text
LABEL0(t) = C(t+2) / C(t+1) - 1
```

其中 `C` 为复权研究价格；收益采用小数表示，`0.01` 表示 1%。它对应在 `t` 日收盘后形成决策、于下一交易日收盘参考买入后的一日价格收益。多日持仓或换手缓冲下的账户收益不能直接等同于单个 LABEL0 的均值。

| 原始划分 | 日期 | 用途 |
| --- | --- | --- |
| train | 2022—2023 年 | 因子筛选、拟合和训练内预处理 |
| valid | 2024 年 | 模型参数选择与开发验证 |
| test | 2025 年 | 初始测试划分；现在已用于开发回放 |

标签跨越切分边界时，根据实际 `label_end_date` 清除不合规样本。原始因子中的合理缺失与可用于监督训练的样本资格分开管理；不能先过滤日期再直接 shift 生成未来标签。

填补、缩放和训练内早停只能使用当时允许的历史样本。当前项目的 40 因子在 2022—2023 年筛选，不能再把该区间内的某一年称为这组因子的独立未见验证期。

交易信号采用允许字段清单，`label`、`y_raw`、`evaluation_label` 等未来监督信息不进入组合决策。

## 8. 六模型与滚动预测

| 模型标识 | 实现与定位 |
| --- | --- |
| `ols` | 普通最小二乘线性回归基线 |
| `lasso` | L1 正则化线性回归 |
| `random_forest` | 随机森林回归 |
| `xgboost` | XGBoost 梯度提升树 |
| `lightgbm` | LightGBM 梯度提升树 |
| `mlp` | scikit-learn 的 MLP 回归模型 |

训练模块提供统一训练、预测、评价、模型保存与加载接口。滚动预测会保存每折时间窗口和模型可用性，后续信号按当时可用的模型结果生成。

当前已有滚动产物使用 **8 个月历史窗口、每 3 个月重训、最多 100,000 行训练样本**。而 notebook 当前滚动参数中 `ROLLING_TRAIN_MONTHS=None` 表示扩展窗口。两者不同，复现已有实验时应读取对应运行的 `config.json` 和 `windows.csv`，不能把当前编辑器参数当成历史实验参数。

只加载本项目可信来源的序列化模型。跨库版本直接加载旧模型不保证兼容。

## 9. 信号、组合与日频回测

当前组合基线配置为 50 个持仓席位、95% 目标股票预算、3% 单股权重上限，以及每次最多 5 个主动完整退出。实际持仓会受到合法股数、现金、停牌、缺失行情、成交和费用约束，不保证每天恰好达到目标仓位。

日频事件顺序：

```text
交易日开始：解锁上一交易日买入的股份，处理已提供的公司事件
  → 执行此前已提交、约定今日收盘参考成交的订单
  → 更新实际现金、持仓与收盘净值
  → 读取今日收盘后可用的信号
  → 规划下一交易日订单，并预留可用现金
```

当前信号约定为收盘后 30 分钟可用。买入股份执行 T+1 可卖约束；卖出失败或部分成交不会被假定成已释放全部资金和持仓席位。期末保留持仓，不默认全部按最后收盘价变现。

费用包含佣金、印花税、过户费、滑点与冲击近似。默认费率实现限定 2022—2025 年；扩展区间前需更新并核验规则。日线成交量参与率和收盘参考价属于日频代理，不能代表真实竞价排队或逐笔执行结果。

研究价格复权用于因子与标签；现金持仓账本使用原始价格。默认 `price_only` 不从复权因子推造分红或送转。可选公司行动事件接口已实现，但历史事件完整性、税费和市场状态仍需补齐。

## 10. 第八模块：结果报告

`result.ipynb` 提供：

- 区间与年化收益、波动、夏普、最大回撤、回撤持续时间、仓位和换手统计。
- 月度收益和源回测指标复算，月收益连乘核对期末净值。
- 现金、沪深300价格指数参考，以及尚未提供的公平对照清单。
- 委托、成交、部分成交、拒单、佣金、税费及执行损耗统计。
- 逐股账面损益贡献、成本分解和逐日净值变化核对。
- 陈旧估值、水平线、异常日收益、持仓估值跳变及仓位超限审计。
- 六组图表、逐模型明细、文字结论和可离线打开的 HTML 报告。

个股归因的核心公式：

```text
毛账面损益 = 当日持仓市值 - 前日持仓市值 + 当日卖出金额 - 当日买入金额
净账面损益 = 毛账面损益 - 该股当日交易成本
个股本金贡献 = 期间净账面损益 / 初始本金
```

尚未解释的账户损益保留为 `unallocated_residual`，不强行分摊到股票。费用已在账户净值扣除，报告分解时不再重复扣费。

**损益桥平衡说明算术一致，不说明输入报价正确。** 当前未做历史行业和风格暴露归因，也没有用最新行业分类回填过去。

主要输出：

```text
data/processed/results/run_.../
├── config.json / source.json / metadata.json
├── performance.csv / monthly_returns.csv
├── metric_reconciliation.csv / attribution_reconciliation.csv
├── quality_summary.csv / flat_nav_episodes.csv
├── suspicious_valuation_jumps.csv / extreme_daily_returns.csv
├── trading_summary.csv / rejection_reasons.csv
├── benchmark_coverage.csv / acceptance_checks.csv
├── conclusion.md
├── result_report.html
├── figures/*.png
└── <model>/
    ├── stock_contribution.csv
    ├── daily_pnl_bridge.parquet
    ├── cumulative_stock_pnl.parquet
    └── daily_quality.parquet
```

`metadata.status=complete` 表示报告生成完成；`final_evaluation_ready=False` 表示当前不应作为最终无偏评价。修复上游并生成新的成功回测后，可以按相同口径重新运行本报告。

## 11. 已知限制与第一版验收状态

| 项目 | 状态与影响 |
| --- | --- |
| 原始行情质量 | 已发现与证券生命周期冲突的报价，需要回查数据源和标的映射；不能只靠文件齐全判断有效 |
| 历史交易状态 | 缺完整历史 ST、停复牌、真实涨跌停及退市处理数据，研究假设不能覆盖全部特殊情形 |
| 陈旧估值 | 缺失行情时沿用旧估值并做审计；长期缺失和退市资产尚未形成完整经济处置，可能产生水平净值线 |
| 仓位约束恢复 | 已有超限持仓阻止新买入的保守逻辑，但超限资产无法处理时可能长期阻断交易，需要修正恢复机制 |
| 公司行动 | 事件接口存在，完整历史分红送转、净分红及相关税费尚未核验，价格收益不等于完整总收益 |
| 公平对照 | 已有现金与沪深300价格指数；同股票池等权、动量及随机排名的同引擎产物尚未提供 |
| 归因范围 | 已完成个股与费用账本分解；尚无历史行业和风格暴露归因 |
| 最终评价 | 2025 年已用于开发，尚未完成冻结方案后的新未见区间评价 |
| 工程复现 | 当前以 notebook 为入口，仍有绝对路径，完整批量下载入口和依赖锁文件待补充 |

截至 2026-10-26，已有的运行验收包括：28 项回测账本与边界检查；结果模块 36 项源指标复算、6 项归因手工案例，以及六模型逐日损益桥核对。它们验证对应代码路径和账本算术，不覆盖所有历史行情、交易状态及市场执行真实性。

第一版的可信结论仍以修复异常数据和交易处理、重新构建受影响的派生产物为前提。风险优化、复杂模型融合和实盘工程化留作后续扩展。

## 12. 常见问题

### `NameError: name 'RESULT_RUN' is not defined`

通常是跳过了创建报告目录的单元格，或重启内核后仅运行导出代码。Notebook 中保存的历史输出不能恢复变量。最直接的方式是从头运行 `result.ipynb`；它只分析已有回测，不重训或重跑交易回测。

### 找不到 `latest_run.json` 或上游数据目录

对应模块尚未成功运行、数据没有复制完整，或项目路径不正确。先固定并核验上游成功运行，再执行下游。只有从 GitHub 下载 notebook 时，通常还没有本地原始数据和模型产物。

### 文件哈希校验失败

源文件或配置在生成元数据后发生变化。不要通过关闭校验或手改摘要绕过问题；应定位变更，并从受影响模块开始生成新的运行版本。

### 曲线长期水平或突然跳升

先看 `quality_summary.csv`、`flat_nav_episodes.csv` 和 `suspicious_valuation_jumps.csv`，再核对实际成交、现金、逐股价格、证券生命周期和公司行动。不能仅据曲线平滑认定风险低。

### 换了目录后无法读取原有产物

部分产物记录了绝对来源路径。只复制一个下游运行目录通常不够，应保留完整来源链并核对路径；必要时在新项目根目录按顺序重新生成。不要直接改动已经有哈希绑定的产物文件。

### 运行很慢

先区分当前是在训练、滚动训练、全量因子计算、信号导出还是回测。随机森林和滚动窗口会显著增加拟合工作量。首次先跑小区间；只看结果时直接运行 `result.ipynb`，不必重跑全部八个模块。

## 13. Git 仓库提交建议

提交 notebook、Markdown 文档及未来补充的依赖文件；原始数据、模型、生成报告和临时文件保留在本地或另行归档。提交前检查 notebook 源码与输出中的凭据。

若希望保留数据字典，可在根目录 `.gitignore` 中采用：

```gitignore
# 忽略数据产物，保留数据说明
/data/*
!/data/README.md

/tmp/
.ipynb_checkpoints/
__pycache__/
*.pyc
.env
.env.*
!.env.example
.venv/
venv/
```

如果已有 `/data/` 整目录忽略规则，应将它替换为上面的两行数据规则；仅增加 `!/data/README.md` 无法穿透被整体忽略的父目录。`.gitignore` 也不会自动移除已被 Git 跟踪的文件。

当前尚未提供 `LICENSE` 文件，不应默认项目或数据按某种开源许可证授权。数据使用范围另受实际服务商授权约束。

## 14. 进一步阅读

- [数据字段、单位和产物接口](data/README.md)
- [回测架构、研究依据与基准规范](bactest_benchmark.md)
- [信号与组合开发规范](signal.md)

这些文档包含制定时的状态快照；判断当前实现，以实际 notebook、固定运行的配置、元数据和审计报告为准。
