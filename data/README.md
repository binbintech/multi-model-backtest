# 数据目录说明

本目录服务于多因子、多模型预测与回测研究，保存下载数据、Alpha158 因子、模型数据集和质量检查结果。

处理顺序：[`data_loader.ipynb`](../data_loader.ipynb) → [`feature.ipynb`](../feature.ipynb) → [`labels&process.ipynb`](../labels&process.ipynb) → 后续因子评价、模型训练和交易回测。

本文依据现有 notebook 和已保存产物编写，核验日期为 **2026-10-24**。配置、数量和运行目录均可能随重建而变化，实际使用以对应运行目录内的 `metadata.json`、`config.json` 和报告为准。

## 1. 从哪里读取数据

| 任务 | 使用位置 | 说明 |
| --- | --- | --- |
| 检查下载结果、研究原始量价 | `raw/` | 接口返回数据，尚未统一股票池、复权和预处理 |
| 单因子评价、检查原始因子 | `processed/alpha158/run_*/` | 158 列原始因子，可包含合理的 NaN |
| 训练、验证和测试模型 | `processed/model_dataset/run_*/` | 已按训练期参数处理，分为 `tree`、`linear` 两种版本 |
| 检查股票覆盖和异常记录 | `checks/` 及运行目录内的报告 | 文件齐全不代表每只股票、每个字段均有效 |
| 快速查看某日市场数据 | `processed/market_preview_20251231.csv` | 单日行情、复权因子和每日指标预览，不是模型训练集 |

新模型优先使用 `model_dataset`。不要直接把分区中的所有列作为 X：标签、未来日期和可用标记均属于辅助列。

## 2. 目录结构

```text
data/
├── README.md
├── raw/
│   ├── basic/                         # 股票池、基本信息、交易日历
│   ├── daily/YYYY/YYYYMMDD.csv        # 个股未复权日线
│   ├── adj_factor/YYYY/YYYYMMDD.csv   # 复权因子
│   ├── daily_basic/YYYY/YYYYMMDD.csv  # 每日估值、市值、换手率
│   └── benchmark/                     # 沪深 300 指数日线，分年及合并版本
├── checks/
│   ├── file_check_*.csv
│   ├── coverage_check_*.csv
│   ├── stock_basic_duplicate_records.csv
│   └── alpha158_input/
│       ├── input_coverage.csv
│       └── excluded_input_rows.csv
└── processed/
    ├── market_preview_20251231.csv
    ├── alpha158/
    │   ├── latest_run.json
    │   └── run_<UTC时间>_<标识>/
    │       ├── part-*.parquet          # 无 pyarrow 时为 part-*.csv.gz
    │       ├── feature_names.json
    │       ├── metadata.json
    │       ├── feature_quality.csv
    │       ├── split_summary.csv
    │       ├── input_coverage.csv
    │       └── excluded_input_rows.csv
    └── model_dataset/
        ├── latest_run.json
        └── run_<UTC时间>_<标识>/
            ├── tree/{train,valid,test}/part-*
            ├── linear/{train,valid,test}/part-*
            ├── feature_names.json
            ├── config.json
            ├── metadata.json
            ├── source.json
            ├── preprocessor.json
            ├── split_summary.csv
            └── training_feature_report.csv
```

每次成功构建生成独立的 `run_*` 目录，最后更新本层的 `latest_run.json`。该指针只记录运行目录名，不包含数据。失败的临时目录不应当作正式产物；不要跨运行目录混合分区或预处理参数。

## 3. 数据范围与主键

- 默认原始日期范围为 `20220101` 至 `20251231`，交易日历有 969 个交易日，首个交易日为 `20220104`。
- 默认研究股票池来自 `raw/basic/stock_codes_a_share.csv`，当前为 5,559 个沪深 A 股代码；包含已退市股票。`INSTRUMENTS=None` 表示使用该池，而非读取接口返回的全部市场代码。
- 原始日线包含池外记录，例如北交所股票；进入因子计算前会过滤。指定 `INSTRUMENTS` 时必须是已保存股票池的子集。
- 使用 `stock_basic.csv` 中的 `list_date` 排除上市前记录，不按当前上市状态删除有效的历史行情。股票池规模不等于研究期间有有效行情或最终有因子样本的股票数。
- 原始个股表以 `(ts_code, trade_date)` 为联合主键。代码按字符串读取以保留前导零，例如 `000001.SZ`；原始日期使用 `YYYYMMDD`。
- 因子及模型表采用 `(datetime, instrument)` 两级索引，分别对应交易日和原股票代码。CSV 分区会将这两个索引写成普通列，读取后需恢复。

变更股票范围时，应统一更新股票池、基本信息和后续派生数据。保存了退市记录有助于保留历史覆盖，但不能据此假定该股票池已完整涵盖所有历史公司或历史指数成分。

## 4. 原始数据字典

### 4.1 股票池、基本信息与交易日历

| 文件 | 用途与关键字段 |
| --- | --- |
| `stock_codes_a_share.csv` | 因子默认股票池；`ts_code`、`returned_in_statuses`、`status_conflict` |
| `stock_basic.csv` | 股票池对应基本信息；代码、简称、地区、行业、`list_date`、`requested_list_status` |
| `stock_basic_a_share_records.csv` | 筛选后的 A 股基本记录，保留状态来源，用于追溯 |
| `stock_basic_raw_L.csv` / `stock_basic_raw_D.csv` | 分别按上市、退市状态请求的原始返回，可含研究池外证券 |
| `trade_calendar_raw_YYYY0101_YYYY1231.csv` | 分年交易日历接口返回，含 `exchange`、`cal_date`、`is_open`、`pretrade_date` |
| `trade_calendar_20220101_20251231.csv` | 当前保存的合并开市日记录，保留日历字段 |
| `trade_dates_20220101_20251231.csv` | 单列 `trade_date` 的交易日列表，供特征和标签对齐 |

其中 `requested_list_status` 和 `returned_in_statuses` 是下载时的状态来源记录，不是每日历史状态。简称、行业等基本信息也不能直接当作严格的历史时点快照。

### 4.2 `daily`：未复权个股日线

| 字段 | 含义与单位 |
| --- | --- |
| `ts_code` / `trade_date` | 股票代码 / 交易日期 |
| `open` / `high` / `low` / `close` | 开盘 / 最高 / 最低 / 收盘价格，元/股 |
| `pre_close` | 接口提供的除权昨收参考价，不一定等于上一条记录的原始收盘价 |
| `pct_chg` | 接口涨跌幅，单位为 %；值 `1` 表示 1% |
| `vol` | 成交量，手 |
| `amount` | 成交额，千元 |

该表保留原始价格。停牌、未上市和源数据缺口均可能使某只股票缺少某日记录，不能只凭缺行就认定为停牌。字段定义参见 [Tushare 日线说明](https://tushare.pro/document/2?doc_id=27)。

### 4.3 `adj_factor`：复权因子

字段为 `ts_code`、`trade_date`、`adj_factor`。与日线按代码和日期合并；复权因子必须为正且有限。缺失记录不会被填成 1，也不会用未来值补齐。接口说明见 [Tushare 复权因子](https://tushare.pro/document/2?doc_id=28)。

### 4.4 `daily_basic`：每日指标

| 字段 | 含义与单位 |
| --- | --- |
| `turnover_rate` | 换手率，% |
| `pe_ttm` | 滚动市盈率；亏损等情况下可为空 |
| `pb` | 市净率 |
| `total_mv` | 总市值，万元 |
| `circ_mv` | 流通市值，万元 |

另有联合主键 `ts_code`、`trade_date`。当前标准 Alpha158 不使用这些指标；若后续加入估值、规模等扩展因子，应另存因子组并注明可得时点，不改变标准 158 列。单位依据 [Tushare 每日指标说明](https://tushare.pro/document/2?doc_id=32)。

### 4.5 `benchmark`：基准指数

保存沪深 300（`000300.SH`）的 OHLC、涨跌、成交量额等字段。`csi300_20220101_20251231.csv` 为合并版本，同时还保存各年版本；**不能将合并文件和分年文件再次直接拼接**，否则会重复日期。

指数数据供后续基准收益和超额收益比较。指数点位不是个股价格，不应直接当作可成交证券或混入个股股票池。

## 5. Alpha158 因子层

### 5.1 计算口径

因子名称和顺序依据 [Qlib Alpha158DL](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py)，由 9 个 K 线因子、4 个当日价格比率，以及 29 类滚动因子 × 5 个窗口构成。窗口为 `5/10/20/30/60` 个市场交易日。

本项目的量价处理为：

```text
原始股数 = vol × 100
原始 VWAP = amount × 1000 / 原始股数
复权倍率 = 当日 adj_factor / 该股票输入期内首个有效 adj_factor
复权 OHLC、VWAP = 原始价格 × 复权倍率
默认复权成交量 = 原始股数 / 复权倍率
```

特征在 t 日收盘后形成，只使用 t 日及之前的信息。每只股票单独按交易日历重建索引，缺失日保留 NaN，滚动窗口不会因删除异常行而压缩。至少累计 61 个有效价格观测后才输出样本，且输出日要求有有效价格和正成交量。

因子列写为 `float32`，名称和顺序以 `feature_names.json` 为唯一读取依据。常数窗口可能使相关系数或 R² 为 NaN，这不等同于计算失败。`ROC5` 表示过去第 5 日收盘价/当前收盘价；`BETA5` 表示价格对时间的回归斜率/当前收盘价，不是相对市场的 beta。

公式参照 Qlib，不代表本地供应商数据、复权基准和最终数值与官方二进制数据逐位相同。

### 5.2 因子分区中的辅助列

| 字段 | 用途 |
| --- | --- |
| `LABEL0` / `LABEL5` / `LABEL10` | 分别为 1、5、10 个交易日的未来收益 |
| `label_entry_date` | 标签进入日期 |
| `LABEL*_end_date` | 对应标签退出日期 |
| `LABEL*_usable` | 上游按自身时间边界计算的监督标签可用标记 |
| `history_count` | 累计有效价格观测数 |
| `split` | 上游初步时间切分标记 |

这些列都不属于 X。第三模块会按自己的配置重新生成切分和可用标记，不能将两层标记混用。

### 5.3 标签定义

令 C 为复权收盘价，h 为持有交易日数，默认进入延迟为 1：

```text
LABEL_h(t) = C(t + 1 + h) / C(t + 1) - 1
LABEL0(t)  = C(t + 2) / C(t + 1) - 1     # h = 1 的兼容命名
LABEL5(t)  = C(t + 6) / C(t + 1) - 1
LABEL10(t) = C(t + 11) / C(t + 1) - 1
```

收益为小数，`0.01` 表示 1%，不同于 `daily.pct_chg` 的百分数单位。进入或退出日无有效成交时标签为空；数据尾部未来价格尚不存在时也为空。不要在已过滤的因子表上重新直接 `shift` 计算标签。

## 6. 模型数据集层

### 6.1 目标与时间划分

当前默认使用 `LABEL0` 回归原始收益；可在 `labels&process.ipynb` 中选择 1/5/10 日周期，或将目标改为收益是否超过阈值的涨跌二分类。修改进入延迟或增加上游未提供的周期，需要先重建因子数据。

| 分段 | 当前默认区间 |
| --- | --- |
| `train` | 2022-01-01 至 2023-12-31 |
| `valid` | 2024-01-01 至 2024-12-31 |
| `test` | 2025-01-01 至 2025-12-31 |

不随机拆分时序样本。标签结束日期晚于所属分段末日时，该行不能作监督样本，避免训练标签使用验证期价格。分段之前的历史行情可用于形成当时已知的因子。

### 6.2 两种预处理版本

| 版本 | 处理 | 典型用途 |
| --- | --- | --- |
| `tree` | 训练中位数填补，不缩放、不截尾 | 树模型输入 |
| `linear` | 训练分位数截尾 → 训练中位数填补 → 训练均值/标准差缩放 | 线性模型等需要缩放的输入 |

默认截尾范围为训练分布的 1%–99%。仅从训练期且标签、特征均可用的行中，按索引稳定哈希选择最多 100,000 行拟合统计量，因此分位数、中位数等为训练抽样估计。`FIT_MAX_ROWS=None` 可使用全部训练样本，但需要更多内存。

验证、测试和未来预测只应用 `preprocessor.json` 的状态，不重新拟合。全空训练列填 0，常数列缩放分母设为 1，保留 158 列结构；原始收益不做标准化。

### 6.3 模型分区中的辅助列

| 字段 | 含义 |
| --- | --- |
| `y_raw` | 所选周期的原始未来收益，保留供审计与评价 |
| `target` | 回归收益或 0/1 分类目标；不合格监督行设为 NaN |
| `label_entry_date` / `label_end_date` | 标签进入和退出日期 |
| `missing_ratio` | 预处理前 158 列特征的缺失比例 |
| `feature_eligible` | 缺失比例不超过阈值，默认 50% |
| `label_eligible` | 标签存在且不跨越分段末日 |
| `supervised_eligible` | 特征和标签均可用，监督训练读取时必须筛选此标记 |
| `split` | 本模块重新计算的 train/valid/test 标记 |
| `label_status` | `ok`、`missing_label` 或 `cross_boundary`；内部审计还可出现 `outside` |

`outside` 行只参与审计，不写入 train/valid/test 分区。最新日期缺标签的行仍可保存并用于预测，预测读取只要求 `feature_eligible=True`。辅助列即使包含可用未来收益，也不得混入模型输入。

### 6.4 配置与报告文件

| 文件 | 记录内容 |
| --- | --- |
| `feature_names.json` | 模型输入的 158 个列名及顺序 |
| `config.json` | 目标、进入延迟、切分、缺失阈值、截尾、抽样和格式配置 |
| `metadata.json` | 目标类型、分区数量、拟合样本数、来源及环境版本 |
| `source.json` | 固定上游版本、元数据哈希、分区大小/修改时间及交易日历 |
| `preprocessor.json` | 两种版本使用的填补值、截尾值、均值、缩放值及拟合索引摘要 |
| `split_summary.csv` | 各分段总行数、可用行数、缺标签及跨界清除数量 |
| `training_feature_report.csv` | 拟合训练样本的缺失率与预处理参数，不是全量测试分布 |

## 7. 质量检查与异常处理

下载阶段的 `file_check_*.csv` 记录各交易日各接口的文件有效性和行数；`coverage_check_*.csv` 记录相关表相对日线的缺失代码数量和示例。这些报告可能包含股票池外记录，不能直接等同于模型股票池内的缺失量。

因子输入审计位于 `checks/alpha158_input/`：

| 报告 | 关键字段 |
| --- | --- |
| `input_coverage.csv` | 日期、原始行数、池外行数、上市前行数、候选行数、隔离行数、保留行数 |
| `excluded_input_rows.csv` | 被排除的具体行情、上市日期、复权因子和 `reason` |

`before_listing` 表示上市前记录；`missing_adj_record` 表示缺少同股票同日复权记录。还会检查非有限值、无效复权因子、非法价格、负量额、OHLC 关系和量额矛盾。

当前默认隔离少量行级异常：上市日期过滤后的候选行中，全期间异常比例上限为 **0.1%**，单日上限为 **2%**。超过任一阈值停止；`INVALID_ROW_POLICY='raise'` 可设为发现异常即失败。阈值分母随配置日期和股票范围变化，小范围调试可能更容易触发上限。

不填造价格或复权因子，不使用未来值补洞。被隔离日期在股票交易日历上仍是 NaN，有效的退市历史行情保留。成功运行目录内另存审计副本；`checks/alpha158_input/` 是最近一次输入检查结果，不能代替历史运行的审计副本。

## 8. 本次已核验产物快照

以下为 2026-10-24 已落盘结果，不是未来每次运行的固定数量。

| 项目 | 本次结果 |
| --- | --- |
| 日线 / 复权 / 每日指标文件 | 各 969 个日文件 |
| 已保存股票池 / 有效输入股票 | 5,559 / 5,350 只 |
| 池外行情 / 上市前记录 / 隔离缺复权记录 | 209,954 / 230 / 66 行 |
| 有效输入行情 | 4,835,886 行 |
| Alpha158 输出 | 4,515,966 行、158 列因子、54 个 Parquet 分区 |
| 模型预处理拟合 | 从 2,033,792 条训练候选中选择 100,000 条 |

| 模型分段 | 保存行数 | 可用监督样本 | 缺标签行数 | 跨界清除行数 |
| --- | ---: | ---: | ---: | ---: |
| train | 2,045,432 | 2,033,792 | 1,535 | 10,105 |
| valid | 1,227,933 | 1,217,014 | 760 | 10,159 |
| test | 1,242,601 | 1,231,144 | 11,457 | 0 |

本次 Alpha158 运行目录为 `run_20261024T123322Z_3e71525d`，模型数据集运行目录为 `run_20261024T125348Z_2adc3b07`。二者均可由各自的 `latest_run.json` 查找；后续成功运行会改变指针。

## 9. 运行与读取

### 9.1 重建顺序

1. 在 notebook 所用 Python 内核环境准备 `numpy`、`pandas`；推荐安装 `pyarrow`，以支持当前 Parquet 产物。
2. 完成下载，核对股票池、基本信息和日历，再检查下载报告。
3. 从头运行 `feature.ipynb`，确认成功输出和输入异常报告。
4. 从头运行 `labels&process.ipynb`，确认三个分段均有可用监督样本。
5. 后续模型固定具体运行目录、目标和 view；不要在同一轮对照中混用不同版本。

修改配置或从外部更新了 notebook 后，重新加载文件、重启内核并从头运行，避免沿用旧函数或旧状态。只需测试代码时可关闭 `RUN_BUILD` 或 `RUN_PROCESS`。

### 9.2 独立分批读取示例

以下示例不依赖 notebook 内核中的辅助函数；每次只读取一个模型分区。先从项目目录启动，或将 `project` 改为项目的绝对路径。读取已有 Parquet 文件仍需可用的 Parquet 引擎，不能仅因当前环境缺少 pyarrow 就按 CSV 方式读取。

```python
from pathlib import Path
import json
import pandas as pd

project = Path.cwd()  # 应指向包含三个 notebook 的项目目录
root = project / 'data' / 'processed' / 'model_dataset'
latest = json.loads((root / 'latest_run.json').read_text(encoding='utf-8'))
run = root / latest['run_dir']  # 正式实验建议固定为明确的历史 run_* 路径
names = json.loads((run / 'feature_names.json').read_text(encoding='utf-8'))
metadata = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))

def iter_supervised_batches(split='train', view='tree'):
    if split not in ('train', 'valid', 'test') or view not in ('tree', 'linear'):
        raise ValueError('split 或 view 无效')
    parquet = metadata['format'] == 'parquet'
    pattern = 'part-*.parquet' if parquet else 'part-*.csv.gz'
    paths = sorted((run / view / split).glob(pattern))
    if not paths or len(paths) != metadata['partitions'][view][split]:
        raise ValueError('数据分区缺失')
    for path in paths:
        if parquet:
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path, dtype={'instrument': str}, parse_dates=['datetime'])
            frame = frame.set_index(['datetime', 'instrument'])
        flag = frame['supervised_eligible']
        if not flag.isin([True, False]).all():
            raise ValueError('监督样本标记格式无效')
        frame = frame.loc[flag.eq(True)]
        yield frame[names].astype('float32'), frame['target']

X_batch, y_batch = next(iter_supervised_batches('train', 'tree'))
print(X_batch.shape, y_batch.shape)
```

在 `labels&process.ipynb` 的内核中，也可直接调用其 `load_model_data`、`iter_model_batches`、`load_prediction_data`。全量 `load_*` 会拼接数据，需要足够内存；大规模预测可逐批进行，不能假定所有模型都支持分批增量训练。

## 10. 后续回测使用约定

保持特征时点、标签进入/退出价格与策略成交时点一致。当前标签为收盘到收盘研究收益；若采用次日开盘成交，应同步修改标签与成交口径，不能直接把现有标签当作该策略的实际收益。

模型数据集不包含完整的交易撮合规则。后续回测还需单独处理原始成交价格与复权研究价格的关系、公司行动、涨跌停、停牌、费用、滑点和持仓约束。新增历史股票池、风险状态或行业约束时，应使用对应历史时点可得的数据。

保留原始数据和成功运行的配置、来源、审计及预处理状态。更新源数据、股票池、日期、标签或预处理配置后，应建立新的派生运行；不要手工覆盖分区使元数据与实际内容不一致。
