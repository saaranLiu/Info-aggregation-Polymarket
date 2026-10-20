# 预测市场中的信息聚合

本仓库提供分析脚本、分层数据样本和已保存数值结果，用于核对在线预测市场中的订单规模、结算边际与条件定价。分析使用 Polymarket，并用独立的 Kalshi 样本检查结论的适用范围。

仓库不包含论文稿件，也不包含完整交易库或报价数据库。

## 仓库内容

| 路径 | 内容 |
|---|---|
| [`analysis/`](analysis/) | 样本构建、分解、聚类推断、方向与时钟检查、预测、跨平台比较 |
| [`data/samples/`](data/samples/) | 6 份样本：1,483 个 Yes/No 市场、200 个体育 moneyline 市场及其窗口，以及已保存预测、结算估计和 Kalshi 摘要 |
| [`data/README.md`](data/README.md) | 样本选择、字段、单位、来源及使用边界 |
| [`results/`](results/) | 已保存的数值汇总、市场层窗口观测和共同市场模型预测；字段见 [`results/README.md`](results/README.md) |
| [`docs/`](docs/) | 测量定义与结果说明 |

## 快速检查

使用 Python 3.10 或更高版本。已在 Python 3.12 上验证。只需标准库：

```sh
git clone <repository-url>
cd Info-aggregation-Polymarket
bash analysis/run.sh check
```

这一命令会：

1. 核对 6 份样本与来源文件及其哈希是否一致。
2. 检查结算边际和后续残差的会计恒等式。
3. 验证已保存预测中的事件分组，并计算演示样本的 Brier 增益。
4. 核对已保存估计、检验和共同市场模型。

演示输出写入被忽略的 `work/sample_demo/summary.json`。样本按主题和最终结果分层抽取，**样本评分不是总体估计**。Kalshi 文件是保留的推断摘要，不是逐笔原始交易。

重新生成样本：

```sh
python3 analysis/build_samples.py
```

## 从已保存的市场观测重估联合模型

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r analysis/requirements.txt
bash analysis/run.sh outputs
```

`analysis/12_economic_extensions.py` 使用仓库中的市场层窗口数据重新估计四个共同样本模型和事件分组交叉验证。其他阶段读取各自所需的保存汇总。该模式不重估全部交易级结果。

## 从原始交易数据库重估

```sh
DATA_ROOT=/path/to/archive bash analysis/run.sh raw
```

`DATA_ROOT` 下须有 `database/daily_aligned/`、`database/OrderFilled/` 和 `database/CTF/`。中间文件位于 `work/base/` 和 `work/events/`。可配置 `ANALYSIS_THREADS` 与 `ANALYSIS_MEMORY`。

外部数据分别放在 `external/d1`、`external/d4`、`external/d5`、`external/d6`，再运行 31–34 号脚本。`MARKETS_PARQUET` 可覆盖市场元数据路径，默认使用 `work/base/results/markets.parquet`。`DA_DIR` 可覆盖日度交易目录，否则从 `DATA_ROOT` 推导。`HF_ENDPOINT` 可配置报价数据下载端点。

## 分析口径

- Polymarket 主样本：链上结算前 1–24 小时、内部价格区间、市场至少 100 笔交易，并有定义的交易前代理中点。
- 实际结果代币份额与美元权重分别保留；结算差额不等于账户净利润。
- 时钟期限使用结算、计划收盘、提前一小时和回溯性收敛截止。
- Kalshi 采用交易所关闭前 1–24 小时的执行价与终值结算边际，并报告绝对现金分箱与市场内现金排序。
- 两平台都出现小单主导参与和正的小单边际；规模梯度和预测收益依赖规模定义及入样群体。两平台没有逐事件匹配，差异不能直接解释为平台因果效应。

单位与样本差异见 [`results/README.md`](results/README.md)。数据来源与第三方许可见 [`data/README.md`](data/README.md)。本仓库不重新授予第三方数据许可。
