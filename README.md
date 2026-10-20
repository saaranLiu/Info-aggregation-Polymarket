# Info-aggregation-Polymarket

预测市场的信息聚合、订单规模与条件定价研究。当前论文为 **Order Size and Conditional Pricing in Online Prediction Markets**，分析 Polymarket，并使用独立 Kalshi 样本检验结论的适用范围。

本仓库包含当前分析脚本、核心数据样本、已保存结果，以及可重建的论文图表。完整版交易和报价数据库需要另行取得。

## 仓库内容

| 路径 | 内容 |
|---|---|
| [`analysis/`](analysis/) | 原始数据处理、分解、聚类推断、方向/时钟/报价检查、预测、Kalshi比较和图表生成 |
| [`data/samples/`](data/samples/) | 6份样本文件：1,483个Yes/No市场、200个体育moneyline市场及其窗口，已保存预测、结算估计和Kalshi摘要 |
| [`data/README.md`](data/README.md) | 样本选择、字段、单位、来源及使用边界 |
| [`results/`](results/) | 当前保存的数值汇总、完整市场层窗口观测和共同市场模型预测；详见[字段说明](results/README.md) |
| [`paper/`](paper/) | 当前TeX、BibTeX、数值宏、统一格式表格、矢量图与12页PDF |
| [`docs/`](docs/) | 原始分析定义及结果说明 |

## 快速检查：不需要原始数据库或TeX

使用Python 3.10或更高版本，已在Python 3.12上验证：

```sh
git clone https://github.com/saaranLiu/Info-aggregation-Polymarket.git
cd Info-aggregation-Polymarket
bash analysis/run.sh check
```

这一命令只用Python标准库，会：

1. 核对6份样本与源CSV及其哈希是否一致。
2. 检查结算边际和后续残差的会计恒等式。
3. 验证已保存预测中的事件分组，并计算演示样本的Brier增益。
4. 校验论文中的源估计、检验、多模型结果、数字精度、图表和引用键。

演示结果写入忽略的`work/sample_demo/summary.json`。样本按主题和最终结果分层抽取，**样本评分不是论文总体估计**。Kalshi文件是保留的推断摘要，不是逐笔原始交易。

重新生成样本：

```sh
python3 analysis/build_samples.py
```

## 重建论文、图表和数值表

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r analysis/requirements.txt
bash analysis/run.sh presentation
```

此外需要TeX Live（含`latexmk`、`pdflatex`、BibTeX）和Poppler的`pdfinfo`。该模式从已保存结果重建图表与表格，再编译并检查：**正文8页，参考文献和附录4页，总计12页**。

图形默认使用Times New Roman；可用`FIGURE_FONT_DIR`指定字体目录。Linux若安装了Liberation Serif，可显式使用：

```sh
FIGURE_FONT_FAMILY="Liberation Serif" bash analysis/run.sh presentation
```

字体替代可能改变图中标签的细微布局；稿件随附的矢量PDF使用原始字体。

## 从完整保存的市场观测重估联合模型

```sh
bash analysis/run.sh outputs
```

其中`analysis/12_economic_extensions.py`使用仓库中完整的市场层窗口数据重新估计四个共同样本模型和事件分组交叉验证；其他阶段读取其所需的保存汇总。该模式不声称重估全部交易级结果。

## 从原始交易数据库重估

```sh
DATA_ROOT=/path/to/archive bash analysis/run.sh raw
```

`DATA_ROOT`下须有`database/daily_aligned/`、`database/OrderFilled/`和`database/CTF/`。处理中间文件位于`work/base/`和`work/events/`。可配置`ANALYSIS_THREADS`与`ANALYSIS_MEMORY`；完整数据库处理需要相应计算资源。

外部数据分别放在`external/d1`、`external/d4`、`external/d5`、`external/d6`，再运行31–34号脚本。`MARKETS_PARQUET`可覆盖外部数据匹配所用的市场元数据路径，默认使用本项目生成的`work/base/results/markets.parquet`。`DA_DIR`可覆盖日度交易目录；否则从`DATA_ROOT`推导。`HF_ENDPOINT`可配置报价数据下载端点。

## 当前分析口径

- Polymarket主样本：链上结算前1–24小时、内部价格区间、市场至少100笔交易，并有定义的交易前代理中点。
- 实际结果代币份额与美元权重分别保留；结算差额不等于账户净利润。
- 时钟期限使用明确的结算、计划收盘、提前一小时和回溯性收敛截止。
- Kalshi采用交易所关闭前1–24小时、执行价与终值的直接结算边际，并报告绝对现金分箱与市场内现金排序。
- 两平台共同出现小单主导与正小单边际；规模梯度和预测收益依赖定义及入样群体。两平台没有逐事件匹配，不能将其差异直接解释为平台因果效应。

保存结果的单位与样本差异见[`results/README.md`](results/README.md)。数据来源与第三方许可见[`data/README.md`](data/README.md)。仓库没有重新授予第三方数据许可。
