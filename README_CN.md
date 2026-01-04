# 数据清洗工具 (Data Cleaning Tool)

**几秒钟内将混乱数据转化为可直接分析的数据集。**

一款强大的 Claude Code 技能工具，自动化处理数据清洗——处理缺失值、重复数据、异常值和格式不一致问题，配置简单。

![GitHub stars](https://img.shields.io/github/stars/guanyugangstar/cleaning-data)
![Version](https://img.shields.io/badge/version-1.0.0-blue)

## 为什么选择这个工具？

数据科学家**60-80%** 的时间都花在数据准备上。这个工具把这个时间缩短到几分钟。

| 清洗前 | 清洗后 |
|--------|--------|
| 手动Excel清理 | 一键自动化 |
| 格式不一致 | 标准输出 |
| 重复性工作耗时 | 可复用配置 |

## 功能特性

- **缺失值处理** — 删除、填充（均值/中位数/众数）或自定义值
- **重复数据** — 智能检测与去除
- **异常值检测** — IQR、Z-score、MAD 或自定义范围
- **格式标准化** — 日期、文本修剪、大小写转换
- **字段值纠正** — 模糊匹配修复输入错误 (如 "住建局限" → "住建局")
- **数据转换** — 编码、分箱、计算新列

## 快速开始

```bash
pip install pandas numpy scipy scikit-learn pyyaml openpyxl

# 一键清洗数据
python scripts/cleaning_data.py -i raw_data.csv

# 或使用自定义规则
python scripts/cleaning_data.py -c configs/custom.yaml -i raw_data.csv
```

## 适用场景

- **数据分析** — 为 pandas、Excel 或 BI 工具准备数据
- **机器学习** — 清洗训练数据集
- **报表制作** — 标准化多来源数据
- **ETL 管道** — 自动化数据质量检查

## 工作原理

```
raw_data.csv → [清洗管道] → cleaned_data.csv
                   ↓
            EDA报告 + 数据血缘
```

## 项目结构

```
├── scripts/           # 核心清洗引擎
├── configs/           # 可复用配置
├── data/              # 输入输出文件
├── reference/         # 提示模板
└── README.md          # English docs
```

## 配置示例

```yaml
missing_values:
  strategy: mean
  columns: [price, quantity]

outliers:
  method: iqr
  ranges:
    age: [0, 120]
    score: [0, 100]

format_standardization:
  value_correction:
    部门:
      allowed_values: [民政局, 人社局, 市场监管局, 住建局, 税务局]
      threshold: 0.6
```

## 贡献

发现 bug 或有新功能建议？提个 issue 或 PR！

---

**❤️ 为数据从业者打造**
