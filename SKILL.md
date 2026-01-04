---
name: analyzing-data-for-cleaning
description: Use when user needs to analyze data - upload data file, specify path, or use data/ directory. Load first 5 rows, generate custom.yaml config, run cleaning script, and review EDA report for anomalies.
---

# Analyzing Data for Cleaning

## 1 概述

**使用流程：** 用户上传数据文件 → 加载前5行 → LLM 智能分析生成 custom.yaml → 调用 cleaning_data.py 执行清洗 → LLM检查脚本生成的EDA报告并修复异常

**核心原则：** 永远不要加载整个数据集到 LLM 上下文，但必须加载前5行让 LLM 理解数据含义和业务语义。

## 2 快速参考

### 2.1 运行命令

```bash

# 优先级1：上传文件后（文件在工作目录）

python3 scripts/cleaning_data.py -c configs/custom.yaml -i 终端默认目录/数据文件名.csv

# 优先级2：指定路径

python3 scripts/cleaning_data.py -c configs/custom.yaml -i /path/to/data.csv

# 优先级3：使用 data/ 目录下的文件

python3 scripts/cleaning_data.py -c configs/custom.yaml

```

### 2.2 数据获取方式

| 优先级 | 方式 | 说明 |
|-------|------|------|
| 1 | 用户上传文件 | 从本地上传文件到 Claude 终端，上传后保存在终端默认的工作目录 |
| 2 | 用户指定路径 | 通过 `-i` 参数指定数据路径 |
| 3 | 缺省目录 | 自动扫描 `data/` 目录下的数据文件 |

### 2.3 支持的文件格式

- `.csv` - CSV 文件（编码 utf-8-sig）

- `.xlsx` / `.xls` - Excel 文件

## 3 核心流程

```dot

digraph workflow {

    "获取数据文件路径" [shape=box];

    "加载前5行样本" [shape=box];

    "转换为LLM可读格式" [shape=box];

    "LLM智能分析并生成custom.yaml" [shape=box];

    "调用cleaning_data.py" [shape=box];

    "LLM检查脚本生成的EDA报告并修复异常" [shape=box];

    "获取数据文件路径" -> "加载前5行样本";

    "加载前5行样本" -> "转换为LLM可读格式";

    "转换为LLM可读格式" -> "LLM智能分析并生成custom.yaml";

    "LLM智能分析并生成custom.yaml" -> "调用cleaning_data.py";

    "调用cleaning_data.py" -> "LLM检查EDA报告并修复异常";

}

```

## 4 LLM 智能分析维度

让 LLM 基于以下维度智能判断：

| 判断维度 | LLM 分析内容 | 生成配置字段 |
|---------|-------------|-------------|
| 异常值范围 | 列名语义 + 业务常识 | `outliers.ranges` |
| 数值分箱 | 数值列的业务含义 | `transformations.binning` |
| 计算列 | 多列的业务关联性 | `transformations.computed_columns` |

**⚠️ LLM 必须严格限制在上述3个维度内分析，禁止主动生成其他配置！**

上述3个维度以外的配置由 `cleaning_data.py` 脚本自动检测和补充

## 5 第一阶段：数据准备

### 5.1 加载并转换样本数据（LLM可读格式）

```python
import pandas as pd
from pathlib import Path

# 根据文件扩展名自动选择读取方式
path = 'data/raw_data.csv'
ext = Path(path).suffix.lower()

if ext == '.csv':
    df = pd.read_csv(path, encoding='utf-8-sig', nrows=5)
else:
    df = pd.read_excel(path, nrows=5)

print('=== 数据样本（仅前5行）===\n')
print('## 列信息 ##')
for col in df.columns:
    dtype = df[col].dtype
    non_null = df[col].dropna()
    sample = non_null.head(5).astype(str).tolist()
    print(f'- {col} ({dtype}), 样本: {sample}')
print('\n## 前5行完整数据 ##')
print(df.head().to_string())
```

## 6 第二阶段：LLM 生成配置

### 6.1 LLM 分析提示词模板

**这是核心步骤！** 将数据样本喂给 LLM，让它基于业务语义判断核心业务规则。

必须主动读取**提示词模板：** [reference/prompt_generate_config.md](reference/prompt_generate_config.md)

### 6.2 LLM 生成 custom.yaml 示例

假设数据有以下列：

- `age` (25, 30, 45, None, 22)

- `gender` ("Male", "Female", "Female", "Male", "Other")

- `birth_date` ("1990-05-15", "1985-08-22", "1992-03-10")

- `income` (5000, 8500.5, 12000, None, 6500)

- `price` + `quantity` → 可计算 `total`

**LLM 只需配置核心业务规则：**

```yaml

# custom.yaml（精简版 - 只需配置核心业务规则）

outliers:

  method: "range"

  ranges:

    age: [0, 120]      # 年龄范围

    income: [0, null]  # 收入非负

transformations:

  binning:

    age:

      bins: [0, 18, 35, 50, 65, 120]

      labels: ["未成年", "青年", "中年", "壮年", "老年"]

  computed_columns:

    total: "price * quantity"  # 计算列

# 其他字段自动补充：

# - date_columns: 自动检测 birth_date

# - trim_columns: 自动填充所有分类列

# - categorical_encoding: 自动用 LabelEncoder 生成

```

### 6.3 保存配置

```python

import yaml

from pathlib import Path

def save_custom_config(yaml_content: str, output_path: str = "configs/custom.yaml"):

    """保存 LLM 生成的 custom.yaml"""

    config = yaml.safe_load(yaml_content)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:

        yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

```

## 7 第三阶段：执行清洗

运行清洗脚本：

```bash

python3 scripts/cleaning_data.py -c configs/custom.yaml

```

## 8 第四阶段：EDA 检查与修复

**目的：** 脚本运行结束后，LLM 自动读取脚本生成 `data/EDA-data_*.json` 报告，检查脚本可能遗漏的异常。

### 8.1 查找最新 EDA 报告

```python

import glob

import json

from pathlib import Path

def load_latest_eda_report(data_dir: str = "data") -> dict:

    """加载最新的 EDA 报告"""

    eda_files = glob.glob(f"{data_dir}/EDA-data_*.json")

    if not eda_files:

        raise FileNotFoundError("未找到 EDA 报告")

    latest_file = max(eda_files, key=lambda f: Path(f).stat().st_mtime)

    with open(latest_file, 'r', encoding='utf-8') as f:

        return json.load(f)

```

### 8.2 EDA 分析提示词

必须主动读取**提示词模板：** [reference/prompt_eda_analysis.md](reference/prompt_eda_analysis.md)

## EDA 检查结果

**状态：** 正常 / 存在问题

### 发现的问题

| 问题类型 | 详情 | 原因分析 | 修复建议 |

|---------|------|---------|---------

| std为NaN | income列 | 数据类型可能是字符串 | 转换为数值类型 |

| 列数过少 | 只有1列 | 列名解析错误 | 检查CSV分隔符 |

### 修复后的 custom.yaml 配置

```yaml

format_standardization:

  numeric_columns: ["income"]  # 强制转换

  drop_columns: []             # 删除错误列

```

## 9 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| LLM生成配置不完整 | 未提供足够样本 | 确保加载前5行，让LLM看到实际数据 |
| 日期列识别错误 | 正则规则太简单 | 让LLM根据列名和样本值综合判断 |
| 范围设置不合理 | 未理解列语义 | 让LLM根据列名的业务含义设置 |
| 分类编码混乱 | 枚举值顺序随意 | 让LLM保持语义一致的映射顺序 |