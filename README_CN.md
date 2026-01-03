# 数据清洗工具 (Data Cleaning Tool)

一个用于清洗原始数据集的Claude Code专业技能工具。

## 功能特性

- **缺失值处理** - 支持删除、均值/中位数/众数填充、固定值填充
- **重复数据处理** - 检测并去除重复记录
- **异常值检测** - 支持IQR、Z-score、MAD、范围检测
- **格式标准化** - 日期格式、文本修剪、大小写转换
- **数据转换** - 类别编码、分箱、计算新列

## 快速开始

```bash
# 安装依赖
pip install pandas numpy scipy scikit-learn pyyaml openpyxl

# 使用默认配置清洗数据
python scripts/cleaning_data.py -i data/raw_data.csv

# 使用自定义配置
python scripts/cleaning_data.py -c configs/custom.yaml -i data/raw_data.csv
```

## 项目结构

```
├── scripts/           # 清洗脚本
├── configs/           # 配置文件
├── data/             # 数据文件
├── reference/        # 提示模板
├── CLAUDE.md         # Claude Code指导
└── SKILL.md          # 技能说明
```

## 配置说明

在 `configs/custom.yaml` 中配置清洗规则：

```yaml
missing_values:
  strategy: mean
  columns: [数值列名]

outliers:
  method: iqr
  ranges:
    列名: [最小值, 最大值]
```

## 使用示例

1. 准备原始数据文件（CSV格式，UTF-8-SIG编码）
2. 根据需要编辑 `configs/custom.yaml`
3. 运行清洗脚本：
   ```bash
   python scripts/cleaning_data.py -c configs/custom.yaml -i your_data.csv
   ```
4. 清洗后的数据保存在 `data/cleaned_data.csv`
