# Data Cleaning Tool

**Turn messy data into analysis-ready datasets in seconds.**

A powerful Claude Code skill that automates data cleaning—handling missing values, duplicates, outliers, and format inconsistencies with minimal configuration.

![GitHub stars](https://img.shields.io/github/stars/guanyugangstar/cleaning-data)
![Version](https://img.shields.io/badge/version-1.0.0-blue)

## Why This Tool?

Data scientists spend **60-80%** of their time on data preparation. This tool cuts that down to minutes.

| Before | After |
|--------|-------|
| Manual Excel cleanup | One-command automation |
| Inconsistent formats | Standardized output |
| Hours of repetitive work | Reusable configurations |

## Features

- **Missing Values** — Drop, fill (mean/median/mode), or custom values
- **Duplicates** — Smart detection and removal
- **Outliers** — IQR, Z-score, MAD, or custom ranges
- **Format Standardization** — Dates, text trimming, case conversion
- **Data Transformation** — Encoding, binning, computed columns

## Quick Start

```bash
pip install pandas numpy scipy scikit-learn pyyaml openpyxl

# Clean your data with one command
python scripts/cleaning_data.py -i raw_data.csv

# Or with custom rules
python scripts/cleaning_data.py -c configs/custom.yaml -i raw_data.csv
```

## Use Cases

- **Data Analysis** — Prepare raw data for pandas, Excel, or BI tools
- **Machine Learning** — Clean training datasets before modeling
- **Reporting** — Standardize data from multiple sources
- **ETL Pipelines** — Automate data quality checks

## How It Works

```
raw_data.csv → [Cleaning Pipeline] → cleaned_data.csv
                   ↓
            EDA Report + Lineage
```

## Project Structure

```
├── scripts/           # Core cleaning engine
├── configs/           # Reusable configurations
├── data/              # Input/output files
├── reference/         # Prompt templates
└── README_CN.md       # 中文文档
```

## Configuration Example

```yaml
missing_values:
  strategy: mean
  columns: [price, quantity]

outliers:
  method: iqr
  ranges:
    age: [0, 120]
    score: [0, 100]
```

## Contributing

Found a bug or have a feature request? Open an issue or PR!

---

**Built with ❤️ for data practitioners**
