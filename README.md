# Data Cleaning Tool

A Claude Code skill for cleaning raw datasets with missing values, duplicates, outliers, inconsistent formats, and other data quality issues.

## Features

- **Missing Values** - Drop, fill with mean/median/mode, or fixed values
- **Duplicates** - Detect and remove duplicate records
- **Outliers** - IQR, Z-score, MAD, or range-based detection
- **Format Standardization** - Date formats, text trimming, case conversion
- **Data Transformation** - Categorical encoding, binning, computed columns

## Quick Start

```bash
# Install dependencies
pip install pandas numpy scipy scikit-learn pyyaml openpyxl

# Run with default config
python scripts/cleaning_data.py -i data/raw_data.csv

# Run with custom config
python scripts/cleaning_data.py -c configs/custom.yaml -i data/raw_data.csv
```

## Project Structure

```
├── scripts/           # Cleaning scripts
├── configs/           # Configuration files
├── data/             # Data files
├── reference/        # Prompt templates
├── CLAUDE.md         # Claude Code guidance
└── SKILL.md          # Skill documentation
```

## Configuration

Edit `configs/custom.yaml` to configure cleaning rules:

```yaml
missing_values:
  strategy: mean
  columns: [column_name]

outliers:
  method: iqr
  ranges:
    column_name: [min, max]
```

## Usage

1. Prepare your raw data file (CSV, UTF-8-SIG encoding)
2. Edit `configs/custom.yaml` as needed
3. Run the cleaning script:
   ```bash
   python scripts/cleaning_data.py -c configs/custom.yaml -i your_data.csv
   ```
4. Cleaned data saved to `data/cleaned_data.csv`
