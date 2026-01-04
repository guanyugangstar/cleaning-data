#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通用数据清洗脚本
============

功能：
- 配置文件驱动的数据清洗
- 支持缺失值、重复值、异常值处理
- 支持格式标准化和数据转换
- 自动生成清洗报告

使用方法：
    python scripts/cleaning_data.py                          # 使用默认配置
    python scripts/cleaning_data.py -c configs/custom.yaml   # 使用自定义配置
    python scripts/cleaning_data.py -i data/raw.csv -o data/clean.csv
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# 确保当前目录在路径中
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import re
import yaml
import pandas as pd
import numpy as np
from scipy import stats

# 导入字段值纠正模块
from scripts.field_correction import correct_field_values, fuzzy_match

# 控制台编码
sys.stdout.reconfigure(encoding='utf-8')


# ============================================================
# 数据血缘追踪器
# ============================================================

class DataLineage:
    """数据血缘追踪器 - 记录每列的清洗历史"""

    def __init__(self):
        self.column_history = {}  # {列名: [操作记录]}

    def record(self, col: str, operation: str, details: dict):
        """
        记录列的操作历史

        Args:
            col: 列名
            operation: 操作类型（如 fill_missing, drop_duplicates, handle_outliers 等）
            details: 操作详情字典
        """
        if col not in self.column_history:
            self.column_history[col] = []

        self.column_history[col].append({
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'details': details,
        })

    def get_report(self) -> dict:
        """获取血缘报告"""
        return self.column_history

    def save_report(self, path: str):
        """保存血缘报告为 YAML 格式"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.get_report(), f, allow_unicode=True, sort_keys=False)

    def get_column_summary(self, col: str) -> dict:
        """获取某列的操作摘要"""
        if col not in self.column_history:
            return {'operation_count': 0, 'operations': []}

        operations = self.column_history[col]
        return {
            'operation_count': len(operations),
            'operations': [op['operation'] for op in operations],
            'first_operation': operations[0] if operations else None,
            'last_operation': operations[-1] if operations else None,
        }


# ============================================================
# 文件加载器 - 支持多种文件格式
# ============================================================

class FileLoader:
    """支持多种文件格式的加载和保存"""

    SUPPORTED_FORMATS = {
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.json': 'json',
        '.parquet': 'parquet',
        '.pkl': 'pickle',
    }

    @staticmethod
    def load(path: str, encoding: str = 'utf-8-sig', **kwargs) -> pd.DataFrame:
        """
        加载数据文件

        Args:
            path: 文件路径
            encoding: 编码（仅CSV有效）
            **kwargs: 其他 pandas read 参数

        Returns:
            pd.DataFrame: 加载的数据
        """
        ext = Path(path).suffix.lower()
        loader = FileLoader.SUPPORTED_FORMATS.get(ext)

        if loader is None:
            raise ValueError(f"不支持的文件格式: {ext}，支持格式: {list(FileLoader.SUPPORTED_FORMATS.keys())}")

        if loader == 'csv':
            return pd.read_csv(path, encoding=encoding, **kwargs)
        elif loader == 'excel':
            return pd.read_excel(path, **kwargs)
        elif loader == 'json':
            return pd.read_json(path, **kwargs)
        elif loader == 'parquet':
            return pd.read_parquet(path, **kwargs)
        elif loader == 'pickle':
            return pd.read_pickle(path, **kwargs)

    @staticmethod
    def save(df: pd.DataFrame, path: str, encoding: str = 'utf-8-sig', **kwargs):
        """
        保存数据文件

        Args:
            df: DataFrame
            path: 输出路径
            encoding: 编码（仅CSV有效）
            **kwargs: 其他 pandas to_* 参数
        """
        ext = Path(path).suffix.lower()
        saver = FileLoader.SUPPORTED_FORMATS.get(ext)

        if saver is None:
            raise ValueError(f"不支持的文件格式: {ext}，支持格式: {list(FileLoader.SUPPORTED_FORMATS.keys())}")

        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if saver == 'csv':
            df.to_csv(path, index=False, encoding=encoding, **kwargs)
        elif saver == 'excel':
            df.to_excel(path, index=False, **kwargs)
        elif saver == 'json':
            df.to_json(path, orient='records', force_ascii=False, **kwargs)
        elif saver == 'parquet':
            df.to_parquet(path, **kwargs)
        elif saver == 'pickle':
            df.to_pickle(path, **kwargs)


# ============================================================
# 动态配置生成器 - 供 Claude Agent 使用
# ============================================================

class ConfigGenerator:
    """根据数据特征自动生成清洗配置"""

    @staticmethod
    def analyze_data(df: pd.DataFrame) -> dict:
        """
        分析数据，返回数据特征摘要

        Args:
            df: pandas DataFrame

        Returns:
            dict: 数据特征摘要
        """
        info = {
            'shape': df.shape,
            'columns': [],
            'numeric_columns': [],
            'categorical_columns': [],
            'date_columns': [],
            'missing_summary': {},
            'duplicate_count': df.duplicated().sum(),
        }

        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'missing_count': int(df[col].isnull().sum()),
                'missing_pct': round(df[col].isnull().sum() / len(df) * 100, 2),
                'unique_count': int(df[col].nunique()),
                'unique_values': list(df[col].dropna().unique()) if df[col].dtype == 'object' else [],
            }

            # 分类列类型 - 使用 pandas 数值类型判断
            if pd.api.types.is_numeric_dtype(df[col]):
                info['numeric_columns'].append(col)
                col_info['min'] = float(df[col].min()) if len(df) > 0 else None
                col_info['max'] = float(df[col].max()) if len(df) > 0 else None
            elif df[col].dtype == 'object':
                # 通过列名识别日期列
                col_lower = col.lower()
                date_keywords = ['date', 'time', '日期', '时间', '日']
                if any(kw in col_lower for kw in date_keywords):
                    info['date_columns'].append(col)
                # 内容模式检测：检查前N个非空值是否符合日期格式
                elif ConfigGenerator._is_date_by_content(df[col]):
                    info['date_columns'].append(col)
                else:
                    info['categorical_columns'].append(col)

            info['columns'].append(col_info)
            info['missing_summary'][col] = col_info['missing_count']

        return info

    @staticmethod
    def _is_date_by_content(series: pd.Series, sample_size: int = 100) -> bool:
        """
        通过列内容检测是否为日期格式

        检测的日期格式:
        - YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
        - DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
        - YYYYMMDD
        - YYYY年MM月DD日
        - MM/DD, MM-DD (无年份)
        - 中文: 2024年1月1日, 2024/1/1

        Args:
            series: pandas Series
            sample_size: 采样数量

        Returns:
            bool: 是否为日期格式
        """
        # 获取非空值样本
        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            return False

        # 采样
        sample = non_null.head(sample_size)

        # 日期正则模式
        date_patterns = [
            # ISO 格式: 2024-01-01, 2024/01/01, 2024.01.01
            r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$',
            # 中文格式: 2024年1月1日, 2024年01月01日
            r'^\d{4}年\d{1,2}月\d{1,2}日$',
            # 无分隔符: 20240101
            r'^\d{8}$',
            # 美式: 01/01/2024, 1-1-2024
            r'^\d{1,2}[-/]\d{1,2}[-/]\d{4}$',
            # 只有月日: 01/01, 1-1
            r'^\d{1,2}[-/]\d{1,2}$',
            # 中文月日: 1月1日
            r'^\d{1,2}月\d{1,2}日$',
        ]

        match_count = 0
        total = len(sample)

        for value in sample:
            value = value.strip()
            for pattern in date_patterns:
                if re.match(pattern, value):
                    match_count += 1
                    break

        # 超过 80% 的值匹配则认为是日期列
        return match_count >= int(total * 0.8) and total > 0

    @staticmethod
    def load_config(config_path: str) -> dict:
        """
        从YAML文件加载用户配置

        Args:
            config_path: 配置文件路径

        Returns:
            dict: 配置字典
        """
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}

    @staticmethod
    def generate_config(
        input_path: str,
        output_path: str = None,
        config_path: str = "configs/cleaning_config.yaml",
        auto_detect_dates: bool = True,
        auto_detect_numeric: bool = True,
        trim_text: bool = True,
    ) -> dict:
        """
        根据数据特征生成清洗配置（先加载用户配置，再自动填充空白字段）

        Args:
            input_path: 输入数据路径
            output_path: 输出数据路径
            config_path: 配置文件路径
            auto_detect_dates: 是否自动检测日期列
            trim_text: 是否自动清理文本空格

        Returns:
            dict: 清洗配置字典
        """
        user_config = ConfigGenerator.load_config(config_path)
        df = pd.read_csv(input_path, encoding='utf-8-sig')
        info = ConfigGenerator.analyze_data(df)

        # 提取配置块
        data = user_config.get('data', {})
        missing = user_config.get('missing_values', {})
        dup = user_config.get('duplicates', {})
        outliers = user_config.get('outliers', {})
        fmt = user_config.get('format_standardization', {})
        trans = user_config.get('transformations', {})

        config = {
            'data': {
                'input_path': input_path,
                'output_path': output_path or data.get('output_path', "data/cleaned_data.csv"),
                'encoding': data.get('encoding', 'utf-8-sig'),
            },
            'missing_values': {
                'strategy': missing.get('strategy', 'median'),
                'fixed_value': missing.get('fixed_value', 0),
                'columns': missing.get('columns', {}),
            },
            'duplicates': {
                'action': dup.get('action', 'drop'),
                'subset': dup.get('subset', []),
            },
            'outliers': {
                'method': outliers.get('method', 'iqr'),
                'treatment': outliers.get('treatment', 'nan'),
                'ranges': outliers.get('ranges', {}),
            },
            'format_standardization': {
                'date_columns': info['date_columns'] if auto_detect_dates else fmt.get('date_columns', []),
                'trim_columns': info['categorical_columns'] if trim_text else fmt.get('trim_columns', []),
                'lowercase_columns': fmt.get('lowercase_columns', []),
                'title_columns': fmt.get('title_columns', []),
                'numeric_columns': info['numeric_columns'] if auto_detect_numeric else fmt.get('numeric_columns', []),
                'drop_columns': fmt.get('drop_columns', []),
                'value_correction': fmt.get('value_correction', {}),
            },
            'transformations': {
                'categorical_encoding': trans.get('categorical_encoding', {}),
                'binning': trans.get('binning', {}),
                'computed_columns': trans.get('computed_columns', {}),
            },
            'business_rules': {
                'enabled': user_config.get('business_rules', {}).get('enabled', False),
                'rules': [],
            },
            'report': {'verbose': True, 'show_stats': True},
        }

        # 自动识别低基数分类变量（仅当用户未定义时）
        user_enc = trans.get('categorical_encoding', {})
        for col in info['columns']:
            if col['dtype'] == 'object' and col['unique_count'] <= 10 and col['name'] not in user_enc:
                # 直接生成映射字典（按字母顺序编码）
                unique_vals = sorted(col.get('unique_values', []))
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                config['transformations']['categorical_encoding'][col['name']] = mapping

        # 高缺失率列自动标记删除（与用户指定的drop_columns合并处理）
        for col in info['columns']:
            if col['missing_pct'] > 50:
                config['missing_values']['columns'][col['name']] = 'drop'
                if col['name'] not in config['format_standardization']['drop_columns']:
                    config['format_standardization']['drop_columns'].append(col['name'])

        return config

    @staticmethod
    def save_config(config: dict, output_path: str):
        """保存配置到YAML文件"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)
        return output_path


class DataCleaner:
    """通用数据清洗器"""

    def __init__(self, config: dict = None, **kwargs):
        """
        初始化

        Args:
            config: 配置字典
            **kwargs: 命令行参数覆盖配置
        """
        self.config = config or {}
        self.df = None
        self.original_shape = None
        self.report = []
        self.stats = {}
        self.lineage = DataLineage()

        # 合并命令行参数
        self._merge_kwargs(kwargs)

    def _merge_kwargs(self, kwargs):
        """合并命令行参数"""
        if not kwargs:
            return

        # 数据路径覆盖
        if kwargs.get('input'):
            self.config.setdefault('data', {})['input_path'] = kwargs['input']
        if kwargs.get('output'):
            self.config.setdefault('data', {})['output_path'] = kwargs['output']
        if kwargs.get('config'):
            # 如果指定了配置文件，重新加载
            self.load_config(kwargs['config'])

    def load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            self.report.append(f"加载配置: {config_path}")
        return self.config

    def _init_log_file(self):
        """初始化日志文件"""
        report_config = self.config.get('report', {})
        self.log_to_file = report_config.get('log_to_file', False)
        self.log_file_path = report_config.get('log_file_path', 'logs/cleaning_log.txt')

        if self.log_to_file:
            log_abs_path = self._get_data_path(self.log_file_path)
            os.makedirs(os.path.dirname(log_abs_path), exist_ok=True)
            # 清空旧日志或创建新文件
            with open(log_abs_path, 'w', encoding='utf-8') as f:
                f.write(f"===== 数据清洗日志 =====\n")
                f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")

    def _write_log_file(self, message: str):
        """写入日志文件"""
        if not getattr(self, 'log_to_file', False):
            return
        log_abs_path = self._get_data_path(self.log_file_path)
        with open(log_abs_path, 'a', encoding='utf-8') as f:
            f.write(message + "\n")

    def _log(self, message: str, level: str = "INFO"):
        """日志记录（支持控制台和文件输出）"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)
        self._write_log_file(log_msg)
        self.report.append(log_msg)

    def _get_data_path(self, path: str) -> str:
        """获取数据文件路径（相对于技能目录）"""
        if os.path.isabs(path):
            return path
        return str(SCRIPT_DIR / path)

    def _regenerate_eda(self):
        """重新生成EDA报告（反映字段纠正后的状态）"""
        self._log(f"[更新] 重新生成清洗后的EDA报告")
        self.inspect_data()

    # ============================================================
    # 1. 加载数据
    # ============================================================

    def load_data(self, path: str = None) -> 'DataCleaner':
        """加载数据"""
        data_config = self.config.get('data', {})
        input_path = self._get_data_path(path or data_config.get('input_path', 'data/raw_data.csv'))
        encoding = data_config.get('encoding', 'utf-8-sig')

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"数据文件不存在: {input_path}")

        # 使用 FileLoader 加载数据（支持多种格式）
        self.df = FileLoader.load(input_path, encoding=encoding)
        self.original_shape = self.df.shape

        self._log(f"加载数据: {input_path}")
        self._log(f"  数据维度: {self.df.shape[0]} 行 × {self.df.shape[1]} 列")

        return self

    # ============================================================
    # 2. 数据概览
    # ============================================================

    def inspect_data(self) -> 'DataCleaner':
        """数据概览报告，生成EDA分析JSON文档"""
        import json
        from datetime import datetime

        verbose = self.config.get('report', {}).get('verbose', True)

        self._log("\n" + "=" * 60)
        self._log("数据概览")
        self._log("=" * 60)

        # 基本统计
        duplicates = self.df.duplicated().sum()
        memory = self.df.memory_usage(deep=True).sum()
        missing = self.df.isnull().sum()

        # 构建EDA报告字典
        eda_report = {
            'generated_at': datetime.now().isoformat(),
            'basic_info': {
                'rows': self.df.shape[0],
                'columns': self.df.shape[1],
                'duplicates': int(duplicates),
                'duplicates_pct': round(duplicates / len(self.df) * 100, 2),
                'memory_mb': round(memory / 1024**2, 2)
            },
            'data_types': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'missing_values': {col: int(count) for col, count in missing.items() if count > 0},
            'numeric_statistics': {},
            'categorical_distributions': {}
        }

        self._log(f"数据维度: {self.df.shape[0]} 行 × {self.df.shape[1]} 列")
        self._log(f"重复行: {duplicates} ({duplicates/len(self.df)*100:.2f}%)")
        self._log(f"内存使用: {memory / 1024**2:.2f} MB")

        # 数据类型
        self._log("\n数据类型:")
        for col, dtype in self.df.dtypes.items():
            self._log(f"  {col}: {dtype}")

        # 缺失值统计
        self._log("\n缺失值统计:")
        for col in self.df.columns:
            if missing[col] > 0:
                pct = missing[col] / len(self.df) * 100
                self._log(f"  {col}: {missing[col]} ({pct:.2f}%)")

        # 详细模式：显示分类列的频率分布
        if verbose:
            self._log("\n分类列频率分布:")
            cat_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns
            for col in cat_cols:
                unique_count = self.df[col].nunique()
                # 优化：跳过基数过高的分类变量，避免报告过于庞大
                if unique_count > 100:
                    self._log(f"  {col}: {unique_count} 个唯一值 (跳过详细分布)")
                    eda_report['categorical_distributions'][col] = {
                        'unique_count': unique_count,
                        'note': '唯一值超过100个，跳过详细分布'
                    }
                    continue

                value_counts = self.df[col].value_counts()
                dist = {str(val): int(count) for val, count in value_counts.items()}
                eda_report['categorical_distributions'][col] = dist

                if len(value_counts) <= 10:
                    self._log(f"  {col}:")
                    for val, count in value_counts.items():
                        pct = count / len(self.df) * 100
                        self._log(f"    {val}: {count} ({pct:.1f}%)")
                else:
                    self._log(f"  {col}: {len(value_counts)} 个类别，前10:")
                    for val, count in value_counts.head(10).items():
                        pct = count / len(self.df) * 100
                        self._log(f"    {val}: {count} ({pct:.1f}%)")

        # 统计数值列摘要
        if self.config.get('report', {}).get('show_stats', True):
            self._log("\n数值列统计:")
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                desc = self.df[numeric_cols].describe().round(2)
                eda_report['numeric_statistics']['describe'] = desc.to_dict()
                self._log(desc.to_string())

                # 百分位数明细
                percentiles = [0.01, 0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
                self._log("\n百分位数明细:")
                percentile_data = self.df[numeric_cols].quantile(percentiles).round(2)
                eda_report['numeric_statistics']['percentiles'] = percentile_data.to_dict()
                self._log(percentile_data.to_string())

                # 偏度和峰度
                self._log("\n偏度 (Skewness):")
                skewness = self.df[numeric_cols].skew().round(2)
                eda_report['numeric_statistics']['skewness'] = skewness.to_dict()
                self._log(skewness.to_string())

                self._log("\n峰度 (Kurtosis):")
                kurtosis = self.df[numeric_cols].kurt().round(2)
                eda_report['numeric_statistics']['kurtosis'] = kurtosis.to_dict()
                self._log(kurtosis.to_string())

        # 数值-分类交叉统计
        cat_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(cat_cols) > 0 and len(numeric_cols) > 0 and verbose:
            self._log("\n数值-分类交叉统计:")
            cross_stats = {}
            for cat_col in cat_cols:
                cat_unique = self.df[cat_col].nunique()
                if cat_unique > 10:
                    continue  # 跳过基数过高的分类变量
                cross_stats[cat_col] = {}
                for num_col in numeric_cols:
                    grouped = self.df.groupby(cat_col)[num_col].agg(['mean', 'std', 'count']).round(2)
                    cross_stats[cat_col][num_col] = grouped.to_dict()
            if cross_stats:
                eda_report['cross_statistics'] = cross_stats
                for cat_col, stats in cross_stats.items():
                    self._log(f"  {cat_col}:")
                    for num_col, agg in stats.items():
                        self._log(f"    {num_col}: mean={agg.get('mean')}, std={agg.get('std')}")

        # 保存EDA报告到JSON文件（带时间戳）
        output_dir = self.config.get('data', {}).get('output_dir', 'data')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'EDA-data_{timestamp}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(eda_report, f, ensure_ascii=False, indent=2, default=str)
        self._log(f"\n[已保存] EDA报告: {output_path}")

        return self

    # ============================================================
    # 3. 缺失值处理
    # ============================================================

    def handle_missing_values(self) -> 'DataCleaner':
        """处理缺失值"""
        missing_config = self.config.get('missing_values', {})
        if not missing_config:
            return self

        strategy = missing_config.get('strategy', 'median')
        fixed_value = missing_config.get('fixed_value', 0)
        column_config = missing_config.get('columns', {})

        total_missing_before = self.df.isnull().sum().sum()
        self._log(f"\n" + "=" * 60)
        self._log(f"处理缺失值 (策略: {strategy})")
        self._log("=" * 60)

        # 按列处理
        for col in self.df.columns:
            if self.df[col].isnull().sum() == 0:
                continue

            # 获取该列的策略
            col_strategy = column_config.get(col, strategy) if col in column_config else strategy
            col_missing = self.df[col].isnull().sum()

            if col_strategy == 'drop':
                self.df = self.df.dropna(subset=[col])
                self._log(f"  {col}: 删除 {col_missing} 行")
                self.lineage.record(col, 'drop_missing', {
                    'rows_removed': col_missing,
                    'strategy': col_strategy,
                })

            elif col_strategy == 'mean':
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    fill_value = self.df[col].mean()
                    self.df[col] = self.df[col].fillna(fill_value)
                    self._log(f"  {col}: 用均值 {fill_value:.2f} 填充")
                    self.lineage.record(col, 'fill_missing', {
                        'filled_count': col_missing,
                        'strategy': col_strategy,
                        'fill_value': fill_value,
                    })
                else:
                    # 自动降级到 mode
                    fill_value = self.df[col].mode()
                    if len(fill_value) > 0:
                        self.df[col] = self.df[col].fillna(fill_value[0])
                        self._log(f"  {col}: [自动降级] 用众数 {fill_value[0]} 填充")
                        self.lineage.record(col, 'fill_missing', {
                            'filled_count': col_missing,
                            'strategy': 'mode (auto-downgrade)',
                            'fill_value': fill_value[0],
                        })

            elif col_strategy == 'median':
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    fill_value = self.df[col].median()
                    self.df[col] = self.df[col].fillna(fill_value)
                    self._log(f"  {col}: 用中位数 {fill_value:.2f} 填充")
                    self.lineage.record(col, 'fill_missing', {
                        'filled_count': col_missing,
                        'strategy': col_strategy,
                        'fill_value': fill_value,
                    })
                else:
                    # 自动降级到 mode
                    fill_value = self.df[col].mode()
                    if len(fill_value) > 0:
                        self.df[col] = self.df[col].fillna(fill_value[0])
                        self._log(f"  {col}: [自动降级] 用众数 {fill_value[0]} 填充")
                        self.lineage.record(col, 'fill_missing', {
                            'filled_count': col_missing,
                            'strategy': 'mode (auto-downgrade)',
                            'fill_value': fill_value[0],
                        })

            elif col_strategy == 'mode':
                fill_value = self.df[col].mode()
                if len(fill_value) > 0:
                    self.df[col] = self.df[col].fillna(fill_value[0])
                    self._log(f"  {col}: 用众数 {fill_value[0]} 填充")
                    self.lineage.record(col, 'fill_missing', {
                        'filled_count': col_missing,
                        'strategy': col_strategy,
                        'fill_value': fill_value[0],
                    })

            elif col_strategy == 'fixed':
                self.df[col] = self.df[col].fillna(fixed_value)
                self._log(f"  {col}: 用固定值 {fixed_value} 填充")
                self.lineage.record(col, 'fill_missing', {
                    'filled_count': col_missing,
                    'strategy': col_strategy,
                    'fill_value': fixed_value,
                })

        total_missing_after = self.df.isnull().sum().sum()
        self._log(f"缺失值处理完成: {total_missing_before} → {total_missing_after}")

        return self

    # ============================================================
    # 4. 重复值处理
    # ============================================================

    def handle_duplicates(self) -> 'DataCleaner':
        """处理重复值"""
        dup_config = self.config.get('duplicates', {})
        if not dup_config:
            return self

        action = dup_config.get('action', 'drop')
        subset = dup_config.get('subset', [])

        before_count = len(self.df)

        if subset:
            # 指定列去重
            if action == 'drop':
                self.df = self.df.drop_duplicates(subset=subset, keep='first')
            else:
                self.df = self.df.drop_duplicates(subset=subset, keep=action.replace('keep_', ''))
        else:
            # 完全重复行
            self.df = self.df.drop_duplicates()

        after_count = len(self.df)
        removed = before_count - after_count

        self._log(f"\n" + "=" * 60)
        self._log(f"处理重复值")
        self._log("=" * 60)
        self._log(f"删除重复行: {removed} 条")

        # 记录血缘
        self.lineage.record('__global__', 'remove_duplicates', {
            'rows_before': before_count,
            'rows_after': after_count,
            'rows_removed': removed,
            'subset': subset if subset else 'all_columns',
            'action': action,
        })

        return self

    # ============================================================
    # 5. 异常值处理
    # ============================================================

    def handle_outliers(self) -> 'DataCleaner':
        """处理异常值"""
        outliers_config = self.config.get('outliers', {})
        if not outliers_config:
            return self

        method = outliers_config.get('method', 'iqr')
        treatment = outliers_config.get('treatment', 'nan')
        ranges = outliers_config.get('ranges', {})

        # 异常值分析报告
        outlier_report = {
            'method': method,
            'treatment': treatment,
            'columns': {}
        }

        self._log(f"\n" + "=" * 60)
        self._log(f"处理异常值 (方法: {method}, 处理方式: {treatment})")
        self._log("=" * 60)

        # 记录处理前数据量
        before_rows = len(self.df)
        total_masked = 0
        total_removed = 0

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col not in ranges and method == 'range':
                continue

            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                multiplier = outliers_config.get('iqr_multiplier', 1.5)
                lower = Q1 - multiplier * IQR
                upper = Q3 + multiplier * IQR

                # 识别异常值
                mask = (self.df[col] < lower) | (self.df[col] > upper)
                outlier_count = mask.sum()
                total_masked += outlier_count

                # 收集异常值分析详情
                outlier_report['columns'][col] = {
                    'method': 'iqr',
                    'q1': round(float(Q1), 2),
                    'q3': round(float(Q3), 2),
                    'iqr': round(float(IQR), 2),
                    'lower': round(float(lower), 2),
                    'upper': round(float(upper), 2),
                    'multiplier': multiplier,
                    'outlier_count': int(outlier_count)
                }

                if treatment == 'nan':
                    self.df.loc[mask, col] = np.nan
                    self._log(f"  {col}: IQR [{lower:.2f}, {upper:.2f}], 标记异常值 {outlier_count} 个")
                elif treatment == 'clip':
                    before_clip = self.df[col].copy()
                    self.df[col] = self.df[col].clip(lower, upper)
                    clipped_count = ((before_clip < lower) | (before_clip > upper)).sum()
                    outlier_report['columns'][col]['clipped_count'] = int(clipped_count)
                    self._log(f"  {col}: IQR [{lower:.2f}, {upper:.2f}], 截断 {clipped_count} 个值")
                elif treatment == 'remove':
                    valid_mask = ~mask
                    removed_rows = (~valid_mask).sum()
                    total_removed += removed_rows
                    outlier_report['columns'][col]['removed_rows'] = int(removed_rows)
                    self.df = self.df[valid_mask]
                    self._log(f"  {col}: IQR [{lower:.2f}, {upper:.2f}], 删除 {removed_rows} 行")

            elif method == 'zscore':
                threshold = outliers_config.get('zscore_threshold', 3)

                # 移除 NaN 后计算 Z-score，再映射回原索引
                col_data = self.df[col].copy()
                valid_mask_data = ~col_data.isna()
                valid_data = col_data[valid_mask_data]

                if len(valid_data) > 0:
                    z_scores_all = np.full(len(col_data), np.nan)
                    z_scores_valid = np.abs(stats.zscore(valid_data))
                    z_scores_all[valid_mask_data] = z_scores_valid
                    mask = z_scores_all > threshold
                    outlier_count = mask.sum()
                    col_mean = float(valid_data.mean())
                    col_std = float(valid_data.std())
                else:
                    mask = np.zeros(len(self.df), dtype=bool)
                    outlier_count = 0
                    col_mean = 0.0
                    col_std = 0.0

                total_masked += outlier_count

                # 收集异常值分析详情
                outlier_report['columns'][col] = {
                    'method': 'zscore',
                    'threshold': threshold,
                    'mean': round(col_mean, 2),
                    'std': round(col_std, 2),
                    'outlier_count': int(outlier_count)
                }

                if treatment == 'nan':
                    self.df.loc[mask, col] = np.nan
                    self._log(f"  {col}: Z-score > {threshold}, 标记异常值 {outlier_count} 个")
                elif treatment == 'clip':
                    before_clip = self.df[col].copy()
                    col_median = valid_data.median() if len(valid_data) > 0 else 0
                    clipped_lower = col_median - threshold * valid_data.std() if len(valid_data) > 0 else lower
                    clipped_upper = col_median + threshold * valid_data.std() if len(valid_data) > 0 else upper
                    self.df[col] = self.df[col].clip(clipped_lower, clipped_upper)
                    clipped_count = ((before_clip < clipped_lower) | (before_clip > clipped_upper)).sum()
                    outlier_report['columns'][col]['clipped_count'] = int(clipped_count)
                    self._log(f"  {col}: Z-score > {threshold}, 截断 {clipped_count} 个值")
                elif treatment == 'remove':
                    valid_mask_rows = ~mask
                    removed_rows = (~valid_mask_rows).sum()
                    total_removed += removed_rows
                    outlier_report['columns'][col]['removed_rows'] = int(removed_rows)
                    self.df = self.df[valid_mask_rows]
                    self._log(f"  {col}: Z-score > {threshold}, 删除 {removed_rows} 行")

            elif method == 'range':
                if col in ranges:
                    min_val, max_val = ranges[col]

                    if min_val is not None and max_val is not None:
                        mask = (self.df[col] < min_val) | (self.df[col] > max_val)
                    elif min_val is not None:
                        mask = self.df[col] < min_val
                    elif max_val is not None:
                        mask = self.df[col] > max_val
                    else:
                        continue

                    outlier_count = mask.sum()
                    total_masked += outlier_count

                    # 收集异常值分析详情
                    range_info = {
                        'min': min_val,
                        'max': max_val
                    }
                    outlier_report['columns'][col] = {
                        'method': 'range',
                        'range': range_info,
                        'outlier_count': int(outlier_count)
                    }

                    if treatment == 'nan':
                        self.df.loc[mask, col] = np.nan
                        range_str = f"[{min_val}, {max_val}]" if min_val is not None and max_val is not None else ("< " + str(min_val) if min_val is not None else "> " + str(max_val))
                        self._log(f"  {col}: 范围 {range_str}, 标记异常值 {outlier_count} 个")
                    elif treatment == 'remove':
                        valid_mask = ~mask
                        removed_rows = (~valid_mask).sum()
                        total_removed += removed_rows
                        outlier_report['columns'][col]['removed_rows'] = int(removed_rows)
                        self.df = self.df[valid_mask]
                        range_str = f"[{min_val}, {max_val}]" if min_val is not None and max_val is not None else ("< " + str(min_val) if min_val is not None else "> " + str(max_val))
                        self._log(f"  {col}: 范围 {range_str}, 删除 {removed_rows} 行")

            elif method == 'mad':
                # MAD (Median Absolute Deviation) 异常值检测
                mad_threshold = outliers_config.get('mad_threshold', 3.5)
                k = 1.4826  # 缩放因子

                col_data = self.df[col].copy()
                valid_mask_data = ~col_data.isna()
                valid_data = col_data[valid_mask_data]

                if len(valid_data) >= 2:
                    col_median = float(valid_data.median())
                    abs_deviation = np.abs(valid_data - col_median)
                    mad = float(abs_deviation.median())

                    if mad == 0:
                        self._log(f"  {col}: MAD=0，数据无变化", "WARN")
                        continue

                    mad_e = k * mad
                    mad_scores = abs_deviation / mad_e
                    mad_scores_all = np.full(len(col_data), np.nan)
                    mad_scores_all[valid_mask_data] = mad_scores.values
                    mask = mad_scores_all > mad_threshold
                    outlier_count = mask.sum()
                else:
                    col_median = 0.0
                    mad = 0.0
                    mad_e = 0.0
                    mask = np.zeros(len(self.df), dtype=bool)
                    outlier_count = 0

                total_masked += outlier_count

                # 收集异常值分析详情
                outlier_report['columns'][col] = {
                    'method': 'mad',
                    'threshold': mad_threshold,
                    'median': round(col_median, 2),
                    'mad': round(mad, 2),
                    'mad_e': round(mad_e, 2),
                    'outlier_count': int(outlier_count)
                }

                if treatment == 'nan':
                    self.df.loc[mask, col] = np.nan
                    self._log(f"  {col}: MAD > {mad_threshold}, 标记异常值 {outlier_count} 个")
                elif treatment == 'clip':
                    before_clip = self.df[col].copy()
                    clip_lower = col_median - mad_threshold * mad_e if mad > 0 else col_median
                    clip_upper = col_median + mad_threshold * mad_e if mad > 0 else col_median
                    self.df[col] = self.df[col].clip(clip_lower, clip_upper)
                    clipped_count = ((before_clip < clip_lower) | (before_clip > clip_upper)).sum()
                    outlier_report['columns'][col]['clipped_count'] = int(clipped_count)
                    self._log(f"  {col}: MAD > {mad_threshold}, 截断 {clipped_count} 个值")
                elif treatment == 'remove':
                    valid_mask_rows = ~mask
                    removed_rows = (~valid_mask_rows).sum()
                    total_removed += removed_rows
                    outlier_report['columns'][col]['removed_rows'] = int(removed_rows)
                    self.df = self.df[valid_mask_rows]
                    self._log(f"  {col}: MAD > {mad_threshold}, 删除 {removed_rows} 行")

        # 输出汇总信息
        after_rows = len(self.df)
        self._log(f"\n异常值处理汇总:")
        if treatment == 'remove':
            self._log(f"  - 删除行数: {total_removed}")
            self._log(f"  - 数据行数: {before_rows} → {after_rows}")
        else:
            self._log(f"  - 标记异常值: {total_masked} 个")
            self._log(f"  - 数据行数: {before_rows} (保持不变)")

        # 记录血缘
        self.lineage.record('__global__', 'handle_outliers', {
            'method': method,
            'treatment': treatment,
            'rows_before': before_rows,
            'rows_after': after_rows,
            'total_masked': total_masked,
            'total_removed': total_removed,
            'numeric_columns_processed': list(numeric_cols),
        })

        # 保存异常值分析报告到EDA-data.json
        import json
        output_dir = self.config.get('data', {}).get('output_dir', 'data')
        json_path = os.path.join(output_dir, 'EDA-data.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                eda_report = json.load(f)
            eda_report['outlier_analysis'] = outlier_report
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(eda_report, f, ensure_ascii=False, indent=2)
            self._log(f"\n[已更新] EDA异常值分析: {json_path}")
        except FileNotFoundError:
            self._log("[警告] 未找到EDA-data.json，跳过异常值报告保存")

        return self

    # ============================================================
    # 6. 格式标准化
    # ============================================================

    def standardize_format(self) -> 'DataCleaner':
        """格式标准化"""
        fmt_config = self.config.get('format_standardization', {})
        if not fmt_config:
            return self

        self._log(f"\n" + "=" * 60)
        self._log("格式标准化")
        self._log("=" * 60)

        # 日期列处理
        date_columns = fmt_config.get('date_columns', [])
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                self._log(f"  {col}: 转换为日期格式")
                self.lineage.record(col, 'convert_to_datetime', {})

        # 文本清理 - 去除空格（跳过 NaN）
        trim_cols = fmt_config.get('trim_columns', [])
        for col in trim_cols:
            if col in self.df.columns:
                mask = self.df[col].notna()
                self.df.loc[mask, col] = self.df.loc[mask, col].astype(str).str.strip()
                self._log(f"  {col}: 去除前后空格")
                self.lineage.record(col, 'trim_whitespace', {})

        # 小写转换
        lowercase_cols = fmt_config.get('lowercase_columns', [])
        for col in lowercase_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.lower()
                self._log(f"  {col}: 转换为小写")
                self.lineage.record(col, 'to_lowercase', {})

        # 首字母大写
        title_cols = fmt_config.get('title_columns', [])
        for col in title_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.title()
                self._log(f"  {col}: 首字母大写")
                self.lineage.record(col, 'to_title', {})

        # 数值列转换
        numeric_cols = fmt_config.get('numeric_columns', [])
        for col in numeric_cols:
            if col in self.df.columns and not pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].astype(str)
                self.df[col] = self.df[col].str.replace(
                    r'[¥$€元,\s％%‰]', '', regex=True
                ).str.replace(r'万', '0000', regex=False)
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self._log(f"  {col}: 转换为数值类型")
                self.lineage.record(col, 'convert_to_numeric', {})

        # 删除列
        drop_cols = fmt_config.get('drop_columns', [])
        for col in drop_cols:
            if col in self.df.columns:
                self.df = self.df.drop(columns=[col])
                self._log(f"  {col}: 删除列")
                self.lineage.record(col, 'drop_column', {'reason': 'format_standardization'})

        # 字段值纠正 - 模糊匹配
        value_corrections = fmt_config.get('value_correction', {})
        if value_corrections:
            self.df, correction_stats = correct_field_values(self.df, value_corrections)
            for col, stats_info in correction_stats.items():
                self._log(f"  {col}: 纠正 {stats_info['corrected_count']} 个值")
                # 记录血缘
                for detail in stats_info['details'][:5]:
                    self.lineage.record(col, 'value_correction', {
                        'original': detail['original'],
                        'corrected': detail['corrected'],
                        'score': detail['score']
                    })

        return self

    # ============================================================
    # 7. 数据转换
    # ============================================================

    def transform_data(self) -> 'DataCleaner':
        """数据转换"""
        trans_config = self.config.get('transformations', {})
        if not trans_config:
            return self

        self._log(f"\n" + "=" * 60)
        self._log("数据转换")
        self._log("=" * 60)

        # 分类编码
        categorical_encoding = trans_config.get('categorical_encoding', {})
        for col, mapping in categorical_encoding.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].map(mapping)
                self._log(f"  {col}: 应用编码映射")

        # 分箱
        binning = trans_config.get('binning', {})
        for col, config in binning.items():
            if col in self.df.columns:
                bins = config.get('bins', [0, 18, 35, 65])
                labels = config.get('labels', ['未成年', '青年', '中年', '老年'])
                try:
                    self.df[f"{col}_group"] = pd.cut(
                        self.df[col],
                        bins=bins,
                        labels=labels,
                        include_lowest=True,
                        duplicates='drop'
                    )
                    self._log(f"  {col}: 分箱为 {labels}")
                except ValueError as e:
                    self._log(f"  {col}: 分箱失败 - {str(e)}", "WARN")

        # 计算列
        computed = trans_config.get('computed_columns', {})
        for new_col, expr in computed.items():
            if '.dt.' in expr:
                # 支持 dt.year, dt.month, dt.day, dt.hour 等
                parts = expr.split('.')
                if len(parts) >= 3:
                    date_col = parts[0]
                    dt_attr = parts[2]  # year, month, day, hour, minute, second 等
                    if date_col in self.df.columns:
                        try:
                            self.df[new_col] = getattr(self.df[date_col].dt, dt_attr)
                            self._log(f"  {new_col}: 从日期提取 {dt_attr}")
                            continue
                        except AttributeError:
                            self._log(f"  {new_col}: 不支持的日期属性 '{dt_attr}'", "WARN")
            else:
                if not self._is_safe_expression(expr):
                    self._log(f"  {new_col}: 计算失败 - 表达式包含非法字符", "WARN")
                    continue
                try:
                    self.df[new_col] = self.df.eval(expr)
                    self._log(f"  {new_col}: 计算列 ({expr})")
                except Exception as e:
                    self._log(f"  {new_col}: 计算失败 - {str(e)}", "WARN")

        return self

    # ============================================================
    # 8. 保存结果
    # ============================================================

    def save_results(self) -> 'DataCleaner':
        """保存清洗后的数据（带时间戳）"""
        data_config = self.config.get('data', {})
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = self._get_data_path(data_config.get('output_path', f'data/cleaned_data_{timestamp}.csv'))

        # 备份原始数据
        if self.original_shape is not None:
            backup_path = self._get_data_path('data/raw_backup.csv')
            if self.original_shape != self.df.shape:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = self._get_data_path(f'data/raw_backup_{timestamp}.csv')

            input_path = data_config.get('input_path')
            if input_path and os.path.exists(input_path):
                import shutil
                shutil.copy2(input_path, backup_path)
                self._log(f"备份原始数据: {backup_path}")

        # 使用 FileLoader 保存数据（支持多种格式）
        encoding = data_config.get('encoding', 'utf-8-sig')
        FileLoader.save(self.df, output_path, encoding=encoding)

        self._log(f"\n" + "=" * 60)
        self._log("保存结果")
        self._log("=" * 60)
        self._log(f"输出路径: {output_path}")
        self._log(f"最终数据: {self.df.shape[0]} 行 × {self.df.shape[1]} 列")

        return self

    # ============================================================
    # 9. 最终统计
    # ============================================================

    def get_final_stats(self) -> dict:
        """获取最终统计信息"""
        stats = {
            'original_rows': self.original_shape[0] if self.original_shape else len(self.df),
            'final_rows': len(self.df),
            'rows_removed': (self.original_shape[0] if self.original_shape else 0) - len(self.df),
            'columns': list(self.df.columns),
            'missing_values': int(self.df.isnull().sum().sum()),
            'duplicates': 0,
        }
        return stats

    # ============================================================
    # 主流程
    # ============================================================

    def clean(self, input_path: str = None) -> 'DataCleaner':
        """
        执行完整清洗流程

        Args:
            input_path: 可选，覆盖配置中的输入路径
        """
        # 初始化日志文件
        self._init_log_file()

        try:
            # 1. 加载数据
            self.load_data(input_path)
            # 2. 数据概览
            self.inspect_data()
            # 3. 处理缺失值
            self.handle_missing_values()
            # 4. 处理重复值
            self.handle_duplicates()
            # 5. 处理异常值
            self.handle_outliers()
            # 5.5 二次处理：异常值可能产生新的缺失值
            self.handle_missing_values()
            # 6. 格式标准化
            self.standardize_format()
            # 6.5 如存在value_correction，重新生成EDA报告
            value_correction = self.config.get('format_standardization', {}).get('value_correction', {})
            if value_correction:
                self._regenerate_eda()
            # 7. 数据转换
            self.transform_data()
            # 8. 保存结果
            self.save_results()
            # 8.5 保存血缘报告（带时间戳）
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            lineage_path = self._get_data_path(f'data/lineage_report_{timestamp}.yaml')
            self.lineage.save_report(lineage_path)
            self._log(f"血缘报告已保存: {lineage_path}")
            # 9. 最终统计
            stats = self.get_final_stats()
            self._log(f"\n清洗完成统计:")
            self._log(f"  原始行数: {stats['original_rows']}")
            self._log(f"  最终行数: {stats['final_rows']}")
            self._log(f"  删除行数: {stats['rows_removed']}")
            self._log(f"  剩余缺失值: {stats['missing_values']}")

            return self

        except Exception as e:
            self._log(f"错误: {str(e)}", "ERROR")
            raise


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='通用数据清洗脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/cleaning_data.py -i data/raw.csv                    # 使用默认配置
  python scripts/cleaning_data.py -i data/raw.csv -c configs/custom.yaml
  python scripts/cleaning_data.py -i data/raw.csv -o data/clean.csv
  python scripts/cleaning_data.py -i data/raw.csv --config configs/custom.yaml
        """
    )

    parser.add_argument('-c', '--config', default='configs/cleaning_config.yaml', help='配置文件路径')
    parser.add_argument('-i', '--input', required=True, help='输入数据路径')
    parser.add_argument('-o', '--output', help='输出数据路径')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')

    args = parser.parse_args()

    # 处理配置路径
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(SCRIPT_DIR / config_path)

    # 生成配置（先加载用户配置，再自动填充空白字段）
    config = ConfigGenerator.generate_config(
        input_path=args.input,
        output_path=args.output,
        config_path=config_path,
    )

    # 保存最终配置
    generated_config_path = str(SCRIPT_DIR / "configs/generated_config.yaml")
    ConfigGenerator.save_config(config, generated_config_path)
    print(f"[配置已保存] {generated_config_path}")

    # 创建清洗器并执行
    cleaner = DataCleaner(config=config)
    cleaner.clean(args.input)

    print("\n" + "=" * 60)
    print("数据清洗完成！")
    print("=" * 60)


# ============================================================
# 便捷函数 - 一键清洗
# ============================================================

def quick_clean(
    input_path: str,
    output_path: str = "data/cleaned_data.csv",
    **kwargs
) -> DataCleaner:
    """
    一键数据清洗（自动生成配置并执行）

    Args:
        input_path: 输入数据路径
        output_path: 输出数据路径
        **kwargs: 其他配置参数

    Returns:
        DataCleaner: 清洗器实例
    """
    # 生成配置
    config = ConfigGenerator.generate_config(
        input_path=input_path,
        output_path=output_path,
        **kwargs
    )

    # 保存配置到文件
    config_output_path = "configs/generated_config.yaml"
    ConfigGenerator.save_config(config, config_output_path)
    print(f"[配置已保存] {config_output_path}")

    # 执行清洗
    cleaner = DataCleaner(config=config)
    cleaner.clean()

    return cleaner


def auto_analyze(input_path: str) -> dict:
    """
    自动分析数据（不执行清洗）

    Args:
        input_path: 数据文件路径

    Returns:
        dict: 数据分析结果
    """
    encoding = 'utf-8-sig'
    df = pd.read_csv(input_path, encoding=encoding)
    return ConfigGenerator.analyze_data(df)


if __name__ == '__main__':
    main()
