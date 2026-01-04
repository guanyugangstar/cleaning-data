"""
字段值模糊匹配纠正模块

提供基于相似度的字段值自动纠正功能，用于处理数据输入错误。
例如: "住建局限" -> "住建局"
"""

from difflib import get_close_matches
from typing import List, Optional, Dict, Any
import pandas as pd


def fuzzy_match(
    value: str,
    allowed_values: List[str],
    threshold: float = 0.6,
    n_matches: int = 1
) -> Optional[str]:
    """
    模糊匹配单个值到允许的值列表

    Args:
        value: 待匹配的值
        allowed_values: 允许的标准值列表
        threshold: 相似度阈值 (0.0-1.0)，默认0.6
        n_matches: 返回的最大匹配数量

    Returns:
        匹配到的值，未匹配到返回None
    """
    if not value or pd.isna(value):
        return None

    # 转换为字符串并清理
    value_str = str(value).strip()

    if not value_str:
        return None

    # 使用 get_close_matches 进行模糊匹配
    # cutoff 参数控制相似度阈值
    matches = get_close_matches(
        value_str,
        allowed_values,
        n=n_matches,
        cutoff=threshold
    )

    return matches[0] if matches else None


def get_match_score(value: str, candidate: str) -> float:
    """
    计算两个字符串的相似度得分

    Args:
        value: 原始值
        candidate: 对比值

    Returns:
        相似度得分 (0.0-1.0)
    """
    from difflib import SequenceMatcher

    if not value or not candidate:
        return 0.0

    return SequenceMatcher(None, str(value), str(candidate)).ratio()


def correct_field_values(
    df: pd.DataFrame,
    correction_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    根据配置批量纠正字段值

    Args:
        df: 原始数据框
        correction_config: 纠正配置
        {
            "列名": {
                "allowed_values": ["标准值1", "标准值2", ...],
                "method": "fuzzy",  # fuzzy 或 exact
                "threshold": 0.6,
                "log_corrections": True  # 是否记录纠正日志
            }
        }

    Returns:
        纠正后的数据框
    """
    if not correction_config:
        return df

    df = df.copy()
    correction_stats = {}

    for column, config in correction_config.items():
        if column not in df.columns:
            continue

        allowed_values = config.get('allowed_values', [])
        if not allowed_values:
            continue

        method = config.get('method', 'fuzzy')
        threshold = config.get('threshold', 0.6)
        log_corrections = config.get('log_corrections', True)

        if not isinstance(allowed_values, list):
            continue

        # 记录原始值分布
        original_counts = df[column].value_counts(dropna=False)

        # 逐行处理
        corrected_count = 0
        correction_details = []

        for idx, cell_value in df[column].items():
            if pd.isna(cell_value):
                continue

            cell_str = str(cell_value).strip()

            # 精确匹配检查
            if cell_str in allowed_values:
                continue

            if method == 'fuzzy':
                # 模糊匹配
                matched = fuzzy_match(cell_str, allowed_values, threshold)

                if matched and matched != cell_str:
                    df.at[idx, column] = matched
                    corrected_count += 1

                    if log_corrections:
                        score = get_match_score(cell_str, matched)
                        correction_details.append({
                            'row': idx,
                            'original': cell_str,
                            'corrected': matched,
                            'score': round(score, 3)
                        })

        if corrected_count > 0:
            correction_stats[column] = {
                'corrected_count': corrected_count,
                'details': correction_details[:10]  # 只保留前10条详细记录
            }

    return df, correction_stats


def validate_field_values(
    df: pd.DataFrame,
    allowed_values: List[str],
    threshold: float = 0.6
) -> Dict[str, Any]:
    """
    验证字段值并返回分析报告

    Args:
        df: 数据框
        allowed_values: 允许的标准值列表
        threshold: 相似度阈值

    Returns:
        验证报告
    """
    report = {
        'total_rows': len(df),
        'valid_count': 0,
        'invalid_count': 0,
        'invalid_values': [],  # {(列名, 值): 建议值}
        'value_distribution': {}
    }

    for col in df.columns:
        col_values = df[col].dropna().unique()
        report['value_distribution'][col] = df[col].value_counts().to_dict()

        for val in col_values:
            val_str = str(val).strip()

            if val_str in allowed_values:
                report['valid_count'] += 1
            else:
                # 尝试模糊匹配获取建议值
                suggested = fuzzy_match(val_str, allowed_values, threshold)
                report['invalid_values'].append({
                    'value': val_str,
                    'suggested': suggested
                })
                report['invalid_count'] += 1

    return report


if __name__ == "__main__":
    # 简单测试
    test_values = ["住建局限", "税物局", "民政局限", "人设局", "市场监督"]
    allowed = ["民政局", "人社局", "市场监管局", "住建局", "税务局"]

    print("=== 模糊匹配测试 ===")
    print(f"标准值: {allowed}")
    print(f"待匹配: {test_values}")
    print()

    for val in test_values:
        matched = fuzzy_match(val, allowed, threshold=0.6)
        score = get_match_score(val, matched) if matched else 0
        print(f"  {val} -> {matched} (相似度: {score:.2f})")
