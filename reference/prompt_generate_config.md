---
type:
category:
tags:
date: 2026-01-03
---
# LLM 分析提示词模板

  

> 出处：[SKILL.md §6.1](../SKILL.md#61-llm-分析提示词模板)

  
```

你是一个数据清洗专家。请分析以下数据样本，生成 custom.yaml 配置文件。

  

=== 数据样本 ===

{数据样本}

  

请基于列名含义和样本值模式，判断以下**核心业务规则**：

  

## 只需要配置以下字段

  

### 1. 异常值范围 outliers.ranges（需要业务语义判断）

根据列名语义设定合理范围：

- age, 年龄 → [0, 120]

- income, salary, 薪资, 收入 → [0, null]

- percentage, rate, 比率, 百分比 → [0, 100]

- score, points, 分数, 成绩 → [0, 100] 或 [0, null]

- year, 年份 → [1900, 当前年份+1]

- price, 价格 → [0, null]

- count, 数量 → [0, null]

  

### 2. 数值分箱 binning（需要业务语义判断）

对有业务含义的数值列进行合理分箱：

- age → [0,18,35,50,65,120] → ["未成年","青年","中年","壮年","老年"]

- score → [0,60,70,80,90,100] → ["不及格","及格","中","良","优"]

  

### 3. 计算列 computed_columns（需要业务语义判断）

识别可计算的新列：

- price × quantity = total_price

- end_date - start_date = duration_days

  

## 自动补充（不需要LLM配置）

- date_columns: 脚本自动通过列名+内容模式检测

- trim_columns: 自动填充所有分类列

- categorical_encoding: 低基数列自动用 LabelEncoder 生成映射

  

## 输出要求

生成 custom.yaml，只需包含上述需要业务判断的字段，未提及的字段脚本会自动补充。

```
