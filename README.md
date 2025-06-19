# SyntheticDiffInDiff | 合成雙重差分法

[English](#english) | [繁體中文](#繁體中文)

---

## English

A Python implementation of Synthetic Difference-in-Differences (SDID) for causal inference and policy evaluation.

### 🌟 Overview

SyntheticDiffInDiff combines the advantages of synthetic control methods and difference-in-differences, providing a robust framework for estimating causal effects of policies or interventions. This method is particularly suitable for panel data and can control for heterogeneity in both unit and time dimensions simultaneously.

### ✨ Features

- **Automatic Weight Calculation**: Estimates unit weights and time weights automatically
- **Event Study Analysis**: Estimate treatment effects at different time points
- **Standard Error Estimation**: Via placebo tests with parallel processing support
- **Visualization Tools**: Generate publication-ready plots with confidence intervals
- **Comprehensive Diagnostics**: Get detailed information about model performance
- **Type Hints & Logging**: Enhanced code quality with type annotations and informative logging

### 📦 Installation

```bash
# Install required packages
pip install numpy pandas cvxpy statsmodels joblib matplotlib scipy

# Clone the repository
git clone https://github.com/yourusername/Synthetic_Difference_in_Difference.git
cd Synthetic_Difference_in_Difference
```

### 🚀 Quick Start

```python
import pandas as pd
from SDID import SyntheticDiffInDiff

# Load your panel data
data = pd.read_csv('your_data.csv')

# Initialize SDID
sdid = SyntheticDiffInDiff(
    data=data,
    outcome_col='outcome',    # Outcome variable
    times_col='time',         # Time identifier
    units_col='unit',         # Unit identifier
    treat_col='treated',      # Treatment indicator (True/1 for treated)
    post_col='post'          # Post-treatment indicator (True/1 for post)
)

# Run analysis
treatment_effect = sdid.run_analysis()
print(f"Estimated treatment effect: {treatment_effect:.4f}")

# Estimate standard error
sdid.estimate_se(bootstrap_rounds=200, n_jobs=4)
print(f"Standard error: {sdid.standard_error:.4f}")
```

### 📊 Advanced Usage

#### Event Study Analysis

```python
# Run event study to see effects over time
post_periods = data[data['post']]['time'].unique()
effects = sdid.run_event_study(post_periods)

# Create visualization
fig = sdid.make_figure(
    times=post_periods,
    bootstrap_rounds=100,
    n_jobs=4,
    confidence_level=0.95,
    figure_size=(14, 8)
)
fig.savefig('event_study.png', dpi=300, bbox_inches='tight')
```

#### Get Weight Summary

```python
# After running analysis, examine the weights
weights_summary = sdid.get_weights_summary()

# Top 5 control units by weight
print("Top 5 control units:")
print(weights_summary['unit_weights'].head())

# Top 5 time periods by weight
print("\nTop 5 time periods:")
print(weights_summary['time_weights'].head())
```

#### Model Diagnostics

```python
# Get comprehensive diagnostics
diagnostics = sdid.get_diagnostics()

print(f"Treatment Effect: {diagnostics['treatment_effect']:.4f}")
print(f"Standard Error: {diagnostics['standard_error']:.4f}")
print(f"Number of effective control units: {diagnostics['effective_control_units']}")
print(f"Number of effective time periods: {diagnostics['effective_time_periods']}")
```

### 📋 Data Requirements

Your data should be in **long format** with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| Outcome | float | The outcome variable you want to analyze |
| Time | any | Time period identifier |
| Unit | any | Unit/entity identifier |
| Treatment | bool/int | 1 if unit is treated, 0 otherwise |
| Post | bool/int | 1 if time is post-treatment, 0 otherwise |

Example data structure:
```
unit | time | outcome | treated | post
-----|------|---------|---------|-----
A    | 2018 | 100.5   | 0       | 0
A    | 2019 | 102.3   | 0       | 0
A    | 2020 | 105.1   | 0       | 1
B    | 2018 | 98.2    | 1       | 0
B    | 2019 | 99.8    | 1       | 0
B    | 2020 | 108.7   | 1       | 1
```

### 🔬 Method Details

The SDID estimator works in three main steps:

1. **Unit Weight Estimation**: Finds weights for control units to match treated units in pre-treatment periods
2. **Time Weight Estimation**: Finds weights for time periods to balance pre/post trends
3. **Weighted DiD Regression**: Estimates treatment effect using the combined weights

The optimization problems include L2 regularization to prevent overfitting, with regularization parameters calculated automatically based on data characteristics.

### 📖 Citation

If you use this implementation in your research, please cite:

```bibtex
@article{arkhangelsky2021synthetic,
  title={Synthetic difference-in-differences},
  author={Arkhangelsky, Dmitry and Athey, Susan and Hirshberg, David A and Imbens, Guido W and Wager, Stefan},
  journal={American Economic Review},
  volume={111},
  number={12},
  pages={4088--4118},
  year={2021}
}
```

---

## 繁體中文

用於因果推論和政策評估的合成雙重差分法（SDID）Python 實作。

### 🌟 概述

SyntheticDiffInDiff 結合了合成控制法和雙重差分法的優點，為估計政策或干預措施的因果效應提供了一個穩健的框架。此方法特別適合於面板數據，能同時控制單位和時間維度的異質性。

### ✨ 功能特色

- **自動權重計算**：自動估計單位權重和時間權重
- **事件研究分析**：估計不同時間點的處理效應
- **標準誤估計**：透過安慰劑檢驗，支援平行處理
- **視覺化工具**：生成可發表品質的圖表，包含信賴區間
- **全面診斷資訊**：獲取模型效能的詳細資訊
- **型別提示與日誌**：透過型別註解和資訊豐富的日誌提升程式碼品質

### 📦 安裝

```bash
# 安裝必要套件
pip install numpy pandas cvxpy statsmodels joblib matplotlib scipy

# 複製儲存庫
git clone https://github.com/yourusername/Synthetic_Difference_in_Difference.git
cd Synthetic_Difference_in_Difference
```

### 🚀 快速開始

```python
import pandas as pd
from SDID import SyntheticDiffInDiff

# 載入面板數據
data = pd.read_csv('your_data.csv')

# 初始化 SDID
sdid = SyntheticDiffInDiff(
    data=data,
    outcome_col='outcome',    # 結果變數
    times_col='time',         # 時間識別碼
    units_col='unit',         # 單位識別碼
    treat_col='treated',      # 處理指標（處理組為 True/1）
    post_col='post'          # 處理後指標（處理後期間為 True/1）
)

# 執行分析
treatment_effect = sdid.run_analysis()
print(f"估計的處理效應: {treatment_effect:.4f}")

# 估計標準誤
sdid.estimate_se(bootstrap_rounds=200, n_jobs=4)
print(f"標準誤: {sdid.standard_error:.4f}")
```

### 📊 進階使用

#### 事件研究分析

```python
# 執行事件研究以查看隨時間變化的效應
post_periods = data[data['post']]['time'].unique()
effects = sdid.run_event_study(post_periods)

# 建立視覺化圖表
fig = sdid.make_figure(
    times=post_periods,
    bootstrap_rounds=100,
    n_jobs=4,
    confidence_level=0.95,
    figure_size=(14, 8)
)
fig.savefig('event_study.png', dpi=300, bbox_inches='tight')
```

#### 取得權重摘要

```python
# 執行分析後，檢查權重
weights_summary = sdid.get_weights_summary()

# 權重前 5 名的控制單位
print("權重前 5 名的控制單位:")
print(weights_summary['unit_weights'].head())

# 權重前 5 名的時間期間
print("\n權重前 5 名的時間期間:")
print(weights_summary['time_weights'].head())
```

#### 模型診斷

```python
# 取得全面的診斷資訊
diagnostics = sdid.get_diagnostics()

print(f"處理效應: {diagnostics['treatment_effect']:.4f}")
print(f"標準誤: {diagnostics['standard_error']:.4f}")
print(f"有效控制單位數: {diagnostics['effective_control_units']}")
print(f"有效時間期間數: {diagnostics['effective_time_periods']}")
```

### 📋 數據要求

您的數據應為**長格式**，包含以下欄位：

| 欄位 | 類型 | 說明 |
|------|------|------|
| 結果變數 | float | 您要分析的結果變數 |
| 時間 | any | 時間期間識別碼 |
| 單位 | any | 單位/實體識別碼 |
| 處理 | bool/int | 若單位接受處理則為 1，否則為 0 |
| 處理後 | bool/int | 若時間為處理後則為 1，否則為 0 |

數據結構範例：
```
unit | time | outcome | treated | post
-----|------|---------|---------|-----
A    | 2018 | 100.5   | 0       | 0
A    | 2019 | 102.3   | 0       | 0
A    | 2020 | 105.1   | 0       | 1
B    | 2018 | 98.2    | 1       | 0
B    | 2019 | 99.8    | 1       | 0
B    | 2020 | 108.7   | 1       | 1
```

### 🔬 方法細節

SDID 估計量的運作分為三個主要步驟：

1. **單位權重估計**：尋找控制單位的權重，使其在處理前期間與處理單位相匹配
2. **時間權重估計**：尋找時間期間的權重，以平衡處理前後的趨勢
3. **加權 DiD 迴歸**：使用組合權重估計處理效應

優化問題包含 L2 正則化以防止過度擬合，正則化參數會根據數據特徵自動計算。

### 📖 引用

如果您在研究中使用此實作，請引用：

```bibtex
@article{arkhangelsky2021synthetic,
  title={Synthetic difference-in-differences},
  author={Arkhangelsky, Dmitry and Athey, Susan and Hirshberg, David A and Imbens, Guido W and Wager, Stefan},
  journal={American Economic Review},
  volume={111},
  number={12},
  pages={4088--4118},
  year={2021}
}
```

### 🤝 貢獻

歡迎提交問題和拉取請求來改進此實作！

### 📄 授權

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案。 