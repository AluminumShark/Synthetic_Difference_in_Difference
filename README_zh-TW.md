# 合成雙重差分法 (SDID)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

用於因果推論的合成雙重差分法 Python 實現。

[English Version](README.md)

---

## 概述

**SDID** 結合了合成控制法與雙重差分法的優點，提供穩健的因果效應估計。它自動計算最佳權重：

- **控制單位權重**：建立與處理組處理前趨勢匹配的合成對照組
- **時間期間權重**：平衡處理前後的比較

### 參考文獻

本實現基於以下論文：

> Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021). Synthetic difference-in-differences. *American Economic Review*, 111(12), 4088-4118.

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

## 安裝

### 使用 uv（推薦）

```bash
git clone https://github.com/yourusername/Synthetic_Difference_in_Difference.git
cd Synthetic_Difference_in_Difference
uv sync
```

### 使用 pip

```bash
pip install numpy pandas cvxpy statsmodels joblib matplotlib scipy
```

---

## 快速開始

```python
import pandas as pd
from SDID import SyntheticDiffInDiff

# 載入面板資料
data = pd.read_csv("your_data.csv")

# 初始化估計器
sdid = SyntheticDiffInDiff(
    data=data,
    outcome_col="outcome",    # 結果變數
    times_col="year",         # 時間識別欄位
    units_col="state",        # 單位識別欄位
    treat_col="treated",      # 處理指標 (0/1)
    post_col="post"           # 處理後指標 (0/1)
)

# 擬合模型
effect = sdid.fit()
print(f"處理效果: {effect:.4f}")

# 估計標準誤（使用平行處理）
sdid.estimate_se(n_bootstrap=200, n_jobs=4)

# 輸出完整摘要
print(sdid.summary())
```

---

## 資料格式

資料須為**長格式**（每個單位-時間組合一列）：

| 欄位 | 說明 | 範例 |
|------|------|------|
| `outcome` | 結果變數 | 銷售額、GDP、考試成績 |
| `times` | 時間期間 | 2018, 2019, 2020 |
| `units` | 單位識別 | "CA", "TX", "NY" |
| `treat` | 處理指標 | 0 = 控制組, 1 = 處理組 |
| `post` | 處理後指標 | 0 = 處理前, 1 = 處理後 |

**範例：**

```
unit  year  outcome  treated  post
CA    2018  100.5    0        0     # 加州，處理前，控制組
CA    2019  102.3    0        0
CA    2020  105.1    0        1     # 處理後期間
TX    2018  98.2     1        0     # 德州，處理組
TX    2019  99.8     1        0
TX    2020  108.7    1        1     # 處理組，處理後
```

---

## API 參考

### 核心方法

| 方法 | 說明 |
|------|------|
| `fit(verbose=False)` | 擬合模型並返回處理效果 |
| `estimate_se(n_bootstrap=400, seed=0, n_jobs=1)` | 透過 placebo bootstrap 估計標準誤 |
| `summary()` | 返回格式化的結果摘要 |
| `get_weights_summary()` | 返回單位和時間權重的 DataFrame |

### 事件研究方法

| 方法 | 說明 |
|------|------|
| `run_event_study(times)` | 估計多個時間點的效果 |
| `plot_event_study(times, n_bootstrap=400, ...)` | 建立帶信賴區間的事件研究圖 |

### 屬性

| 屬性 | 說明 |
|------|------|
| `treatment_effect` | 估計的 ATT（呼叫 `fit()` 後） |
| `standard_error` | 估計的標準誤（呼叫 `estimate_se()` 後） |
| `unit_weights` | 分配給控制單位的權重 |
| `time_weights` | 分配給時間期間的權重 |
| `is_fitted` | 布林值，表示模型是否已擬合 |

---

## 使用範例

### 基本用法

```python
# 擬合並取得結果
effect = sdid.fit()
sdid.estimate_se(n_bootstrap=400, n_jobs=-1)  # 使用所有 CPU 核心
print(sdid.summary())
```

### 事件研究

```python
# 分析處理效果隨時間的變化
post_periods = [2020, 2021, 2022]
effects = sdid.run_event_study(post_periods)

# 建立可發表的圖表
fig = sdid.plot_event_study(
    times=post_periods,
    n_bootstrap=200,
    confidence_level=0.95,
    n_jobs=4
)
fig.savefig("event_study.png", dpi=300, bbox_inches="tight")
```

### 檢視權重

```python
weights = sdid.get_weights_summary()

print("權重最高的控制單位：")
print(weights["unit_weights"].head(10))

print("\n時間期間權重：")
print(weights["time_weights"])
```

---

## 關鍵假設

SDID 依賴以下關鍵假設：

1. **無預期效應**：單位不會因預期處理而提前改變行為
2. **SUTVA**：處理組與控制組之間無外溢效應
3. **重疊性**：控制單位能合理地近似處理單位

在您的具體情境中，請務必考慮這些假設是否成立。

---

## 開發

```bash
# 安裝開發相依套件
uv sync --dev

# 執行 linter
uv run ruff check SDID.py

# 格式化程式碼
uv run ruff format SDID.py
```

---

## 授權

MIT 授權 - 詳見 [LICENSE](LICENSE)。

