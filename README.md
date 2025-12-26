# Synthetic Difference-in-Differences (SDID)

[![PyPI version](https://img.shields.io/pypi/v/sdid.svg)](https://pypi.org/project/sdid/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

[English](#english) | [繁體中文](#繁體中文)

---

## English

A Python implementation of Synthetic Difference-in-Differences for causal inference.

### Overview

**SDID** combines synthetic control methods with difference-in-differences to provide robust causal effect estimates. It automatically finds optimal weights for:

- **Control units**: Creates a synthetic comparison group
- **Time periods**: Balances pre/post treatment comparisons

### Mathematical Formulation

SDID estimates the Average Treatment Effect on the Treated (ATT) through a weighted two-way fixed effects regression.

#### Unit Weights Optimization

We find optimal unit weights $\hat{\omega}$ by solving:

$$\hat{\omega}, \hat{\alpha} = \underset{\omega, \alpha}{\arg\min} \sum_{t=1}^{T_{pre}} \left( \sum_{i \in \mathcal{C}} \omega_i Y_{it} + \alpha - \bar{Y}_{treated,t} \right)^2 + \zeta_{unit} \|\omega\|_2^2$$

subject to $\omega_i \geq 0$ for all $i \in \mathcal{C}$

where:
- $\mathcal{C}$ is the set of control units
- $Y_{it}$ is the outcome for unit $i$ at time $t$
- $\alpha$ is an intercept term (allows level differences)
- $\zeta_{unit}$ is the regularization parameter

#### Time Weights Optimization

Similarly, we estimate time weights $\hat{\lambda}$:

$$\hat{\lambda}, \hat{\beta} = \underset{\lambda, \beta}{\arg\min} \left( \sum_{t=1}^{T_{pre}} \lambda_t \Delta_t + \beta \right)^2 + \zeta_{time} \|\lambda\|_2^2$$

subject to $\lambda_t \geq 0$

where $\Delta_t = \bar{Y}_{treated,t} - \bar{Y}_{control,t}$

#### Treatment Effect Estimation

The ATT is estimated via weighted two-way fixed effects regression:

$$Y_{it} = \alpha_i + \gamma_t + \tau \cdot D_{it} + \varepsilon_{it}$$

where:
- $\alpha_i$ are unit fixed effects
- $\gamma_t$ are time fixed effects
- $D_{it} = \mathbf{1}\{i \in treated\} \cdot \mathbf{1}\{t \geq T_0\}$ is the treatment indicator
- $\tau$ is the **treatment effect** (our target parameter)

Observations are weighted by $w_{it} = \hat{\omega}_i \cdot \hat{\lambda}_t$.

### Installation

**From PyPI (Recommended)**

```bash
pip install sdid
```

**From Source**

```bash
git clone https://github.com/yourusername/sdid.git
cd sdid
pip install .
```

**Development Installation**

```bash
git clone https://github.com/yourusername/sdid.git
cd sdid
uv sync --extra dev
```

### Quick Start

```python
import pandas as pd
from sdid import SyntheticDiffInDiff

# Load your panel data
data = pd.read_csv("your_data.csv")

# Initialize and fit
sdid = SyntheticDiffInDiff(
    data=data,
    outcome_col="outcome",
    times_col="year",
    units_col="state",
    treat_col="treated",
    post_col="post"
)

# Estimate treatment effect
effect = sdid.fit()
print(f"Treatment Effect: {effect:.4f}")

# Estimate standard error
sdid.estimate_se(n_bootstrap=200, n_jobs=4)
print(sdid.summary())
```

### Data Format

Your data should be in **long format** (one row per unit-time observation):

| Column | Description | Example |
|--------|-------------|---------|
| `outcome` | Outcome variable | Sales, GDP |
| `times` | Time period | 2018, 2019, 2020 |
| `units` | Unit identifier | "CA", "TX", "NY" |
| `treat` | Treatment indicator | 0 = control, 1 = treated |
| `post` | Post-treatment period | 0 = before, 1 = after |

**Example:**

```
unit  year  outcome  treated  post
CA    2018  100.5    0        0
CA    2019  102.3    0        0
CA    2020  105.1    0        1
TX    2018  98.2     1        0
TX    2019  99.8     1        0
TX    2020  108.7    1        1
```

### API Reference

| Method | Description |
|--------|-------------|
| `fit(verbose=False)` | Fit model and return treatment effect |
| `estimate_se(n_bootstrap, seed, n_jobs)` | Estimate standard error via placebo bootstrap |
| `summary(confidence_level=0.95)` | Return formatted results with customizable CI |
| `run_event_study(times)` | Estimate effects for multiple periods |
| `plot_event_study(times, ...)` | Create event study plot with confidence intervals |
| `plot_raw_trends(...)` | Plot raw trends for treated vs control units |
| `plot_synthetic_control(...)` | Plot treated unit vs synthetic control |
| `get_weights_summary()` | Return unit and time weights |
| `is_fitted` | Property to check if model is fitted |

### Visualization

```python
# Plot raw trends: treated unit vs all control units
sdid.plot_raw_trends(
    title="Raw Data: Treated vs Controls",
    treated_color="red",
    control_color="gray"
)

# Plot synthetic control comparison
sdid.plot_synthetic_control(
    title="SDID: Actual vs Synthetic",
    treated_color="blue",
    synthetic_color="orange"
)
```

### Customizable Confidence Interval

```python
# Default 95% confidence interval
print(sdid.summary())

# Custom 90% confidence interval
print(sdid.summary(confidence_level=0.90))

# Custom 99% confidence interval
print(sdid.summary(confidence_level=0.99))
```

### Event Study

```python
# Estimate dynamic treatment effects
effects = sdid.run_event_study([2020, 2021, 2022])

# Plot with confidence intervals
fig = sdid.plot_event_study(
    times=[2020, 2021, 2022],
    n_bootstrap=200,
    confidence_level=0.95,
    n_jobs=4
)
fig.savefig("event_study.png", dpi=300)
```

### Assumptions

SDID relies on these key assumptions:

1. **No anticipation**: Units don't change behavior before treatment
2. **SUTVA**: No spillover effects between units
3. **Overlap**: Control units can approximate treated units

### Reference

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

### Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Run linter
uv run ruff check .

# Format code
uv run ruff format .

# Build package
uv build

# Upload to PyPI (requires API token)
uv publish
```

### License

MIT License - see [LICENSE](LICENSE) for details.

---

## 繁體中文

合成雙重差分法 (SDID) 的 Python 實現，用於因果推論。

### 概述

**SDID** 結合了合成控制法與雙重差分法的優點，提供穩健的因果效應估計。它自動計算最佳權重：

- **控制單位權重**：建立合成對照組
- **時間期間權重**：平衡處理前後的比較

### 數學公式

SDID 透過加權雙向固定效應迴歸估計處理組的平均處理效果 (ATT)。

#### 單位權重優化

我們通過求解以下問題找到最佳單位權重 $\hat{\omega}$：

$$\hat{\omega}, \hat{\alpha} = \underset{\omega, \alpha}{\arg\min} \sum_{t=1}^{T_{pre}} \left( \sum_{i \in \mathcal{C}} \omega_i Y_{it} + \alpha - \bar{Y}_{treated,t} \right)^2 + \zeta_{unit} \|\omega\|_2^2$$

限制條件：$\omega_i \geq 0$（對所有控制單位 $i \in \mathcal{C}$）

其中：
- $\mathcal{C}$ 為控制單位集合
- $Y_{it}$ 為單位 $i$ 在時間 $t$ 的結果
- $\alpha$ 為截距項（允許水平差異）
- $\zeta_{unit}$ 為正則化參數

#### 時間權重優化

類似地，我們估計時間權重 $\hat{\lambda}$：

$$\hat{\lambda}, \hat{\beta} = \underset{\lambda, \beta}{\arg\min} \left( \sum_{t=1}^{T_{pre}} \lambda_t \Delta_t + \beta \right)^2 + \zeta_{time} \|\lambda\|_2^2$$

限制條件：$\lambda_t \geq 0$

其中 $\Delta_t = \bar{Y}_{treated,t} - \bar{Y}_{control,t}$

#### 處理效果估計

ATT 透過加權雙向固定效應迴歸估計：

$$Y_{it} = \alpha_i + \gamma_t + \tau \cdot D_{it} + \varepsilon_{it}$$

其中：
- $\alpha_i$ 為單位固定效應
- $\gamma_t$ 為時間固定效應
- $D_{it} = \mathbf{1}\{i \in treated\} \cdot \mathbf{1}\{t \geq T_0\}$ 為處理指標
- $\tau$ 為**處理效果**（目標參數）

觀測值權重為 $w_{it} = \hat{\omega}_i \cdot \hat{\lambda}_t$。

### 安裝

**從 PyPI 安裝（推薦）**

```bash
pip install sdid
```

**從原始碼安裝**

```bash
git clone https://github.com/yourusername/sdid.git
cd sdid
pip install .
```

**開發安裝**

```bash
git clone https://github.com/yourusername/sdid.git
cd sdid
uv sync --extra dev
```

### 快速開始

```python
import pandas as pd
from sdid import SyntheticDiffInDiff

# 載入面板資料
data = pd.read_csv("your_data.csv")

# 初始化並擬合
sdid = SyntheticDiffInDiff(
    data=data,
    outcome_col="outcome",    # 結果變數
    times_col="year",         # 時間
    units_col="state",        # 單位識別
    treat_col="treated",      # 處理指標
    post_col="post"           # 處理後指標
)

# 估計處理效果
effect = sdid.fit()
print(f"處理效果: {effect:.4f}")

# 估計標準誤
sdid.estimate_se(n_bootstrap=200, n_jobs=4)
print(sdid.summary())
```

### 資料格式

資料須為**長格式**（每個單位-時間組合一列）：

| 欄位 | 說明 | 範例 |
|------|------|------|
| `outcome` | 結果變數 | 銷售額、GDP |
| `times` | 時間期間 | 2018, 2019, 2020 |
| `units` | 單位識別 | "CA", "TX", "NY" |
| `treat` | 處理指標 | 0 = 控制組, 1 = 處理組 |
| `post` | 處理後期間 | 0 = 處理前, 1 = 處理後 |

**範例：**

```
unit  year  outcome  treated  post
CA    2018  100.5    0        0
CA    2019  102.3    0        0
CA    2020  105.1    0        1
TX    2018  98.2     1        0
TX    2019  99.8     1        0
TX    2020  108.7    1        1
```

### API 參考

| 方法 | 說明 |
|------|------|
| `fit(verbose=False)` | 擬合模型並回傳處理效果 |
| `estimate_se(n_bootstrap, seed, n_jobs)` | 透過安慰劑 bootstrap 估計標準誤 |
| `summary(confidence_level=0.95)` | 回傳格式化結果，可自訂信賴區間 |
| `run_event_study(times)` | 估計多個時間點的效果 |
| `plot_event_study(times, ...)` | 繪製帶信賴區間的事件研究圖 |
| `plot_raw_trends(...)` | 繪製處理組與控制組的原始趨勢 |
| `plot_synthetic_control(...)` | 繪製處理單位與合成控制的比較 |
| `get_weights_summary()` | 回傳單位和時間權重 |
| `is_fitted` | 檢查模型是否已擬合的屬性 |

### 視覺化

```python
# 繪製原始趨勢：處理單位 vs 所有控制單位
sdid.plot_raw_trends(
    title="原始資料：處理組 vs 控制組",
    treated_color="red",
    control_color="gray"
)

# 繪製合成控制比較圖
sdid.plot_synthetic_control(
    title="SDID：實際值 vs 合成控制",
    treated_color="blue",
    synthetic_color="orange"
)
```

### 自訂信賴區間

```python
# 預設 95% 信賴區間
print(sdid.summary())

# 自訂 90% 信賴區間
print(sdid.summary(confidence_level=0.90))

# 自訂 99% 信賴區間
print(sdid.summary(confidence_level=0.99))
```

### 事件研究

```python
# 估計動態處理效果
effects = sdid.run_event_study([2020, 2021, 2022])

# 繪製帶信賴區間的圖表
fig = sdid.plot_event_study(
    times=[2020, 2021, 2022],
    n_bootstrap=200,
    confidence_level=0.95,
    n_jobs=4
)
fig.savefig("event_study.png", dpi=300)
```

### 假設條件

SDID 依賴以下關鍵假設：

1. **無預期效應**：單位不會在處理前改變行為
2. **SUTVA**：單位間無外溢效應
3. **重疊性**：控制單位能近似處理單位

### 參考文獻

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

### 開發

```bash
# 安裝開發相依套件
uv sync --extra dev

# 執行測試
uv run pytest tests/ -v

# 執行 linter
uv run ruff check .

# 格式化程式碼
uv run ruff format .

# 建置套件
uv build

# 上傳到 PyPI（需要 API token）
uv publish
```

### 授權

MIT 授權 - 詳見 [LICENSE](LICENSE)。
