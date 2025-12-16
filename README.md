# Synthetic Difference-in-Differences (SDID)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A Python implementation of Synthetic Difference-in-Differences for causal inference.

[繁體中文版](README_zh-TW.md)

---

## Overview

**SDID** combines synthetic control methods with difference-in-differences to provide robust causal effect estimates. It automatically finds optimal weights for:

- **Control units**: Creates a synthetic comparison group that matches treated units' pre-treatment trends
- **Time periods**: Balances pre/post treatment comparisons

### Reference

This implementation is based on:

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

## Installation

### Using uv (Recommended)

```bash
git clone https://github.com/yourusername/Synthetic_Difference_in_Difference.git
cd Synthetic_Difference_in_Difference
uv sync
```

### Using pip

```bash
pip install numpy pandas cvxpy statsmodels joblib matplotlib scipy
```

---

## Quick Start

```python
import pandas as pd
from SDID import SyntheticDiffInDiff

# Load your panel data
data = pd.read_csv("your_data.csv")

# Initialize the estimator
sdid = SyntheticDiffInDiff(
    data=data,
    outcome_col="outcome",    # Outcome variable
    times_col="year",         # Time period identifier
    units_col="state",        # Unit identifier
    treat_col="treated",      # Treatment indicator (0/1)
    post_col="post"           # Post-treatment indicator (0/1)
)

# Fit the model
effect = sdid.fit()
print(f"Treatment Effect: {effect:.4f}")

# Estimate standard error (with parallel processing)
sdid.estimate_se(n_bootstrap=200, n_jobs=4)

# Print full summary
print(sdid.summary())
```

---

## Data Format

Your data should be in **long format** (one row per unit-time observation):

| Column | Description | Example |
|--------|-------------|---------|
| `outcome` | Outcome variable | Sales, GDP, Test scores |
| `times` | Time period | 2018, 2019, 2020 |
| `units` | Unit identifier | "CA", "TX", "NY" |
| `treat` | Treatment indicator | 0 = control, 1 = treated |
| `post` | Post-treatment indicator | 0 = before, 1 = after |

**Example:**

```
unit  year  outcome  treated  post
CA    2018  100.5    0        0     # California, pre-treatment, control
CA    2019  102.3    0        0
CA    2020  105.1    0        1     # Post-treatment period
TX    2018  98.2     1        0     # Texas, treated unit
TX    2019  99.8     1        0
TX    2020  108.7    1        1     # Treated, post-treatment
```

---

## API Reference

### Core Methods

| Method | Description |
|--------|-------------|
| `fit(verbose=False)` | Fit the model and return the treatment effect |
| `estimate_se(n_bootstrap=400, seed=0, n_jobs=1)` | Estimate standard error via placebo bootstrap |
| `summary()` | Return a formatted summary of results |
| `get_weights_summary()` | Return DataFrames of unit and time weights |

### Event Study Methods

| Method | Description |
|--------|-------------|
| `run_event_study(times)` | Estimate effects for multiple time periods |
| `plot_event_study(times, n_bootstrap=400, ...)` | Create event study plot with confidence intervals |

### Properties

| Property | Description |
|----------|-------------|
| `treatment_effect` | Estimated ATT (after calling `fit()`) |
| `standard_error` | Estimated SE (after calling `estimate_se()`) |
| `unit_weights` | Weights assigned to control units |
| `time_weights` | Weights assigned to time periods |
| `is_fitted` | Boolean indicating if model has been fitted |

---

## Examples

### Basic Usage

```python
# Fit and get results
effect = sdid.fit()
sdid.estimate_se(n_bootstrap=400, n_jobs=-1)  # Use all CPU cores
print(sdid.summary())
```

### Event Study

```python
# Analyze treatment effects over time
post_periods = [2020, 2021, 2022]
effects = sdid.run_event_study(post_periods)

# Create publication-ready plot
fig = sdid.plot_event_study(
    times=post_periods,
    n_bootstrap=200,
    confidence_level=0.95,
    n_jobs=4
)
fig.savefig("event_study.png", dpi=300, bbox_inches="tight")
```

### Inspect Weights

```python
weights = sdid.get_weights_summary()

print("Top control units by weight:")
print(weights["unit_weights"].head(10))

print("\nTime period weights:")
print(weights["time_weights"])
```

---

## Key Assumptions

SDID relies on several key assumptions:

1. **No anticipation**: Units do not change behavior in anticipation of treatment
2. **SUTVA**: No spillover effects between treated and control units
3. **Overlap**: Control units can reasonably approximate treated units

Always consider whether these assumptions hold in your specific context.

---

## Development

```bash
# Install with dev dependencies
uv sync --dev

# Run linter
uv run ruff check SDID.py

# Format code
uv run ruff format SDID.py
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
