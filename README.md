# SyntheticDiffInDiff

A Python implementation of Synthetic Difference-in-Differences (SDID) for causal inference and policy evaluation.

## Overview

SyntheticDiffInDiff combines the advantages of synthetic control methods and difference-in-differences, providing a robust framework for estimating causal effects of policies or interventions. This method is particularly suitable for panel data and can control for heterogeneity in both unit and time dimensions simultaneously.

## Features

- Automatic calculation of unit weights and time weights
- Event study analysis to estimate treatment effects at different time points
- Standard error estimation via placebo tests
- Visualization with confidence intervals

## Installation Requirements

```bash
pip install numpy pandas cvxpy statsmodels joblib matplotlib
```

## Usage

### Basic Usage

```python
import pandas as pd
from synthetic_diff_in_diff import SyntheticDiffInDiff

# Prepare data
data = pd.read_csv('your_data.csv')

# Initialize SDID object
sdid = SyntheticDiffInDiff(
    data=data,
    outcome_col='outcome',  # Outcome variable column name
    times_col='time',       # Time column name
    units_col='unit',       # Unit column name
    treat_col='treated',    # Treatment indicator column name
    post_col='post'         # Post-treatment indicator column name
)

# Run analysis
treatment_effect = sdid.run_analysis()
print(f"Estimated treatment effect: {treatment_effect}")

# Estimate standard error
sdid.estimate_se(bootstrap_rounds=200, n_jobs=4)
print(f"Standard error: {sdid.standard_error}")

# Run event study
times = data[data['post']]['time'].unique()
effects = sdid.run_event_study(times)
print("Treatment effects at different time points:")
print(effects)

# Generate figure
fig = sdid.make_figure(times, bootstrap_rounds=100, n_jobs=4)
fig.savefig('treatment_effects.png')
```

### Data Format

Your data should be in long-format panel data with the following columns:

- Outcome variable: The outcome measure you want to analyze
- Time identifier: Column indicating the time point of observation
- Unit identifier: Column indicating the unit of observation
- Treatment indicator: Boolean or 0/1 column indicating whether a unit receives treatment
- Post-treatment indicator: Boolean or 0/1 column indicating whether a time point is after treatment

## Method Details

### Unit Weight Calculation

The method uses regularized quadratic programming to estimate unit weights with the goal of matching the weighted average of control units to the treated units in pre-treatment periods.

### Time Weight Calculation

Similarly, time weights aim to control for time-specific effects, making the weighted average of time points comparable between control and treatment groups.

### Synthetic Difference-in-Differences

SDID estimates the treatment effect through the following steps:
1. Calculating unit weights
2. Calculating time weights
3. Performing weighted least squares regression using the combined weights
4. Extracting the coefficient of the treatment and post-treatment interaction term as the treatment effect

## Citation

If you use this implementation in your research, please consider citing the original SDID methodology paper:

Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021). Synthetic difference-in-differences. American Economic Review, 111(12), 4088-4118.

## Contributing

Issues and pull requests to improve this implementation are welcome!
