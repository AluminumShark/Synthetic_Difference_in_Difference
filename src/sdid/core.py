"""
Synthetic Difference-in-Differences (SDID) Implementation.

This module provides a Python implementation of the SDID estimator
based on Arkhangelsky et al. (2021).

Reference:
    Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021).
    Synthetic difference-in-differences. American Economic Review, 111(12), 4088-4118.
"""

import logging
import warnings
from functools import partial

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from scipy import stats

# Library logging: attach a NullHandler so importing this package does not
# mutate the root logger configuration of the host application. Consumers can
# enable output by configuring their own handlers / levels for this logger.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SyntheticDiffInDiff:
    """
    Synthetic Difference-in-Differences (SDID) Estimator.

    SDID combines the strengths of synthetic control methods and traditional
    difference-in-differences to provide more robust causal effect estimates.
    It works by:

    1. Finding optimal weights for control units to match pre-treatment trends
    2. Finding optimal weights for time periods to balance comparisons
    3. Running a weighted diff-in-diff regression

    Requirements:
        - Panel data in long format (one row per unit-time observation)
        - Binary treatment indicator (treated vs control units)
        - Binary post-treatment indicator (pre vs post periods)

    Example:
        >>> sdid = SyntheticDiffInDiff(
        ...     data=df,
        ...     outcome_col="outcome",
        ...     times_col="year",
        ...     units_col="state",
        ...     treat_col="treated",
        ...     post_col="post"
        ... )
        >>> effect = sdid.fit()
        >>> print(f"Treatment effect: {effect:.4f}")

    Attributes:
        treatment_effect: Estimated average treatment effect (after fitting)
        standard_error: Estimated standard error (after calling estimate_se)
        unit_weights: Weights assigned to control units
        time_weights: Weights assigned to time periods
    """

    # Class constants
    WEIGHT_THRESHOLD = 1e-6  # Minimum weight to keep
    DEFAULT_NOISE_LEVEL = 0.01  # Fallback when noise can't be estimated

    def __init__(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        times_col: str,
        units_col: str,
        treat_col: str,
        post_col: str,
    ):
        """
        Initialize the SDID estimator.

        Args:
            data: Panel data in long format
            outcome_col: Name of the outcome variable column
            times_col: Name of the time period column
            units_col: Name of the unit identifier column
            treat_col: Name of the treatment indicator column (1/True = treated)
            post_col: Name of the post-treatment indicator column (1/True = post)

        Raises:
            ValueError: If required columns are missing or contain NaN values
        """
        self._validate_columns(data, outcome_col, times_col, units_col, treat_col, post_col)

        # Store configuration
        self.data = data.copy()
        self.outcome_col = outcome_col
        self.times_col = times_col
        self.units_col = units_col
        self.treat_col = treat_col
        self.post_col = post_col

        # Results (populated after fitting)
        self.unit_weights: pd.Series | None = None
        self.time_weights: pd.Series | None = None
        self.merged_data: pd.DataFrame | None = None
        self.treatment_effect: float | None = None
        self.standard_error: float | None = None
        self._unit_intercept: float | None = None  # Intercept from unit weight optimization
        self._time_intercept: float | None = None  # Intercept from time weight optimization

        # Ensure boolean types for indicators
        self.data[self.treat_col] = self.data[self.treat_col].astype(bool)
        self.data[self.post_col] = self.data[self.post_col].astype(bool)

        # Validate data structure
        self._validate_data_structure()

    def _validate_columns(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        times_col: str,
        units_col: str,
        treat_col: str,
        post_col: str,
    ) -> None:
        """Validate that required columns exist and contain no NaN values."""
        required_cols = [outcome_col, times_col, units_col, treat_col, post_col]

        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        for col in required_cols:
            if data[col].isna().any():
                raise ValueError(f"Column '{col}' contains NaN values. Please clean your data.")

    def _validate_data_structure(self) -> None:
        """Validate that data has required structure for SDID analysis."""
        # Count units
        n_treated = self.data[self.data[self.treat_col]][self.units_col].nunique()
        n_control = self.data[~self.data[self.treat_col]][self.units_col].nunique()

        if n_treated == 0:
            raise ValueError("No treated units found. Check your treatment indicator.")
        if n_control == 0:
            raise ValueError("No control units found. SDID requires a comparison group.")

        # Count periods
        n_pre = self.data[~self.data[self.post_col]][self.times_col].nunique()
        n_post = self.data[self.data[self.post_col]][self.times_col].nunique()

        if n_pre == 0:
            raise ValueError("No pre-treatment periods found.")
        if n_post == 0:
            raise ValueError("No post-treatment periods found.")

        logger.info(
            f"Data validated: {n_treated} treated units, {n_control} controls, "
            f"{n_pre} pre-periods, {n_post} post-periods"
        )

    # =========================================================================
    # Core Estimation Methods
    # =========================================================================

    def fit(self, verbose: bool = False) -> float | None:
        """
        Fit the SDID model and estimate the treatment effect.

        This is the main method to run the complete analysis. It:
        1. Estimates optimal unit weights
        2. Estimates optimal time weights
        3. Runs weighted difference-in-differences regression

        Args:
            verbose: If True, print detailed optimization output

        Returns:
            Estimated average treatment effect on the treated (ATT)
        """
        logger.info("Starting SDID analysis...")

        self._estimate_unit_weights(verbose=verbose)
        self._estimate_time_weights(verbose=verbose)
        self._merge_weights_with_data()
        self._run_weighted_regression(verbose=verbose)

        logger.info("SDID analysis complete!")

        return self.treatment_effect

    def _calculate_unit_regularization(self) -> float:
        """
        Calculate regularization parameter (zeta) for unit weights.

        Following Arkhangelsky et al. (2021):

            zeta = (N_treated_post)^(1/4) * sigma

        where ``sigma`` is the standard deviation of the first differences of
        control-unit outcomes in the pre-treatment period. The squared value is
        applied as the ridge penalty in the weight optimization.

        Returns:
            Regularization parameter value (zeta)
        """
        # Count treated observations in post-treatment period
        n_treated_post = self.data[self.data[self.post_col] & self.data[self.treat_col]].shape[0]

        # Estimate noise from control units in pre-treatment period
        control_pre = self.data[(~self.data[self.post_col]) & (~self.data[self.treat_col])]

        if control_pre.empty:
            raise ValueError("No pre-treatment control data available for regularization.")

        # Noise level: std of first differences
        noise_level = (
            control_pre.sort_values(self.times_col)
            .groupby(self.units_col)[self.outcome_col]
            .diff()
            .std()
        )

        if np.isnan(noise_level) or noise_level == 0:
            warnings.warn(
                "Cannot estimate noise level from data. Using default value.",
                stacklevel=2,
            )
            noise_level = self.DEFAULT_NOISE_LEVEL

        return (n_treated_post**0.25) * noise_level

    def _calculate_time_regularization(self) -> float:
        """
        Calculate regularization parameter for time weights.

        Returns:
            Regularization parameter value
        """
        pre_data = self.data[~self.data[self.post_col]]

        if pre_data.empty:
            return self.DEFAULT_NOISE_LEVEL

        # Volatility across units within time periods
        time_volatility = pre_data.groupby(self.times_col)[self.outcome_col].std().mean()

        if np.isnan(time_volatility) or time_volatility == 0:
            time_volatility = self.DEFAULT_NOISE_LEVEL

        n_periods = pre_data[self.times_col].nunique()
        return time_volatility * (n_periods**-0.25)

    def _estimate_unit_weights(self, verbose: bool = False) -> None:
        """
        Estimate optimal weights for control units.

        Solves a quadratic programming problem to find weights that make
        the weighted average of control units match treated unit TRENDS in
        pre-treatment periods. Includes an intercept term to allow for
        level differences (SDID matches trends, not absolute levels).
        """
        logger.info("Estimating unit weights...")

        zeta = self._calculate_unit_regularization()
        logger.info(f"Unit regularization: {zeta:.4f}")

        # Get pre-treatment data
        pre_data = self.data[~self.data[self.post_col]]

        # Pivot control outcomes: time x units
        control_matrix = pre_data[~pre_data[self.treat_col]].pivot(
            index=self.times_col, columns=self.units_col, values=self.outcome_col
        )

        if control_matrix.empty:
            raise ValueError("No pre-treatment control data available.")

        # Average treated outcomes by time
        treated_avg = (
            pre_data[pre_data[self.treat_col]].groupby(self.times_col)[self.outcome_col].mean()
        )

        if treated_avg.empty:
            raise ValueError("No pre-treatment treated data available.")

        # Align time periods
        common_times = control_matrix.index.intersection(treated_avg.index)
        if len(common_times) == 0:
            raise ValueError("No overlapping time periods between treated and control groups.")

        control_matrix = control_matrix.loc[common_times]
        treated_avg = treated_avg.loc[common_times]

        # Set up optimization problem with intercept on the unit simplex.
        # The intercept allows for level differences between treated and controls
        # (SDID learns trends, not absolute levels), while the simplex constraint
        # (weights >= 0, sum to 1) follows Arkhangelsky et al. (2021).
        n_units = control_matrix.shape[1]
        weights = cp.Variable(n_units, nonneg=True)
        intercept = cp.Variable(1)  # Can be positive or negative

        Y = control_matrix.values
        y = treated_avg.values

        # Objective: minimize squared error with intercept + ridge penalty (zeta^2)
        objective = cp.Minimize(
            cp.sum_squares(Y @ weights + intercept - y) + (zeta**2) * cp.sum_squares(weights)
        )
        constraints = [cp.sum(weights) == 1]

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(verbose=verbose)

            if problem.status in ["infeasible", "unbounded"]:
                raise ValueError(f"Optimization failed: {problem.status}")

            weight_series = pd.Series(weights.value, index=control_matrix.columns, name="weight")
            self.unit_weights = weight_series[weight_series > self.WEIGHT_THRESHOLD]
            intercept_value = intercept.value
            if intercept_value is not None:
                self._unit_intercept = float(intercept_value[0])

            logger.info(
                f"Unit weights estimated: {len(self.unit_weights)} units with non-zero weights"
            )
            logger.info(f"Unit intercept: {self._unit_intercept:.4f}")

        except Exception as e:
            raise ValueError(f"Unit weight optimization failed: {e!s}") from e

    def _estimate_time_weights(self, verbose: bool = False) -> None:
        """
        Estimate optimal weights for time periods.

        Following Arkhangelsky et al. (2021), time weights are chosen so that,
        for the *control* units, a weighted average of pre-treatment outcomes
        predicts their average post-treatment outcome. An intercept absorbs
        level differences and the weights are constrained to the unit simplex
        (non-negative and summing to one).
        """
        logger.info("Estimating time weights...")

        zeta = self._calculate_time_regularization()
        logger.info(f"Time regularization: {zeta:.4f}")

        control_data = self.data[~self.data[self.treat_col]]
        pre_control = control_data[~control_data[self.post_col]]
        post_control = control_data[control_data[self.post_col]]

        if pre_control.empty or post_control.empty:
            raise ValueError("Insufficient control data for time weight estimation.")

        # Pre-treatment outcomes (control units x pre-periods)
        pre_matrix = pre_control.pivot(
            index=self.units_col, columns=self.times_col, values=self.outcome_col
        )
        # Average post-treatment outcome per control unit
        post_avg = post_control.groupby(self.units_col)[self.outcome_col].mean()

        # Align control units present in both pre- and post-treatment periods
        common_units = pre_matrix.index.intersection(post_avg.index)
        if len(common_units) == 0:
            raise ValueError("No control units with both pre- and post-treatment data.")

        pre_matrix = pre_matrix.loc[common_units]
        post_avg = post_avg.loc[common_units]

        pre_times = pre_matrix.columns
        n_periods = len(pre_times)

        # Optimization: weighted pre-period outcomes predict the post-period
        # average, on the time simplex with an intercept for level adjustment.
        weights = cp.Variable(n_periods, nonneg=True)
        intercept = cp.Variable(1)

        X = pre_matrix.values
        y = post_avg.values

        objective = cp.Minimize(
            cp.sum_squares(X @ weights + intercept - y) + zeta * cp.sum_squares(weights)
        )
        constraints = [cp.sum(weights) == 1]

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(verbose=verbose)

            if problem.status in ["infeasible", "unbounded"]:
                raise ValueError(f"Optimization failed: {problem.status}")

            weight_series = pd.Series(weights.value, index=pre_times, name="time_weight")
            self.time_weights = weight_series[weight_series > self.WEIGHT_THRESHOLD]
            intercept_value = intercept.value
            if intercept_value is not None:
                self._time_intercept = float(intercept_value[0])

            logger.info(
                f"Time weights estimated: {len(self.time_weights)} periods with non-zero weights"
            )
            logger.info(f"Time intercept: {self._time_intercept:.4f}")

        except Exception as e:
            raise ValueError(f"Time weight optimization failed: {e!s}") from e

    def _merge_weights_with_data(self) -> None:
        """Merge estimated weights with the original data."""
        logger.info("Merging weights with data...")

        if self.unit_weights is None or self.time_weights is None:
            raise ValueError("Weights must be estimated before merging.")

        df = self.data.copy()

        # Counts used to balance the treated/post mass against the synthetic
        # control / pre-period mass (Arkhangelsky et al., 2021 regression form).
        n_treated_units = df[df[self.treat_col]][self.units_col].nunique()
        n_post_periods = df[df[self.post_col]][self.times_col].nunique()

        # Initialize weight columns
        df["unit_weight"] = 0.0
        df["time_weight"] = 0.0

        # Assign unit weights: control units get estimated omega (sum to 1),
        # treated units share uniform weight 1 / N_treated.
        control_mask = ~df[self.treat_col]
        for unit, weight in self.unit_weights.items():
            mask = control_mask & (df[self.units_col] == unit)
            df.loc[mask, "unit_weight"] = weight

        df.loc[~control_mask, "unit_weight"] = 1.0 / n_treated_units

        # Assign time weights: pre-treatment periods get estimated lambda
        # (sum to 1), post-treatment periods share uniform weight 1 / T_post.
        for period, weight in self.time_weights.items():
            df.loc[df[self.times_col] == period, "time_weight"] = weight

        df.loc[df[self.post_col], "time_weight"] = 1.0 / n_post_periods

        # Combined weight
        df["combined_weight"] = df["unit_weight"] * df["time_weight"]

        self.merged_data = df
        logger.info("Weights merged successfully")

    def _run_weighted_regression(self, verbose: bool = False) -> None:
        """
        Run weighted two-way fixed effects regression.

        Uses unit and time fixed effects as required by SDID methodology.
        The treatment effect is identified from the treat_post interaction term.
        """
        logger.info("Running weighted regression...")

        if self.merged_data is None:
            raise ValueError("Data must be merged before running regression.")

        # Work directly on merged_data to avoid memory overhead
        # Create interaction term in-place
        self.merged_data["treat_post"] = (
            self.merged_data[self.treat_col] & self.merged_data[self.post_col]
        )

        # Filter to observations with positive weights (view, not copy)
        df = self.merged_data[self.merged_data["combined_weight"] > 0]

        if df.empty:
            raise ValueError("No observations with positive weights.")

        # Two-way fixed effects regression (unit + time FE)
        # C() indicates categorical/factor variables for fixed effects
        formula = f"{self.outcome_col} ~ treat_post + C({self.units_col}) + C({self.times_col})"

        try:
            model = smf.wls(formula, data=df, weights=df["combined_weight"])
            results = model.fit()

            self.treatment_effect = results.params["treat_post[T.True]"]

            if verbose:
                print("\n" + "=" * 60)
                print("SDID REGRESSION RESULTS (Two-Way Fixed Effects)")
                print("=" * 60)
                print(results.summary())
                print("=" * 60)

            logger.info(f"Treatment effect: {self.treatment_effect:.4f}")

        except Exception as e:
            raise ValueError(f"Weighted regression failed: {e!s}") from e

    # =========================================================================
    # Inference Methods
    # =========================================================================

    def estimate_se(
        self,
        n_bootstrap: int = 400,
        seed: int | None = 0,
        n_jobs: int = 1,
    ) -> float:
        """
        Estimate standard error using a placebo bootstrap.

        Following Arkhangelsky et al. (2021, Algorithm 4), the real treated
        units are discarded and placebo treatments are assigned at random among
        the control units. The spread of the resulting placebo effects
        approximates the sampling distribution of the estimator.

        Args:
            n_bootstrap: Number of bootstrap iterations
            seed: Random seed for reproducibility (None for nondeterministic)
            n_jobs: Number of parallel jobs (-1 for all cores)

        Returns:
            Estimated standard error
        """
        logger.info(f"Estimating standard error with {n_bootstrap} bootstrap samples...")

        # Local generator avoids mutating the caller's global NumPy RNG state.
        rng = np.random.default_rng(seed)

        placebo_fn = partial(
            self._run_placebo,
            outcome_col=self.outcome_col,
            times_col=self.times_col,
            units_col=self.units_col,
            treat_col=self.treat_col,
            post_col=self.post_col,
        )

        # Placebo datasets are drawn in the parent process, so results are
        # deterministic given the seed regardless of n_jobs.
        effects = Parallel(n_jobs=n_jobs)(
            delayed(placebo_fn)(self._create_placebo_data(rng)) for _ in range(n_bootstrap)
        )

        valid_effects = [e for e in effects if e is not None and not np.isnan(e)]

        if len(valid_effects) < 10:
            warnings.warn(
                f"Only {len(valid_effects)} valid placebo effects. "
                "Standard error estimate may be unreliable.",
                stacklevel=2,
            )

        self.standard_error = np.std(valid_effects, ddof=1) if valid_effects else np.nan
        logger.info(f"Standard error: {self.standard_error:.4f}")

        return self.standard_error

    def _create_placebo_data(self, rng: np.random.Generator) -> pd.DataFrame:
        """
        Create a placebo dataset for inference.

        Discards the real treated units and randomly designates the same number
        of control units as placebo-treated (Arkhangelsky et al., 2021). This
        keeps the placebo effects free of the real treatment signal.
        """
        control_units = self.data[~self.data[self.treat_col]][self.units_col].unique()

        if len(control_units) < 2:
            raise ValueError("Need at least 2 control units for placebo inference.")

        n_treated_units = self.data[self.data[self.treat_col]][self.units_col].nunique()

        # At least one placebo-treated unit, leaving at least one as control.
        n_placebo = max(1, min(n_treated_units, len(control_units) - 1))
        placebo_units = rng.choice(control_units, size=n_placebo, replace=False)

        # Keep only control units, then relabel the chosen ones as treated.
        df = self.data[~self.data[self.treat_col]].copy()
        df[self.treat_col] = df[self.units_col].isin(placebo_units).astype(bool)

        return df

    def _run_placebo(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        times_col: str,
        units_col: str,
        treat_col: str,
        post_col: str,
    ) -> float | None:
        """Run SDID on placebo data and return estimated effect."""
        try:
            sdid = SyntheticDiffInDiff(
                data=data,
                outcome_col=outcome_col,
                times_col=times_col,
                units_col=units_col,
                treat_col=treat_col,
                post_col=post_col,
            )
            return sdid.fit()
        except Exception:
            return np.nan

    # =========================================================================
    # Event Study Methods
    # =========================================================================

    def run_event_study(self, times: list[int | float | str]) -> pd.Series:
        """
        Run SDID for multiple time periods (event study).

        Estimates treatment effects at each specified time point,
        useful for examining dynamic treatment effects.

        Args:
            times: List of time periods to analyze

        Returns:
            Series of treatment effects indexed by time
        """
        logger.info(f"Running event study for {len(times)} time periods...")

        effects = {}

        for t in times:
            # Filter: pre-treatment periods + current period (view first, copy only when needed)
            mask = (~self.data[self.post_col]) | (self.data[self.times_col] == t)
            filtered = self.data.loc[mask]

            # Check data availability (no copy needed for counting)
            n_treated = (filtered[self.treat_col]).sum()
            n_control = (~filtered[self.treat_col]).sum()

            if n_treated == 0 or n_control == 0:
                logger.warning(f"Time {t}: Insufficient data, skipping.")
                effects[t] = np.nan
                continue

            try:
                # Copy only happens inside SyntheticDiffInDiff.__init__
                sdid = SyntheticDiffInDiff(
                    data=filtered,
                    outcome_col=self.outcome_col,
                    times_col=self.times_col,
                    units_col=self.units_col,
                    treat_col=self.treat_col,
                    post_col=self.post_col,
                )
                effects[t] = sdid.fit()
            except Exception as e:
                logger.warning(f"Time {t}: Analysis failed - {e}")
                effects[t] = np.nan

        return pd.Series(effects, name="treatment_effect")

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    def plot_event_study(
        self,
        times: list[int | float | str],
        n_bootstrap: int = 400,
        seed: int | None = 0,
        n_jobs: int = 1,
        confidence_level: float = 0.90,
        figsize: tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Create event study plot with confidence intervals.

        Args:
            times: List of time periods to analyze
            n_bootstrap: Number of bootstrap iterations for SE estimation
            seed: Random seed for reproducibility
            n_jobs: Number of parallel jobs
            confidence_level: Confidence level for intervals (default: 0.90)
            figsize: Figure size as (width, height)

        Returns:
            matplotlib Figure object
        """
        logger.info("Creating event study plot...")

        # Get point estimates
        effects = self.run_event_study(times)

        # Estimate standard errors for each period
        se_dict = {}
        for t in times:
            mask = (~self.data[self.post_col]) | (self.data[self.times_col] == t)
            filtered = self.data.loc[mask]  # View, copy happens in __init__

            try:
                sdid = SyntheticDiffInDiff(
                    data=filtered,
                    outcome_col=self.outcome_col,
                    times_col=self.times_col,
                    units_col=self.units_col,
                    treat_col=self.treat_col,
                    post_col=self.post_col,
                )
                sdid.estimate_se(n_bootstrap=n_bootstrap, seed=seed, n_jobs=n_jobs)
                se_dict[t] = sdid.standard_error
            except Exception as e:
                logger.warning(f"SE estimation at time {t} failed: {e}")
                se_dict[t] = np.nan

        se = pd.Series(se_dict)

        # Calculate confidence intervals
        z = stats.norm.ppf((1 + confidence_level) / 2)
        ci_lower = effects - z * se
        ci_upper = effects + z * se

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            effects.index,
            effects.values,
            marker="o",
            linewidth=2,
            markersize=8,
            label="Treatment Effect",
            color="#2563eb",
        )

        ax.fill_between(
            effects.index,
            ci_lower,
            ci_upper,
            alpha=0.3,
            color="#2563eb",
            label=f"{int(confidence_level * 100)}% CI",
        )

        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)

        # Add intervention start line (first post-treatment period)
        intervention_time = self.data[self.data[self.post_col]][self.times_col].min()
        ax.axvline(
            x=intervention_time,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="Intervention Start",
        )

        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Treatment Effect", fontsize=12)
        ax.set_title("Synthetic Difference-in-Differences: Event Study", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        if len(times) > 10:
            plt.xticks(rotation=45)

        plt.tight_layout()
        return fig

    def plot_raw_trends(
        self,
        treatment_time: int | float | None = None,
        figsize: tuple[int, int] = (10, 6),
        control_color: str = "lightgray",
        control_alpha: float = 0.3,
        avg_control_color: str = "gray",
        treated_color: str = "red",
        title: str | None = None,
    ) -> plt.Figure:
        """
        Plot raw trends comparing treated unit(s) against control units.

        This visualization shows the original data without any weighting,
        displaying all control units, their average, and the treated unit(s).

        Args:
            treatment_time: Time point of intervention. If None, uses the first
                post-treatment period from the data.
            figsize: Figure size as (width, height)
            control_color: Color for individual control unit lines
            control_alpha: Transparency for individual control unit lines
            avg_control_color: Color for average control line
            treated_color: Color for treated unit line
            title: Custom title for the plot. If None, uses default title.

        Returns:
            matplotlib Figure object
        """
        logger.info("Creating raw trends plot...")

        fig, ax = plt.subplots(figsize=figsize)

        # Determine treatment time
        plot_treatment_time: int | float
        if treatment_time is None:
            plot_treatment_time = float(self.data[self.data[self.post_col]][self.times_col].min())
        else:
            plot_treatment_time = treatment_time

        # Get control and treated data
        control_data = self.data[~self.data[self.treat_col]]
        treated_data = self.data[self.data[self.treat_col]]

        # Plot all control units
        for unit in control_data[self.units_col].unique():
            unit_data = control_data[control_data[self.units_col] == unit]
            ax.plot(
                unit_data[self.times_col],
                unit_data[self.outcome_col],
                color=control_color,
                alpha=control_alpha,
                linewidth=1,
            )

        # Plot average control
        avg_control = control_data.groupby(self.times_col)[self.outcome_col].mean()
        ax.plot(
            avg_control.index,
            avg_control.values,
            color=avg_control_color,
            linestyle="--",
            linewidth=2,
            label="Avg Control",
        )

        # Plot treated unit(s) - average if multiple
        treated_avg = treated_data.groupby(self.times_col)[self.outcome_col].mean()
        ax.plot(
            treated_avg.index,
            treated_avg.values,
            color=treated_color,
            linewidth=3,
            label="Treated",
        )

        # Add intervention line
        ax.axvline(x=plot_treatment_time, color="black", linestyle=":", label="Intervention")

        # Formatting
        plot_title = title if title is not None else "Raw Trends: Treated vs Controls"
        ax.set_title(plot_title, fontsize=14)
        ax.set_xlabel(self.times_col.capitalize(), fontsize=12)
        ax.set_ylabel(self.outcome_col.capitalize(), fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_synthetic_control(
        self,
        treatment_time: int | float | None = None,
        figsize: tuple[int, int] = (10, 6),
        treated_color: str = "red",
        synthetic_color: str = "blue",
        title: str | None = None,
    ) -> plt.Figure:
        """
        Plot the treated unit(s) against the SDID synthetic control.

        This visualization shows how well the synthetic control (weighted
        average of control units) matches the treated unit's trend before
        intervention and the divergence after intervention.

        Note: The model must be fitted before calling this method.

        Args:
            treatment_time: Time point of intervention. If None, uses the first
                post-treatment period from the data.
            figsize: Figure size as (width, height)
            treated_color: Color for treated unit line
            synthetic_color: Color for synthetic control line
            title: Custom title for the plot. If None, uses default title.

        Returns:
            matplotlib Figure object

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError(
                "Model must be fitted before plotting synthetic control. Call fit() first."
            )

        logger.info("Creating synthetic control plot...")

        if self.unit_weights is None:
            raise ValueError("Unit weights not estimated. Call fit() first.")

        fig, ax = plt.subplots(figsize=figsize)

        # Determine treatment time
        plot_treatment_time: int | float
        if treatment_time is None:
            plot_treatment_time = float(self.data[self.data[self.post_col]][self.times_col].min())
        else:
            plot_treatment_time = treatment_time

        # Get pre-treatment periods
        pre_periods = self.data[~self.data[self.post_col]][self.times_col].unique()

        # Pivot control outcomes: time x units
        control_wide = self.data[~self.data[self.treat_col]].pivot(
            index=self.times_col, columns=self.units_col, values=self.outcome_col
        )

        # Get valid control units (those with non-zero weights)
        valid_controls = self.unit_weights.index.intersection(control_wide.columns)

        if len(valid_controls) == 0:
            raise ValueError("No valid control units found with non-zero weights.")

        # Calculate synthetic control outcome (weighted average)
        synthetic_trend = control_wide[valid_controls].dot(self.unit_weights[valid_controls])

        # Get treated unit outcome
        treated_outcome = (
            self.data[self.data[self.treat_col]].groupby(self.times_col)[self.outcome_col].mean()
        )

        # Adjust level (intercept) - SDID matches trends, not levels
        # Align them in the pre-treatment period
        common_pre = [
            p for p in pre_periods if p in treated_outcome.index and p in synthetic_trend.index
        ]

        if len(common_pre) == 0:
            raise ValueError("No common pre-treatment periods for level adjustment.")

        treated_pre_mean = treated_outcome.loc[common_pre].mean()
        synthetic_pre_mean = synthetic_trend.loc[common_pre].mean()
        diff_mean = treated_pre_mean - synthetic_pre_mean
        synthetic_control = synthetic_trend + diff_mean

        # Get max time for post-treatment shading
        max_time = self.data[self.times_col].max()

        # Plot
        ax.plot(
            treated_outcome.index,
            treated_outcome.values,
            color=treated_color,
            linewidth=3,
            label="Treated Unit (Actual)",
        )

        ax.plot(
            synthetic_control.index,
            synthetic_control.values,
            color=synthetic_color,
            linestyle="--",
            linewidth=2,
            label="Synthetic Control (SDID)",
        )

        # Add intervention line and post-treatment shading
        plot_max_time = float(max_time)
        ax.axvline(x=plot_treatment_time, color="black", alpha=0.3)
        ax.axvspan(
            plot_treatment_time, plot_max_time, color="gray", alpha=0.1, label="Post-Treatment"
        )

        # Formatting
        plot_title = title if title is not None else "SDID Match: Treated vs Synthetic Control"
        ax.set_title(plot_title, fontsize=14)
        ax.set_xlabel(self.times_col.capitalize(), fontsize=12)
        ax.set_ylabel(self.outcome_col.capitalize(), fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_weights_summary(self) -> dict[str, pd.DataFrame]:
        """
        Get summary of estimated weights.

        Returns:
            Dictionary with 'unit_weights' and 'time_weights' DataFrames
        """
        if self.unit_weights is None or self.time_weights is None:
            raise ValueError("Weights not yet estimated. Call fit() first.")

        unit_df = pd.DataFrame(
            {
                "weight": self.unit_weights,
                "rank": self.unit_weights.rank(ascending=False, method="min"),
            }
        ).sort_values("weight", ascending=False)

        time_df = pd.DataFrame(
            {
                "weight": self.time_weights,
                "rank": self.time_weights.rank(ascending=False, method="min"),
            }
        ).sort_values("weight", ascending=False)

        return {"unit_weights": unit_df, "time_weights": time_df}

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self.treatment_effect is not None

    def summary(self, confidence_level: float = 0.95) -> str:
        """
        Generate a text summary of the analysis results.

        Args:
            confidence_level: Confidence level for the confidence interval
                (default 0.95 for 95% CI). Must be between 0 and 1.

        Returns:
            Formatted summary string
        """
        if not self.is_fitted:
            return "Model not yet fitted. Call fit() first."

        lines = [
            "=" * 50,
            "Synthetic Difference-in-Differences Results",
            "=" * 50,
            f"Treatment Effect (ATT): {self.treatment_effect:.4f}",
        ]

        if self.standard_error is not None and self.treatment_effect is not None:
            lines.append(f"Standard Error:        {self.standard_error:.4f}")

            # Calculate confidence interval
            z = stats.norm.ppf((1 + confidence_level) / 2)
            ci_lower = self.treatment_effect - z * self.standard_error
            ci_upper = self.treatment_effect + z * self.standard_error
            ci_pct = int(confidence_level * 100)
            lines.append(f"{ci_pct}% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

            t_stat = self.treatment_effect / self.standard_error
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            lines.append(f"t-statistic:           {t_stat:.4f}")
            lines.append(f"p-value:               {p_value:.4f}")

        if self.unit_weights is not None:
            lines.append(f"Control units used:    {len(self.unit_weights)}")
        if self.time_weights is not None:
            lines.append(f"Time periods used:     {len(self.time_weights)}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"SyntheticDiffInDiff(outcome='{self.outcome_col}', status={status})"
