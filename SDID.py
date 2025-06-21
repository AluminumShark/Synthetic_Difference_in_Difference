import numpy as np
import pandas as pd
import cvxpy as cp
import statsmodels.formula.api as smf
from functools import partial
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Dict, Tuple, Any
import warnings
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticDiffInDiff:
    """
    Synthetic Difference-in-Differences (SDID) estimator for causal inference.
    
    This class implements the SDID method proposed by Arkhangelsky et al. (2021),
    which combines the advantages of synthetic control and difference-in-differences
    approaches for estimating treatment effects in panel data settings.
    
    Attributes:
        data (pd.DataFrame): The input panel data
        outcome_col (str): Name of the outcome variable column
        times_col (str): Name of the time variable column
        units_col (str): Name of the unit identifier column
        treat_col (str): Name of the treatment indicator column
        post_col (str): Name of the post-treatment period indicator column
        unit_weights (pd.Series): Estimated weights for control units
        time_weights (pd.Series): Estimated weights for time periods
        merged_data (pd.DataFrame): Data with weights merged
        treatment_effect (float): Estimated treatment effect
        standard_error (float): Standard error of the treatment effect
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 outcome_col: str, 
                 times_col: str, 
                 units_col: str, 
                 treat_col: str, 
                 post_col: str):
        """
        Initialize the SyntheticDiffInDiff object with the dataset and relevant column names.
        
        Args:
            data: Panel data in long format
            outcome_col: Name of the outcome variable column
            times_col: Name of the time variable column
            units_col: Name of the unit identifier column
            treat_col: Name of the treatment indicator column (True/1 for treated units)
            post_col: Name of the post-treatment period indicator column (True/1 for post periods)
            
        Raises:
            ValueError: If required columns are missing from the data
            ValueError: If data contains NaN values in critical columns
        """
        # Validate input columns
        required_columns = [outcome_col, times_col, units_col, treat_col, post_col]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"The following required columns are missing from the data: {missing_columns}")
        
        # Check for NaN values in critical columns
        for col in required_columns:
            if data[col].isna().any():
                raise ValueError(f"Column '{col}' contains NaN values. Please handle missing data before analysis.")
        
        self.data = data.copy()
        self.outcome_col = outcome_col
        self.times_col = times_col
        self.units_col = units_col
        self.treat_col = treat_col
        self.post_col = post_col
        self.unit_weights: Optional[pd.Series] = None
        self.time_weights: Optional[pd.Series] = None
        self.merged_data: Optional[pd.DataFrame] = None
        self.treatment_effect: Optional[float] = None
        self.standard_error: Optional[float] = None

        # Ensure that treat_col and post_col are boolean
        self.data[self.treat_col] = self.data[self.treat_col].astype(bool)
        self.data[self.post_col] = self.data[self.post_col].astype(bool)
        
        # Validate data structure
        self._validate_data_structure()
        
    def _validate_data_structure(self) -> None:
        """Validate the basic structure of the input data."""
        # Check if there are both treated and control units
        n_treated = self.data[self.data[self.treat_col]][self.units_col].nunique()
        n_control = self.data[~self.data[self.treat_col]][self.units_col].nunique()
        
        if n_treated == 0:
            raise ValueError("No treated units found in the data.")
        if n_control == 0:
            raise ValueError("No control units found in the data.")
            
        # Check if there are both pre and post periods
        n_pre = self.data[~self.data[self.post_col]][self.times_col].nunique()
        n_post = self.data[self.data[self.post_col]][self.times_col].nunique()
        
        if n_pre == 0:
            raise ValueError("No pre-treatment periods found in the data.")
        if n_post == 0:
            raise ValueError("No post-treatment periods found in the data.")
            
        logger.info(f"Data validated: {n_treated} treated units, {n_control} control units, "
                   f"{n_pre} pre-treatment periods, {n_post} post-treatment periods")

    def calculate_regularization(self, penalty_factor: float = 1.0) -> float:
        """
        Calculate the regularization parameter zeta for the L2 penalty on unit weights.
        
        Args:
            penalty_factor: Multiplicative factor for the regularization parameter (default: 1.0)
            
        Returns:
            float: The calculated regularization parameter zeta
            
        Raises:
            ValueError: If standard deviation cannot be calculated
        """
        # Number of treated units in post-treatment period
        n_treated_post = self.data.query(f"({self.post_col}) & ({self.treat_col})").shape[0]

        # Calculate the standard deviation of the first differences for control units
        control_pre_data = self.data.query(f"(~{self.post_col}) & (~{self.treat_col})")
        
        if control_pre_data.empty:
            raise ValueError("No pre-treatment control data available for calculating regularization.")
        
        first_diff_std = (control_pre_data
                          .sort_values(self.times_col)
                          .groupby(self.units_col)[self.outcome_col]
                          .diff()
                          .std())

        # Handle cases where first_diff_std is NaN or zero
        if np.isnan(first_diff_std) or first_diff_std == 0:
            # Use a small default value
            warnings.warn("Standard deviation of first differences is NaN or zero. Using default value.")
            first_diff_std = 0.01

        # Calculate the regularization parameter zeta
        zeta = penalty_factor * n_treated_post ** (1 / 4) * first_diff_std

        return zeta

    def fit_unit_weights(self, verbose: bool = False) -> None:
        """
        Estimate unit weights (w_i) to adjust for unit heterogeneity.
        
        This method solves a regularized quadratic programming problem to find weights
        that make the weighted average of control units similar to treated units
        in the pre-treatment period.
        
        Args:
            verbose: Whether to print optimization details
            
        Raises:
            ValueError: If optimization fails or data is insufficient
        """
        logger.info("Starting unit weight estimation...")
        
        # Calculate the regularization parameter zeta
        zeta = self.calculate_regularization()
        logger.info(f"Regularization parameter zeta: {zeta:.4f}")

        # Extract pre-treatment data
        pre_data = self.data[~self.data[self.post_col]]

        # Construct the pre-treatment control group outcome matrix
        y_pre_control = (pre_data[~pre_data[self.treat_col]]
                         .pivot(index=self.times_col, columns=self.units_col, values=self.outcome_col))

        # Check if y_pre_control is empty
        if y_pre_control.empty:
            raise ValueError("No pre-treatment control data available to fit unit weights.")

        # Calculate the average outcome for the treatment group in the pre-treatment period
        y_pre_treat_mean = (pre_data[pre_data[self.treat_col]]
                            .groupby(self.times_col)[self.outcome_col]
                            .mean())

        # Check if y_pre_treat_mean is empty
        if y_pre_treat_mean.empty:
            raise ValueError("No pre-treatment treatment group data available to fit unit weights.")

        # Find common time periods
        common_times = y_pre_control.index.intersection(y_pre_treat_mean.index)

        if len(common_times) == 0:
            raise ValueError("No common time periods between pre-treatment control and treated groups.")

        # Filter data to only include common times
        y_pre_control = y_pre_control.loc[common_times]
        y_pre_treat_mean = y_pre_treat_mean.loc[common_times]

        # Add a column of ones to the left of the matrix as the intercept term
        T_pre = y_pre_control.shape[0]
        X = np.concatenate([np.ones((T_pre, 1)), y_pre_control.values], axis=1)

        # Define the optimization variable (unit weights, including intercept)
        w = cp.Variable(X.shape[1])

        # Define the objective function
        objective = cp.Minimize(
            cp.sum_squares(X @ w - y_pre_treat_mean.values) +
            T_pre * zeta ** 2 * cp.sum_squares(w[1:])
        )

        # Define constraints
        constraints = [
            cp.sum(w[1:]) == 1,
            w[1:] >= 0
        ]

        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(verbose=verbose)
        except Exception as e:
            raise ValueError(f"Optimization for unit weights failed: {str(e)}")

        # Check if the problem was solved successfully
        if w.value is None or problem.status not in ['optimal', 'optimal_inaccurate']:
            raise ValueError(f"Optimization for unit weights did not converge. Status: {problem.status}")

        # Extract unit weights (excluding intercept)
        self.unit_weights = pd.Series(
            w.value[1:],  # Exclude intercept
            name="unit_weights",
            index=y_pre_control.columns  # Units as index
        )
        
        # Log summary statistics
        logger.info(f"Unit weights estimated successfully. Mean: {self.unit_weights.mean():.4f}, "
                   f"Std: {self.unit_weights.std():.4f}, Non-zero: {(self.unit_weights > 1e-6).sum()}")

    def fit_time_weights(self, verbose: bool = False) -> None:
        """
        Estimate time weights (lambda_t) to adjust for time-specific effects.
        
        This method solves a regularized quadratic programming problem to find weights
        that make the weighted average of post-treatment periods similar to
        pre-treatment periods for control units.
        
        Args:
            verbose: Whether to print optimization details
            
        Raises:
            ValueError: If optimization fails or data is insufficient
        """
        logger.info("Starting time weight estimation...")
        
        # Extract post-treatment data for control units
        post_data = self.data[self.data[self.post_col]]
        y_post_control = (post_data[~post_data[self.treat_col]]
                          .pivot(index=self.units_col, columns=self.times_col, values=self.outcome_col))

        if y_post_control.empty:
            raise ValueError("No post-treatment control data available to fit time weights.")

        # Extract pre-treatment data for control units
        pre_data = self.data[~self.data[self.post_col]]
        y_pre_control_mean = (pre_data[~pre_data[self.treat_col]]
                             .groupby(self.units_col)[self.outcome_col]
                             .mean())

        # Find common units
        common_units = y_post_control.index.intersection(y_pre_control_mean.index)
        if len(common_units) == 0:
            raise ValueError("No common units between post-treatment control and pre-treatment control.")

        # Filter data to only include common units
        y_post_control = y_post_control.loc[common_units]
        y_pre_control_mean = y_pre_control_mean.loc[common_units]

        # Calculate regularization parameter
        n_control_post = post_data[~post_data[self.treat_col]].shape[0]
        first_diff_std_time = self._calculate_time_first_diff_std()
        omega = n_control_post ** (1 / 4) * first_diff_std_time
        logger.info(f"Regularization parameter omega: {omega:.4f}")

        # Add intercept term
        N_control = y_post_control.shape[0]
        Z = np.concatenate([np.ones((N_control, 1)), y_post_control.values], axis=1)

        # Define optimization problem
        mu = cp.Variable(Z.shape[1])
        objective = cp.Minimize(
            cp.sum_squares(Z @ mu - y_pre_control_mean.values) +
            N_control * omega ** 2 * cp.sum_squares(mu[1:])
        )
        constraints = [
            cp.sum(mu[1:]) == 1,
            mu[1:] >= 0
        ]

        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(verbose=verbose)
        except Exception as e:
            raise ValueError(f"Optimization for time weights failed: {str(e)}")

        # Check if the problem was solved successfully
        if mu.value is None or problem.status not in ['optimal', 'optimal_inaccurate']:
            raise ValueError(f"Optimization for time weights did not converge. Status: {problem.status}")

        # Extract time weights (excluding intercept)
        self.time_weights = pd.Series(
            mu.value[1:],
            name="time_weights",
            index=y_post_control.columns
        )
        
        logger.info(f"Time weights estimated successfully. Mean: {self.time_weights.mean():.4f}, "
                   f"Std: {self.time_weights.std():.4f}, Non-zero: {(self.time_weights > 1e-6).sum()}")
        
    def _calculate_time_first_diff_std(self) -> float:
        """Calculate standard deviation of first differences for time regularization."""
        try:
            first_diff_std_time = (self.data
                                  .query(f"(~{self.post_col}) & (~{self.treat_col})")
                                  .sort_values(self.units_col)
                                  .groupby(self.times_col)[self.outcome_col]
                                  .diff()
                                  .std())
            
            if np.isnan(first_diff_std_time) or first_diff_std_time == 0:
                warnings.warn("Using default regularization for time weights.")
                return 1.0
                
            return first_diff_std_time
        except Exception:
            warnings.warn("Failed to calculate time first differences. Using default value.")
            return 1.0

    def join_weights(self) -> None:
        """
        Merge unit weights and time weights into the dataset and calculate combined weights.
        
        Missing values are filled with uniform weights (1 / number of unique units/times).
        
        Raises:
            ValueError: If weights have not been fitted
        """
        if self.unit_weights is None or self.time_weights is None:
            raise ValueError("Unit weights and time weights must be fitted before joining.")

        # Merge weights into the dataset
        merged_data = (self.data
                       .set_index([self.times_col, self.units_col])
                       .join(self.time_weights)
                       .join(self.unit_weights)
                       .reset_index())

        # Fill missing weights with uniform weights
        num_unique_times = self.data[self.times_col].nunique()
        num_unique_units = self.data[self.units_col].nunique()

        merged_data[self.time_weights.name] = merged_data[self.time_weights.name].fillna(1 / num_unique_times)
        merged_data[self.unit_weights.name] = merged_data[self.unit_weights.name].fillna(1 / num_unique_units)

        # Calculate combined weights
        merged_data["weights"] = (merged_data[self.time_weights.name] * 
                                 merged_data[self.unit_weights.name]).round(10)

        # Convert boolean columns to int for regression
        merged_data = merged_data.astype({self.treat_col: int, self.post_col: int})

        self.merged_data = merged_data
        
        logger.info("Weights successfully joined to data")

    def synthetic_diff_in_diff_analysis(self, verbose: bool = False) -> None:
        """
        Implement the Synthetic Difference-in-Differences (SDID) method to estimate treatment effects.
        
        This method performs the following steps:
        1. Estimates unit weights
        2. Estimates time weights
        3. Merges weights with data
        4. Runs weighted regression to estimate treatment effect
        
        Args:
            verbose: Whether to print detailed information
        """
        logger.info("Starting SDID analysis...")
        
        # Estimate unit weights
        self.fit_unit_weights(verbose=verbose)

        # Estimate time weights
        self.fit_time_weights(verbose=verbose)

        # Merge weights and calculate combined weights
        self.join_weights()

        # Construct the regression formula
        formula = f"{self.outcome_col} ~ {self.post_col} * {self.treat_col}"

        # Perform Weighted Least Squares regression (WLS)
        did_model = smf.wls(
            formula,
            data=self.merged_data,
            weights=self.merged_data["weights"] + 1e-10  # Prevent weights from being zero
        ).fit()

        # Extract the treatment effect (coefficient of the interaction term)
        interaction_term = f"{self.post_col}:{self.treat_col}"
        if interaction_term in did_model.params:
            self.treatment_effect = did_model.params[interaction_term]
            self.did_model = did_model  # Store the model for further analysis
            logger.info(f"Treatment effect estimated: {self.treatment_effect:.4f}")
        else:
            raise KeyError(f"Interaction term '{interaction_term}' not found in the model parameters.")

    def get_treatment_effect(self) -> float:
        """
        Retrieve the estimated treatment effect.
        
        Returns:
            float: The estimated treatment effect
            
        Raises:
            ValueError: If treatment effect has not been estimated
        """
        if self.treatment_effect is None:
            raise ValueError("Treatment effect has not been estimated. Call synthetic_diff_in_diff_analysis() first.")
        return self.treatment_effect

    def run_analysis(self, verbose: bool = False) -> float:
        """
        Execute the full SDID analysis pipeline to estimate the treatment effect.
        
        Args:
            verbose: Whether to print detailed information
            
        Returns:
            float: The estimated treatment effect
        """
        self.synthetic_diff_in_diff_analysis(verbose=verbose)
        return self.get_treatment_effect()

    def run_event_study(self, times: List[Union[int, float, str]]) -> pd.Series:
        """
        Run the SDID analysis for each specified time period to create an event study plot.
        
        This method estimates treatment effects at different time points by
        treating each time as the "post" period.
        
        Args:
            times: List of time periods to analyze
            
        Returns:
            pd.Series: Treatment effects indexed by time
        """
        logger.info(f"Running event study for {len(times)} time periods...")
        effects_dict = {}
        
        for time in times:
            # Filter data: include observations not in post-treatment or in the current time
            filtered_data = self.data[(~self.data[self.post_col]) | 
                                     (self.data[self.times_col] == time)].copy()

            # Check if filtered_data has both treated and control groups
            treated_count = filtered_data[filtered_data[self.treat_col]].shape[0]
            control_count = filtered_data[~filtered_data[self.treat_col]].shape[0]
            
            if treated_count == 0 or control_count == 0:
                logger.warning(f"Time {time}: Insufficient treated or control observations. Skipping.")
                effects_dict[time] = np.nan
                continue

            # Initialize a new SDID instance with the filtered data
            sdid_instance = SyntheticDiffInDiff(
                data=filtered_data,
                outcome_col=self.outcome_col,
                times_col=self.times_col,
                units_col=self.units_col,
                treat_col=self.treat_col,
                post_col=self.post_col
            )

            try:
                # Run the analysis and get the treatment effect
                effect = sdid_instance.run_analysis()
                effects_dict[time] = effect
            except Exception as e:
                # Handle exceptions and assign NaN
                logger.warning(f"Time {time}: Analysis failed with error: {e}")
                effects_dict[time] = np.nan

        # Convert the dictionary to a Pandas Series
        effects = pd.Series(effects_dict, name="treatment_effect")
        return effects

    def make_random_placebo(self) -> pd.DataFrame:
        """
        Create a placebo dataset by randomly selecting a control unit and marking it as treated.
        
        This is used for placebo tests to estimate standard errors.
        
        Returns:
            pd.DataFrame: Placebo dataset with a randomly selected control unit marked as treated
            
        Raises:
            ValueError: If no control units are available
        """
        # Extract control group data
        control_data = self.data[~self.data[self.treat_col]]
        # Get unique control units
        control_units = control_data[self.units_col].unique()
        if len(control_units) == 0:
            raise ValueError("No control units available to create a placebo.")
        # Randomly select a control unit
        placebo_unit = np.random.choice(control_units)
        # Mark the selected unit as treated throughout
        placebo_data = self.data.copy()
        mask = (placebo_data[self.units_col] == placebo_unit)
        placebo_data.loc[mask, self.treat_col] = True
        # Ensure treat_col is boolean
        placebo_data[self.treat_col] = placebo_data[self.treat_col].astype(bool)
        return placebo_data

    def estimate_se(self, bootstrap_rounds: int = 400, seed: int = 0, n_jobs: int = 1) -> None:
        """
        Estimate the standard error of the treatment effect using placebo tests.
        
        This method runs multiple placebo tests where control units are randomly
        marked as treated, and computes the standard deviation of the resulting
        placebo effects.
        
        Args:
            bootstrap_rounds: Number of placebo tests to run
            seed: Random seed for reproducibility
            n_jobs: Number of parallel jobs to run
        """
        logger.info(f"Estimating standard error with {bootstrap_rounds} placebo tests...")
        np.random.seed(seed)
        
        sdid_fn = partial(
            self._synthetic_diff_in_diff_placebo,
            outcome_col=self.outcome_col,
            times_col=self.times_col,
            units_col=self.units_col,
            treat_col=self.treat_col,
            post_col=self.post_col
        )

        effects = Parallel(n_jobs=n_jobs)(
            delayed(sdid_fn)(self.make_random_placebo())
            for _ in range(bootstrap_rounds)
        )

        # Remove NaN values and compute standard error
        valid_effects = [e for e in effects if not np.isnan(e)]
        if len(valid_effects) < 10:
            warnings.warn(f"Only {len(valid_effects)} valid placebo effects. Standard error may be unreliable.")
        
        self.standard_error = np.std(valid_effects, ddof=1) if valid_effects else np.nan
        logger.info(f"Standard error estimated: {self.standard_error:.4f}")

    def _synthetic_diff_in_diff_placebo(self, 
                                       placebo_data: pd.DataFrame, 
                                       outcome_col: str, 
                                       times_col: str, 
                                       units_col: str, 
                                       treat_col: str, 
                                       post_col: str) -> float:
        """
        Helper function to compute the SDID treatment effect on placebo data.
        
        Args:
            placebo_data: Placebo dataset
            outcome_col: Name of outcome column
            times_col: Name of time column
            units_col: Name of unit column
            treat_col: Name of treatment column
            post_col: Name of post-treatment column
            
        Returns:
            float: Placebo treatment effect or NaN if estimation fails
        """
        # Initialize a new SDID instance with placebo data
        sdid_placebo = SyntheticDiffInDiff(
            data=placebo_data,
            outcome_col=outcome_col,
            times_col=times_col,
            units_col=units_col,
            treat_col=treat_col,
            post_col=post_col
        )
        try:
            effect = sdid_placebo.run_analysis()
            return effect
        except Exception:
            return np.nan

    def make_figure(self, 
                   times: List[Union[int, float, str]], 
                   bootstrap_rounds: int = 400, 
                   seed: int = 0, 
                   n_jobs: int = 1,
                   confidence_level: float = 0.90,
                   figure_size: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot the treatment effect over time with confidence intervals.
        
        Args:
            times: List of time periods to analyze
            bootstrap_rounds: Number of bootstrap rounds for standard error estimation
            seed: Random seed for reproducibility
            n_jobs: Number of parallel jobs
            confidence_level: Confidence level for intervals (default: 0.90)
            figure_size: Figure size as (width, height) tuple
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        logger.info("Creating event study figure...")
        
        # Run event study to get treatment effects over time
        effects = self.run_event_study(times)

        # Estimate standard errors for each time point
        standard_errors = {}
        for time in times:
            # Filter data
            filtered_data = self.data[(~self.data[self.post_col]) | 
                                     (self.data[self.times_col] == time)].copy()
            
            # Initialize SDID instance
            sdid_instance = SyntheticDiffInDiff(
                data=filtered_data,
                outcome_col=self.outcome_col,
                times_col=self.times_col,
                units_col=self.units_col,
                treat_col=self.treat_col,
                post_col=self.post_col
            )
            try:
                # Estimate standard error
                sdid_instance.estimate_se(
                    bootstrap_rounds=bootstrap_rounds,
                    seed=seed,
                    n_jobs=n_jobs
                )
                standard_errors[time] = sdid_instance.standard_error
            except Exception as e:
                logger.warning(f"Standard error estimation at time {time} failed: {e}")
                standard_errors[time] = np.nan

        # Convert standard errors to Series
        standard_errors = pd.Series(standard_errors)
        
        # Calculate critical value for confidence intervals
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Plotting
        fig, ax = plt.subplots(figsize=figure_size)
        ax.plot(effects.index, effects.values, marker='o', linewidth=2, 
                markersize=8, label='Treatment Effect')
        
        # Add confidence intervals
        ci_lower = effects - z_score * standard_errors
        ci_upper = effects + z_score * standard_errors
        ax.fill_between(effects.index, ci_lower, ci_upper, 
                       color='skyblue', alpha=0.4, 
                       label=f'{int(confidence_level*100)}% Confidence Interval')
        
        # Add reference line at zero
        ax.axhline(0, color='grey', linestyle='--', alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Treatment Effect', fontsize=12)
        ax.set_title('Synthetic DiD Treatment Effect Over Time', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if necessary
        if len(times) > 10:
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        return fig
    
    def get_weights_summary(self) -> Dict[str, pd.DataFrame]:
        """
        Get a summary of the estimated weights.
        
        Returns:
            Dict containing DataFrames with unit and time weights information
        """
        if self.unit_weights is None or self.time_weights is None:
            raise ValueError("Weights have not been estimated yet.")
        
        # Unit weights summary
        unit_summary = pd.DataFrame({
            'weight': self.unit_weights,
            'rank': self.unit_weights.rank(ascending=False, method='min')
        }).sort_values('weight', ascending=False)
        
        # Time weights summary  
        time_summary = pd.DataFrame({
            'weight': self.time_weights,
            'rank': self.time_weights.rank(ascending=False, method='min')
        }).sort_values('weight', ascending=False)
        
        return {
            'unit_weights': unit_summary,
            'time_weights': time_summary
        }
