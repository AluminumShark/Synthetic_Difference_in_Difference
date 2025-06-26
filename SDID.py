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

# Set up logging - because debugging is half the battle
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticDiffInDiff:
    """
    Your friendly neighborhood SDID estimator! 🎉
    
    Look, causal inference is hard. Traditional diff-in-diff assumes parallel trends,
    synthetic control assumes treatment timing doesn't matter. This combines both
    approaches to give you something more robust. It's like having your cake and eating it too!
    
    The math is from Arkhangelsky et al. (2021) - those folks are way smarter than me.
    This implementation tries to make their brilliant ideas actually usable for us mortals.
    
    What this does:
    - Finds the best weights for control units (so they look like treated units pre-treatment)
    - Finds the best weights for time periods (to balance pre/post comparisons)
    - Combines everything into a weighted diff-in-diff that's hopefully less biased
    
    What you need:
    - Panel data in long format (one row per unit-time combo)
    - Clear treatment and control groups
    - Pre and post treatment periods
    - Some faith that your control units provide a decent counterfactual
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 outcome_col: str, 
                 times_col: str, 
                 units_col: str, 
                 treat_col: str, 
                 post_col: str):
        """
        Set up the SDID analysis - just tell me which column is which!
        
        Args:
            data: Your precious panel data
            outcome_col: The thing you actually care about measuring
            times_col: When stuff happened (quarters, years, whatever)
            units_col: Who you're studying (states, firms, individuals)
            treat_col: Who got the treatment (True/1 for treated folks)
            post_col: When treatment kicked in (True/1 for post-treatment periods)
            
        I'll do some basic sanity checks because nobody likes cryptic error messages
        """
        # First, let's make sure you didn't forget any columns
        required_cols = [outcome_col, times_col, units_col, treat_col, post_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Oops! These columns are missing: {missing_cols}")
        
        # NaNs are the enemy of all econometric analysis
        for col in required_cols:
            if data[col].isna().any():
                raise ValueError(f"Found some NaNs in '{col}' - please clean your data first!")
        
        # Store everything and set up our instance variables
        self.data = data.copy()  # Don't mess with the original data
        self.outcome_col = outcome_col
        self.times_col = times_col
        self.units_col = units_col
        self.treat_col = treat_col
        self.post_col = post_col
        
        # These will get filled in when we run the analysis
        self.unit_weights = None  # Weights for control units
        self.time_weights = None  # Weights for time periods  
        self.merged_data = None   # Data with weights attached
        self.treatment_effect = None  # The money shot
        self.standard_error = None    # How confident should we be?

        # Make sure treatment indicators are actually boolean
        # (You'd be surprised how often this trips people up)
        self.data[self.treat_col] = self.data[self.treat_col].astype(bool)
        self.data[self.post_col] = self.data[self.post_col].astype(bool)
        
        # Run some basic validation
        self._check_data_structure()
        
    def _check_data_structure(self) -> None:
        """
        Just making sure your data makes sense before we dive in
        Because there's nothing worse than running a 2-hour analysis on garbage data
        """
        # Count treated vs control units
        treated_units = self.data[self.data[self.treat_col]][self.units_col].nunique()
        control_units = self.data[~self.data[self.treat_col]][self.units_col].nunique()
        
        if treated_units == 0:
            raise ValueError("No treated units found - did you code the treatment variable correctly?")
        if control_units == 0:
            raise ValueError("No control units found - need some comparison group!")
            
        # Count pre vs post periods
        pre_periods = self.data[~self.data[self.post_col]][self.times_col].nunique()
        post_periods = self.data[self.data[self.post_col]][self.times_col].nunique()
        
        if pre_periods == 0:
            raise ValueError("No pre-treatment periods - we need baseline data!")
        if post_periods == 0:
            raise ValueError("No post-treatment periods - what are we even measuring?")
            
        logger.info(f"Data looks good! {treated_units} treated units, {control_units} controls, "
                   f"{pre_periods} pre-periods, {post_periods} post-periods")

    def _calculate_regularization_penalty(self, penalty_multiplier: float = 1.0) -> float:
        """
        Calculate the regularization parameter (zeta) for unit weights
        
        This prevents overfitting by penalizing extreme weights. The formula comes from 
        the original paper, but honestly the intuition is "don't put all your eggs 
        in one basket when weighting control units"
        
        Args:
            penalty_multiplier: Dial this up if you want more regularization
            
        Returns:
            The penalty parameter (higher = more regularization)
        """
        # How many treated observations do we have post-treatment?
        n_treated_post = self.data.query(f"({self.post_col}) & ({self.treat_col})").shape[0]

        # Calculate volatility from control units in pre-period
        # (This gives us a sense of how noisy the data is)
        control_pre_data = self.data.query(f"(~{self.post_col}) & (~{self.treat_col})")
        
        if control_pre_data.empty:
            raise ValueError("Need some pre-treatment control data to calculate regularization!")
        
        # Standard deviation of first differences (a measure of noise)
        noise_level = (control_pre_data
                      .sort_values(self.times_col)
                      .groupby(self.units_col)[self.outcome_col]
                      .diff()
                      .std())

        # Handle edge cases (because real data is messy)
        if np.isnan(noise_level) or noise_level == 0:
            warnings.warn("Can't estimate noise level from data - using conservative default")
            noise_level = 0.01

        # The magic formula from the paper
        penalty = penalty_multiplier * n_treated_post ** (1/4) * noise_level
        
        return penalty

    def _estimate_unit_weights(self, verbose: bool = False) -> None:
        """
        Find the optimal weights for control units
        
        The goal: weight control units so their pre-treatment average looks as much
        like the treated units as possible. It's like creating a "synthetic treated unit"
        from your controls.
        
        This solves a quadratic programming problem with L2 regularization.
        Don't worry if that sounds scary - the computer does the heavy lifting.
        """
        logger.info("Finding optimal unit weights... (this might take a moment)")
        
        # Calculate how much regularization we need
        regularization_strength = self._calculate_regularization_penalty()
        logger.info(f"Using regularization parameter: {regularization_strength:.4f}")

        # Get pre-treatment data only
        pre_treatment_data = self.data[~self.data[self.post_col]]

        # Create outcome matrix for control units (time x units)
        control_outcomes = (pre_treatment_data[~pre_treatment_data[self.treat_col]]
                           .pivot(index=self.times_col, columns=self.units_col, values=self.outcome_col))

        if control_outcomes.empty:
            raise ValueError("No pre-treatment control data - can't estimate unit weights!")

        # Average outcome for treated units by time period
        treated_avg_by_time = (pre_treatment_data[pre_treatment_data[self.treat_col]]
                              .groupby(self.times_col)[self.outcome_col]
                              .mean())

        if treated_avg_by_time.empty:
            raise ValueError("No pre-treatment treated data - what are we weighting towards?")

        # Find common time periods (sometimes data is unbalanced)
        common_times = control_outcomes.index.intersection(treated_avg_by_time.index)
        
        if len(common_times) == 0:
            raise ValueError("No overlapping time periods between treated and control groups!")

        # Filter to common periods
        control_outcomes = control_outcomes.loc[common_times]
        treated_avg_by_time = treated_avg_by_time.loc[common_times]

        # Set up the optimization problem
        n_control_units = control_outcomes.shape[1]
        n_time_periods = len(common_times)
        
        # Decision variables: weights for each control unit
        unit_weights = cp.Variable(n_control_units, nonneg=True)
        
        # Convert to numpy for the optimization
        Y_control = control_outcomes.values  # T x N_co matrix
        y_treated = treated_avg_by_time.values  # T x 1 vector
        
        # Objective: minimize difference between weighted controls and treated
        # Plus L2 penalty on weights to prevent overfitting
        fit_error = cp.sum_squares(Y_control @ unit_weights - y_treated)
        penalty = regularization_strength * cp.sum_squares(unit_weights)
        objective = cp.Minimize(fit_error + penalty)
        
        # Constraints: weights must be non-negative (we set this above)
        # No need to constrain weights to sum to 1 - let the data decide
        constraints = []
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(verbose=verbose)
            
            if problem.status not in ["infeasible", "unbounded"]:
                estimated_weights = unit_weights.value
                
                # Create a nice series with unit names as index
                weight_series = pd.Series(
                    estimated_weights,
                    index=control_outcomes.columns,
                    name='weight'
                )
                
                # Only keep units with meaningful weights (reduces noise)
                self.unit_weights = weight_series[weight_series > 1e-6]
                
                logger.info(f"Unit weight estimation completed! Using {len(self.unit_weights)} control units")
                
            else:
                raise ValueError(f"Optimization failed with status: {problem.status}")
                
        except Exception as e:
            raise ValueError(f"Something went wrong with unit weight optimization: {str(e)}")

    def _estimate_time_weights(self, verbose: bool = False) -> None:
        """
        Find optimal weights for time periods
        
        This is like the unit weights but for time periods. We want to weight
        time periods so that the pre/post comparison is as balanced as possible.
        
        The intuition: some time periods might be more informative than others
        for identifying the treatment effect.
        """
        logger.info("Finding optimal time weights...")
        
        # Calculate regularization for time weights
        time_regularization = self._calculate_time_regularization()
        logger.info(f"Time regularization parameter: {time_regularization:.4f}")

        # Get pre-treatment data
        pre_data = self.data[~self.data[self.post_col]]

        # Create matrices for pre-treatment period
        treated_outcomes = (pre_data[pre_data[self.treat_col]]
                           .pivot(index=self.units_col, columns=self.times_col, values=self.outcome_col))
        
        control_outcomes = (pre_data[~pre_data[self.treat_col]]
                           .pivot(index=self.units_col, columns=self.times_col, values=self.outcome_col))

        if treated_outcomes.empty or control_outcomes.empty:
            raise ValueError("Need both treated and control data to estimate time weights!")

        # Find common time periods
        common_times = treated_outcomes.columns.intersection(control_outcomes.columns)
        
        if len(common_times) == 0:
            raise ValueError("No common time periods for time weight estimation!")

        # Filter to common periods
        treated_outcomes = treated_outcomes[common_times]
        control_outcomes = control_outcomes[common_times]

        # Calculate average outcomes by time
        treated_time_avg = treated_outcomes.mean(axis=0)
        control_time_avg = control_outcomes.mean(axis=0)

        # Set up optimization
        n_time_periods = len(common_times)
        time_weights = cp.Variable(n_time_periods, nonneg=True)
        
        # Convert to numpy
        y_treated_time = treated_time_avg.values
        y_control_time = control_time_avg.values
        
        # Objective: balance treated and control time trends
        balance_error = cp.sum_squares(time_weights.T @ (y_treated_time - y_control_time))
        penalty = time_regularization * cp.sum_squares(time_weights)
        objective = cp.Minimize(balance_error + penalty)
        
        # Solve
        problem = cp.Problem(objective)
        
        try:
            problem.solve(verbose=verbose)
            
            if problem.status not in ["infeasible", "unbounded"]:
                estimated_time_weights = time_weights.value
                
                # Create series with time period names
                time_weight_series = pd.Series(
                    estimated_time_weights,
                    index=common_times,
                    name='time_weight'
                )
                
                # Keep meaningful weights only
                self.time_weights = time_weight_series[time_weight_series > 1e-6]
                
                logger.info(f"Time weight estimation done! Using {len(self.time_weights)} time periods")
                
            else:
                raise ValueError(f"Time weight optimization failed: {problem.status}")
                
        except Exception as e:
            raise ValueError(f"Time weight optimization error: {str(e)}")

    def _calculate_time_regularization(self) -> float:
        """
        Calculate regularization parameter for time weights
        
        Similar logic to unit weights but adapted for the time dimension
        """
        # Use volatility across units as a guide
        pre_data = self.data[~self.data[self.post_col]]
        
        if pre_data.empty:
            return 0.01  # Conservative default
        
        # Standard deviation across units within each time period
        time_volatility = (pre_data.groupby(self.times_col)[self.outcome_col]
                          .std()
                          .mean())
        
        if np.isnan(time_volatility) or time_volatility == 0:
            time_volatility = 0.01
            
        # Scale by number of time periods (more periods = less regularization needed)
        n_time_periods = pre_data[self.times_col].nunique()
        regularization = time_volatility * (n_time_periods ** (-1/4))
        
        return regularization

    def _merge_weights_with_data(self) -> None:
        """
        Combine the estimated weights with our original data
        
        This creates a single dataset with both unit and time weights attached.
        Makes the final regression much cleaner.
        """
        logger.info("Merging weights with data...")
        
        if self.unit_weights is None or self.time_weights is None:
            raise ValueError("Need to estimate weights first! Call fit_unit_weights() and fit_time_weights()")
        
        # Start with a copy of the original data
        working_data = self.data.copy()
        
        # Add unit weights (only for control units)
        working_data['unit_weight'] = 0.0  # Default weight
        control_mask = ~working_data[self.treat_col]
        
        for unit, weight in self.unit_weights.items():
            unit_mask = working_data[self.units_col] == unit
            working_data.loc[control_mask & unit_mask, 'unit_weight'] = weight
        
        # Treated units get weight of 1 (they represent themselves)
        working_data.loc[~control_mask, 'unit_weight'] = 1.0
        
        # Add time weights
        working_data['time_weight'] = 0.0  # Default
        
        for time_period, weight in self.time_weights.items():
            time_mask = working_data[self.times_col] == time_period
            working_data.loc[time_mask, 'time_weight'] = weight
        
        # Combine unit and time weights
        working_data['combined_weight'] = working_data['unit_weight'] * working_data['time_weight']
        
        # Store the result
        self.merged_data = working_data
        
        logger.info("Weights successfully merged with data")

    def _run_weighted_regression(self, verbose: bool = False) -> None:
        """
        Run the final weighted difference-in-differences regression
        
        This is where the magic happens! We use the estimated weights to run
        a standard diff-in-diff regression that should be less biased than
        the unweighted version.
        """
        logger.info("Running weighted diff-in-diff regression...")
        
        if self.merged_data is None:
            raise ValueError("Need merged data! Call _merge_weights_with_data() first")
        
        # Create interaction term for treatment effect
        self.merged_data['treat_post'] = (self.merged_data[self.treat_col] & 
                                         self.merged_data[self.post_col])
        
        # Only use observations with positive weights
        regression_data = self.merged_data[self.merged_data['combined_weight'] > 0].copy()
        
        if regression_data.empty:
            raise ValueError("No observations with positive weights - something went wrong!")
        
        # Set up the regression formula
        # This is a standard diff-in-diff specification
        formula = (f"{self.outcome_col} ~ {self.treat_col} + {self.post_col} + treat_post")
        
        try:
            # Run weighted OLS
            model = smf.wls(formula, 
                           data=regression_data, 
                           weights=regression_data['combined_weight'])
            results = model.fit()
            
            # Extract the treatment effect (coefficient on the interaction term)
            self.treatment_effect = results.params['treat_post[T.True]']
            
            if verbose:
                print("\n" + "="*50)
                print("SDID REGRESSION RESULTS")
                print("="*50)
                print(results.summary())
                print("="*50)
                
            logger.info(f"Treatment effect estimated: {self.treatment_effect:.4f}")
            
        except Exception as e:
            raise ValueError(f"Weighted regression failed: {str(e)}")

    def get_treatment_effect(self) -> float:
        """
        Get the estimated treatment effect
        
        Returns:
            The SDID estimate of the average treatment effect
        """
        if self.treatment_effect is None:
            raise ValueError("Haven't run the analysis yet! Call run_analysis() first")
        
        return self.treatment_effect

    def run_full_analysis(self, verbose: bool = False) -> float:
        """
        Run the complete SDID analysis from start to finish
        
        This is the main method you'll probably want to use. It does everything:
        1. Estimates unit weights
        2. Estimates time weights  
        3. Merges weights with data
        4. Runs weighted regression
        5. Returns treatment effect
        
        Args:
            verbose: Print detailed output?
            
        Returns:
            Estimated treatment effect
        """
        logger.info("Starting complete SDID analysis...")
        
        # Step 1: Unit weights
        self._estimate_unit_weights(verbose=verbose)
        
        # Step 2: Time weights
        self._estimate_time_weights(verbose=verbose)
        
        # Step 3: Merge everything
        self._merge_weights_with_data()
        
        # Step 4: Final regression
        self._run_weighted_regression(verbose=verbose)
        
        logger.info("SDID analysis complete! 🎉")
        
        return self.treatment_effect

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
