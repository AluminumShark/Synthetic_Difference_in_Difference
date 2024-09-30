import numpy as np
import pandas as pd
import cvxpy as cp
import statsmodels.formula.api as smf
from functools import partial
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

class SyntheticDiffInDiff:
    def __init__(self, data, outcome_col, times_col, units_col, treat_col, post_col):
        """
        Initialize the SyntheticDiffInDiff object with the dataset and relevant column names.
        """
        required_columns = [outcome_col, times_col, units_col, treat_col, post_col]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"The following required columns are missing from the data: {missing_columns}")
        
        self.data = data.copy()
        self.outcome_col = outcome_col
        self.times_col = times_col
        self.units_col = units_col
        self.treat_col = treat_col
        self.post_col = post_col
        self.unit_weights = None
        self.time_weights = None
        self.merged_data = None
        self.treatment_effect = None
        self.standard_error = None

        # Ensure that treat_col and post_col are boolean
        self.data[self.treat_col] = self.data[self.treat_col].astype(bool)
        self.data[self.post_col] = self.data[self.post_col].astype(bool)

    def fit_unit_weights(self):
        """
        Estimate unit weights (w_i) to adjust for unit heterogeneity.
        """
        # Calculate the regularization parameter zeta
        zeta = self.calculate_regularization()

        # Extract pre-treatment data
        pre_data = self.data[~self.data[self.post_col]]

        # Construct the pre-treatment control group outcome matrix
        y_pre_control = (pre_data[~pre_data[self.treat_col]]
                        .pivot(index=self.times_col, columns=self.units_col, values=self.outcome_col))

        # Calculate the average outcome for the treatment group in the pre-treatment period
        y_pre_treat_mean = (pre_data[pre_data[self.treat_col]]
                            .groupby(self.times_col)[self.outcome_col]
                            .mean())

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
        problem.solve(verbose=False)

        # Check if the problem was solved successfully
        if w.value is None:
            raise ValueError("Optimization for unit weights did not converge.")

        # Extract unit weights (excluding intercept)
        self.unit_weights = pd.Series(
            w.value[1:],  # Exclude intercept
            name="unit_weights",
            index=y_pre_control.columns  # Units as index
        )


    def calculate_regularization(self):
        """
        Calculate the regularization parameter zeta for the L2 penalty on unit weights.
        """
        # Corrected the query here using & and parentheses
        n_treated_post = self.data.query(f"({self.post_col}) & ({self.treat_col})").shape[0]

        # Calculate the standard deviation of the first differences
        first_diff_std = (self.data.query(f"(~{self.post_col}) & (~{self.treat_col})")
                          .sort_values(self.times_col)
                          .groupby(self.units_col)[self.outcome_col]
                          .diff()
                          .std())

        # Handle cases where first_diff_std is NaN
        if np.isnan(first_diff_std):
            raise ValueError("Standard deviation of first differences is NaN. Check your data for sufficient variation.")

        # Calculate the regularization parameter zeta
        zeta = n_treated_post ** (1 / 4) * first_diff_std

        return zeta

    def fit_unit_weights(self):
        """
        Estimate unit weights (w_i) to adjust for unit heterogeneity.
        """
        # Calculate the regularization parameter zeta
        zeta = self.calculate_regularization()

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
        problem.solve(verbose=False)

        # Check if the problem was solved successfully
        if w.value is None:
            raise ValueError("Optimization for unit weights did not converge.")

        # Extract unit weights (excluding intercept)
        self.unit_weights = pd.Series(
            w.value[1:],  # Exclude intercept
            name="unit_weights",
            index=y_pre_control.columns  # Units as index
        )

    def join_weights(self):
        """
        Merge unit weights and time weights into the dataset and calculate combined weights.
        Missing values are filled with the average weights (1 / number of unique units/times).
        """
        if self.unit_weights is None or self.time_weights is None:
            raise ValueError("Unit weights and time weights must be fitted before joining.")

        # Merge weights into the dataset
        merged_data = (self.data
                       .set_index([self.times_col, self.units_col])
                       .join(self.time_weights)
                        .join(self.unit_weights)
                       .reset_index())

        num_unique_times = self.data[self.times_col].nunique()
        num_unique_units = self.data[self.units_col].nunique()

        merged_data[self.time_weights.name] = merged_data[self.time_weights.name].fillna(1 / num_unique_times)
        merged_data[self.unit_weights.name] = merged_data[self.unit_weights.name].fillna(1 / num_unique_units)

        merged_data["weights"] = (merged_data[self.time_weights.name] * merged_data[self.unit_weights.name]).round(10)

        merged_data = merged_data.astype({self.treat_col: int, self.post_col: int})

        self.merged_data = merged_data



    def synthetic_diff_in_diff_analysis(self):
        """
        Implement the Synthetic Difference-in-Differences (SDID) method to estimate treatment effects.
        """
        # Estimate unit weights
        self.fit_unit_weights()

        # Estimate time weights
        self.fit_time_weights()

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
        else:
            raise KeyError(f"Interaction term '{interaction_term}' not found in the model parameters.")

    def get_treatment_effect(self):
        """
        Retrieve the estimated treatment effect.
        """
        if self.treatment_effect is None:
            raise ValueError("Treatment effect has not been estimated. Call synthetic_diff_in_diff_analysis() first.")
        return self.treatment_effect

    def run_analysis(self):
        """
        Execute the full SDID analysis pipeline to estimate the treatment effect.
        """
        self.synthetic_diff_in_diff_analysis()
        return self.get_treatment_effect()

    def run_event_study(self, times):
        """
        Run the SDID analysis for each specified time by filtering the data accordingly.
        """
        effects_dict = {}
        for time in times:
            # Filter data: include observations not in post-treatment or in the current time
            filtered_data = self.data[(~self.data[self.post_col]) | (self.data[self.times_col] == time)].copy()

            # Check if filtered_data has both treated and control groups
            treated_count = filtered_data[filtered_data[self.treat_col]].shape[0]
            control_count = filtered_data[~filtered_data[self.treat_col]].shape[0]
            if treated_count == 0 or control_count == 0:
                print(f"Time {time}: Insufficient treated or control observations. Skipping.")
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
                print(f"Time {time}: Analysis failed with error: {e}")
                effects_dict[time] = np.nan

        # Convert the dictionary to a Pandas Series
        effects = pd.Series(effects_dict, name="treatment_effect")
        return effects

    def make_random_placebo(self):
        """
        Create a placebo dataset by randomly selecting a control unit and marking it as treated throughout.
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

    def estimate_se(self, bootstrap_rounds=400, seed=0, n_jobs=1):
        """
        Estimate the standard error of the treatment effect using placebo tests.
        """
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

        # Compute standard error
        self.standard_error = np.nanstd(effects, ddof=1)  # Sample standard deviation, ignoring NaNs

    def _synthetic_diff_in_diff_placebo(self, placebo_data, outcome_col, times_col, units_col, treat_col, post_col):
        """
        Helper function to compute the SDID treatment effect on placebo data.
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
        effect = sdid_placebo.run_analysis()
        return effect

    def make_figure(self, times, bootstrap_rounds=400, seed=0, n_jobs=1):
        """
        Plot the treatment effect over time with confidence intervals.
        """
        # Run event study to get treatment effects over time
        effects = self.run_event_study(times)

        # Estimate standard errors for each time point
        standard_errors = {}
        for time in times:
            # Filter data: include observations not in post-treatment or in the current time
            filtered_data = self.data[(~self.data[self.post_col]) | (self.data[self.times_col] == time)].copy()
            # Initialize SDID instance
            sdid_instance = SyntheticDiffInDiff(
                data=filtered_data,
                outcome_col=self.outcome_col,
                times_col=self.times_col,
                units_col=self.units_col,
                treat_col=self.treat_col,
                post_col=self.post_col
            )
            # Estimate standard error
            sdid_instance.estimate_se(
                bootstrap_rounds=bootstrap_rounds,
                seed=seed,
                n_jobs=n_jobs
            )
            standard_errors[time] = sdid_instance.standard_error

        # Convert standard errors to Series
        standard_errors = pd.Series(standard_errors)
        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(effects.index, effects.values, marker='o', label='Treatment Effect')
        ci_lower = effects - 1.65 * standard_errors
        ci_upper = effects + 1.65 * standard_errors
        ax.fill_between(effects.index, ci_lower, ci_upper, color='skyblue', alpha=0.4, label='90% Confidence Interval')
        ax.axhline(0, color='grey', linestyle='--')
        ax.set_xlabel('Time')
        ax.set_ylabel('Treatment Effect')
        ax.set_title('Synthetic DiD Treatment Effect Over Time with 90% Confidence Intervals')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return fig
