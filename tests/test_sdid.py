"""
Unit tests for Synthetic Difference-in-Differences (SDID) implementation.

These tests verify:
1. Data validation and initialization
2. Weight estimation (unit and time weights)
3. Treatment effect estimation
4. Standard error estimation
5. Event study functionality
6. Utility methods
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from sdid import SyntheticDiffInDiff

# =============================================================================
# Fixtures - Test Data Generation
# =============================================================================


@pytest.fixture
def simple_panel_data():
    """
    Create a simple panel dataset with known treatment effect.

    Structure:
    - 5 control units, 1 treated unit
    - 4 pre-treatment periods, 2 post-treatment periods
    - True treatment effect: +10
    """
    np.random.seed(42)

    units = ["control_1", "control_2", "control_3", "control_4", "control_5", "treated_1"]
    times = [2015, 2016, 2017, 2018, 2019, 2020]

    data = []
    for unit in units:
        is_treated = unit.startswith("treated")
        base_value = 100 + np.random.randn() * 5  # Unit-specific baseline

        for t in times:
            is_post = t >= 2019
            # Outcome = base + time trend + treatment effect + noise
            time_effect = (t - 2015) * 2
            treatment_effect = 10 if (is_treated and is_post) else 0
            noise = np.random.randn() * 1

            outcome = base_value + time_effect + treatment_effect + noise

            data.append(
                {
                    "unit": unit,
                    "time": t,
                    "outcome": outcome,
                    "treated": is_treated,
                    "post": is_post,
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def larger_panel_data():
    """
    Create a larger panel dataset for more robust testing.

    Structure:
    - 20 control units, 5 treated units
    - 8 pre-treatment periods, 4 post-treatment periods
    - True treatment effect: +15
    """
    np.random.seed(123)

    n_control = 20
    n_treated = 5
    pre_periods = 8
    post_periods = 4

    control_units = [f"control_{i}" for i in range(n_control)]
    treated_units = [f"treated_{i}" for i in range(n_treated)]
    all_units = control_units + treated_units

    times = list(range(2010, 2010 + pre_periods + post_periods))
    post_start = 2010 + pre_periods

    data = []
    for unit in all_units:
        is_treated = unit.startswith("treated")
        base_value = 50 + np.random.randn() * 10
        unit_trend = np.random.randn() * 0.5

        for t in times:
            is_post = t >= post_start
            time_effect = (t - 2010) * (2 + unit_trend)
            treatment_effect = 15 if (is_treated and is_post) else 0
            noise = np.random.randn() * 2

            outcome = base_value + time_effect + treatment_effect + noise

            data.append(
                {
                    "unit": unit,
                    "time": t,
                    "outcome": outcome,
                    "treated": is_treated,
                    "post": is_post,
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def minimal_panel_data():
    """Minimal valid dataset for edge case testing."""
    return pd.DataFrame(
        {
            "unit": ["A", "A", "B", "B"],
            "time": [1, 2, 1, 2],
            "outcome": [10.0, 12.0, 11.0, 18.0],
            "treated": [False, False, True, True],
            "post": [False, True, False, True],
        }
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInitialization:
    """Tests for SDID initialization and data validation."""

    def test_valid_initialization(self, simple_panel_data):
        """Test that valid data initializes without errors."""
        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        assert sdid.outcome_col == "outcome"
        assert sdid.times_col == "time"
        assert sdid.units_col == "unit"
        assert sdid.treat_col == "treated"
        assert sdid.post_col == "post"
        assert not sdid.is_fitted

    def test_missing_column_raises_error(self, simple_panel_data):
        """Test that missing columns raise ValueError."""
        with pytest.raises(ValueError, match="Missing required columns"):
            SyntheticDiffInDiff(
                data=simple_panel_data,
                outcome_col="nonexistent",
                times_col="time",
                units_col="unit",
                treat_col="treated",
                post_col="post",
            )

    def test_nan_values_raise_error(self, simple_panel_data):
        """Test that NaN values in required columns raise ValueError."""
        data_with_nan = simple_panel_data.copy()
        data_with_nan.loc[0, "outcome"] = np.nan

        with pytest.raises(ValueError, match="contains NaN"):
            SyntheticDiffInDiff(
                data=data_with_nan,
                outcome_col="outcome",
                times_col="time",
                units_col="unit",
                treat_col="treated",
                post_col="post",
            )

    def test_no_treated_units_raises_error(self, simple_panel_data):
        """Test that data with no treated units raises ValueError."""
        data_no_treated = simple_panel_data.copy()
        data_no_treated["treated"] = False

        with pytest.raises(ValueError, match="No treated units"):
            SyntheticDiffInDiff(
                data=data_no_treated,
                outcome_col="outcome",
                times_col="time",
                units_col="unit",
                treat_col="treated",
                post_col="post",
            )

    def test_no_control_units_raises_error(self, simple_panel_data):
        """Test that data with no control units raises ValueError."""
        data_no_control = simple_panel_data.copy()
        data_no_control["treated"] = True

        with pytest.raises(ValueError, match="No control units"):
            SyntheticDiffInDiff(
                data=data_no_control,
                outcome_col="outcome",
                times_col="time",
                units_col="unit",
                treat_col="treated",
                post_col="post",
            )

    def test_no_pre_periods_raises_error(self, simple_panel_data):
        """Test that data with no pre-treatment periods raises ValueError."""
        data_no_pre = simple_panel_data.copy()
        data_no_pre["post"] = True

        with pytest.raises(ValueError, match="No pre-treatment periods"):
            SyntheticDiffInDiff(
                data=data_no_pre,
                outcome_col="outcome",
                times_col="time",
                units_col="unit",
                treat_col="treated",
                post_col="post",
            )

    def test_no_post_periods_raises_error(self, simple_panel_data):
        """Test that data with no post-treatment periods raises ValueError."""
        data_no_post = simple_panel_data.copy()
        data_no_post["post"] = False

        with pytest.raises(ValueError, match="No post-treatment periods"):
            SyntheticDiffInDiff(
                data=data_no_post,
                outcome_col="outcome",
                times_col="time",
                units_col="unit",
                treat_col="treated",
                post_col="post",
            )

    def test_data_copy_is_made(self, simple_panel_data):
        """Test that original data is not modified."""
        original_data = simple_panel_data.copy()

        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )
        sdid.fit()

        # Original data should be unchanged
        pd.testing.assert_frame_equal(
            simple_panel_data[["unit", "time", "outcome"]],
            original_data[["unit", "time", "outcome"]],
        )


# =============================================================================
# Fit and Treatment Effect Tests
# =============================================================================


class TestFit:
    """Tests for the fit method and treatment effect estimation."""

    def test_fit_returns_float(self, simple_panel_data):
        """Test that fit() returns a float treatment effect."""
        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()

        assert isinstance(effect, float)
        assert not np.isnan(effect)

    def test_is_fitted_after_fit(self, simple_panel_data):
        """Test that is_fitted is True after calling fit()."""
        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        assert not sdid.is_fitted
        sdid.fit()
        assert sdid.is_fitted

    def test_treatment_effect_reasonable(self, simple_panel_data):
        """Test that estimated treatment effect is in reasonable range."""
        # True effect is +10, should be within reasonable bounds
        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()

        # Effect should be positive and roughly in the right ballpark
        # With noise, we allow a wide range
        assert effect > 0
        assert effect < 30  # Should not be wildly off

    def test_larger_dataset_effect(self, larger_panel_data):
        """Test treatment effect estimation on larger dataset."""
        # True effect is +15
        sdid = SyntheticDiffInDiff(
            data=larger_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()

        # With more data, estimate should be in positive direction
        # Allow wider range due to noise and regularization
        assert 0 < effect < 30

    def test_weights_populated_after_fit(self, simple_panel_data):
        """Test that unit and time weights are populated after fit."""
        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        assert sdid.unit_weights is None
        assert sdid.time_weights is None

        sdid.fit()

        # Weights should be Series objects (may be empty if all below threshold)
        assert sdid.unit_weights is not None
        assert sdid.time_weights is not None
        assert isinstance(sdid.unit_weights, pd.Series)
        assert isinstance(sdid.time_weights, pd.Series)
        # At least unit weights should have some non-zero values
        assert len(sdid.unit_weights) > 0

    def test_unit_weights_are_nonnegative(self, simple_panel_data):
        """Test that all unit weights are non-negative."""
        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )
        sdid.fit()

        assert (sdid.unit_weights >= 0).all()

    def test_time_weights_are_nonnegative(self, simple_panel_data):
        """Test that all time weights are non-negative."""
        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )
        sdid.fit()

        assert (sdid.time_weights >= 0).all()


# =============================================================================
# Standard Error Estimation Tests
# =============================================================================


class TestStandardError:
    """Tests for standard error estimation."""

    def test_estimate_se_returns_float(self, simple_panel_data):
        """Test that estimate_se returns a float."""
        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )
        sdid.fit()

        se = sdid.estimate_se(n_bootstrap=20, seed=42)

        assert isinstance(se, float)
        assert se > 0

    def test_standard_error_stored(self, simple_panel_data):
        """Test that standard error is stored in instance."""
        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )
        sdid.fit()

        assert sdid.standard_error is None
        sdid.estimate_se(n_bootstrap=20, seed=42)
        assert sdid.standard_error is not None
        assert sdid.standard_error > 0

    def test_reproducibility_with_seed(self, simple_panel_data):
        """Test that results are reproducible with same seed."""
        sdid1 = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )
        sdid1.fit()
        se1 = sdid1.estimate_se(n_bootstrap=20, seed=42)

        sdid2 = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )
        sdid2.fit()
        se2 = sdid2.estimate_se(n_bootstrap=20, seed=42)

        assert se1 == se2


# =============================================================================
# Event Study Tests
# =============================================================================


class TestEventStudy:
    """Tests for event study functionality."""

    def test_run_event_study_returns_series(self, simple_panel_data):
        """Test that run_event_study returns a pandas Series."""
        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        times = [2019, 2020]
        effects = sdid.run_event_study(times)

        assert isinstance(effects, pd.Series)
        assert len(effects) == len(times)

    def test_event_study_index_matches_times(self, simple_panel_data):
        """Test that event study index matches input times."""
        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        times = [2019, 2020]
        effects = sdid.run_event_study(times)

        assert list(effects.index) == times

    def test_event_study_effects_reasonable(self, larger_panel_data):
        """Test that event study effects are reasonable."""
        sdid = SyntheticDiffInDiff(
            data=larger_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        # Post-treatment periods
        post_times = [2018, 2019, 2020, 2021]
        effects = sdid.run_event_study(post_times)

        # All effects should be positive (true effect is +15)
        valid_effects = effects.dropna()
        assert len(valid_effects) > 0
        assert (valid_effects > 0).all()


# =============================================================================
# Utility Method Tests
# =============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_get_weights_summary_structure(self, simple_panel_data):
        """Test that get_weights_summary returns correct structure."""
        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )
        sdid.fit()

        summary = sdid.get_weights_summary()

        assert isinstance(summary, dict)
        assert "unit_weights" in summary
        assert "time_weights" in summary
        assert isinstance(summary["unit_weights"], pd.DataFrame)
        assert isinstance(summary["time_weights"], pd.DataFrame)

    def test_get_weights_summary_before_fit_raises(self, simple_panel_data):
        """Test that get_weights_summary raises error before fit."""
        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        with pytest.raises(ValueError, match="not yet estimated"):
            sdid.get_weights_summary()

    def test_summary_before_fit(self, simple_panel_data):
        """Test summary() before fit returns appropriate message."""
        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        summary = sdid.summary()
        assert "not yet fitted" in summary.lower()

    def test_summary_after_fit(self, simple_panel_data):
        """Test summary() after fit contains expected information."""
        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )
        sdid.fit()

        summary = sdid.summary()

        assert "Treatment Effect" in summary
        assert "Control units" in summary

    def test_summary_with_se(self, simple_panel_data):
        """Test summary() includes SE information when available."""
        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )
        sdid.fit()
        sdid.estimate_se(n_bootstrap=20, seed=42)

        summary = sdid.summary()

        assert "Standard Error" in summary
        assert "95% Confidence Interval" in summary
        assert "t-statistic" in summary
        assert "p-value" in summary

    def test_summary_custom_ci(self, simple_panel_data):
        """Test summary() with custom confidence level."""
        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )
        sdid.fit()
        sdid.estimate_se(n_bootstrap=20, seed=42)

        summary_90 = sdid.summary(confidence_level=0.90)
        assert "90% Confidence Interval" in summary_90

        summary_99 = sdid.summary(confidence_level=0.99)
        assert "99% Confidence Interval" in summary_99

    def test_repr(self, simple_panel_data):
        """Test __repr__ method."""
        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        repr_str = repr(sdid)
        assert "SyntheticDiffInDiff" in repr_str
        assert "outcome" in repr_str
        assert "not fitted" in repr_str

        sdid.fit()
        repr_str = repr(sdid)
        assert "fitted" in repr_str


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_minimal_data(self, minimal_panel_data):
        """Test with minimal valid dataset."""
        sdid = SyntheticDiffInDiff(
            data=minimal_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        # Should not raise
        effect = sdid.fit()
        assert isinstance(effect, float)

    def test_integer_treatment_indicator(self, simple_panel_data):
        """Test that integer treatment indicators work (converted to bool)."""
        data = simple_panel_data.copy()
        data["treated"] = data["treated"].astype(int)
        data["post"] = data["post"].astype(int)

        sdid = SyntheticDiffInDiff(
            data=data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)

    def test_verbose_mode(self, simple_panel_data, capsys):
        """Test verbose mode produces output."""
        sdid = SyntheticDiffInDiff(
            data=simple_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        sdid.fit(verbose=True)
        captured = capsys.readouterr()

        # Verbose output should contain regression results
        assert "SDID REGRESSION RESULTS" in captured.out


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(self, larger_panel_data):
        """Test complete analysis workflow."""
        # Initialize
        sdid = SyntheticDiffInDiff(
            data=larger_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        # Fit
        effect = sdid.fit()
        assert sdid.is_fitted
        assert isinstance(effect, float)

        # Standard error
        se = sdid.estimate_se(n_bootstrap=30, seed=42)
        assert se > 0

        # Summary
        summary = sdid.summary()
        assert "Treatment Effect" in summary
        assert "Standard Error" in summary

        # Weights - unit weights should exist, time weights may be empty
        weights = sdid.get_weights_summary()
        assert len(weights["unit_weights"]) > 0
        # Time weights can be empty if pre-treatment trends are parallel
        assert isinstance(weights["time_weights"], pd.DataFrame)

    def test_event_study_workflow(self, larger_panel_data):
        """Test event study workflow."""
        sdid = SyntheticDiffInDiff(
            data=larger_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        # Event study
        times = [2018, 2019, 2020, 2021]
        effects = sdid.run_event_study(times)

        assert len(effects) == len(times)
        assert effects.name == "treatment_effect"


class TestPlotMethods:
    """Tests for visualization methods."""

    @pytest.fixture(autouse=True)
    def setup_matplotlib_backend(self):
        """Use non-interactive backend for tests."""
        matplotlib.use("Agg")

    def test_plot_raw_trends_returns_figure(self, larger_panel_data):
        """Test that plot_raw_trends returns a matplotlib Figure."""
        sdid = SyntheticDiffInDiff(
            data=larger_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        # plot_raw_trends can be called before fit
        fig = sdid.plot_raw_trends()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_raw_trends_with_custom_params(self, larger_panel_data):
        """Test plot_raw_trends with custom parameters."""
        sdid = SyntheticDiffInDiff(
            data=larger_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        fig = sdid.plot_raw_trends(
            treatment_time=2018,
            figsize=(12, 8),
            control_color="blue",
            control_alpha=0.5,
            avg_control_color="green",
            treated_color="orange",
            title="Custom Title",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_synthetic_control_requires_fit(self, larger_panel_data):
        """Test that plot_synthetic_control raises error if not fitted."""
        sdid = SyntheticDiffInDiff(
            data=larger_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        with pytest.raises(ValueError, match="Model must be fitted"):
            sdid.plot_synthetic_control()

    def test_plot_synthetic_control_returns_figure(self, larger_panel_data):
        """Test that plot_synthetic_control returns a matplotlib Figure after fit."""
        sdid = SyntheticDiffInDiff(
            data=larger_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        sdid.fit()
        fig = sdid.plot_synthetic_control()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_synthetic_control_with_custom_params(self, larger_panel_data):
        """Test plot_synthetic_control with custom parameters."""
        sdid = SyntheticDiffInDiff(
            data=larger_panel_data,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        sdid.fit()
        fig = sdid.plot_synthetic_control(
            treatment_time=2018,
            figsize=(14, 7),
            treated_color="orange",
            synthetic_color="green",
            title="Custom SDID Plot",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# Extended Testing with Various Data Scenarios
# =============================================================================


class TestVariousDataScenarios:
    """Tests with different data structures and edge cases."""

    @pytest.fixture(autouse=True)
    def setup_matplotlib_backend(self):
        """Use non-interactive backend for tests."""
        matplotlib.use("Agg")

    def generate_panel_data(
        self,
        n_control: int = 10,
        n_treated: int = 1,
        pre_periods: int = 5,
        post_periods: int = 3,
        treatment_effect: float = 10.0,
        noise_level: float = 1.0,
        trend_strength: float = 2.0,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate panel data with customizable parameters."""
        np.random.seed(seed)

        units = [f"control_{i}" for i in range(n_control)] + [
            f"treated_{i}" for i in range(n_treated)
        ]
        times = list(range(2010, 2010 + pre_periods + post_periods))
        treatment_start = 2010 + pre_periods

        data = []
        for unit in units:
            is_treated = unit.startswith("treated")
            base_value = 100 + np.random.randn() * 10

            for t in times:
                is_post = t >= treatment_start
                time_effect = (t - 2010) * trend_strength
                effect = treatment_effect if (is_treated and is_post) else 0
                noise = np.random.randn() * noise_level

                data.append({
                    "unit": unit,
                    "time": t,
                    "outcome": base_value + time_effect + effect + noise,
                    "treated": is_treated,
                    "post": is_post,
                })

        return pd.DataFrame(data)

    def test_single_treated_unit(self):
        """Test with 1 treated unit and 10 controls."""
        df = self.generate_panel_data(n_control=10, n_treated=1, treatment_effect=15)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)
        assert not np.isnan(effect)
        # Effect should be somewhat close to true value (15)
        assert 5 < effect < 25

    def test_multiple_treated_units(self):
        """Test with 5 treated units and 20 controls."""
        df = self.generate_panel_data(n_control=20, n_treated=5, treatment_effect=20)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)
        assert not np.isnan(effect)
        # Effect should be somewhat close to true value (20)
        assert 10 < effect < 30

    def test_many_control_units(self):
        """Test with 50 control units."""
        df = self.generate_panel_data(n_control=50, n_treated=1, treatment_effect=12)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)
        assert not np.isnan(effect)

    def test_long_pre_period(self):
        """Test with 15 pre-treatment periods."""
        df = self.generate_panel_data(pre_periods=15, post_periods=3, treatment_effect=8)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)
        assert not np.isnan(effect)

    def test_long_post_period(self):
        """Test with 10 post-treatment periods."""
        df = self.generate_panel_data(pre_periods=5, post_periods=10, treatment_effect=18)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)
        assert not np.isnan(effect)

    def test_high_noise_data(self):
        """Test with high noise level."""
        df = self.generate_panel_data(noise_level=10.0, treatment_effect=20)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)
        assert not np.isnan(effect)

    def test_low_noise_data(self):
        """Test with very low noise level."""
        df = self.generate_panel_data(noise_level=0.01, treatment_effect=10)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)
        assert not np.isnan(effect)

    def test_zero_treatment_effect(self):
        """Test when true treatment effect is zero."""
        df = self.generate_panel_data(treatment_effect=0)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)
        # Effect should be close to zero (within noise range)
        assert -5 < effect < 5

    def test_negative_treatment_effect(self):
        """Test with negative treatment effect."""
        df = self.generate_panel_data(treatment_effect=-15)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)
        assert effect < 0

    def test_large_treatment_effect(self):
        """Test with very large treatment effect."""
        df = self.generate_panel_data(treatment_effect=100)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)
        # Should capture large positive effect
        assert effect > 30

    def test_no_trend(self):
        """Test with no time trend."""
        df = self.generate_panel_data(trend_strength=0, treatment_effect=10)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)
        assert not np.isnan(effect)

    def test_strong_trend(self):
        """Test with strong time trend."""
        df = self.generate_panel_data(trend_strength=10, treatment_effect=15)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)
        assert not np.isnan(effect)

    def test_full_workflow_with_se_estimation(self):
        """Test complete workflow including standard error."""
        df = self.generate_panel_data(n_control=15, n_treated=3, treatment_effect=12)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        # Fit
        sdid.fit()
        assert sdid.is_fitted
        
        # SE estimation
        se = sdid.estimate_se(n_bootstrap=50, seed=42)
        assert se > 0
        assert not np.isnan(se)

        # Summary
        summary = sdid.summary()
        assert "Treatment Effect" in summary
        assert "Standard Error" in summary

    def test_all_plotting_methods(self):
        """Test all plotting methods work together."""
        df = self.generate_panel_data(n_control=10, n_treated=2, treatment_effect=15)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        # Raw trends before fit
        fig1 = sdid.plot_raw_trends()
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        # Fit
        sdid.fit()

        # Raw trends after fit
        fig2 = sdid.plot_raw_trends(title="After Fit")
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

        # Synthetic control
        fig3 = sdid.plot_synthetic_control()
        assert isinstance(fig3, plt.Figure)
        plt.close(fig3)

    def test_event_study_with_many_periods(self):
        """Test event study with many time periods."""
        df = self.generate_panel_data(pre_periods=5, post_periods=8, treatment_effect=10)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        # Use all post-treatment periods for event study
        post_times = list(range(2015, 2023))
        effects = sdid.run_event_study(post_times)

        assert len(effects) == len(post_times)
        # At least some effects should be valid
        valid_effects = effects.dropna()
        assert len(valid_effects) > 0


class TestDataTypeVariations:
    """Tests with different data types and column configurations."""

    def test_string_time_column(self):
        """Test with string time values."""
        np.random.seed(42)

        data = []
        for unit in ["A", "B", "C", "D"]:
            is_treated = unit == "D"
            for t in ["2019-Q1", "2019-Q2", "2019-Q3", "2019-Q4", "2020-Q1", "2020-Q2"]:
                is_post = t.startswith("2020")
                effect = 10 if (is_treated and is_post) else 0
                data.append({
                    "unit": unit,
                    "time": t,
                    "outcome": 100 + np.random.randn() + effect,
                    "treated": is_treated,
                    "post": is_post,
                })

        df = pd.DataFrame(data)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)

    def test_integer_unit_ids(self):
        """Test with integer unit IDs."""
        np.random.seed(42)

        data = []
        for unit in range(10):
            is_treated = unit == 9
            for t in range(6):
                is_post = t >= 4
                effect = 15 if (is_treated and is_post) else 0
                data.append({
                    "unit": unit,
                    "time": t,
                    "outcome": 50 + t * 2 + np.random.randn() + effect,
                    "treated": is_treated,
                    "post": is_post,
                })

        df = pd.DataFrame(data)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)

    def test_float_outcome(self):
        """Test with float outcome values including decimals."""
        np.random.seed(42)

        data = []
        for unit in ["ctrl_1", "ctrl_2", "ctrl_3", "treat_1"]:
            is_treated = unit.startswith("treat")
            for t in range(8):
                is_post = t >= 5
                effect = 3.14159 if (is_treated and is_post) else 0
                data.append({
                    "unit": unit,
                    "time": t,
                    "outcome": 10.5 + t * 0.75 + np.random.randn() * 0.1 + effect,
                    "treated": is_treated,
                    "post": is_post,
                })

        df = pd.DataFrame(data)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)
        # Should be close to 3.14159
        assert 1 < effect < 6

    def test_different_column_names(self):
        """Test with non-standard column names."""
        np.random.seed(42)

        data = []
        for entity in ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]:
            is_policy = entity == "Epsilon"
            for period in [1, 2, 3, 4, 5, 6]:
                after_policy = period > 4
                effect = 20 if (is_policy and after_policy) else 0
                data.append({
                    "entity_id": entity,
                    "time_period": period,
                    "Y": 80 + period * 3 + np.random.randn() * 2 + effect,
                    "policy_group": is_policy,
                    "after_policy": after_policy,
                })

        df = pd.DataFrame(data)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="Y",
            times_col="time_period",
            units_col="entity_id",
            treat_col="policy_group",
            post_col="after_policy",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)

    def test_with_01_indicators(self):
        """Test with 0/1 instead of True/False indicators."""
        np.random.seed(42)

        data = []
        for unit in range(8):
            is_treated = 1 if unit >= 6 else 0
            for t in range(6):
                is_post = 1 if t >= 4 else 0
                effect = 12 if (is_treated == 1 and is_post == 1) else 0
                data.append({
                    "unit": f"unit_{unit}",
                    "time": t,
                    "outcome": 100 + np.random.randn() * 2 + effect,
                    "treated": is_treated,
                    "post": is_post,
                })

        df = pd.DataFrame(data)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)


class TestStressTests:
    """Stress tests for large datasets and edge cases."""

    def test_large_dataset(self):
        """Test with large dataset (100 units, 20 periods)."""
        np.random.seed(42)

        n_control = 95
        n_treated = 5
        n_periods = 20

        data = []
        for i in range(n_control + n_treated):
            is_treated = i >= n_control
            unit_effect = np.random.randn() * 10

            for t in range(n_periods):
                is_post = t >= 15
                treatment_effect = 25 if (is_treated and is_post) else 0

                data.append({
                    "unit": f"unit_{i}",
                    "time": t,
                    "outcome": 100 + unit_effect + t * 2 + np.random.randn() + treatment_effect,
                    "treated": is_treated,
                    "post": is_post,
                })

        df = pd.DataFrame(data)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)
        assert not np.isnan(effect)
        # Effect should be reasonably close to 25
        assert 15 < effect < 35

    def test_minimum_viable_dataset(self):
        """Test with absolute minimum data (2 controls, 1 treated, 2+1 periods)."""
        np.random.seed(42)

        data = [
            # Control 1
            {"unit": "C1", "time": 1, "outcome": 10, "treated": False, "post": False},
            {"unit": "C1", "time": 2, "outcome": 12, "treated": False, "post": False},
            {"unit": "C1", "time": 3, "outcome": 14, "treated": False, "post": True},
            # Control 2
            {"unit": "C2", "time": 1, "outcome": 11, "treated": False, "post": False},
            {"unit": "C2", "time": 2, "outcome": 13, "treated": False, "post": False},
            {"unit": "C2", "time": 3, "outcome": 15, "treated": False, "post": True},
            # Treated
            {"unit": "T1", "time": 1, "outcome": 10, "treated": True, "post": False},
            {"unit": "T1", "time": 2, "outcome": 12, "treated": True, "post": False},
            {"unit": "T1", "time": 3, "outcome": 24, "treated": True, "post": True},  # +10 effect
        ]

        df = pd.DataFrame(data)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        effect = sdid.fit()
        assert isinstance(effect, float)

    def test_weights_summary_structure(self):
        """Test that weight summaries have correct structure."""
        np.random.seed(42)

        data = []
        for i in range(12):
            is_treated = i >= 10
            for t in range(8):
                is_post = t >= 5
                effect = 10 if (is_treated and is_post) else 0
                data.append({
                    "unit": f"U{i}",
                    "time": t,
                    "outcome": 100 + np.random.randn() + effect,
                    "treated": is_treated,
                    "post": is_post,
                })

        df = pd.DataFrame(data)

        sdid = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )

        sdid.fit()
        weights = sdid.get_weights_summary()

        assert "unit_weights" in weights
        assert "time_weights" in weights
        assert "weight" in weights["unit_weights"].columns
        assert "rank" in weights["unit_weights"].columns

    def test_multiple_runs_consistency(self):
        """Test that multiple runs on same data give same results."""
        np.random.seed(42)

        data = []
        for i in range(10):
            is_treated = i >= 8
            for t in range(6):
                is_post = t >= 4
                effect = 15 if (is_treated and is_post) else 0
                data.append({
                    "unit": f"U{i}",
                    "time": t,
                    "outcome": 100 + t * 2 + np.random.randn() + effect,
                    "treated": is_treated,
                    "post": is_post,
                })

        df = pd.DataFrame(data)

        # Run 1
        sdid1 = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )
        effect1 = sdid1.fit()

        # Run 2
        sdid2 = SyntheticDiffInDiff(
            data=df,
            outcome_col="outcome",
            times_col="time",
            units_col="unit",
            treat_col="treated",
            post_col="post",
        )
        effect2 = sdid2.fit()

        # Should be identical
        assert effect1 == effect2


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
