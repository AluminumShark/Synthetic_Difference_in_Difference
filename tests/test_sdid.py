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
        assert "t-statistic" in summary
        assert "p-value" in summary

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


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
