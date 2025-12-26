"""
Synthetic Difference-in-Differences (SDID) for Python.

A Python implementation of the Synthetic Difference-in-Differences estimator
for causal inference, based on Arkhangelsky et al. (2021).

Example:
    >>> from sdid import SyntheticDiffInDiff
    >>> sdid = SyntheticDiffInDiff(
    ...     data=df,
    ...     outcome_col="outcome",
    ...     times_col="year",
    ...     units_col="state",
    ...     treat_col="treated",
    ...     post_col="post"
    ... )
    >>> effect = sdid.fit()
"""

from sdid.core import SyntheticDiffInDiff

__all__ = ["SyntheticDiffInDiff"]
__version__ = "0.3.3"
