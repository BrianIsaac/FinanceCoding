"""Multiple comparison corrections framework for portfolio performance testing.

Implements various multiple hypothesis testing correction procedures to control
family-wise error rate (FWER) and false discovery rate (FDR) when conducting
simultaneous statistical tests across multiple portfolio strategies.
"""

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd


class MultipleComparisonCorrections:
    """Framework for multiple comparison corrections in portfolio performance testing."""

    @staticmethod
    def bonferroni_correction(
        p_values: Union[list[float], np.ndarray, pd.Series], alpha: float = 0.05
    ) -> dict[str, Union[np.ndarray, bool, float]]:
        """Bonferroni correction for multiple hypothesis testing.

        Controls family-wise error rate (FWER) by adjusting significance level.
        Conservative but widely accepted method.

        Args:
            p_values: Array of p-values from individual tests
            alpha: Overall significance level

        Returns:
            Dictionary containing corrected results
        """
        p_vals = np.asarray(p_values)

        # Remove NaN values
        valid_mask = ~np.isnan(p_vals)
        valid_p_vals = p_vals[valid_mask]

        if len(valid_p_vals) == 0:
            warnings.warn("No valid p-values for Bonferroni correction", stacklevel=2)
            return {
                "corrected_p_values": p_vals,
                "rejected": np.array([False] * len(p_vals)),
                "corrected_alpha": alpha,
                "method": "bonferroni",
                "n_comparisons": 0,
            }

        n_comparisons = len(valid_p_vals)
        corrected_alpha = alpha / n_comparisons

        # Adjust p-values
        corrected_p_vals = np.full_like(p_vals, np.nan)
        corrected_p_vals[valid_mask] = np.minimum(valid_p_vals * n_comparisons, 1.0)

        # Determine rejections
        rejected = np.full_like(p_vals, False, dtype=bool)
        rejected[valid_mask] = valid_p_vals < corrected_alpha

        return {
            "corrected_p_values": corrected_p_vals,
            "rejected": rejected,
            "corrected_alpha": corrected_alpha,
            "method": "bonferroni",
            "n_comparisons": n_comparisons,
            "original_alpha": alpha,
        }

    @staticmethod
    def benjamini_hochberg_fdr(
        p_values: Union[list[float], np.ndarray, pd.Series], alpha: float = 0.05
    ) -> dict[str, Union[np.ndarray, bool, float]]:
        """Benjamini-Hochberg False Discovery Rate (FDR) control.

        Controls the expected proportion of false discoveries among rejected hypotheses.
        Less conservative than Bonferroni, providing better power while controlling FDR.

        Args:
            p_values: Array of p-values from individual tests
            alpha: Desired FDR level

        Returns:
            Dictionary containing FDR-corrected results
        """
        p_vals = np.asarray(p_values)

        # Handle NaN values
        valid_mask = ~np.isnan(p_vals)
        valid_p_vals = p_vals[valid_mask]
        valid_indices = np.where(valid_mask)[0]

        if len(valid_p_vals) == 0:
            warnings.warn("No valid p-values for FDR correction", stacklevel=2)
            return {
                "corrected_p_values": p_vals,
                "rejected": np.array([False] * len(p_vals)),
                "critical_values": np.full_like(p_vals, np.nan),
                "method": "benjamini_hochberg_fdr",
                "fdr_level": alpha,
            }

        n_comparisons = len(valid_p_vals)

        # Sort p-values and track original indices
        sort_indices = np.argsort(valid_p_vals)
        sorted_p_vals = valid_p_vals[sort_indices]
        sorted_original_indices = valid_indices[sort_indices]

        # Calculate critical values for each test
        ranks = np.arange(1, n_comparisons + 1)
        critical_values = (ranks / n_comparisons) * alpha

        # Find largest i such that p(i) <= (i/m) * alpha
        significant_mask = sorted_p_vals <= critical_values

        if np.any(significant_mask):
            # Find the largest significant index
            max_significant_idx = np.max(np.where(significant_mask)[0])
            # All tests up to and including this index are rejected
            rejected_sorted_indices = np.arange(max_significant_idx + 1)
        else:
            rejected_sorted_indices = np.array([], dtype=int)

        # Create results arrays
        corrected_p_vals = np.full_like(p_vals, np.nan)
        rejected = np.full_like(p_vals, False, dtype=bool)
        critical_vals_array = np.full_like(p_vals, np.nan)

        # Calculate corrected p-values (using step-up procedure)
        for i in range(len(sorted_p_vals)):
            corrected_p_val = min(1.0, sorted_p_vals[i] * n_comparisons / (i + 1))
            original_idx = sorted_original_indices[i]
            corrected_p_vals[original_idx] = corrected_p_val
            critical_vals_array[original_idx] = critical_values[i]

        # Set rejections
        if len(rejected_sorted_indices) > 0:
            rejected_original_indices = sorted_original_indices[rejected_sorted_indices]
            rejected[rejected_original_indices] = True

        return {
            "corrected_p_values": corrected_p_vals,
            "rejected": rejected,
            "critical_values": critical_vals_array,
            "method": "benjamini_hochberg_fdr",
            "fdr_level": alpha,
            "n_comparisons": n_comparisons,
            "n_rejected": np.sum(rejected),
        }

    @staticmethod
    def holm_sidak_correction(
        p_values: Union[list[float], np.ndarray, pd.Series], alpha: float = 0.05
    ) -> dict[str, Union[np.ndarray, bool, float]]:
        """Holm-Sidak step-down correction for improved power over Bonferroni.

        Step-down procedure that provides better power than Bonferroni while
        still controlling family-wise error rate (FWER).

        Args:
            p_values: Array of p-values from individual tests
            alpha: Overall significance level

        Returns:
            Dictionary containing Holm-Sidak corrected results
        """
        p_vals = np.asarray(p_values)

        # Handle NaN values
        valid_mask = ~np.isnan(p_vals)
        valid_p_vals = p_vals[valid_mask]
        valid_indices = np.where(valid_mask)[0]

        if len(valid_p_vals) == 0:
            warnings.warn("No valid p-values for Holm-Sidak correction", stacklevel=2)
            return {
                "corrected_p_values": p_vals,
                "rejected": np.array([False] * len(p_vals)),
                "step_down_alphas": np.full_like(p_vals, np.nan),
                "method": "holm_sidak",
                "original_alpha": alpha,
            }

        n_comparisons = len(valid_p_vals)

        # Sort p-values and track original indices
        sort_indices = np.argsort(valid_p_vals)
        sorted_p_vals = valid_p_vals[sort_indices]
        sorted_original_indices = valid_indices[sort_indices]

        # Step-down critical values using Sidak correction
        # Alpha_i = 1 - (1 - alpha)^(1/(m - i + 1))
        step_alphas = np.zeros(n_comparisons)
        for i in range(n_comparisons):
            remaining_tests = n_comparisons - i
            step_alphas[i] = 1 - (1 - alpha) ** (1 / remaining_tests)

        # Apply step-down procedure
        rejected_sorted_mask = np.zeros(n_comparisons, dtype=bool)

        for i in range(n_comparisons):
            if sorted_p_vals[i] <= step_alphas[i]:
                # Reject this and all remaining (more significant) hypotheses
                rejected_sorted_mask[: i + 1] = True
                break

        # Create results arrays
        corrected_p_vals = np.full_like(p_vals, np.nan)
        rejected = np.full_like(p_vals, False, dtype=bool)
        step_alphas_array = np.full_like(p_vals, np.nan)

        # Calculate corrected p-values and set results
        for i in range(n_comparisons):
            original_idx = sorted_original_indices[i]

            # Corrected p-value using reverse step-down logic
            remaining_tests = n_comparisons - i
            corrected_p_val = 1 - (1 - sorted_p_vals[i]) ** remaining_tests
            corrected_p_vals[original_idx] = min(1.0, corrected_p_val)

            step_alphas_array[original_idx] = step_alphas[i]
            rejected[original_idx] = rejected_sorted_mask[i]

        return {
            "corrected_p_values": corrected_p_vals,
            "rejected": rejected,
            "step_down_alphas": step_alphas_array,
            "method": "holm_sidak",
            "original_alpha": alpha,
            "n_comparisons": n_comparisons,
            "n_rejected": np.sum(rejected),
        }

    @staticmethod
    def correction_selection_framework(
        p_values: Union[list[float], np.ndarray, pd.Series],
        alpha: float = 0.05,
        scenario: str = "exploratory",
        power_priority: bool = True,
    ) -> dict[str, dict]:
        """Framework for selecting appropriate multiple comparison correction.

        Args:
            p_values: Array of p-values from individual tests
            alpha: Significance level
            scenario: Testing scenario ('exploratory', 'confirmatory', 'publication')
            power_priority: Whether statistical power is prioritized over strict error control

        Returns:
            Dictionary containing results from recommended correction methods
        """
        p_vals = np.asarray(p_values)
        n_tests = np.sum(~np.isnan(p_vals))

        results = {}
        recommendations = []

        # Always calculate all methods for comparison
        results["bonferroni"] = MultipleComparisonCorrections.bonferroni_correction(p_vals, alpha)
        results["benjamini_hochberg"] = MultipleComparisonCorrections.benjamini_hochberg_fdr(
            p_vals, alpha
        )
        results["holm_sidak"] = MultipleComparisonCorrections.holm_sidak_correction(p_vals, alpha)

        # Scenario-based recommendations
        if scenario == "exploratory":
            if power_priority:
                primary_method = "benjamini_hochberg"
                recommendations.append(
                    "FDR control recommended for exploratory analysis with power priority"
                )
            else:
                primary_method = "holm_sidak"
                recommendations.append("Holm-Sidak recommended for balanced exploratory analysis")

        elif scenario == "confirmatory":
            if n_tests <= 5:
                primary_method = "holm_sidak"
                recommendations.append(
                    "Holm-Sidak recommended for small number of confirmatory tests"
                )
            else:
                primary_method = "bonferroni"
                recommendations.append("Bonferroni recommended for many confirmatory tests")

        elif scenario == "publication":
            primary_method = "bonferroni"
            recommendations.append("Bonferroni recommended for publication standards")

        else:
            primary_method = "benjamini_hochberg"
            recommendations.append("FDR control as default recommendation")

        # Add comparison summary
        comparison_summary = {
            "n_tests": n_tests,
            "bonferroni_rejected": np.sum(results["bonferroni"]["rejected"]),
            "fdr_rejected": np.sum(results["benjamini_hochberg"]["rejected"]),
            "holm_sidak_rejected": np.sum(results["holm_sidak"]["rejected"]),
            "primary_recommendation": primary_method,
            "recommendations": recommendations,
        }

        results["comparison_summary"] = comparison_summary
        results["primary_method"] = results[primary_method]

        return results


class PerformanceTestingCorrections:
    """Specialized multiple comparison corrections for performance testing scenarios."""

    def __init__(self, alpha: float = 0.05):
        """Initialize with significance level.

        Args:
            alpha: Overall significance level
        """
        self.alpha = alpha

    def portfolio_comparison_corrections(
        self,
        comparison_results: pd.DataFrame,
        p_value_column: str = "p_value",
        groupby_column: Optional[str] = None,
        method: str = "benjamini_hochberg",
    ) -> pd.DataFrame:
        """Apply multiple comparison corrections to portfolio comparison results.

        Args:
            comparison_results: DataFrame with pairwise comparison results
            p_value_column: Name of column containing p-values
            groupby_column: Optional column to group corrections by (e.g., 'metric')
            method: Correction method to apply

        Returns:
            DataFrame with corrected p-values and significance flags
        """
        df = comparison_results.copy()

        correction_methods = {
            "bonferroni": MultipleComparisonCorrections.bonferroni_correction,
            "benjamini_hochberg": MultipleComparisonCorrections.benjamini_hochberg_fdr,
            "holm_sidak": MultipleComparisonCorrections.holm_sidak_correction,
        }

        if method not in correction_methods:
            raise ValueError(f"Unknown correction method: {method}")

        correction_func = correction_methods[method]

        if groupby_column is not None and groupby_column in df.columns:
            # Apply corrections within each group
            corrected_results = []

            for group_name, group_data in df.groupby(groupby_column):
                p_values = group_data[p_value_column].values
                correction_result = correction_func(p_values, self.alpha)

                group_corrected = group_data.copy()
                group_corrected[f"{p_value_column}_corrected"] = correction_result[
                    "corrected_p_values"
                ]
                group_corrected[f"rejected_{method}"] = correction_result["rejected"]
                group_corrected["correction_method"] = method
                group_corrected["correction_group"] = group_name

                corrected_results.append(group_corrected)

            return pd.concat(corrected_results, ignore_index=True)

        else:
            # Apply correction to all p-values
            p_values = df[p_value_column].values
            correction_result = correction_func(p_values, self.alpha)

            df[f"{p_value_column}_corrected"] = correction_result["corrected_p_values"]
            df[f"rejected_{method}"] = correction_result["rejected"]
            df["correction_method"] = method

            return df

    def multi_metric_corrections(
        self,
        performance_comparisons: dict[str, pd.DataFrame],
        p_value_column: str = "p_value",
        global_correction: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Apply corrections across multiple performance metrics.

        Args:
            performance_comparisons: Dict mapping metric names to comparison DataFrames
            p_value_column: Name of p-value column
            global_correction: Whether to correct across all metrics simultaneously

        Returns:
            Dictionary of corrected comparison DataFrames
        """
        if global_correction:
            # Collect all p-values across all metrics
            all_p_values = []
            metric_indices = []

            for metric_name, comparison_df in performance_comparisons.items():
                p_vals = comparison_df[p_value_column].values
                all_p_values.extend(p_vals)
                metric_indices.extend([metric_name] * len(p_vals))

            # Apply global correction
            global_correction_result = MultipleComparisonCorrections.benjamini_hochberg_fdr(
                all_p_values, self.alpha
            )

            # Distribute corrected results back to individual DataFrames
            corrected_results = {}
            start_idx = 0

            for metric_name, comparison_df in performance_comparisons.items():
                n_comparisons = len(comparison_df)
                end_idx = start_idx + n_comparisons

                df_corrected = comparison_df.copy()
                df_corrected[f"{p_value_column}_global_corrected"] = global_correction_result[
                    "corrected_p_values"
                ][start_idx:end_idx]
                df_corrected["rejected_global_fdr"] = global_correction_result["rejected"][
                    start_idx:end_idx
                ]
                df_corrected["global_correction"] = True

                corrected_results[metric_name] = df_corrected
                start_idx = end_idx

            return corrected_results

        else:
            # Apply corrections within each metric separately
            corrected_results = {}

            for metric_name, comparison_df in performance_comparisons.items():
                corrected_df = self.portfolio_comparison_corrections(
                    comparison_df, p_value_column, method="benjamini_hochberg"
                )
                corrected_df["global_correction"] = False
                corrected_results[metric_name] = corrected_df

            return corrected_results

    def rolling_window_corrections(
        self,
        rolling_comparison_results: pd.DataFrame,
        p_value_column: str = "p_value",
        window_column: str = "window_start",
        method: str = "benjamini_hochberg",
    ) -> pd.DataFrame:
        """Apply corrections to rolling window comparison results.

        Args:
            rolling_comparison_results: DataFrame with rolling window comparisons
            p_value_column: Name of p-value column
            window_column: Column identifying different windows
            method: Correction method

        Returns:
            DataFrame with corrections applied within each window
        """
        return self.portfolio_comparison_corrections(
            rolling_comparison_results,
            p_value_column=p_value_column,
            groupby_column=window_column,
            method=method,
        )
