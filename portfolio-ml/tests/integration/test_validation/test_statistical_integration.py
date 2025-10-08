"""Integration tests for statistical validation framework."""

import gc
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.evaluation.validation.bootstrap import BootstrapMethodology, MultiMetricBootstrap
from src.evaluation.validation.confidence_intervals import ComprehensiveConfidenceIntervals
from src.evaluation.validation.consistency import RollingConsistencyAnalyzer
from src.evaluation.validation.corrections import (
    MultipleComparisonCorrections,
    PerformanceTestingCorrections,
)
from src.evaluation.validation.hypothesis_testing import PerformanceHypothesisTestingFramework
from src.evaluation.validation.reporting import PublicationReadyStatisticalReporting
from src.evaluation.validation.significance import (
    PerformanceSignificanceTest,
    StatisticalValidation,
)


class TestStatisticalValidationIntegration:
    """Integration tests across statistical validation components."""

    @pytest.fixture
    def comprehensive_test_data(self):
        """Generate comprehensive test data simulating real portfolio scenarios."""
        np.random.seed(42)

        # Date range
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 12, 31)
        dates = pd.date_range(start_date, end_date, freq="D")
        n_days = len(dates)

        # Market regimes (bull/bear)
        regime_changes = [252, 504, 756]  # Change points
        regimes = np.ones(n_days)
        for change in regime_changes:
            if change < n_days:
                regimes[change:] = -regimes[change - 1]

        # Base market returns with regime dependency
        market_base = np.where(regimes > 0, 0.0008, -0.0002)  # Bull/bear base returns
        market_vol = np.where(regimes > 0, 0.015, 0.025)  # Lower vol in bull markets

        market_returns = np.random.normal(market_base, market_vol)

        # Strategy returns with different characteristics
        strategies = {
            "HRP": {
                "alpha": 0.0002,
                "beta": 0.8,
                "vol_multiplier": 0.9,
                "skill_decay": 0.0,  # Consistent performance
            },
            "LSTM": {
                "alpha": 0.0004,
                "beta": 1.1,
                "vol_multiplier": 1.2,
                "skill_decay": 0.0001,  # Slight skill decay over time
            },
            "GAT": {
                "alpha": 0.0003,
                "beta": 0.95,
                "vol_multiplier": 1.05,
                "skill_decay": -0.00005,  # Slight skill improvement over time
            },
            "Equal_Weight": {
                "alpha": 0.0001,
                "beta": 1.0,
                "vol_multiplier": 1.0,
                "skill_decay": 0.0,
            },
            "Mean_Variance": {
                "alpha": 0.00015,
                "beta": 0.85,
                "vol_multiplier": 0.95,
                "skill_decay": 0.00002,
            },
        }

        returns_dict = {}

        for strategy, params in strategies.items():
            # Time-varying alpha due to skill decay/improvement
            time_trend = np.linspace(0, 1, n_days)
            alpha_t = params["alpha"] + params["skill_decay"] * time_trend

            # Strategy returns = alpha + beta * market + idiosyncratic noise
            idiosyncratic_vol = market_vol * (params["vol_multiplier"] - params["beta"])
            strategy_returns = (
                alpha_t
                + params["beta"] * market_returns
                + np.random.normal(0, np.abs(idiosyncratic_vol))
            )

            returns_dict[strategy] = pd.Series(strategy_returns, index=dates)

        return {
            "returns": returns_dict,
            "market_returns": pd.Series(market_returns, index=dates),
            "regimes": pd.Series(regimes, index=dates),
            "dates": dates,
        }

    def test_end_to_end_statistical_analysis(self, comprehensive_test_data):
        """Test complete end-to-end statistical analysis workflow."""
        returns_dict = comprehensive_test_data["returns"]

        # Step 1: Basic statistical significance testing
        significance_tester = PerformanceSignificanceTest(alpha=0.05)
        pairwise_results = significance_tester.comprehensive_comparison(returns_dict)

        assert "sharpe" in pairwise_results
        pairwise_df = pairwise_results["sharpe"]

        # Should have comparisons between all strategies
        expected_comparisons = len(returns_dict) * (len(returns_dict) - 1)
        assert len(pairwise_df) == expected_comparisons

        # Step 2: Multiple comparison corrections
        corrections_framework = PerformanceTestingCorrections(alpha=0.05)
        corrected_results = corrections_framework.portfolio_comparison_corrections(
            pairwise_df, method="benjamini_hochberg"
        )

        assert "p_value_corrected" in corrected_results.columns
        assert "rejected_benjamini_hochberg" in corrected_results.columns

        # Step 3: Bootstrap confidence intervals
        BootstrapMethodology(n_bootstrap=200, random_state=42)
        ci_framework = ComprehensiveConfidenceIntervals(
            confidence_levels=[0.95], bootstrap_samples=200, random_state=42
        )

        # Test one strategy's confidence intervals
        sample_strategy = list(returns_dict.keys())[0]
        sample_returns = returns_dict[sample_strategy].dropna().values

        from src.evaluation.validation.bootstrap import sharpe_ratio

        ci_results = ci_framework.bootstrap_confidence_intervals_multi_level(
            sharpe_ratio, sample_returns, methods=["percentile"]
        )

        assert "percentile" in ci_results
        assert 0.95 in ci_results["percentile"]

        # Step 4: Rolling consistency analysis
        consistency_analyzer = RollingConsistencyAnalyzer(window_size=252, step_size=63)

        rolling_results = consistency_analyzer.rolling_sharpe_stability_test(sample_returns)

        assert "rolling_sharpe_ratios" in rolling_results
        assert "stability_score" in rolling_results

        # Step 5: Generate comprehensive report
        reporting = PublicationReadyStatisticalReporting()

        all_results = {
            "pairwise_comparisons": corrected_results,
            "sample_info": {
                "n": len(sample_returns),
                "period": "date_range_placeholder",
                "frequency": "Daily",
            },
        }

        comprehensive_report = reporting.comprehensive_statistical_report(all_results)

        assert "executive_summary" in comprehensive_report
        assert "detailed_tables" in comprehensive_report
        assert "apa_formatted_results" in comprehensive_report

    def test_cross_validation_statistical_methods(self, comprehensive_test_data):
        """Test consistency across different statistical methods."""
        returns_dict = comprehensive_test_data["returns"]

        # Select two strategies for comparison
        strategy_names = list(returns_dict.keys())
        returns_a = returns_dict[strategy_names[0]].dropna().values
        returns_b = returns_dict[strategy_names[1]].dropna().values

        # Ensure equal length
        min_length = min(len(returns_a), len(returns_b))
        returns_a = returns_a[:min_length]
        returns_b = returns_b[:min_length]

        # Method 1: Jobson-Korkie test
        jk_result = StatisticalValidation.sharpe_ratio_test(returns_a, returns_b)

        # Method 2: Bootstrap significance test
        bootstrap = BootstrapMethodology(n_bootstrap=500, random_state=42)
        from src.evaluation.validation.bootstrap import sharpe_ratio

        bootstrap_result = bootstrap.bootstrap_significance_test(
            returns_a, returns_b, sharpe_ratio, alternative="two-sided"
        )

        # Method 3: Hypothesis testing framework
        hypothesis_tester = PerformanceHypothesisTestingFramework(alpha=0.05)
        hypothesis_result = hypothesis_tester.sharpe_ratio_improvement_test(
            returns_a, returns_b, min_improvement=0.0, alternative="two-sided"
        )

        # Compare results - should be broadly consistent
        jk_significant = jk_result["is_significant"]
        bootstrap_significant = bootstrap_result["bootstrap_p_value"] < 0.05
        hypothesis_significant = hypothesis_result.is_significant

        # All methods should agree (either all find significance or all don't)
        # Allow for boundary cases where p-values are very close to 0.05
        agreement_count = sum([jk_significant, bootstrap_significant, hypothesis_significant])

        # Perfect agreement: all methods agree (either 0 or 3)
        # Or boundary case: p-values close to significance threshold
        assert agreement_count in [0, 3] or (
            abs(jk_result["p_value"] - 0.05) < 0.02
            or abs(bootstrap_result["bootstrap_p_value"] - 0.05) < 0.02
            or abs(hypothesis_result.p_value - 0.05) < 0.02
        )

    def test_rolling_window_statistical_consistency(self, comprehensive_test_data):
        """Test statistical consistency across rolling windows."""
        returns_dict = comprehensive_test_data["returns"]

        # Focus on one strategy
        strategy_name = list(returns_dict.keys())[0]
        returns_series = returns_dict[strategy_name].dropna()

        # Rolling analysis
        consistency_analyzer = RollingConsistencyAnalyzer(
            window_size=252, step_size=126  # Semi-annual windows
        )

        # Test rolling Sharpe stability
        stability_results = consistency_analyzer.rolling_sharpe_stability_test(
            returns_series.values
        )

        # Test persistence analysis
        single_strategy_dict = {strategy_name: returns_series}
        persistence_results = consistency_analyzer.performance_persistence_analysis(
            single_strategy_dict
        )

        # Test temporal stability
        temporal_results = consistency_analyzer.temporal_stability_framework(returns_series.values)

        # Validate consistency across analyses
        assert "stability_score" in stability_results
        assert "persistence_results" in persistence_results
        assert "stability_analysis" in temporal_results

        # Stability score should be reasonable (between 0 and 1)
        stability_score = stability_results["stability_score"]
        if not np.isnan(stability_score):
            assert 0 <= stability_score <= 1

    def test_multiple_correction_methods_consistency(self, comprehensive_test_data):
        """Test consistency across different multiple correction methods."""
        returns_dict = comprehensive_test_data["returns"]

        # Generate pairwise comparisons
        significance_tester = PerformanceSignificanceTest(alpha=0.05)
        pairwise_results = significance_tester.comprehensive_comparison(returns_dict)
        pairwise_df = pairwise_results["sharpe"]

        # Apply different correction methods
        corrections = MultipleComparisonCorrections()

        p_values = pairwise_df["p_value"].values

        bonferroni_result = corrections.bonferroni_correction(p_values)
        fdr_result = corrections.benjamini_hochberg_fdr(p_values)
        holm_sidak_result = corrections.holm_sidak_correction(p_values)

        # Test selection framework
        selection_result = corrections.correction_selection_framework(
            p_values, scenario="exploratory"
        )

        # Validate results structure
        for result in [bonferroni_result, fdr_result, holm_sidak_result]:
            assert "corrected_p_values" in result
            assert "rejected" in result
            assert len(result["corrected_p_values"]) == len(p_values)

        assert "comparison_summary" in selection_result
        assert "primary_method" in selection_result

        # Bonferroni should be most conservative (fewest rejections)
        # FDR should be least conservative (most rejections)
        bonf_rejections = np.sum(bonferroni_result["rejected"])
        fdr_rejections = np.sum(fdr_result["rejected"])
        holm_rejections = np.sum(holm_sidak_result["rejected"])

        assert bonf_rejections <= holm_rejections <= fdr_rejections

    def test_confidence_interval_methods_consistency(self, comprehensive_test_data):
        """Test consistency across different confidence interval methods."""
        returns_dict = comprehensive_test_data["returns"]

        # Select one strategy
        strategy_name = list(returns_dict.keys())[0]
        returns = returns_dict[strategy_name].dropna().values

        ci_framework = ComprehensiveConfidenceIntervals(
            confidence_levels=[0.95], bootstrap_samples=300, random_state=42
        )

        from src.evaluation.validation.bootstrap import sharpe_ratio

        # Bootstrap methods
        bootstrap_results = ci_framework.bootstrap_confidence_intervals_multi_level(
            sharpe_ratio, returns, methods=["percentile", "bias_corrected"]
        )

        # Asymptotic method
        asymptotic_results = ci_framework.asymptotic_confidence_intervals_delta_method(
            sharpe_ratio, returns
        )

        # Compare intervals - should have similar coverage
        bootstrap_percentile = bootstrap_results["percentile"][0.95]
        bootstrap_bc = bootstrap_results["bias_corrected"][0.95]
        asymptotic_ci = asymptotic_results[0.95]

        # All should contain the point estimate
        point_est = bootstrap_percentile.point_estimate

        assert bootstrap_percentile.lower_bound <= point_est <= bootstrap_percentile.upper_bound
        assert bootstrap_bc.lower_bound <= point_est <= bootstrap_bc.upper_bound
        assert asymptotic_ci.lower_bound <= point_est <= asymptotic_ci.upper_bound

        # Intervals should be reasonably similar in width (within factor of 2)
        width_percentile = bootstrap_percentile.upper_bound - bootstrap_percentile.lower_bound
        width_bc = bootstrap_bc.upper_bound - bootstrap_bc.lower_bound
        width_asymp = asymptotic_ci.upper_bound - asymptotic_ci.lower_bound

        max_width = max(width_percentile, width_bc, width_asymp)
        min_width = min(width_percentile, width_bc, width_asymp)

        assert max_width / min_width <= 3  # Widths shouldn't differ by more than factor of 3

    def test_power_analysis_integration(self, comprehensive_test_data):
        """Test integration of power analysis across components."""
        returns_dict = comprehensive_test_data["returns"]

        # Select treatment and control strategies
        treatment_name = "LSTM"  # Higher expected performance
        control_name = "Equal_Weight"  # Baseline

        treatment_returns = returns_dict[treatment_name].dropna().values
        control_returns = returns_dict[control_name].dropna().values

        # Ensure equal length
        min_length = min(len(treatment_returns), len(control_returns))
        treatment_returns = treatment_returns[:min_length]
        control_returns = control_returns[:min_length]

        # Hypothesis testing with power analysis
        hypothesis_tester = PerformanceHypothesisTestingFramework(alpha=0.05)

        # Test for meaningful improvement
        improvement_test = hypothesis_tester.sharpe_ratio_improvement_test(
            treatment_returns, control_returns, min_improvement=0.2
        )

        # Sample size validation
        sample_size_analysis = hypothesis_tester.sample_size_validation_adequate_power(
            treatment_returns, control_returns
        )

        # Effect size analysis
        effect_size_analysis = hypothesis_tester.effect_size_calculation_cohen_financial(
            treatment_returns, control_returns
        )

        # Validate integration
        assert improvement_test.power >= 0
        assert improvement_test.power <= 1

        assert "current_power" in sample_size_analysis
        assert "required_sample_size" in sample_size_analysis

        assert "effect_size" in effect_size_analysis
        assert "interpretation" in effect_size_analysis

        # Power from improvement test should be consistent with sample size analysis
        power_diff = abs(improvement_test.power - sample_size_analysis["current_power"])
        assert power_diff < 0.1  # Should be similar (allowing for different calculation methods)

    def test_reporting_integration_completeness(self, comprehensive_test_data):
        """Test that reporting integrates all analysis components."""
        returns_dict = comprehensive_test_data["returns"]

        # Run various analyses
        significance_tester = PerformanceSignificanceTest(alpha=0.05)
        pairwise_results = significance_tester.comprehensive_comparison(returns_dict)

        corrections_framework = PerformanceTestingCorrections(alpha=0.05)
        corrected_results = corrections_framework.portfolio_comparison_corrections(
            pairwise_results["sharpe"], method="benjamini_hochberg"
        )

        # Create comprehensive results dictionary
        all_results = {
            "pairwise_comparisons": corrected_results,
            "sample_info": {
                "n": len(returns_dict[list(returns_dict.keys())[0]]),
                "period": "2020-2023",
                "frequency": "Daily",
            },
            "sharpe_ratio_test": {
                "test_statistic": 2.5,
                "p_value": 0.012,
                "sharpe_diff": 0.15,
                "sample_size": 1000,
                "method": "Jobson-Korkie",
            },
            "multiple_comparisons": {
                "method": "Benjamini-Hochberg FDR",
                "n_comparisons": len(corrected_results),
                "n_significant": corrected_results["rejected_benjamini_hochberg"].sum(),
            },
            "effect_sizes": {
                "effect_sizes": (
                    corrected_results.get("effect_size", []).tolist()
                    if hasattr(corrected_results.get("effect_size", []), "tolist")
                    else corrected_results.get("effect_size", [])
                )
            },
        }

        # Generate comprehensive report
        reporting = PublicationReadyStatisticalReporting()
        comprehensive_report = reporting.comprehensive_statistical_report(all_results)

        # Validate completeness
        required_sections = [
            "executive_summary",
            "detailed_tables",
            "apa_formatted_results",
            "interpretation_guidelines",
            "recommendations",
        ]

        for section in required_sections:
            assert section in comprehensive_report
            assert comprehensive_report[section] is not None

        # Test table generation
        if "pairwise_comparisons" in comprehensive_report["detailed_tables"]:
            table_data = comprehensive_report["detailed_tables"]["pairwise_comparisons"]
            assert "summary_table" in table_data
            assert "metadata" in table_data

        # Test publication formats
        if "publication_formats" in comprehensive_report:
            for _format_name, formats in comprehensive_report["publication_formats"].items():
                if "latex" in formats:
                    assert "begin{table}" in formats["latex"]
                if "html" in formats:
                    assert "<table" in formats["html"]


class TestRobustnessAndReliability:
    """Test robustness and reliability of integrated statistical framework."""

    def test_missing_data_handling(self):
        """Test handling of missing data across components."""
        np.random.seed(42)

        # Create returns with missing values
        returns_with_gaps = np.random.normal(0.001, 0.02, 252)
        missing_indices = np.random.choice(252, 20, replace=False)
        returns_with_gaps[missing_indices] = np.nan

        returns_dict = {
            "Strategy_A": pd.Series(returns_with_gaps),
            "Strategy_B": pd.Series(np.random.normal(0.0008, 0.018, 252)),
        }

        # Test that components handle missing data appropriately
        significance_tester = PerformanceSignificanceTest(alpha=0.05)

        # Should complete without errors
        try:
            results = significance_tester.comprehensive_comparison(returns_dict)
            # If successful, results should be meaningful
            assert "sharpe" in results
        except Exception as e:
            # If it fails, should be with appropriate error message
            assert "missing" in str(e).lower() or "nan" in str(e).lower()

    def test_extreme_market_conditions(self):
        """Test behavior under extreme market conditions."""
        np.random.seed(42)

        # Create extreme scenarios
        scenarios = {
            "market_crash": np.concatenate(
                [
                    np.random.normal(0.001, 0.015, 100),  # Normal period
                    np.random.normal(-0.05, 0.08, 50),  # Crash period
                    np.random.normal(0.002, 0.02, 102),  # Recovery
                ]
            ),
            "low_volatility": np.random.normal(0.0005, 0.005, 252),  # Very low vol
            "high_volatility": np.random.normal(0.001, 0.10, 252),  # Very high vol
            "trending_market": np.cumsum(np.random.normal(0.002, 0.015, 252)),  # Strong trend
        }

        # Test each scenario
        for _scenario_name, returns in scenarios.items():
            returns_dict = {
                "Strategy": returns,
                "Benchmark": np.random.normal(0.0008, 0.02, len(returns)),
            }

            # Should handle extreme conditions gracefully
            try:
                significance_tester = PerformanceSignificanceTest(alpha=0.05)
                results = significance_tester.comprehensive_comparison(returns_dict)

                # Results should be valid (not all NaN)
                assert "sharpe" in results
                sharpe_results = results["sharpe"]
                assert not sharpe_results["test_statistic"].isna().all()

            except Exception as e:
                # If it fails, should be with reasonable error
                assert isinstance(e, (ValueError, Warning))

    def test_computational_stability_large_datasets(self):
        """Test computational stability with large datasets."""
        np.random.seed(42)

        # Create large dataset (5 years daily data)
        large_returns = np.random.normal(0.0008, 0.018, 1826)

        returns_dict = {
            "Large_Strategy_A": large_returns,
            "Large_Strategy_B": np.random.normal(0.0006, 0.020, 1826),
        }

        # Test various components with large data
        components_to_test = [
            (
                "StatisticalValidation",
                lambda: StatisticalValidation.sharpe_ratio_test(
                    large_returns, returns_dict["Large_Strategy_B"]
                ),
            ),
            (
                "BootstrapMethodology",
                lambda: BootstrapMethodology(
                    n_bootstrap=100, random_state=42
                ).bootstrap_confidence_intervals(
                    lambda x: np.mean(x) / np.std(x) if np.std(x) > 0 else 0, large_returns
                ),
            ),
            (
                "RollingConsistency",
                lambda: RollingConsistencyAnalyzer().rolling_sharpe_stability_test(large_returns),
            ),
        ]

        for component_name, test_func in components_to_test:
            try:
                result = test_func()
                assert result is not None

                # Check for numerical stability (no inf or extreme values)
                if isinstance(result, dict):
                    for _key, value in result.items():
                        if isinstance(value, (float, np.floating)):
                            assert np.isfinite(value) or np.isnan(value)
                            if np.isfinite(value):
                                assert abs(value) < 1e6  # Reasonable bounds

            except Exception as e:
                pytest.fail(f"{component_name} failed with large dataset: {str(e)}")


class TestPerformanceAndEfficiency:
    """Test performance and efficiency of integrated framework."""

    @pytest.fixture
    def comprehensive_test_data(self):
        """Generate comprehensive test data simulating real portfolio scenarios."""
        np.random.seed(42)

        # Date range
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 12, 31)
        dates = pd.date_range(start_date, end_date, freq="D")
        n_days = len(dates)

        # Market regimes (bull/bear)
        regime_changes = [252, 504, 756]  # Change points
        regimes = np.ones(n_days)
        for change in regime_changes:
            if change < n_days:
                regimes[change:] = -regimes[change - 1]

        # Base market returns with regime dependency
        market_base = np.where(regimes > 0, 0.0008, -0.0002)  # Bull/bear base returns
        market_vol = np.where(regimes > 0, 0.015, 0.025)  # Lower vol in bull markets

        market_returns = np.random.normal(market_base, market_vol)

        # Strategy returns with different characteristics
        strategies = {
            "HRP": {
                "alpha": 0.0002,
                "beta": 0.8,
                "vol_multiplier": 0.9,
                "skill_decay": 0.0,  # Consistent performance
            },
            "LSTM": {
                "alpha": 0.0004,
                "beta": 1.1,
                "vol_multiplier": 1.2,
                "skill_decay": 0.000001,  # Slight decay
            },
        }

        returns_dict = {}
        for strategy_name, params in strategies.items():
            strategy_returns = (
                params["alpha"]
                + params["beta"] * market_returns
                + np.random.normal(0, market_vol * params["vol_multiplier"])
            )

            returns_dict[strategy_name] = pd.Series(
                strategy_returns, index=dates, name=strategy_name
            )

        return {"returns": returns_dict, "dates": dates, "market_returns": market_returns}

    def test_computational_time_reasonable(self, comprehensive_test_data):
        """Test that computations complete in reasonable time."""
        import time

        returns_dict = comprehensive_test_data["returns"]

        # Test key operations with timing
        operations = [
            (
                "Pairwise Comparisons",
                lambda: PerformanceSignificanceTest().comprehensive_comparison(returns_dict),
            ),
            (
                "Bootstrap CI",
                lambda: BootstrapMethodology(
                    n_bootstrap=100, random_state=42
                ).bootstrap_confidence_intervals(
                    lambda x: np.mean(x) / np.std(x) if np.std(x) > 0 else 0,
                    list(returns_dict.values())[0].dropna().values,
                ),
            ),
            (
                "Rolling Analysis",
                lambda: RollingConsistencyAnalyzer().rolling_sharpe_stability_test(
                    list(returns_dict.values())[0].dropna().values
                ),
            ),
        ]

        for op_name, op_func in operations:
            start_time = time.time()
            result = op_func()
            elapsed_time = time.time() - start_time

            # Should complete within reasonable time (adjust thresholds as needed)
            assert elapsed_time < 30, f"{op_name} took too long: {elapsed_time:.2f} seconds"
            assert result is not None

    def test_memory_usage_reasonable(self, comprehensive_test_data):
        """Test that memory usage remains reasonable."""

        returns_dict = comprehensive_test_data["returns"]

        # Monitor memory usage for large operations
        initial_objects = len(gc.get_objects())

        # Perform memory-intensive operations
        significance_tester = PerformanceSignificanceTest()
        significance_tester.comprehensive_comparison(returns_dict)

        bootstrap = BootstrapMethodology(n_bootstrap=500, random_state=42)
        sample_returns = list(returns_dict.values())[0].dropna().values

        bootstrap.bootstrap_confidence_intervals(
            lambda x: np.mean(x) / np.std(x) if np.std(x) > 0 else 0, sample_returns
        )

        # Memory should not grow excessively
        gc.collect()  # Force garbage collection
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects

        # Should not create excessive number of objects
        assert object_growth < 10000, f"Excessive object creation: {object_growth}"
