"""Comprehensive data quality validation framework.

This module provides automated quality reports, flagging system, statistical outlier
detection, and data integrity monitoring for financial time series data.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from src.config.data import ValidationConfig


class DataQualityValidator:
    """
    Comprehensive data quality validation system for financial time series.

    Provides automated quality reports, statistical outlier detection,
    and data integrity monitoring with configurable thresholds.
    """

    def __init__(self, config: ValidationConfig):
        """
        Initialize data quality validator.

        Args:
            config: Validation configuration with quality thresholds
        """
        self.config = config
        self.validation_results = {}
        self.quality_flags = []

        # Ensure report directory exists
        if self.config.generate_reports:
            Path(self.config.report_output_dir).mkdir(parents=True, exist_ok=True)

    def validate_complete_dataset(
        self,
        data_dict: dict[str, pd.DataFrame],
        universe_tickers: list[str] | None = None,
        generate_report: bool = True,
    ) -> dict[str, Any]:
        """Perform comprehensive validation on complete dataset.

        Args:
            data_dict: Dictionary of DataFrames to validate
            universe_tickers: Expected universe of tickers
            generate_report: Whether to generate detailed validation report

        Returns:
            Comprehensive validation results
        """

        validation_summary = {
            "timestamp": datetime.now().isoformat(),
            "overall_quality_score": 0.0,
            "data_types_validated": list(data_dict.keys()),
            "validation_flags": [],
            "component_scores": {},
            "ticker_analysis": {},
            "recommendations": [],
            "auto_fixes_applied": [],
        }

        # Primary data validation (prices/close)
        primary_data = data_dict.get("close")
        if primary_data is None:
            primary_data = data_dict.get("prices")
        if primary_data is not None and not primary_data.empty:
            primary_results = self.validate_price_data(primary_data, universe_tickers)
            validation_summary["component_scores"]["prices"] = primary_results

        # Returns validation
        returns_data = data_dict.get("returns")
        if returns_data is not None and not returns_data.empty:
            returns_results = self.validate_returns_data(returns_data)
            validation_summary["component_scores"]["returns"] = returns_results

        # Volume validation
        volume_data = data_dict.get("volume")
        if volume_data is not None and not volume_data.empty:
            volume_results = self.validate_volume_data(volume_data, primary_data)
            validation_summary["component_scores"]["volume"] = volume_results

        # Cross-validation between data types
        if len(data_dict) > 1:
            cross_validation = self.cross_validate_datasets(data_dict)
            validation_summary["component_scores"]["cross_validation"] = cross_validation

        # Calculate overall quality score
        validation_summary["overall_quality_score"] = self._calculate_overall_score(
            validation_summary["component_scores"]
        )

        # Generate recommendations
        validation_summary["recommendations"] = self._generate_recommendations(validation_summary)

        # Apply auto-fixes if enabled
        if self.config.auto_fix_enabled:
            validation_summary["auto_fixes_applied"] = self._apply_auto_fixes(data_dict)

        # Generate detailed report
        if generate_report and self.config.generate_reports:
            report_path = self._generate_validation_report(validation_summary, data_dict)
            validation_summary["report_path"] = report_path

        return validation_summary

    def validate_price_data(
        self, prices_df: pd.DataFrame, universe_tickers: list[str] | None = None
    ) -> dict[str, Any]:
        """Validate price data comprehensively.

        Args:
            prices_df: Price DataFrame to validate
            universe_tickers: Expected universe tickers

        Returns:
            Price validation results
        """
        results = {
            "data_completeness": 0.0,
            "price_range_validation": {"passed": True, "issues": []},
            "temporal_consistency": {"passed": True, "issues": []},
            "outlier_detection": {"method": self.config.outlier_detection_method, "outliers": {}},
            "business_day_alignment": {"passed": True, "issues": []},
            "ticker_coverage": {"expected": 0, "actual": 0, "missing": []},
            "quality_score": 0.0,
            "recommendations": [],
        }

        if prices_df.empty:
            results["quality_score"] = 0.0
            results["recommendations"].append("Price data is empty")
            return results

        # Data completeness check
        total_cells = prices_df.size
        missing_cells = prices_df.isna().sum().sum()
        results["data_completeness"] = 1.0 - (missing_cells / total_cells)

        # Ticker coverage check
        if universe_tickers:
            results["ticker_coverage"]["expected"] = len(universe_tickers)
            results["ticker_coverage"]["actual"] = len(prices_df.columns)
            results["ticker_coverage"]["missing"] = [
                t for t in universe_tickers if t not in prices_df.columns
            ]

        # Price range validation
        if self.config.price_range_validation:
            for ticker in prices_df.columns:
                price_series = prices_df[ticker].dropna()
                if len(price_series) == 0:
                    continue

                min_price = price_series.min()
                max_price = price_series.max()

                if min_price < self.config.min_price:
                    results["price_range_validation"]["issues"].append(
                        f"{ticker}: Price below minimum ({min_price:.4f} < {self.config.min_price})"
                    )
                    results["price_range_validation"]["passed"] = False

                if max_price > self.config.max_price:
                    results["price_range_validation"]["issues"].append(
                        f"{ticker}: Price above maximum ({max_price:.2f} > {self.config.max_price})"
                    )
                    results["price_range_validation"]["passed"] = False

        # Outlier detection
        results["outlier_detection"]["outliers"] = self._detect_price_outliers(prices_df)

        # Temporal consistency
        if self.config.temporal_consistency_check:
            temporal_issues = self._check_temporal_consistency(prices_df)
            results["temporal_consistency"]["issues"] = temporal_issues
            results["temporal_consistency"]["passed"] = len(temporal_issues) == 0

        # Business day alignment
        if self.config.validate_business_days:
            business_day_issues = self._check_business_day_alignment(prices_df)
            results["business_day_alignment"]["issues"] = business_day_issues
            results["business_day_alignment"]["passed"] = len(business_day_issues) == 0

        # Calculate quality score
        score_components = [
            results["data_completeness"],
            1.0 if results["price_range_validation"]["passed"] else 0.5,
            1.0 if results["temporal_consistency"]["passed"] else 0.7,
            1.0 if results["business_day_alignment"]["passed"] else 0.8,
        ]

        # Penalize for outliers
        total_outliers = sum(
            len(outliers) for outliers in results["outlier_detection"]["outliers"].values()
        )
        total_data_points = prices_df.notna().sum().sum()
        outlier_penalty = (
            min(0.3, (total_outliers / total_data_points) * 2) if total_data_points > 0 else 0
        )

        results["quality_score"] = max(0.0, np.mean(score_components) - outlier_penalty)

        # Generate recommendations
        if results["data_completeness"] < 0.9:
            results["recommendations"].append(
                f"Low data completeness: {results['data_completeness']:.2%}. Consider gap filling."
            )

        if not results["price_range_validation"]["passed"]:
            results["recommendations"].append("Price range validation failed. Review data sources.")

        if total_outliers > total_data_points * 0.01:  # > 1% outliers
            results["recommendations"].append(
                f"High outlier count: {total_outliers}. Consider outlier cleaning."
            )

        return results

    def validate_returns_data(self, returns_df: pd.DataFrame) -> dict[str, Any]:
        """Validate returns data.

        Args:
            returns_df: Returns DataFrame to validate

        Returns:
            Returns validation results
        """
        results = {
            "statistical_properties": {},
            "extreme_returns": {},
            "return_distribution": {},
            "autocorrelation_test": {},
            "quality_score": 0.0,
            "recommendations": [],
        }

        if returns_df.empty:
            results["quality_score"] = 0.0
            return results

        # Statistical properties per ticker
        for ticker in returns_df.columns:
            returns_series = returns_df[ticker].dropna()
            if len(returns_series) < 10:
                continue

            # Basic statistics
            stats_info = {
                "mean": float(returns_series.mean()),
                "std": float(returns_series.std()),
                "skewness": float(stats.skew(returns_series)),
                "kurtosis": float(stats.kurtosis(returns_series)),
                "min": float(returns_series.min()),
                "max": float(returns_series.max()),
            }
            results["statistical_properties"][ticker] = stats_info

            # Extreme returns detection
            extreme_threshold = self.config.price_change_threshold
            extreme_returns = returns_series[abs(returns_series) > extreme_threshold]
            if len(extreme_returns) > 0:
                results["extreme_returns"][ticker] = {
                    "count": len(extreme_returns),
                    "dates": extreme_returns.index.strftime("%Y-%m-%d").tolist(),
                    "values": extreme_returns.tolist(),
                }

        # Overall return distribution analysis
        all_returns = returns_df.values.flatten()
        all_returns = all_returns[~np.isnan(all_returns)]

        if len(all_returns) > 0:
            results["return_distribution"] = {
                "mean": float(np.mean(all_returns)),
                "std": float(np.std(all_returns)),
                "skewness": float(stats.skew(all_returns)),
                "kurtosis": float(stats.kurtosis(all_returns)),
                "normality_test": {
                    "statistic": float(stats.jarque_bera(all_returns)[0]),
                    "p_value": float(stats.jarque_bera(all_returns)[1]),
                    "is_normal": bool(stats.jarque_bera(all_returns)[1] > 0.05),
                },
            }

        # Quality score calculation
        quality_factors = []

        # Check for reasonable return statistics
        if results["return_distribution"]:
            daily_vol = results["return_distribution"]["std"]
            annual_vol = daily_vol * np.sqrt(252)

            # Reasonable volatility range (5% to 100% annually)
            vol_score = 1.0 if 0.05 <= annual_vol <= 1.0 else max(0.3, 1.0 - abs(annual_vol - 0.3))
            quality_factors.append(vol_score)

            # Reasonable skewness (between -2 and 2)
            skew_score = 1.0 if abs(results["return_distribution"]["skewness"]) <= 2 else 0.7
            quality_factors.append(skew_score)

        # Extreme returns penalty
        extreme_count = sum(info["count"] for info in results["extreme_returns"].values())
        total_returns = len(all_returns)
        extreme_ratio = extreme_count / total_returns if total_returns > 0 else 0
        extreme_score = max(
            0.5, 1.0 - extreme_ratio * 5
        )  # Penalize heavily for > 20% extreme returns
        quality_factors.append(extreme_score)

        results["quality_score"] = np.mean(quality_factors) if quality_factors else 0.0

        # Recommendations
        if extreme_count > total_returns * 0.05:  # > 5% extreme returns
            results["recommendations"].append(
                f"High extreme returns: {extreme_count} ({extreme_ratio:.1%}). Review data quality."
            )

        if (
            results["return_distribution"]
            and not results["return_distribution"]["normality_test"]["is_normal"]
        ):
            results["recommendations"].append(
                "Returns are not normally distributed. Consider transformations."
            )

        return results

    def validate_volume_data(
        self, volume_df: pd.DataFrame, prices_df: pd.DataFrame | None = None
    ) -> dict[str, Any]:
        """Validate volume data.

        Args:
            volume_df: Volume DataFrame to validate
            prices_df: Optional price data for consistency checks

        Returns:
            Volume validation results
        """
        results = {
            "zero_volume_analysis": {},
            "volume_price_consistency": {},
            "volume_outliers": {},
            "quality_score": 0.0,
            "recommendations": [],
        }

        if volume_df.empty:
            results["quality_score"] = 0.0
            return results

        # Zero volume analysis
        for ticker in volume_df.columns:
            volume_series = volume_df[ticker]
            total_days = len(volume_series)
            zero_volume_days = (volume_series == 0).sum()
            na_volume_days = volume_series.isna().sum()

            results["zero_volume_analysis"][ticker] = {
                "total_days": total_days,
                "zero_volume_days": int(zero_volume_days),
                "na_volume_days": int(na_volume_days),
                "zero_volume_ratio": (
                    float(zero_volume_days / total_days) if total_days > 0 else 0.0
                ),
            }

        # Volume-price consistency check
        if prices_df is not None and self.config.volume_consistency_check:
            for ticker in volume_df.columns:
                if ticker not in prices_df.columns:
                    continue

                volume_series = volume_df[ticker]
                price_series = prices_df[ticker]

                # Check for zero volume on days with price changes
                price_changes = price_series.pct_change(fill_method=None).abs()
                significant_price_changes = price_changes > 0.05  # > 5% price change
                zero_volume_on_price_change = (
                    (volume_series == 0) & significant_price_changes
                ).sum()

                if zero_volume_on_price_change > 0:
                    results["volume_price_consistency"][ticker] = {
                        "zero_volume_on_price_change_days": int(zero_volume_on_price_change),
                        "inconsistency_ratio": (
                            float(zero_volume_on_price_change / significant_price_changes.sum())
                            if significant_price_changes.sum() > 0
                            else 0.0
                        ),
                    }

        # Volume outlier detection
        for ticker in volume_df.columns:
            volume_series = volume_df[ticker].replace(0, np.nan).dropna()
            if len(volume_series) < 10:
                continue

            outliers = self._detect_volume_outliers(volume_series, ticker)
            if outliers:
                results["volume_outliers"][ticker] = outliers

        # Quality score calculation
        quality_factors = []

        # Penalize for excessive zero volume
        avg_zero_ratio = np.mean(
            [info["zero_volume_ratio"] for info in results["zero_volume_analysis"].values()]
        )
        zero_volume_score = max(0.3, 1.0 - avg_zero_ratio * 2)
        quality_factors.append(zero_volume_score)

        # Penalize for volume-price inconsistencies
        if results["volume_price_consistency"]:
            avg_inconsistency = np.mean(
                [
                    info["inconsistency_ratio"]
                    for info in results["volume_price_consistency"].values()
                ]
            )
            consistency_score = max(0.5, 1.0 - avg_inconsistency)
            quality_factors.append(consistency_score)
        else:
            quality_factors.append(0.8)  # No price data to check against

        # Penalize for excessive outliers
        total_outliers = sum(
            len(outliers.get("outlier_dates", []))
            for outliers in results["volume_outliers"].values()
        )
        total_volume_points = volume_df.notna().sum().sum()
        outlier_ratio = total_outliers / total_volume_points if total_volume_points > 0 else 0
        outlier_score = max(0.5, 1.0 - outlier_ratio * 10)
        quality_factors.append(outlier_score)

        results["quality_score"] = np.mean(quality_factors)

        # Recommendations
        if avg_zero_ratio > 0.1:
            results["recommendations"].append(
                f"High zero volume ratio: {avg_zero_ratio:.1%}. Review data sources."
            )

        if results["volume_price_consistency"]:
            high_inconsistency_tickers = [
                ticker
                for ticker, info in results["volume_price_consistency"].items()
                if info["inconsistency_ratio"] > 0.2
            ]
            if high_inconsistency_tickers:
                results["recommendations"].append(
                    f"Volume-price inconsistencies in {len(high_inconsistency_tickers)} tickers"
                )

        return results

    def cross_validate_datasets(self, data_dict: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Cross-validate consistency between different data types.

        Args:
            data_dict: Dictionary of DataFrames to cross-validate

        Returns:
            Cross-validation results
        """
        results = {
            "index_alignment": {"passed": True, "issues": []},
            "column_alignment": {"passed": True, "issues": []},
            "data_consistency": {"passed": True, "issues": []},
            "correlation_analysis": {},
            "quality_score": 0.0,
        }

        list(data_dict.keys())
        non_empty_data = {k: v for k, v in data_dict.items() if not v.empty}

        if len(non_empty_data) < 2:
            results["quality_score"] = 0.8  # Cannot cross-validate with single dataset
            return results

        # Index alignment check
        reference_index = None
        for data_type, df in non_empty_data.items():
            if reference_index is None:
                reference_index = df.index
                continue

            if not df.index.equals(reference_index):
                results["index_alignment"]["issues"].append(
                    f"{data_type} has different index than reference"
                )
                results["index_alignment"]["passed"] = False

        # Column alignment check
        reference_columns = None
        for data_type, df in non_empty_data.items():
            if reference_columns is None:
                reference_columns = set(df.columns)
                continue

            current_columns = set(df.columns)
            missing_cols = reference_columns - current_columns
            extra_cols = current_columns - reference_columns

            if missing_cols or extra_cols:
                results["column_alignment"]["issues"].append(
                    f"{data_type}: missing={list(missing_cols)}, extra={list(extra_cols)}"
                )
                results["column_alignment"]["passed"] = False

        # Data consistency checks (e.g., returns vs price changes)
        if "close" in non_empty_data and "returns" in non_empty_data:
            consistency_check = self._check_price_return_consistency(
                non_empty_data["close"], non_empty_data["returns"]
            )
            if not consistency_check["passed"]:
                results["data_consistency"]["issues"].extend(consistency_check["issues"])
                results["data_consistency"]["passed"] = False

        # Calculate quality score
        alignment_score = 1.0 if results["index_alignment"]["passed"] else 0.7
        column_score = 1.0 if results["column_alignment"]["passed"] else 0.8
        consistency_score = 1.0 if results["data_consistency"]["passed"] else 0.6

        results["quality_score"] = np.mean([alignment_score, column_score, consistency_score])

        return results

    def _detect_price_outliers(self, prices_df: pd.DataFrame) -> dict[str, list]:
        """Detect price outliers using configured method.

        Args:
            prices_df: Price DataFrame

        Returns:
            Dictionary of outliers per ticker
        """
        outliers = {}

        for ticker in prices_df.columns:
            price_series = prices_df[ticker].dropna()
            if len(price_series) < 10:
                continue

            if self.config.outlier_detection_method == "iqr":
                ticker_outliers = self._detect_outliers_iqr(price_series, ticker)
            elif self.config.outlier_detection_method == "zscore":
                ticker_outliers = self._detect_outliers_zscore(price_series, ticker)
            elif self.config.outlier_detection_method == "isolation_forest":
                ticker_outliers = self._detect_outliers_isolation_forest(price_series, ticker)
            else:
                ticker_outliers = []

            if ticker_outliers:
                outliers[ticker] = ticker_outliers

        return outliers

    def _detect_outliers_iqr(self, series: pd.Series, ticker: str) -> list[dict]:
        """Detect outliers using IQR method."""
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - self.config.outlier_threshold * iqr
        upper_bound = q3 + self.config.outlier_threshold * iqr

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outliers = series[outlier_mask]

        return [
            {"date": date.strftime("%Y-%m-%d"), "value": float(value), "type": "IQR"}
            for date, value in outliers.items()
        ]

    def _detect_outliers_zscore(self, series: pd.Series, ticker: str) -> list[dict]:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(series))
        outlier_mask = z_scores > self.config.outlier_threshold
        outliers = series[outlier_mask]

        return [
            {"date": date.strftime("%Y-%m-%d"), "value": float(value), "type": "Z-score"}
            for date, value in outliers.items()
        ]

    def _detect_outliers_isolation_forest(self, series: pd.Series, ticker: str) -> list[dict]:
        """Detect outliers using Isolation Forest."""
        try:
            clf = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = clf.fit_predict(series.values.reshape(-1, 1))
            outlier_mask = outlier_labels == -1
            outliers = series[outlier_mask]

            return [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "value": float(value),
                    "type": "Isolation Forest",
                }
                for date, value in outliers.items()
            ]
        except Exception:
            # Fallback to IQR method
            return self._detect_outliers_iqr(series, ticker)

    def _detect_volume_outliers(self, volume_series: pd.Series, ticker: str) -> dict[str, Any]:
        """Detect volume outliers."""
        # Use log-transformed volume for better outlier detection
        log_volume = np.log(volume_series)

        q1, q3 = log_volume.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 2.0 * iqr  # More lenient for volume
        upper_bound = q3 + 2.0 * iqr

        outlier_mask = (log_volume < lower_bound) | (log_volume > upper_bound)
        outliers = volume_series[outlier_mask]

        if len(outliers) > 0:
            return {
                "outlier_count": len(outliers),
                "outlier_dates": outliers.index.strftime("%Y-%m-%d").tolist(),
                "outlier_values": outliers.tolist(),
                "outlier_ratio": len(outliers) / len(volume_series),
            }

        return {}

    def _check_temporal_consistency(self, prices_df: pd.DataFrame) -> list[str]:
        """Check temporal consistency of price data."""
        issues = []

        # Check for duplicate dates
        duplicate_dates = prices_df.index.duplicated()
        if duplicate_dates.any():
            issues.append(f"Found {duplicate_dates.sum()} duplicate dates in index")

        # Check for non-monotonic dates
        if not prices_df.index.is_monotonic_increasing:
            issues.append("Index is not monotonic increasing")

        # Check for reasonable date range
        date_range = prices_df.index.max() - prices_df.index.min()
        if date_range.days > 365 * 20:  # More than 20 years
            issues.append(f"Very large date range: {date_range.days} days")

        return issues

    def _check_business_day_alignment(self, prices_df: pd.DataFrame) -> list[str]:
        """Check business day alignment."""
        issues = []

        # Check for weekends in data
        weekend_mask = prices_df.index.weekday >= 5  # Saturday=5, Sunday=6
        weekend_count = weekend_mask.sum()

        if weekend_count > 0:
            issues.append(f"Found {weekend_count} weekend dates in data")

        return issues

    def _check_price_return_consistency(
        self, prices_df: pd.DataFrame, returns_df: pd.DataFrame
    ) -> dict[str, Any]:
        """Check consistency between prices and returns."""
        result = {"passed": True, "issues": []}

        # Check a sample of tickers
        sample_tickers = list(set(prices_df.columns) & set(returns_df.columns))[:5]

        for ticker in sample_tickers:
            price_series = prices_df[ticker].dropna()
            return_series = returns_df[ticker].dropna()

            if len(price_series) < 2 or len(return_series) < 2:
                continue

            # Calculate returns from prices
            calculated_returns = price_series.pct_change().dropna()

            # Find common dates
            common_dates = calculated_returns.index.intersection(return_series.index)

            if len(common_dates) < 10:
                continue

            # Compare calculated vs provided returns
            calc_rets = calculated_returns.loc[common_dates]
            prov_rets = return_series.loc[common_dates]

            # Allow for small numerical differences
            differences = abs(calc_rets - prov_rets)
            large_differences = (differences > 0.001).sum()  # > 0.1% difference

            if large_differences > len(common_dates) * 0.05:  # > 5% of data
                result["passed"] = False
                result["issues"].append(
                    f"{ticker}: {large_differences}/{len(common_dates)} "
                    f"large price-return inconsistencies"
                )

        return result

    def _calculate_overall_score(self, component_scores: dict[str, Any]) -> float:
        """Calculate overall quality score from component scores."""
        scores = []
        weights = {"prices": 0.4, "returns": 0.25, "volume": 0.2, "cross_validation": 0.15}

        for component, weight in weights.items():
            if component in component_scores:
                component_score = component_scores[component].get("quality_score", 0.0)
                scores.append(component_score * weight)

        return sum(scores) / sum(
            weights[comp] for comp in component_scores.keys() if comp in weights
        )

    def _generate_recommendations(self, validation_summary: dict[str, Any]) -> list[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []

        overall_score = validation_summary["overall_quality_score"]

        if overall_score < self.config.quality_score_threshold:
            recommendations.append(
                f"Overall quality score ({overall_score:.3f}) below threshold "
                f"({self.config.quality_score_threshold})"
            )

        # Collect recommendations from components
        for component, results in validation_summary["component_scores"].items():
            if "recommendations" in results:
                recommendations.extend(
                    [f"{component}: {rec}" for rec in results["recommendations"]]
                )

        return recommendations

    def _apply_auto_fixes(self, data_dict: dict[str, pd.DataFrame]) -> list[str]:
        """Apply automatic fixes if enabled."""
        fixes_applied = []

        # This is a placeholder for auto-fix functionality
        # In practice, you would implement specific fixes based on common issues

        return fixes_applied

    def _generate_validation_report(
        self, validation_summary: dict[str, Any], data_dict: dict[str, pd.DataFrame]
    ) -> str:
        """Generate detailed validation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"data_quality_report_{timestamp}.json"
        report_path = os.path.join(self.config.report_output_dir, report_filename)

        # Add data summary to report
        data_summary = {}
        for data_type, df in data_dict.items():
            if not df.empty:
                data_summary[data_type] = {
                    "shape": df.shape,
                    "date_range": [df.index.min().isoformat(), df.index.max().isoformat()],
                    "columns": list(df.columns),
                    "missing_data_pct": float(df.isna().sum().sum() / df.size * 100),
                }

        validation_summary["data_summary"] = data_summary

        # Write report
        with open(report_path, "w") as f:
            json.dump(validation_summary, f, indent=2, default=str)

        return report_path

    def generate_quality_dashboard(
        self, validation_results: dict[str, Any], output_path: str | None = None
    ) -> str:
        """Generate HTML quality dashboard.

        Args:
            validation_results: Validation results dictionary
            output_path: Optional output path for dashboard

        Returns:
            Path to generated dashboard
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.config.report_output_dir, f"quality_dashboard_{timestamp}.html"
            )

        # Simple HTML dashboard template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{
                    background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px;
                }}
                .score {{ font-size: 24px; font-weight: bold; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .poor {{ color: red; }}
                .recommendations {{ background: #fff3cd; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Data Quality Dashboard</h1>

            <div class="metric">
                <h2>Overall Quality Score</h2>
                <div class="score {
                    self._get_score_class(validation_results['overall_quality_score'])
                }">
                    {validation_results['overall_quality_score']:.3f}
                </div>
            </div>

            <div class="metric">
                <h2>Component Scores</h2>
                <ul>
        """

        for component, results in validation_results.get("component_scores", {}).items():
            score = results.get("quality_score", 0.0)
            score_class = self._get_score_class(score)
            html_content += f'<li>{component}: <span class="{score_class}">{score:.3f}</span></li>'

        html_content += """
                </ul>
            </div>
        """

        if validation_results.get("recommendations"):
            html_content += """
            <div class="recommendations">
                <h2>Recommendations</h2>
                <ul>
            """
            for rec in validation_results["recommendations"]:
                html_content += f"<li>{rec}</li>"

            html_content += """
                </ul>
            </div>
            """

        html_content += f"""
            <div class="metric">
                <h2>Report Details</h2>
                <p>Generated: {validation_results['timestamp']}</p>
                <p>Data types validated: {', '.join(validation_results['data_types_validated'])}</p>
            </div>
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html_content)

        return output_path

    def _get_score_class(self, score: float) -> str:
        """Get CSS class for score color coding."""
        if score >= 0.8:
            return "good"
        elif score >= 0.6:
            return "warning"
        else:
            return "poor"
