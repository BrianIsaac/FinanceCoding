"""
Unit tests for rolling window generation and temporal validation.

Tests the core functionality of rolling window generation with strict
temporal validation and integrity monitoring.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.validation.rolling_validation import RollSplit, ValidationPeriod
from src.evaluation.validation.rolling_window_generator import (
    RollingWindowGenerator,
    WalkForwardAnalyzer,
    WindowGenerationConfig,
)


class TestRollingWindowGenerator:
    """Test rolling window generator functionality."""

    @pytest.fixture
    def sample_timestamps(self) -> list[pd.Timestamp]:
        """Generate sample timestamps for testing."""
        return pd.date_range(start="2020-01-01", end="2024-12-31", freq="D").tolist()

    @pytest.fixture
    def config(self) -> WindowGenerationConfig:
        """Default configuration for testing."""
        return WindowGenerationConfig(
            training_months=12,  # Shorter for testing
            validation_months=6,
            test_months=6,
            step_months=6,
            min_training_samples=100,  # Lower for testing
            min_validation_samples=20,
            min_test_samples=20,
        )

    @pytest.fixture
    def generator(self, config: WindowGenerationConfig) -> RollingWindowGenerator:
        """Rolling window generator instance."""
        return RollingWindowGenerator(config)

    def test_initialization(self, config: WindowGenerationConfig):
        """Test generator initialization."""
        generator = RollingWindowGenerator(config)
        assert generator.config == config
        assert generator._generation_stats == {}

    def test_generate_basic_windows(
        self, generator: RollingWindowGenerator, sample_timestamps: list[pd.Timestamp]
    ):
        """Test basic window generation."""
        windows = generator.generate_rolling_windows(sample_timestamps)

        assert len(windows) > 0
        assert all(isinstance(window, RollSplit) for window in windows)

        # Check first window structure
        first_window = windows[0]
        assert isinstance(first_window.train_period, ValidationPeriod)
        assert isinstance(first_window.validation_period, ValidationPeriod)
        assert isinstance(first_window.test_period, ValidationPeriod)

    def test_temporal_ordering(
        self, generator: RollingWindowGenerator, sample_timestamps: list[pd.Timestamp]
    ):
        """Test that windows maintain proper temporal ordering."""
        windows = generator.generate_rolling_windows(sample_timestamps)

        for window in windows:
            # Check period ordering within window
            assert window.train_period.start_date < window.train_period.end_date
            assert window.validation_period.start_date < window.validation_period.end_date
            assert window.test_period.start_date < window.test_period.end_date

            # Check sequential ordering
            assert window.train_period.end_date <= window.validation_period.start_date
            assert window.validation_period.end_date <= window.test_period.start_date

    def test_no_look_ahead_bias(
        self, generator: RollingWindowGenerator, sample_timestamps: list[pd.Timestamp]
    ):
        """Test that windows prevent look-ahead bias."""
        windows = generator.generate_rolling_windows(sample_timestamps)

        for window in windows:
            # Training should not overlap with validation
            assert window.train_period.end_date <= window.validation_period.start_date

            # Validation should not overlap with test
            assert window.validation_period.end_date <= window.test_period.start_date

    def test_window_progression(
        self, generator: RollingWindowGenerator, sample_timestamps: list[pd.Timestamp]
    ):
        """Test proper window progression."""
        windows = generator.generate_rolling_windows(sample_timestamps)

        if len(windows) > 1:
            for i in range(1, len(windows)):
                prev_window = windows[i - 1]
                curr_window = windows[i]

                # Current window should start after previous
                assert curr_window.train_period.start_date > prev_window.train_period.start_date

                # Check step size (approximately step_months)
                step_days = (
                    curr_window.train_period.start_date - prev_window.train_period.start_date
                ).days

                # Allow some tolerance for month variations (±30 days)
                expected_days = generator.config.step_months * 30.44  # Average days per month
                assert abs(step_days - expected_days) < 60

    def test_data_sufficiency_validation(self, config: WindowGenerationConfig):
        """Test data sufficiency validation."""
        # Create sparse timestamps
        sparse_timestamps = pd.date_range(
            start="2020-01-01", end="2020-06-30", freq="10D"  # Every 10 days - insufficient
        ).tolist()

        generator = RollingWindowGenerator(config)

        # Should generate fewer or no windows due to insufficient data
        windows = generator.generate_rolling_windows(sparse_timestamps)

        # Verify that remaining windows (if any) have sufficient data
        for window in windows:
            train_samples = sum(
                1 for ts in sparse_timestamps if window.train_period.contains_date(ts)
            )
            # If window exists, it should have been validated for sufficiency
            assert train_samples > 0  # At least some data should be present

    def test_date_range_constraints(
        self, generator: RollingWindowGenerator, sample_timestamps: list[pd.Timestamp]
    ):
        """Test date range constraints."""
        start_date = pd.Timestamp("2021-01-01")
        end_date = pd.Timestamp("2023-12-31")

        windows = generator.generate_rolling_windows(
            sample_timestamps, start_date=start_date, end_date=end_date
        )

        for window in windows:
            # Windows should respect date constraints
            assert window.train_period.start_date >= start_date
            assert window.test_period.end_date <= end_date

    def test_adaptive_stepping(self, sample_timestamps: list[pd.Timestamp]):
        """Test adaptive stepping functionality."""
        config = WindowGenerationConfig(
            training_months=36,
            validation_months=12,
            test_months=12,
            step_months=12,
            enable_adaptive_stepping=True,
            adaptive_step_factor=0.8,
        )

        generator = RollingWindowGenerator(config)
        windows = generator.generate_rolling_windows(sample_timestamps)

        # Should generate windows (adaptive stepping is internal logic)
        assert len(windows) > 0

    def test_overlap_detection(self, sample_timestamps: list[pd.Timestamp]):
        """Test overlap detection and resolution."""
        config = WindowGenerationConfig(
            training_months=36,
            validation_months=12,
            test_months=12,
            step_months=6,  # Smaller step to create overlaps
            enable_overlap_detection=True,
        )

        generator = RollingWindowGenerator(config)
        windows = generator.generate_rolling_windows(sample_timestamps)

        # Should generate windows with overlap resolution
        assert len(windows) > 0

    def test_empty_timestamps(self, generator: RollingWindowGenerator):
        """Test handling of empty timestamp list."""
        with pytest.raises(ValueError, match="No valid timestamps"):
            generator.generate_rolling_windows([])

    def test_insufficient_timestamps(self, generator: RollingWindowGenerator):
        """Test handling of insufficient timestamps."""
        few_timestamps = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D").tolist()

        with pytest.raises(ValueError, match="Insufficient timestamps"):
            generator.generate_rolling_windows(few_timestamps)

    def test_generation_statistics(
        self, generator: RollingWindowGenerator, sample_timestamps: list[pd.Timestamp]
    ):
        """Test generation statistics collection."""
        windows = generator.generate_rolling_windows(sample_timestamps)
        stats = generator.get_generation_stats()

        assert isinstance(stats, dict)
        assert "total_windows" in stats
        assert stats["total_windows"] == len(windows)
        assert "data_utilization_rate" in stats
        assert "avg_training_period_days" in stats

    def test_partial_windows_config(self, sample_timestamps: list[pd.Timestamp]):
        """Test configuration for partial windows."""
        config = WindowGenerationConfig(
            training_months=36,
            validation_months=12,
            test_months=12,
            step_months=12,
            allow_partial_windows=True,
            require_full_periods=False,
        )

        generator = RollingWindowGenerator(config)
        windows = generator.generate_rolling_windows(sample_timestamps)

        assert len(windows) > 0


class TestWalkForwardAnalyzer:
    """Test walk-forward analysis functionality."""

    @pytest.fixture
    def sample_windows(self) -> list[RollSplit]:
        """Generate sample windows for testing."""
        windows = []

        for i in range(3):
            start_date = pd.Timestamp(f"202{i}-01-01")
            train_end = start_date + pd.DateOffset(months=36)
            val_start = train_end
            val_end = val_start + pd.DateOffset(months=12)
            test_start = val_end
            test_end = test_start + pd.DateOffset(months=12)

            train_period = ValidationPeriod(start_date, train_end)
            val_period = ValidationPeriod(val_start, val_end)
            test_period = ValidationPeriod(test_start, test_end)

            windows.append(RollSplit(train_period, val_period, test_period))

        return windows

    @pytest.fixture
    def sample_timestamps(self) -> list[pd.Timestamp]:
        """Generate sample timestamps."""
        return pd.date_range(start="2020-01-01", end="2024-12-31", freq="D").tolist()

    @pytest.fixture
    def analyzer(self) -> WalkForwardAnalyzer:
        """Walk-forward analyzer instance."""
        return WalkForwardAnalyzer()

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = WalkForwardAnalyzer()
        assert analyzer._analysis_cache == {}

    def test_analyze_walk_forward_progression(
        self,
        analyzer: WalkForwardAnalyzer,
        sample_windows: list[RollSplit],
        sample_timestamps: list[pd.Timestamp],
    ):
        """Test walk-forward progression analysis."""
        analysis = analyzer.analyze_walk_forward_progression(
            sample_windows, sample_timestamps, expected_step_months=12
        )

        assert isinstance(analysis, dict)
        assert "progression_validation" in analysis
        assert "temporal_coverage" in analysis
        assert "bias_detection" in analysis
        assert "consistency_metrics" in analysis

    def test_step_progression_validation(
        self,
        analyzer: WalkForwardAnalyzer,
        sample_windows: list[RollSplit],
        sample_timestamps: list[pd.Timestamp],
    ):
        """Test step progression validation."""
        analysis = analyzer.analyze_walk_forward_progression(
            sample_windows, sample_timestamps, expected_step_months=12
        )

        progression = analysis["progression_validation"]
        assert "valid_progression" in progression
        assert "expected_step_days" in progression
        assert "actual_mean_step_days" in progression
        assert isinstance(progression["valid_progression"], bool)

    def test_temporal_coverage_analysis(
        self,
        analyzer: WalkForwardAnalyzer,
        sample_windows: list[RollSplit],
        sample_timestamps: list[pd.Timestamp],
    ):
        """Test temporal coverage analysis."""
        analysis = analyzer.analyze_walk_forward_progression(sample_windows, sample_timestamps)

        coverage = analysis["temporal_coverage"]
        assert "total_data_range_days" in coverage
        assert "unique_coverage_days" in coverage
        assert "coverage_efficiency" in coverage
        assert coverage["coverage_efficiency"] >= 0.0
        assert coverage["coverage_efficiency"] <= 1.0

    def test_bias_detection(
        self,
        analyzer: WalkForwardAnalyzer,
        sample_windows: list[RollSplit],
        sample_timestamps: list[pd.Timestamp],
    ):
        """Test bias detection analysis."""
        analysis = analyzer.analyze_walk_forward_progression(sample_windows, sample_timestamps)

        bias_detection = analysis["bias_detection"]
        assert "overall_bias_free" in bias_detection
        assert "bias_checks" in bias_detection

        bias_checks = bias_detection["bias_checks"]
        assert "look_ahead_bias" in bias_checks
        assert "data_snooping_bias" in bias_checks
        assert "survivorship_bias" in bias_checks

    def test_consistency_metrics(
        self,
        analyzer: WalkForwardAnalyzer,
        sample_windows: list[RollSplit],
        sample_timestamps: list[pd.Timestamp],
    ):
        """Test consistency metrics calculation."""
        analysis = analyzer.analyze_walk_forward_progression(sample_windows, sample_timestamps)

        consistency = analysis["consistency_metrics"]
        assert "train_period_consistency" in consistency
        assert "validation_period_consistency" in consistency
        assert "test_period_consistency" in consistency

        # Check structure of consistency metrics
        train_consistency = consistency["train_period_consistency"]
        assert "mean_days" in train_consistency
        assert "std_days" in train_consistency
        assert "cv" in train_consistency  # Coefficient of variation

    def test_insufficient_windows_error(
        self, analyzer: WalkForwardAnalyzer, sample_timestamps: list[pd.Timestamp]
    ):
        """Test error handling for insufficient windows."""
        single_window = [
            RollSplit(
                ValidationPeriod(pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01")),
                ValidationPeriod(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")),
                ValidationPeriod(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
            )
        ]

        analysis = analyzer.analyze_walk_forward_progression(single_window, sample_timestamps)

        # Should handle gracefully but note insufficient data
        assert "error" in analysis or "progression_validation" in analysis

    def test_look_ahead_bias_detection(self, analyzer: WalkForwardAnalyzer):
        """Test look-ahead bias detection."""
        # Create windows with overlapping periods (bias)
        biased_windows = [
            RollSplit(
                ValidationPeriod(
                    pd.Timestamp("2020-01-01"), pd.Timestamp("2023-02-01")
                ),  # Overlap!
                ValidationPeriod(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")),
                ValidationPeriod(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
            )
        ]

        timestamps = pd.date_range("2020-01-01", "2025-12-31", freq="D").tolist()

        analysis = analyzer.analyze_walk_forward_progression(biased_windows, timestamps)
        bias_checks = analysis["bias_detection"]["bias_checks"]

        # Should detect look-ahead bias
        assert bias_checks["look_ahead_bias"]["detected"] is True

    def test_data_snooping_detection(self, analyzer: WalkForwardAnalyzer):
        """Test data snooping bias detection."""
        # Create highly overlapping windows (potential snooping)
        overlapping_windows = []

        for i in range(5):
            # Each window starts only 1 month after the previous (high overlap)
            start_date = pd.Timestamp("2020-01-01") + pd.DateOffset(months=i)
            train_end = start_date + pd.DateOffset(months=36)
            val_start = train_end
            val_end = val_start + pd.DateOffset(months=12)
            test_start = val_end
            test_end = test_start + pd.DateOffset(months=12)

            overlapping_windows.append(
                RollSplit(
                    ValidationPeriod(start_date, train_end),
                    ValidationPeriod(val_start, val_end),
                    ValidationPeriod(test_start, test_end),
                )
            )

        timestamps = pd.date_range("2020-01-01", "2025-12-31", freq="D").tolist()

        analysis = analyzer.analyze_walk_forward_progression(overlapping_windows, timestamps)
        bias_checks = analysis["bias_detection"]["bias_checks"]

        # Should detect potential data snooping
        assert "data_snooping_bias" in bias_checks
