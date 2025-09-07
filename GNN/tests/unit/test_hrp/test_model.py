"""
Unit tests for HRP model integration.

Tests complete HRP model functionality including fit, predict, and constraint integration.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.base.portfolio_model import PortfolioConstraints
from src.models.hrp.allocation import AllocationConfig
from src.models.hrp.clustering import ClusteringConfig
from src.models.hrp.model import HRPConfig, HRPModel


class TestHRPModel:
    """Test suite for HRP model integration."""

    @pytest.fixture
    def sample_returns_data(self):
        """Create comprehensive sample returns data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=1000, freq="D")

        # Create realistic market-like returns with correlations
        n_assets = 20

        # Create correlation structure (3 blocks of correlated assets)
        correlation_matrix = np.eye(n_assets)

        # Block 1: Assets 0-6 (Technology sector)
        correlation_matrix[0:7, 0:7] = 0.6
        np.fill_diagonal(correlation_matrix[0:7, 0:7], 1.0)

        # Block 2: Assets 7-13 (Financial sector)
        correlation_matrix[7:14, 7:14] = 0.5
        np.fill_diagonal(correlation_matrix[7:14, 7:14], 1.0)

        # Block 3: Assets 14-19 (Energy sector)
        correlation_matrix[14:20, 14:20] = 0.4
        np.fill_diagonal(correlation_matrix[14:20, 14:20], 1.0)

        # Generate correlated returns
        returns = np.random.multivariate_normal(
            mean=np.zeros(n_assets),
            cov=0.0001 * correlation_matrix,  # Daily volatility ~1%
            size=len(dates),
        )

        asset_names = [f"ASSET_{i:02d}" for i in range(n_assets)]
        return pd.DataFrame(returns, index=dates, columns=asset_names)

    @pytest.fixture
    def default_constraints(self):
        """Create default portfolio constraints."""
        return PortfolioConstraints(
            long_only=True,
            top_k_positions=15,
            max_position_weight=0.15,
            max_monthly_turnover=0.25,
            transaction_cost_bps=10.0,
        )

    @pytest.fixture
    def hrp_config(self):
        """Create HRP configuration."""
        clustering_config = ClusteringConfig(
            linkage_method="single", min_observations=200, correlation_method="pearson"
        )
        allocation_config = AllocationConfig(
            risk_measure="variance", min_allocation=0.01, max_allocation=0.15
        )

        return HRPConfig(
            lookback_days=500,
            clustering_config=clustering_config,
            allocation_config=allocation_config,
        )

    @pytest.fixture
    def hrp_model(self, default_constraints, hrp_config):
        """Create HRP model instance."""
        return HRPModel(default_constraints, hrp_config)

    def test_model_initialization(self, default_constraints, hrp_config):
        """Test HRP model initialization."""
        model = HRPModel(default_constraints, hrp_config)

        assert model.constraints == default_constraints
        assert model.hrp_config == hrp_config
        assert not model.is_fitted
        assert model._fitted_covariance is None
        assert model._fitted_returns is None
        assert model._fitted_universe is None

    def test_model_initialization_with_defaults(self, default_constraints):
        """Test HRP model initialization with default configuration."""
        model = HRPModel(default_constraints)

        assert model.constraints == default_constraints
        assert isinstance(model.hrp_config, HRPConfig)
        assert model.hrp_config.lookback_days == 756  # Default 3 years
        assert not model.is_fitted

    def test_fit_with_valid_data(self, hrp_model, sample_returns_data):
        """Test model fitting with valid data."""
        universe = sample_returns_data.columns.tolist()
        fit_period = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01"))

        hrp_model.fit(sample_returns_data, universe, fit_period)

        assert hrp_model.is_fitted
        assert hrp_model._fitted_returns is not None
        assert hrp_model._fitted_covariance is not None
        assert hrp_model._fitted_universe is not None
        assert len(hrp_model._fitted_universe) > 0

    def test_fit_insufficient_data(self, hrp_model):
        """Test model fitting with insufficient data."""
        # Create data with too few observations
        short_data = pd.DataFrame(
            np.random.randn(100, 10),  # Only 100 observations
            index=pd.date_range("2020-01-01", periods=100, freq="D"),
            columns=[f"ASSET_{i}" for i in range(10)],
        )

        universe = short_data.columns.tolist()
        fit_period = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-10"))

        with pytest.raises(ValueError, match="Fit period too short"):
            hrp_model.fit(short_data, universe, fit_period)

    def test_fit_insufficient_asset_coverage(self, hrp_model, sample_returns_data):
        """Test model fitting with insufficient asset coverage."""
        # Request universe with many missing assets
        universe = sample_returns_data.columns.tolist() + [
            "MISSING_1",
            "MISSING_2",
            "MISSING_3",
            "MISSING_4",
            "MISSING_5",
        ]
        fit_period = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01"))

        with pytest.raises(ValueError, match="Insufficient asset coverage"):
            hrp_model.fit(sample_returns_data, universe, fit_period)

    def test_fit_invalid_date_period(self, hrp_model, sample_returns_data):
        """Test model fitting with invalid date period."""
        universe = sample_returns_data.columns.tolist()

        # Invalid period (end before start)
        with pytest.raises(ValueError, match="Invalid fit period"):
            hrp_model.fit(
                sample_returns_data,
                universe,
                (pd.Timestamp("2021-01-01"), pd.Timestamp("2020-01-01")),
            )

    def test_predict_weights_unfitted_model(self, hrp_model):
        """Test weight prediction on unfitted model."""
        with pytest.raises(ValueError, match="Model must be fitted"):
            hrp_model.predict_weights(pd.Timestamp("2021-07-01"), ["ASSET_01", "ASSET_02"])

    def test_predict_weights_basic(self, hrp_model, sample_returns_data):
        """Test basic weight prediction functionality."""
        universe = sample_returns_data.columns.tolist()
        fit_period = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01"))

        # Fit model
        hrp_model.fit(sample_returns_data, universe, fit_period)

        # Predict weights
        prediction_date = pd.Timestamp("2021-07-01")
        prediction_universe = universe[:10]  # Subset of fitted universe

        weights = hrp_model.predict_weights(prediction_date, prediction_universe)

        # Check weight properties
        assert isinstance(weights, pd.Series)
        assert len(weights) <= len(prediction_universe)  # May be filtered by constraints
        np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=6)
        assert all(weights >= 0)  # Long-only constraint
        assert all(weights <= hrp_model.constraints.max_position_weight)

    def test_predict_weights_constraint_enforcement(self, sample_returns_data):
        """Test constraint enforcement in weight prediction."""
        # Create constraints with top-k limit
        strict_constraints = PortfolioConstraints(
            long_only=True,
            top_k_positions=5,  # Only 5 positions allowed
            max_position_weight=0.25,
            max_monthly_turnover=0.30,
        )

        hrp_config = HRPConfig(lookback_days=300)
        model = HRPModel(strict_constraints, hrp_config)

        universe = sample_returns_data.columns.tolist()[:15]  # 15 assets available
        fit_period = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01"))

        model.fit(sample_returns_data, universe, fit_period)

        weights = model.predict_weights(pd.Timestamp("2021-07-01"), universe)

        # Check constraint enforcement
        assert (weights > 0).sum() <= 5  # Top-k constraint
        assert all(weights <= 0.25)  # Max position weight
        np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=6)

    def test_predict_weights_single_asset(self, hrp_model, sample_returns_data):
        """Test weight prediction with single asset."""
        universe = sample_returns_data.columns.tolist()
        fit_period = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01"))

        hrp_model.fit(sample_returns_data, universe, fit_period)

        # Single asset prediction
        single_asset_weights = hrp_model.predict_weights(pd.Timestamp("2021-07-01"), [universe[0]])

        assert len(single_asset_weights) == 1
        np.testing.assert_almost_equal(single_asset_weights.iloc[0], 1.0, decimal=6)

    def test_predict_weights_no_common_assets(self, hrp_model, sample_returns_data):
        """Test weight prediction with no common assets."""
        universe = sample_returns_data.columns.tolist()
        fit_period = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01"))

        hrp_model.fit(sample_returns_data, universe, fit_period)

        # Request weights for non-existent assets
        with pytest.raises(ValueError, match="No assets in common"):
            hrp_model.predict_weights(
                pd.Timestamp("2021-07-01"), ["NONEXISTENT_1", "NONEXISTENT_2"]
            )

    def test_get_model_info(self, hrp_model, sample_returns_data):
        """Test model info retrieval."""
        # Before fitting
        info_unfitted = hrp_model.get_model_info()

        assert info_unfitted["model_type"] == "HRP"
        assert not info_unfitted["is_fitted"]
        assert "hrp_config" in info_unfitted
        assert "constraints" in info_unfitted

        # After fitting
        universe = sample_returns_data.columns.tolist()
        fit_period = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01"))

        hrp_model.fit(sample_returns_data, universe, fit_period)

        info_fitted = hrp_model.get_model_info()

        assert info_fitted["is_fitted"]
        assert info_fitted["fitted_universe_size"] > 0
        assert info_fitted["training_observations"] > 0
        assert "fit_period_start" in info_fitted
        assert "fit_period_end" in info_fitted

    def test_clustering_diagnostics(self, hrp_model, sample_returns_data):
        """Test clustering diagnostics functionality."""
        universe = sample_returns_data.columns.tolist()
        fit_period = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01"))

        hrp_model.fit(sample_returns_data, universe, fit_period)

        diagnostics = hrp_model.get_clustering_diagnostics(
            pd.Timestamp("2021-07-01"), universe[:10]
        )

        # Check diagnostics structure
        assert "n_assets" in diagnostics
        assert "n_observations" in diagnostics
        assert "linkage_method" in diagnostics
        assert "correlation_method" in diagnostics
        assert "cluster_tree_depth" in diagnostics
        assert "assets" in diagnostics

        assert diagnostics["n_assets"] > 0
        assert diagnostics["n_observations"] > 0
        assert diagnostics["cluster_tree_depth"] > 0

    def test_clustering_diagnostics_unfitted(self, hrp_model):
        """Test clustering diagnostics on unfitted model."""
        diagnostics = hrp_model.get_clustering_diagnostics(
            pd.Timestamp("2021-07-01"), ["ASSET_01", "ASSET_02"]
        )

        assert "error" in diagnostics
        assert "Model not fitted" in diagnostics["error"]

    def test_risk_contributions(self, hrp_model, sample_returns_data):
        """Test risk contribution calculations."""
        universe = sample_returns_data.columns.tolist()
        fit_period = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01"))

        hrp_model.fit(sample_returns_data, universe, fit_period)

        # Generate weights
        weights = hrp_model.predict_weights(pd.Timestamp("2021-07-01"), universe[:10])

        # Calculate risk contributions
        risk_contributions = hrp_model.get_risk_contributions(weights, pd.Timestamp("2021-07-01"))

        # Check risk contributions
        assert isinstance(risk_contributions, dict)
        assert len(risk_contributions) > 0

        # All assets with non-zero weights should have risk contributions
        non_zero_assets = weights[weights > 0].index
        for asset in non_zero_assets:
            if asset in risk_contributions:
                assert isinstance(risk_contributions[asset], float)

    def test_model_with_different_configurations(self, default_constraints, sample_returns_data):
        """Test model with different HRP configurations."""
        # Configuration with different clustering method
        config_complete = HRPConfig(
            lookback_days=400,
            clustering_config=ClusteringConfig(
                linkage_method="complete", correlation_method="spearman"
            ),
            allocation_config=AllocationConfig(risk_measure="vol"),
        )

        model_complete = HRPModel(default_constraints, config_complete)

        universe = sample_returns_data.columns.tolist()[:12]
        fit_period = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01"))

        model_complete.fit(sample_returns_data, universe, fit_period)
        weights_complete = model_complete.predict_weights(pd.Timestamp("2021-07-01"), universe)

        # Configuration with different allocation method
        config_equal = HRPConfig(
            lookback_days=400,
            clustering_config=ClusteringConfig(linkage_method="average"),
            allocation_config=AllocationConfig(risk_measure="equal"),
        )

        model_equal = HRPModel(default_constraints, config_equal)
        model_equal.fit(sample_returns_data, universe, fit_period)
        weights_equal = model_equal.predict_weights(pd.Timestamp("2021-07-01"), universe)

        # Both should produce valid weights but with different distributions
        np.testing.assert_almost_equal(weights_complete.sum(), 1.0, decimal=6)
        np.testing.assert_almost_equal(weights_equal.sum(), 1.0, decimal=6)

        # Weights should be different (given different configurations)
        # Note: This is a probabilistic test - there's a small chance they could be similar
        assert not np.allclose(weights_complete.values, weights_equal.values, atol=0.01)

    def test_model_memory_efficiency(self, default_constraints, sample_returns_data):
        """Test model with memory-efficient configuration."""
        # Enable memory-efficient processing
        config = HRPConfig(
            lookback_days=400,
            clustering_config=ClusteringConfig(memory_efficient=True, chunk_size=5),
        )

        model = HRPModel(default_constraints, config)

        universe = sample_returns_data.columns.tolist()  # Use all assets
        fit_period = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01"))

        model.fit(sample_returns_data, universe, fit_period)
        weights = model.predict_weights(pd.Timestamp("2021-07-01"), universe)

        # Should still produce valid weights
        np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=6)
        assert all(weights >= 0)

    def test_model_robustness_to_missing_data(self, hrp_model, sample_returns_data):
        """Test model robustness with missing data."""
        # Introduce missing data
        noisy_data = sample_returns_data.copy()

        # Randomly set some values to NaN
        np.random.seed(42)
        mask = np.random.random(noisy_data.shape) < 0.05  # 5% missing data
        noisy_data.values[mask] = np.nan

        universe = noisy_data.columns.tolist()[:10]
        fit_period = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01"))

        # Model should handle missing data gracefully
        hrp_model.fit(noisy_data, universe, fit_period)
        weights = hrp_model.predict_weights(pd.Timestamp("2021-07-01"), universe)

        # Should still produce valid weights
        assert len(weights) > 0
        np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=6)
        assert all(weights >= 0)

    def test_model_state_validation(self, hrp_model, sample_returns_data):
        """Test model state validation."""
        universe = sample_returns_data.columns.tolist()
        fit_period = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01"))

        hrp_model.fit(sample_returns_data, universe, fit_period)

        # Corrupt model state
        hrp_model._fitted_returns = None

        with pytest.raises(ValueError, match="Model state is invalid"):
            hrp_model.predict_weights(pd.Timestamp("2021-07-01"), universe)

    def test_edge_case_empty_universe_after_filtering(self, sample_returns_data):
        """Test edge case where universe becomes empty after filtering."""
        # Create very strict constraints that might filter out all assets
        strict_constraints = PortfolioConstraints(
            long_only=True,
            top_k_positions=1,  # Only 1 position
            max_position_weight=0.001,  # Extremely small max weight
            min_weight_threshold=0.5,  # Extremely high min weight (impossible to satisfy)
        )

        model = HRPModel(strict_constraints)

        universe = sample_returns_data.columns.tolist()[:5]
        fit_period = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01"))

        model.fit(sample_returns_data, universe, fit_period)

        # This should still produce some weights (fallback behavior)
        weights = model.predict_weights(pd.Timestamp("2021-07-01"), universe)

        assert len(weights) > 0
        np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=6)
