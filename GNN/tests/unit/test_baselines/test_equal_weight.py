"""
Unit tests for equal-weight portfolio allocation model.

Tests cover weight distribution accuracy, top-k selection logic,
constraint enforcement, and integration with the portfolio model interface.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.models.base.portfolio_model import PortfolioConstraints
from src.models.baselines.equal_weight import EqualWeightModel


class TestEqualWeightModel:
    """Test suite for EqualWeightModel class."""

    @pytest.fixture
    def sample_constraints(self):
        """Create sample portfolio constraints for testing."""
        return PortfolioConstraints(
            long_only=True,
            top_k_positions=50,
            max_position_weight=0.10,
            max_monthly_turnover=0.20,
            transaction_cost_bps=10.0,
        )

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data for testing."""
        dates = pd.date_range(start='2020-01-01', end='2020-03-31', freq='D')
        assets = [f"ASSET_{i:03d}" for i in range(100)]
        
        # Generate random returns
        np.random.seed(42)
        returns_data = np.random.normal(0.0005, 0.02, size=(len(dates), len(assets)))
        
        return pd.DataFrame(returns_data, index=dates, columns=assets)

    @pytest.fixture
    def equal_weight_model(self, sample_constraints):
        """Create EqualWeightModel instance for testing."""
        return EqualWeightModel(constraints=sample_constraints, top_k=50)

    def test_model_initialization(self, sample_constraints):
        """Test proper model initialization."""
        model = EqualWeightModel(constraints=sample_constraints, top_k=30)
        
        assert model.constraints == sample_constraints
        assert model.top_k == 30
        assert model.is_fitted is False
        assert model.fitted_universe is None
        assert model.selection_method == "alphabetical"

    def test_fit_with_valid_data(self, equal_weight_model, sample_returns):
        """Test fitting model with valid returns data."""
        universe = sample_returns.columns[:50].tolist()
        fit_period = (sample_returns.index[0], sample_returns.index[-1])
        
        equal_weight_model.fit(sample_returns, universe, fit_period)
        
        assert equal_weight_model.is_fitted is True
        assert isinstance(equal_weight_model.fitted_universe, list)
        assert len(equal_weight_model.fitted_universe) > 0
        assert all(asset in universe for asset in equal_weight_model.fitted_universe)

    def test_fit_with_empty_returns(self, equal_weight_model):
        """Test fitting with empty returns data raises error."""
        empty_returns = pd.DataFrame()
        universe = ["ASSET_001", "ASSET_002"]
        fit_period = (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-02-01'))
        
        with pytest.raises(ValueError, match="Returns data cannot be empty"):
            equal_weight_model.fit(empty_returns, universe, fit_period)

    def test_fit_with_empty_universe(self, equal_weight_model, sample_returns):
        """Test fitting with empty universe raises error."""
        fit_period = (sample_returns.index[0], sample_returns.index[-1])
        
        with pytest.raises(ValueError, match="Universe cannot be empty"):
            equal_weight_model.fit(sample_returns, [], fit_period)

    def test_fit_with_invalid_date_period(self, equal_weight_model, sample_returns):
        """Test fitting with invalid date period raises error."""
        universe = sample_returns.columns[:10].tolist()
        invalid_period = (sample_returns.index[-1], sample_returns.index[0])  # End before start
        
        with pytest.raises(ValueError, match="Start date must be before end date"):
            equal_weight_model.fit(sample_returns, universe, invalid_period)

    def test_predict_weights_unfitted_model(self, equal_weight_model):
        """Test weight prediction on unfitted model raises error."""
        date = pd.Timestamp('2020-02-01')
        universe = ["ASSET_001", "ASSET_002"]
        
        with pytest.raises(ValueError, match="Model must be fitted before generating predictions"):
            equal_weight_model.predict_weights(date, universe)

    def test_predict_weights_equal_distribution(self, equal_weight_model, sample_returns):
        """Test that weights are equally distributed among selected assets."""
        # Fit the model
        universe = sample_returns.columns[:20].tolist()
        fit_period = (sample_returns.index[0], sample_returns.index[-1])
        equal_weight_model.fit(sample_returns, universe, fit_period)
        
        # Predict weights
        prediction_date = sample_returns.index[50]
        weights = equal_weight_model.predict_weights(prediction_date, universe)
        
        # Check equal weight distribution
        active_weights = weights[weights > 0]
        assert len(active_weights) > 0
        
        # All non-zero weights should be equal (within tolerance)
        expected_weight = 1.0 / len(active_weights)
        np.testing.assert_allclose(active_weights.values, expected_weight, atol=1e-10)
        
        # Total weights should sum to 1.0
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_predict_weights_top_k_selection(self, sample_returns):
        """Test that only top-k assets are selected."""
        # Create constraints with small top_k
        from src.models.base.portfolio_model import PortfolioConstraints
        constraints = PortfolioConstraints(
            long_only=True,
            top_k_positions=10,  # This will be the effective limit
            max_position_weight=0.15,
        )
        model = EqualWeightModel(constraints=constraints, top_k=10)
        
        # Fit the model
        universe = sample_returns.columns[:50].tolist()
        fit_period = (sample_returns.index[0], sample_returns.index[-1])
        model.fit(sample_returns, universe, fit_period)
        
        # Predict weights
        prediction_date = sample_returns.index[50]
        weights = model.predict_weights(prediction_date, universe)
        
        # Check that only top_k assets have non-zero weights
        active_positions = (weights > 0).sum()
        assert active_positions <= 10

    def test_predict_weights_constraint_enforcement(self, equal_weight_model, sample_returns):
        """Test that portfolio constraints are properly enforced."""
        # Fit the model
        universe = sample_returns.columns[:30].tolist()
        fit_period = (sample_returns.index[0], sample_returns.index[-1])
        equal_weight_model.fit(sample_returns, universe, fit_period)
        
        # Predict weights
        prediction_date = sample_returns.index[50]
        weights = equal_weight_model.predict_weights(prediction_date, universe)
        
        # Test long-only constraint
        assert all(weights >= 0), "All weights should be non-negative (long-only)"
        
        # Test maximum position weight constraint
        max_weight = equal_weight_model.constraints.max_position_weight
        assert all(weights <= max_weight), f"No weight should exceed {max_weight}"
        
        # Test weights sum to 1.0
        assert abs(weights.sum() - 1.0) < 1e-10, "Weights should sum to 1.0"

    def test_predict_weights_empty_universe(self, equal_weight_model, sample_returns):
        """Test prediction with empty universe raises error."""
        # Fit the model first
        universe = sample_returns.columns[:10].tolist()
        fit_period = (sample_returns.index[0], sample_returns.index[-1])
        equal_weight_model.fit(sample_returns, universe, fit_period)
        
        # Try to predict with empty universe
        prediction_date = sample_returns.index[50]
        
        with pytest.raises(ValueError, match="Universe cannot be empty"):
            equal_weight_model.predict_weights(prediction_date, [])

    def test_predict_weights_no_valid_assets(self, equal_weight_model, sample_returns):
        """Test prediction when no assets from universe were in fitted universe."""
        # Fit the model with one set of assets
        original_universe = sample_returns.columns[:10].tolist()
        fit_period = (sample_returns.index[0], sample_returns.index[-1])
        equal_weight_model.fit(sample_returns, original_universe, fit_period)
        
        # Try to predict with completely different universe
        different_universe = sample_returns.columns[90:].tolist()
        prediction_date = sample_returns.index[50]
        
        with pytest.raises(ValueError, match="No assets in current universe were present during fitting"):
            equal_weight_model.predict_weights(prediction_date, different_universe)

    def test_get_model_info(self, equal_weight_model, sample_returns):
        """Test model metadata retrieval."""
        # Test before fitting
        info_before = equal_weight_model.get_model_info()
        assert info_before["model_type"] == "EqualWeight"
        assert info_before["top_k"] == 50
        assert info_before["is_fitted"] is False
        assert info_before["fitted_universe_size"] == 0
        
        # Fit the model
        universe = sample_returns.columns[:20].tolist()
        fit_period = (sample_returns.index[0], sample_returns.index[-1])
        equal_weight_model.fit(sample_returns, universe, fit_period)
        
        # Test after fitting
        info_after = equal_weight_model.get_model_info()
        assert info_after["is_fitted"] is True
        assert info_after["fitted_universe_size"] > 0
        assert "constraints" in info_after
        assert info_after["selection_method"] == "alphabetical"

    def test_select_top_k_assets_alphabetical(self):
        """Test alphabetical asset selection logic."""
        # Create model with small top_k for testing
        constraints = PortfolioConstraints(top_k_positions=3)
        model = EqualWeightModel(constraints=constraints, top_k=3)
        
        universe = ["ZULU", "ALPHA", "BETA", "CHARLIE", "DELTA"]
        
        # Test with k=3
        selected = model._select_top_k_assets(universe)
        expected = ["ALPHA", "BETA", "CHARLIE"]  # First 3 alphabetically
        assert selected == expected

    def test_select_top_k_assets_limits_k(self, equal_weight_model):
        """Test that selection respects universe size limits."""
        small_universe = ["ASSET_A", "ASSET_B"]
        
        # Model's top_k is 50, but universe only has 2 assets
        selected = equal_weight_model._select_top_k_assets(small_universe)
        assert len(selected) == 2
        assert set(selected) == set(small_universe)

    def test_constraint_top_k_override(self, sample_returns):
        """Test that constraint's top_k_positions overrides model's top_k."""
        constraints = PortfolioConstraints(top_k_positions=15)
        model = EqualWeightModel(constraints=constraints, top_k=50)  # Model top_k should be ignored
        
        universe = [f"ASSET_{i:03d}" for i in range(30)]
        
        # Fit model
        fit_period = (sample_returns.index[0], sample_returns.index[-1])
        model.fit(sample_returns, universe, fit_period)
        
        # Predict weights
        prediction_date = sample_returns.index[50]
        weights = model.predict_weights(prediction_date, universe)
        
        # Should have at most 15 active positions (from constraints)
        active_positions = (weights > 0).sum()
        assert active_positions <= 15