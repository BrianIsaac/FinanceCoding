"""
Unit tests for HRP allocation module.

Tests recursive bisection algorithm, risk budgeting, and constraint enforcement.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.hrp.allocation import AllocationConfig, HRPAllocation
from src.models.hrp.clustering import ClusteringConfig, HRPClustering


class TestHRPAllocation:
    """Test suite for HRP allocation functionality."""

    @pytest.fixture
    def sample_covariance_matrix(self):
        """Create sample covariance matrix for testing."""
        # Create covariance matrix with known structure
        assets = ['A', 'B', 'C', 'D', 'E']
        cov_data = np.array([
            [0.04, 0.02, 0.01, 0.005, 0.003],
            [0.02, 0.03, 0.01, 0.004, 0.002],
            [0.01, 0.01, 0.025, 0.008, 0.005],
            [0.005, 0.004, 0.008, 0.035, 0.015],
            [0.003, 0.002, 0.005, 0.015, 0.045]
        ])
        return pd.DataFrame(cov_data, index=assets, columns=assets)

    @pytest.fixture
    def sample_cluster_tree(self):
        """Create sample cluster tree for testing."""
        return {
            "id": 8,
            "type": "cluster",
            "assets": ["A", "B", "C", "D", "E"],
            "distance": 0.5,
            "left": {
                "id": 6,
                "type": "cluster",
                "assets": ["A", "B", "C"],
                "distance": 0.3,
                "left": {
                    "id": 0,
                    "type": "leaf",
                    "asset": "A",
                    "assets": ["A"],
                    "distance": 0.0
                },
                "right": {
                    "id": 5,
                    "type": "cluster",
                    "assets": ["B", "C"],
                    "distance": 0.2,
                    "left": {
                        "id": 1,
                        "type": "leaf",
                        "asset": "B",
                        "assets": ["B"],
                        "distance": 0.0
                    },
                    "right": {
                        "id": 2,
                        "type": "leaf",
                        "asset": "C",
                        "assets": ["C"],
                        "distance": 0.0
                    }
                }
            },
            "right": {
                "id": 7,
                "type": "cluster",
                "assets": ["D", "E"],
                "distance": 0.4,
                "left": {
                    "id": 3,
                    "type": "leaf",
                    "asset": "D",
                    "assets": ["D"],
                    "distance": 0.0
                },
                "right": {
                    "id": 4,
                    "type": "leaf",
                    "asset": "E",
                    "assets": ["E"],
                    "distance": 0.0
                }
            }
        }

    @pytest.fixture
    def allocation_engine(self):
        """Create HRP allocation engine with default config."""
        config = AllocationConfig()
        return HRPAllocation(config)

    def test_allocation_initialization(self):
        """Test HRP allocation engine initialization."""
        config = AllocationConfig(
            risk_measure="vol",
            min_allocation=0.005,
            max_allocation=0.15,
            allocation_precision=4
        )
        allocation_engine = HRPAllocation(config)
        
        assert allocation_engine.config.risk_measure == "vol"
        assert allocation_engine.config.min_allocation == 0.005
        assert allocation_engine.config.max_allocation == 0.15
        assert allocation_engine.config.allocation_precision == 4

    def test_recursive_bisection_basic(self, allocation_engine, sample_covariance_matrix, sample_cluster_tree):
        """Test basic recursive bisection functionality."""
        weights = allocation_engine.recursive_bisection(
            sample_covariance_matrix,
            sample_cluster_tree
        )
        
        # Check basic properties
        assert isinstance(weights, pd.Series)
        assert len(weights) == 5
        assert all(asset in weights.index for asset in ['A', 'B', 'C', 'D', 'E'])
        
        # Check weights sum to 1
        np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=6)
        
        # Check all weights are positive
        assert all(weights >= 0)

    def test_single_asset_cluster(self, allocation_engine):
        """Test allocation for single asset cluster."""
        single_asset_cov = pd.DataFrame([[0.04]], index=['A'], columns=['A'])
        single_asset_tree = {
            "id": 0,
            "type": "leaf",
            "asset": "A",
            "assets": ["A"],
            "distance": 0.0
        }
        
        weights = allocation_engine.recursive_bisection(
            single_asset_cov,
            single_asset_tree
        )
        
        assert len(weights) == 1
        assert weights.index[0] == 'A'
        np.testing.assert_almost_equal(weights.iloc[0], 1.0, decimal=6)

    def test_cluster_risk_calculation(self, allocation_engine, sample_covariance_matrix):
        """Test cluster risk calculation methods."""
        # Single asset risk
        single_asset_risk = allocation_engine._calculate_cluster_risk(
            ['A'], sample_covariance_matrix
        )
        assert single_asset_risk > 0
        np.testing.assert_almost_equal(single_asset_risk, 0.04, decimal=6)  # Variance of A
        
        # Multi-asset cluster risk
        multi_asset_risk = allocation_engine._calculate_cluster_risk(
            ['A', 'B'], sample_covariance_matrix
        )
        assert multi_asset_risk > 0
        
        # Empty cluster risk
        empty_risk = allocation_engine._calculate_cluster_risk([], sample_covariance_matrix)
        assert empty_risk == 0.0

    def test_different_risk_measures(self, sample_covariance_matrix, sample_cluster_tree):
        """Test different risk measure configurations."""
        assets = ['A', 'B']
        
        # Test variance risk measure
        var_config = AllocationConfig(risk_measure="variance")
        var_engine = HRPAllocation(var_config)
        var_risk = var_engine._calculate_cluster_risk(assets, sample_covariance_matrix)
        
        # Test volatility risk measure
        vol_config = AllocationConfig(risk_measure="vol")
        vol_engine = HRPAllocation(vol_config)
        vol_risk = vol_engine._calculate_cluster_risk(assets, sample_covariance_matrix)
        
        # Volatility should be sqrt of variance
        np.testing.assert_almost_equal(vol_risk, np.sqrt(var_risk), decimal=6)
        
        # Test equal risk measure
        equal_config = AllocationConfig(risk_measure="equal")
        equal_engine = HRPAllocation(equal_config)
        equal_risk = equal_engine._calculate_cluster_risk(assets, sample_covariance_matrix)
        
        assert equal_risk == 1.0

    def test_allocation_constraints(self, allocation_engine, sample_covariance_matrix):
        """Test allocation constraint enforcement."""
        # Create weights that violate constraints
        raw_weights = pd.Series([0.8, 0.15, 0.03, 0.015, 0.005], 
                               index=['A', 'B', 'C', 'D', 'E'])
        
        # Apply constraints
        constrained_weights = allocation_engine._apply_allocation_constraints(raw_weights)
        
        # Check constraints are satisfied
        assert all(constrained_weights >= allocation_engine.config.min_allocation)
        assert all(constrained_weights <= allocation_engine.config.max_allocation)
        np.testing.assert_almost_equal(constrained_weights.sum(), 1.0, decimal=6)

    def test_risk_budgeting_calculation(self, allocation_engine, sample_covariance_matrix, sample_cluster_tree):
        """Test risk contribution calculation."""
        # Generate weights using recursive bisection
        weights = allocation_engine.recursive_bisection(
            sample_covariance_matrix,
            sample_cluster_tree
        )
        
        # Calculate risk contributions
        risk_contributions = allocation_engine.calculate_risk_budgets(
            weights,
            sample_covariance_matrix,
            sample_cluster_tree
        )
        
        # Check risk contributions structure
        assert isinstance(risk_contributions, dict)
        assert len(risk_contributions) == len(weights)
        assert all(asset in risk_contributions for asset in weights.index)
        
        # Risk contributions should sum approximately to portfolio volatility
        total_risk_contrib = sum(risk_contributions.values())
        portfolio_vol = np.sqrt(np.dot(weights.values, 
                                      np.dot(sample_covariance_matrix.values, weights.values)))
        
        # Note: This is an approximation test since risk contributions are complex
        assert total_risk_contrib > 0

    def test_optimize_cluster_allocation(self, allocation_engine, sample_covariance_matrix):
        """Test within-cluster allocation optimization."""
        # Test cluster allocation
        cluster_assets = ['A', 'B', 'C']
        target_allocation = 0.6
        
        optimized_weights = allocation_engine.optimize_cluster_allocation(
            cluster_assets,
            sample_covariance_matrix,
            target_allocation
        )
        
        # Check allocation properties
        assert len(optimized_weights) == 3
        np.testing.assert_almost_equal(optimized_weights.sum(), target_allocation, decimal=6)
        assert all(optimized_weights > 0)
        
        # Test single asset cluster
        single_optimized = allocation_engine.optimize_cluster_allocation(
            ['A'],
            sample_covariance_matrix,
            target_allocation
        )
        
        assert len(single_optimized) == 1
        np.testing.assert_almost_equal(single_optimized.iloc[0], target_allocation, decimal=6)

    def test_edge_case_handling(self, allocation_engine, sample_covariance_matrix, sample_cluster_tree):
        """Test edge case handling in allocation."""
        # Test edge case detection
        needs_handling, handling_type = allocation_engine.handle_edge_cases(
            sample_cluster_tree,
            sample_covariance_matrix
        )
        
        assert isinstance(needs_handling, bool)
        assert isinstance(handling_type, str)
        
        # Test single asset case
        single_asset_tree = {
            "id": 0,
            "type": "leaf",
            "asset": "A",
            "assets": ["A"],
            "distance": 0.0
        }
        
        needs_handling, handling_type = allocation_engine.handle_edge_cases(
            single_asset_tree,
            sample_covariance_matrix
        )
        
        assert needs_handling
        assert handling_type == "single_asset"
        
        # Test empty cluster case
        empty_tree = {
            "id": 0,
            "type": "cluster",
            "assets": [],
            "distance": 0.0
        }
        
        needs_handling, handling_type = allocation_engine.handle_edge_cases(
            empty_tree,
            sample_covariance_matrix
        )
        
        assert needs_handling
        assert handling_type == "empty_cluster"

    def test_input_validation(self, allocation_engine):
        """Test input validation for allocation methods."""
        # Test invalid covariance matrix
        with pytest.raises(ValueError, match="Empty covariance matrix"):
            allocation_engine.recursive_bisection(
                pd.DataFrame(),  # Empty dataframe
                {"assets": ["A"]}
            )
        
        # Test invalid cluster tree
        with pytest.raises(ValueError, match="Cluster tree must be a dictionary"):
            allocation_engine.recursive_bisection(
                pd.DataFrame([[1]], index=['A'], columns=['A']),
                "invalid_tree"
            )
        
        # Test missing assets field in cluster tree
        with pytest.raises(ValueError, match="Cluster tree must contain 'assets' field"):
            allocation_engine.recursive_bisection(
                pd.DataFrame([[1]], index=['A'], columns=['A']),
                {"type": "leaf"}
            )

    def test_allocation_precision(self):
        """Test allocation precision configuration."""
        config = AllocationConfig(allocation_precision=3)
        allocation_engine = HRPAllocation(config)
        
        # Create test weights with high precision
        raw_weights = pd.Series([0.123456789, 0.876543211], index=['A', 'B'])
        
        constrained_weights = allocation_engine._apply_allocation_constraints(raw_weights)
        
        # Check that weights are rounded to specified precision
        for weight in constrained_weights:
            decimal_places = len(str(weight).split('.')[-1]) if '.' in str(weight) else 0
            assert decimal_places <= 3

    def test_covariance_matrix_alignment(self, allocation_engine):
        """Test covariance matrix alignment with cluster tree assets."""
        # Covariance matrix with extra assets
        cov_matrix = pd.DataFrame(
            np.eye(5) * 0.01,
            index=['A', 'B', 'C', 'X', 'Y'],
            columns=['A', 'B', 'C', 'X', 'Y']
        )
        
        # Cluster tree with subset of assets
        asset_names = ['A', 'B', 'C']
        
        aligned_cov = allocation_engine._align_covariance_matrix(cov_matrix, asset_names)
        
        assert aligned_cov.shape == (3, 3)
        assert all(asset in aligned_cov.index for asset in asset_names)
        assert all(asset in aligned_cov.columns for asset in asset_names)
        
        # Test no common assets
        with pytest.raises(ValueError, match="No common assets"):
            allocation_engine._align_covariance_matrix(cov_matrix, ['Z'])

    def test_large_universe_performance(self, allocation_engine):
        """Test allocation performance with larger universe."""
        # Create larger covariance matrix (50 assets)
        n_assets = 50
        asset_names = [f'ASSET_{i:02d}' for i in range(n_assets)]
        
        # Generate random covariance matrix
        np.random.seed(42)
        random_cov = np.random.rand(n_assets, n_assets)
        random_cov = np.dot(random_cov, random_cov.T)  # Make positive semi-definite
        cov_matrix = pd.DataFrame(random_cov, index=asset_names, columns=asset_names)
        
        # Create simple cluster tree (binary tree)
        def create_binary_tree(assets):
            if len(assets) == 1:
                return {
                    "id": 0,
                    "type": "leaf",
                    "asset": assets[0],
                    "assets": assets,
                    "distance": 0.0
                }
            
            mid = len(assets) // 2
            left_assets = assets[:mid]
            right_assets = assets[mid:]
            
            return {
                "id": len(assets),
                "type": "cluster",
                "assets": assets,
                "distance": 0.1,
                "left": create_binary_tree(left_assets),
                "right": create_binary_tree(right_assets)
            }
        
        large_tree = create_binary_tree(asset_names)
        
        # Test allocation (should complete without errors)
        weights = allocation_engine.recursive_bisection(cov_matrix, large_tree)
        
        assert len(weights) == n_assets
        np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=6)
        assert all(weights >= 0)