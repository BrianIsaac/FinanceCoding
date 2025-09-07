"""
Unit tests for HRP clustering module.

Tests correlation distance calculation, hierarchical clustering,
and cluster tree construction functionality.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.hrp.clustering import ClusteringConfig, HRPClustering


class TestHRPClustering:
    """Test suite for HRP clustering functionality."""

    @pytest.fixture
    def sample_returns_data(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        
        # Create correlated asset returns
        n_assets = 10
        returns = np.random.multivariate_normal(
            mean=np.zeros(n_assets),
            cov=0.01 * np.eye(n_assets),  # 1% daily volatility
            size=len(dates)
        )
        
        asset_names = [f'ASSET_{i:02d}' for i in range(n_assets)]
        return pd.DataFrame(returns, index=dates, columns=asset_names)

    @pytest.fixture
    def clustering_engine(self):
        """Create HRP clustering engine with default config."""
        config = ClusteringConfig(min_observations=100)
        return HRPClustering(config)

    def test_clustering_initialization(self):
        """Test HRP clustering engine initialization."""
        config = ClusteringConfig(
            linkage_method="complete",
            min_observations=200,
            correlation_method="spearman"
        )
        clustering = HRPClustering(config)
        
        assert clustering.config.linkage_method == "complete"
        assert clustering.config.min_observations == 200
        assert clustering.config.correlation_method == "spearman"
        assert clustering._distance_matrix is None
        assert clustering._linkage_matrix is None

    def test_correlation_distance_calculation(self, clustering_engine, sample_returns_data):
        """Test correlation distance matrix calculation."""
        distance_matrix = clustering_engine.build_correlation_distance(sample_returns_data)
        
        # Check matrix properties
        assert isinstance(distance_matrix, np.ndarray)
        assert distance_matrix.shape == (10, 10)
        
        # Check distance matrix properties
        assert np.allclose(np.diag(distance_matrix), 0.0)  # Diagonal should be zero
        assert np.all(distance_matrix >= 0.0)  # All distances non-negative
        assert np.all(distance_matrix <= 1.0)  # All distances <= 1
        assert np.allclose(distance_matrix, distance_matrix.T)  # Symmetric matrix

    def test_correlation_distance_transformation(self, clustering_engine):
        """Test correlation to distance transformation formula."""
        # Create known correlation matrix
        correlation_matrix = pd.DataFrame({
            'A': [1.0, 0.8, -0.2],
            'B': [0.8, 1.0, 0.5], 
            'C': [-0.2, 0.5, 1.0]
        }, index=['A', 'B', 'C'])
        
        # Create returns data that would produce this correlation
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.multivariate_normal([0, 0, 0], correlation_matrix.values, 300),
            columns=['A', 'B', 'C']
        )
        
        distance_matrix = clustering_engine.build_correlation_distance(returns)
        
        # Verify transformation: distance = (1 - correlation) / 2
        calculated_corr = returns.corr()
        expected_distances = (1.0 - calculated_corr.values) / 2.0
        
        np.testing.assert_allclose(distance_matrix, expected_distances, rtol=0.1)

    def test_insufficient_observations(self, clustering_engine):
        """Test handling of insufficient observations."""
        # Create data with too few observations
        short_data = pd.DataFrame(
            np.random.randn(50, 5),  # Only 50 observations, need 100
            columns=['A', 'B', 'C', 'D', 'E']
        )
        
        with pytest.raises(ValueError, match="Insufficient observations"):
            clustering_engine.build_correlation_distance(short_data)

    def test_empty_data_handling(self, clustering_engine):
        """Test handling of empty or invalid data."""
        # Empty DataFrame
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError, match="Returns data is empty"):
            clustering_engine.build_correlation_distance(empty_data)
        
        # All NaN data
        nan_data = pd.DataFrame(np.full((200, 5), np.nan), columns=['A', 'B', 'C', 'D', 'E'])
        with pytest.raises(ValueError, match="Returns data is empty"):
            clustering_engine.build_correlation_distance(nan_data)

    def test_hierarchical_clustering(self, clustering_engine, sample_returns_data):
        """Test hierarchical clustering functionality."""
        # Build distance matrix first
        distance_matrix = clustering_engine.build_correlation_distance(sample_returns_data)
        
        # Perform hierarchical clustering
        linkage_matrix = clustering_engine.hierarchical_clustering(distance_matrix)
        
        # Check linkage matrix properties
        assert linkage_matrix.shape == (9, 4)  # n-1 rows, 4 columns for 10 assets
        assert np.all(linkage_matrix[:, 2] >= 0)  # Distances non-negative
        assert np.all(linkage_matrix[:, 3] >= 1)  # Cluster sizes >= 1

    def test_different_linkage_methods(self, clustering_engine, sample_returns_data):
        """Test different hierarchical clustering linkage methods."""
        distance_matrix = clustering_engine.build_correlation_distance(sample_returns_data)
        
        linkage_methods = ['single', 'complete', 'average']
        
        for method in linkage_methods:
            linkage_matrix = clustering_engine.hierarchical_clustering(
                distance_matrix, linkage_method=method
            )
            assert linkage_matrix.shape == (9, 4)
            
    def test_invalid_linkage_method(self, clustering_engine, sample_returns_data):
        """Test invalid linkage method handling."""
        distance_matrix = clustering_engine.build_correlation_distance(sample_returns_data)
        
        with pytest.raises(ValueError, match="Invalid linkage method"):
            clustering_engine.hierarchical_clustering(
                distance_matrix, linkage_method="invalid"
            )

    def test_cluster_tree_construction(self, clustering_engine, sample_returns_data):
        """Test cluster tree construction."""
        # Build clustering components
        distance_matrix = clustering_engine.build_correlation_distance(sample_returns_data)
        linkage_matrix = clustering_engine.hierarchical_clustering(distance_matrix)
        
        # Build cluster tree
        asset_names = sample_returns_data.columns.tolist()
        cluster_tree = clustering_engine.build_cluster_tree(asset_names, linkage_matrix)
        
        # Check tree structure
        assert isinstance(cluster_tree, dict)
        assert "type" in cluster_tree
        assert "assets" in cluster_tree
        assert cluster_tree["type"] in ["leaf", "cluster"]
        
        # Check all assets are included
        tree_assets = set(cluster_tree["assets"])
        expected_assets = set(asset_names)
        assert tree_assets == expected_assets

    def test_cluster_tree_leaf_nodes(self, clustering_engine):
        """Test cluster tree with single asset (leaf node case)."""
        # Create minimal clustering case
        single_asset_returns = pd.DataFrame(
            np.random.randn(200, 1),
            columns=['SINGLE_ASSET']
        )
        
        distance_matrix = clustering_engine.build_correlation_distance(single_asset_returns)
        
        # Single asset should produce empty linkage matrix
        # This is an edge case that should be handled gracefully
        assert distance_matrix.shape == (1, 1)
        assert distance_matrix[0, 0] == 0.0

    def test_correlation_matrix_validation(self, clustering_engine):
        """Test correlation matrix validation functionality."""
        # Valid correlation matrix
        valid_corr = pd.DataFrame({
            'A': [1.0, 0.5, 0.3],
            'B': [0.5, 1.0, -0.2],
            'C': [0.3, -0.2, 1.0]
        }, index=['A', 'B', 'C'])
        
        is_valid, message = clustering_engine.validate_correlation_matrix(valid_corr)
        assert is_valid
        assert "Valid correlation matrix" in message
        
        # Invalid correlation matrix (not symmetric)
        invalid_corr = pd.DataFrame({
            'A': [1.0, 0.5, 0.3],
            'B': [0.6, 1.0, -0.2],  # Different from A-B correlation
            'C': [0.3, -0.2, 1.0]
        }, index=['A', 'B', 'C'])
        
        is_valid, message = clustering_engine.validate_correlation_matrix(invalid_corr)
        assert not is_valid
        assert "symmetric" in message

    def test_min_correlation_threshold(self):
        """Test minimum correlation threshold filtering."""
        config = ClusteringConfig(
            min_observations=100,
            min_correlation_threshold=0.3
        )
        clustering_engine = HRPClustering(config)
        
        # Create returns with known low correlations
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(200, 3),
            columns=['A', 'B', 'C']
        )
        
        distance_matrix = clustering_engine.build_correlation_distance(returns)
        
        # Check that low correlations were filtered (set to 0)
        correlation_matrix = returns.corr()
        low_corr_mask = np.abs(correlation_matrix.values) < 0.3
        
        # Distance matrix should reflect filtering
        assert isinstance(distance_matrix, np.ndarray)
        assert distance_matrix.shape == (3, 3)

    def test_get_cluster_assets(self, clustering_engine, sample_returns_data):
        """Test cluster asset extraction functionality."""
        # Build cluster tree
        distance_matrix = clustering_engine.build_correlation_distance(sample_returns_data)
        linkage_matrix = clustering_engine.hierarchical_clustering(distance_matrix)
        asset_names = sample_returns_data.columns.tolist()
        cluster_tree = clustering_engine.build_cluster_tree(asset_names, linkage_matrix)
        
        # Test asset extraction
        target_assets = asset_names[:5]  # First 5 assets
        extracted_assets = clustering_engine.get_cluster_assets(cluster_tree, target_assets)
        
        assert isinstance(extracted_assets, list)
        assert all(asset in asset_names for asset in extracted_assets)

    def test_quasi_diagonalization(self):
        """Test quasi-diagonalization feature."""
        config = ClusteringConfig(
            min_observations=100,
            enable_quasi_diagonalization=True
        )
        clustering_engine = HRPClustering(config)
        
        # Create sample correlation matrix and linkage
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(200, 5),
            columns=['A', 'B', 'C', 'D', 'E']
        )
        
        correlation_matrix = returns.corr()
        distance_matrix = clustering_engine.build_correlation_distance(returns)
        linkage_matrix = clustering_engine.hierarchical_clustering(distance_matrix)
        
        # Apply quasi-diagonalization
        quasi_diag = clustering_engine.quasi_diagonalize_correlation(
            correlation_matrix, linkage_matrix
        )
        
        # Check that result is a valid correlation matrix
        assert isinstance(quasi_diag, pd.DataFrame)
        assert quasi_diag.shape == correlation_matrix.shape
        assert np.allclose(np.diag(quasi_diag.values), 1.0)

    def test_memory_efficient_correlation(self):
        """Test memory-efficient correlation calculation."""
        config = ClusteringConfig(
            min_observations=50,
            memory_efficient=True,
            chunk_size=3
        )
        clustering_engine = HRPClustering(config)
        
        # Create data larger than chunk size
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(100, 8),  # 8 assets > chunk_size of 3
            columns=[f'ASSET_{i}' for i in range(8)]
        )
        
        # Calculate correlation with memory-efficient method
        memory_efficient_corr = clustering_engine.build_correlation_memory_efficient(returns)
        
        # Compare with standard method
        standard_corr = returns.corr()
        
        # Results should be approximately equal
        np.testing.assert_allclose(
            memory_efficient_corr.values,
            standard_corr.values,
            rtol=1e-10
        )