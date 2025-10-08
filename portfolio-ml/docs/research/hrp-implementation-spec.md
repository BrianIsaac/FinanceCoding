# HRP Implementation Specification
## Portfolio Optimization with Machine Learning Techniques

**Version:** 1.0  
**Date:** September 5, 2025  
**Dependencies:** Existing data pipeline, unified constraint system

---

## Overview

The HRP (Hierarchical Risk Parity) module implements clustering-aware portfolio allocation using correlation distance matrices and recursive bisection. This approach avoids the unstable covariance matrix inversion problems of traditional mean-variance optimization while respecting natural asset groupings and relationships discovered through hierarchical clustering.

## Theoretical Foundation

### Core HRP Algorithm
1. **Distance Matrix Construction**: Convert correlation matrix to distance metric using `d = √((1 - ρ)/2)`
2. **Hierarchical Clustering**: Build asset hierarchy using correlation distances and specified linkage method
3. **Quasi-Diagonalization**: Reorder assets according to clustering hierarchy to minimize off-diagonal correlations
4. **Recursive Bisection**: Allocate capital through recursive cluster bisection using inverse variance weighting

### Mathematical Framework
```python
# Distance metric from correlation
d_ij = sqrt((1 - ρ_ij) / 2)

# Recursive bisection allocation
def allocate_weight(cluster_weights, left_cluster, right_cluster):
    # Calculate cluster variances
    var_left = calculate_cluster_variance(left_cluster)
    var_right = calculate_cluster_variance(right_cluster) 
    
    # Inverse variance allocation
    total_var = var_left + var_right
    alpha = var_right / total_var  # Allocation to left cluster
    
    return alpha * cluster_weights, (1 - alpha) * cluster_weights
```

## Data Integration

### Input Data Requirements
```python
@dataclass
class HRPDataConfig:
    lookback_days: int = 756                     # 3 years of daily data (252*3)
    min_observations: int = 252                  # Minimum overlap for correlation
    correlation_method: str = 'pearson'         # Correlation calculation method
    handle_missing: str = 'pairwise'            # Missing data handling
    min_asset_coverage: float = 0.8              # Minimum data coverage per asset
    rebalance_frequency: str = 'M'               # Monthly rebalancing
    
    # Data preprocessing
    winsorize_returns: bool = True               # Cap extreme returns
    winsorize_quantiles: Tuple[float, float] = (0.01, 0.99)  # Winsorization limits
    standardize_assets: bool = False             # Don't standardize for HRP
```

### Data Pipeline Integration
```python
class HRPDataProcessor:
    def __init__(self, config: HRPDataConfig):
        self.config = config
        
    def prepare_returns_matrix(self, 
                              returns_df: pd.DataFrame,
                              universe: List[str],
                              end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Prepare clean returns matrix for HRP calculation.
        
        Returns:
            returns_matrix: [dates x assets] aligned returns
        """
        # Calculate lookback period
        start_date = end_date - pd.Timedelta(days=self.config.lookback_days + 30)
        
        # Filter returns to universe and time period
        period_returns = returns_df.loc[start_date:end_date, universe]
        
        # Remove assets with insufficient data
        coverage = period_returns.notna().mean()
        valid_assets = coverage[coverage >= self.config.min_asset_coverage].index
        period_returns = period_returns[valid_assets]
        
        # Handle missing values
        if self.config.handle_missing == 'forward_fill':
            period_returns = period_returns.fillna(method='ffill')
        elif self.config.handle_missing == 'drop':
            period_returns = period_returns.dropna(axis=1)
        
        # Winsorize extreme returns
        if self.config.winsorize_returns:
            for asset in period_returns.columns:
                lower, upper = period_returns[asset].quantile(self.config.winsorize_quantiles)
                period_returns[asset] = period_returns[asset].clip(lower=lower, upper=upper)
        
        # Get final lookback period with clean data
        clean_returns = period_returns.tail(self.config.lookback_days)
        
        # Ensure minimum observations
        if len(clean_returns) < self.config.min_observations:
            raise ValueError(f"Insufficient data: {len(clean_returns)} < {self.config.min_observations}")
            
        return clean_returns
    
    def calculate_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix with proper handling of missing data."""
        if self.config.handle_missing == 'pairwise':
            return returns_df.corr(method=self.config.correlation_method)
        else:
            return returns_df.corr(method=self.config.correlation_method, min_periods=self.config.min_observations)
```

## Core HRP Implementation

### Distance Matrix and Clustering
```python
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from scipy.spatial.distance import squareform
from typing import Dict, List, Tuple, Optional

class HRPClustering:
    def __init__(self, linkage_method: str = 'single'):
        """
        Initialize HRP clustering.
        
        Args:
            linkage_method: 'single', 'complete', 'average', 'weighted', 'ward'
        """
        self.linkage_method = linkage_method
        
    def correlation_to_distance(self, correlation_matrix: pd.DataFrame) -> np.ndarray:
        """
        Convert correlation matrix to distance matrix.
        
        Distance formula: d = sqrt((1 - ρ) / 2)
        This ensures distance properties: d ∈ [0, 1], d(i,i) = 0
        """
        # Clip correlations to [-1, 1] for numerical stability
        corr_clipped = np.clip(correlation_matrix.values, -1, 1)
        
        # Convert to distance
        distance_matrix = np.sqrt((1 - corr_clipped) / 2)
        
        # Ensure diagonal is zero
        np.fill_diagonal(distance_matrix, 0)
        
        return distance_matrix
    
    def perform_clustering(self, correlation_matrix: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform hierarchical clustering on correlation distance matrix.
        
        Returns:
            linkage_matrix: Hierarchical clustering linkage matrix
            distance_matrix: Distance matrix used for clustering
        """
        # Convert correlation to distance
        distance_matrix = self.correlation_to_distance(correlation_matrix)
        
        # Convert to condensed distance matrix for scipy
        condensed_distances = squareform(distance_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distances, method=self.linkage_method)
        
        return linkage_matrix, distance_matrix
    
    def get_cluster_ordering(self, linkage_matrix: np.ndarray) -> List[int]:
        """
        Get optimal asset ordering from hierarchical clustering.
        
        This reorders assets to minimize off-diagonal correlation magnitudes.
        """
        # Convert linkage matrix to tree structure
        tree = to_tree(linkage_matrix)
        
        # Get leaf ordering from tree traversal
        def get_leaf_order(node):
            if node.is_leaf():
                return [node.id]
            else:
                left_leaves = get_leaf_order(node.left)
                right_leaves = get_leaf_order(node.right)
                return left_leaves + right_leaves
        
        return get_leaf_order(tree)
    
    def quasi_diagonalize(self, 
                         correlation_matrix: pd.DataFrame,
                         asset_order: List[int]) -> pd.DataFrame:
        """
        Reorder correlation matrix to quasi-diagonal form.
        
        This groups similar assets together, reducing off-diagonal correlations.
        """
        assets = correlation_matrix.index
        ordered_assets = [assets[i] for i in asset_order]
        
        # Reorder both rows and columns
        quasi_diag_corr = correlation_matrix.loc[ordered_assets, ordered_assets]
        
        return quasi_diag_corr
```

### Recursive Bisection Allocation
```python
class HRPAllocation:
    def __init__(self, risk_measure: str = 'variance'):
        """
        Initialize HRP allocation engine.
        
        Args:
            risk_measure: 'variance', 'mad' (mean absolute deviation), or 'cvar'
        """
        self.risk_measure = risk_measure
    
    def calculate_cluster_risk(self, 
                              returns_df: pd.DataFrame,
                              cluster_assets: List[str]) -> float:
        """Calculate risk measure for a cluster of assets."""
        if len(cluster_assets) == 1:
            asset_returns = returns_df[cluster_assets[0]]
            if self.risk_measure == 'variance':
                return asset_returns.var()
            elif self.risk_measure == 'mad':
                return asset_returns.mad()
            elif self.risk_measure == 'cvar':
                return -asset_returns.quantile(0.05)  # 5% CVaR
        else:
            # For multi-asset clusters, use equally weighted portfolio risk
            cluster_returns = returns_df[cluster_assets]
            equal_weights = np.ones(len(cluster_assets)) / len(cluster_assets)
            portfolio_returns = cluster_returns.dot(equal_weights)
            
            if self.risk_measure == 'variance':
                return portfolio_returns.var()
            elif self.risk_measure == 'mad':
                return portfolio_returns.mad()
            elif self.risk_measure == 'cvar':
                return -portfolio_returns.quantile(0.05)
    
    def recursive_bisection(self, 
                          returns_df: pd.DataFrame,
                          correlation_matrix: pd.DataFrame,
                          linkage_matrix: np.ndarray,
                          initial_weight: float = 1.0) -> pd.Series:
        """
        Perform recursive bisection to allocate portfolio weights.
        
        This is the core HRP allocation algorithm.
        """
        assets = list(correlation_matrix.index)
        n_assets = len(assets)
        
        # Initialize weights
        weights = pd.Series(0.0, index=assets)
        
        # Convert linkage matrix to tree for traversal
        tree = to_tree(linkage_matrix)
        
        def allocate_node(node, available_weight: float, asset_subset: List[str]):
            """Recursively allocate weights through the cluster tree."""
            
            if node.is_leaf():
                # Leaf node - assign weight to single asset
                asset_name = assets[node.id]
                weights[asset_name] = available_weight
                return
            
            # Internal node - split weight between children
            left_assets = self._get_cluster_assets(node.left, assets)
            right_assets = self._get_cluster_assets(node.right, assets)
            
            # Filter to assets in current subset
            left_assets = [a for a in left_assets if a in asset_subset]
            right_assets = [a for a in right_assets if a in asset_subset]
            
            if not left_assets or not right_assets:
                # Degenerate case - allocate all weight to non-empty cluster
                if left_assets:
                    self.allocate_node(node.left, available_weight, left_assets)
                elif right_assets:
                    self.allocate_node(node.right, available_weight, right_assets)
                return
            
            # Calculate cluster risks
            left_risk = self.calculate_cluster_risk(returns_df, left_assets)
            right_risk = self.calculate_cluster_risk(returns_df, right_assets)
            
            # Inverse risk weighting
            total_inv_risk = (1/left_risk) + (1/right_risk)
            left_weight_fraction = (1/left_risk) / total_inv_risk
            right_weight_fraction = (1/right_risk) / total_inv_risk
            
            # Recursive allocation to children
            allocate_node(node.left, available_weight * left_weight_fraction, left_assets)
            allocate_node(node.right, available_weight * right_weight_fraction, right_assets)
        
        # Start recursive allocation from root
        allocate_node(tree, initial_weight, assets)
        
        return weights
    
    def _get_cluster_assets(self, node, assets: List[str]) -> List[str]:
        """Get all assets in a cluster subtree."""
        if node.is_leaf():
            return [assets[node.id]]
        else:
            left_assets = self._get_cluster_assets(node.left, assets)
            right_assets = self._get_cluster_assets(node.right, assets)
            return left_assets + right_assets
```

### Complete HRP Model Implementation
```python
class HRPPortfolioModel(PortfolioModel):
    def __init__(self, 
                 constraints: PortfolioConstraints,
                 data_config: HRPDataConfig,
                 clustering_config: Dict = None,
                 allocation_config: Dict = None):
        super().__init__(constraints)
        self.data_config = data_config
        
        # Initialize components
        self.data_processor = HRPDataProcessor(data_config)
        
        clustering_params = clustering_config or {'linkage_method': 'single'}
        self.clustering = HRPClustering(**clustering_params)
        
        allocation_params = allocation_config or {'risk_measure': 'variance'}
        self.allocation = HRPAllocation(**allocation_params)
        
        # Store latest model state
        self.last_correlation_matrix = None
        self.last_linkage_matrix = None
        self.last_asset_order = None
        
    def fit(self, 
            returns: pd.DataFrame,
            universe: List[str], 
            fit_period: Tuple[pd.Timestamp, pd.Timestamp]) -> None:
        """
        Fit HRP model on historical data.
        
        For HRP, 'fitting' means calculating the hierarchical structure
        that will be used for the next rebalancing period.
        """
        end_date = fit_period[1]
        
        # Prepare returns matrix
        try:
            clean_returns = self.data_processor.prepare_returns_matrix(
                returns, universe, end_date
            )
        except ValueError as e:
            print(f"Data preparation failed: {e}")
            # Fallback to equal weights if insufficient data
            self.last_correlation_matrix = None
            return
        
        # Calculate correlation matrix
        correlation_matrix = self.data_processor.calculate_correlation_matrix(clean_returns)
        
        # Perform clustering
        linkage_matrix, distance_matrix = self.clustering.perform_clustering(correlation_matrix)
        
        # Get optimal asset ordering
        asset_order = self.clustering.get_cluster_ordering(linkage_matrix)
        
        # Store results for prediction
        self.last_correlation_matrix = correlation_matrix
        self.last_linkage_matrix = linkage_matrix  
        self.last_asset_order = asset_order
        self.last_returns = clean_returns
        
    def predict_weights(self, 
                       date: pd.Timestamp,
                       universe: List[str]) -> pd.Series:
        """Generate HRP portfolio weights for rebalancing date."""
        
        # Check if model was fitted successfully
        if self.last_correlation_matrix is None:
            print("HRP model not fitted or insufficient data. Using equal weights.")
            equal_weights = pd.Series(1.0 / len(universe), index=universe)
            return self._apply_constraints(equal_weights)
        
        # Filter to available assets
        available_assets = [asset for asset in universe 
                          if asset in self.last_correlation_matrix.index]
        
        if len(available_assets) < 2:
            print("Insufficient assets for HRP. Using equal weights.")
            equal_weights = pd.Series(1.0 / len(universe), index=universe)
            return self._apply_constraints(equal_weights)
        
        # Get subset correlation matrix and returns
        subset_correlation = self.last_correlation_matrix.loc[available_assets, available_assets]
        subset_returns = self.last_returns[available_assets]
        
        # Recalculate clustering for available assets
        linkage_matrix, _ = self.clustering.perform_clustering(subset_correlation)
        
        # Perform recursive bisection allocation
        hrp_weights = self.allocation.recursive_bisection(
            subset_returns, subset_correlation, linkage_matrix
        )
        
        # Extend to full universe (zero weights for unavailable assets)
        full_weights = pd.Series(0.0, index=universe)
        full_weights.loc[available_assets] = hrp_weights
        
        # Apply constraints
        final_weights = self._apply_constraints(full_weights)
        
        return final_weights
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for analysis."""
        clustering_info = {
            "model_type": "HRP",
            "linkage_method": self.clustering.linkage_method,
            "risk_measure": self.allocation.risk_measure,
            "lookback_days": self.data_config.lookback_days,
            "min_observations": self.data_config.min_observations,
            "correlation_method": self.data_config.correlation_method
        }
        
        if self.last_correlation_matrix is not None:
            clustering_info.update({
                "n_assets_last_fit": len(self.last_correlation_matrix),
                "avg_correlation": self.last_correlation_matrix.values[np.triu_indices_from(self.last_correlation_matrix.values, k=1)].mean(),
                "min_correlation": self.last_correlation_matrix.min().min(),
                "max_correlation": self.last_correlation_matrix.max().max()
            })
            
        return clustering_info
    
    def get_cluster_analysis(self) -> Dict[str, Any]:
        """Return detailed clustering analysis for interpretation."""
        if self.last_correlation_matrix is None or self.last_linkage_matrix is None:
            return {"error": "Model not fitted"}
        
        # Asset ordering from clustering
        assets = list(self.last_correlation_matrix.index)
        ordered_assets = [assets[i] for i in self.last_asset_order]
        
        # Quasi-diagonal correlation matrix
        quasi_diag_corr = self.clustering.quasi_diagonalize(
            self.last_correlation_matrix, self.last_asset_order
        )
        
        return {
            "asset_order": ordered_assets,
            "quasi_diagonal_correlation": quasi_diag_corr,
            "linkage_matrix": self.last_linkage_matrix,
            "clustering_quality": self._assess_clustering_quality(quasi_diag_corr)
        }
    
    def _assess_clustering_quality(self, quasi_diag_corr: pd.DataFrame) -> Dict[str, float]:
        """Assess quality of hierarchical clustering."""
        n = len(quasi_diag_corr)
        
        # Calculate on-diagonal vs off-diagonal correlation strength
        diagonal_strength = np.mean([abs(quasi_diag_corr.iloc[i, i+1]) for i in range(n-1)])
        
        # Off-diagonal correlations (excluding immediate neighbors)
        off_diagonal_vals = []
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1:  # Skip diagonal and immediate neighbors
                    off_diagonal_vals.append(abs(quasi_diag_corr.iloc[i, j]))
        
        off_diagonal_strength = np.mean(off_diagonal_vals) if off_diagonal_vals else 0
        
        return {
            "diagonal_strength": diagonal_strength,
            "off_diagonal_strength": off_diagonal_strength,
            "clustering_ratio": diagonal_strength / (off_diagonal_strength + 1e-8)
        }
```

## Hydra Configuration

### Default HRP Configuration
```yaml
# configs/models/hrp_default.yaml
model:
  _target_: src.models.hrp.model.HRPPortfolioModel
  
  data_config:
    lookback_days: 756                    # 3 years
    min_observations: 252                 # 1 year minimum
    correlation_method: 'pearson'
    handle_missing: 'pairwise'
    min_asset_coverage: 0.8
    rebalance_frequency: 'M'
    winsorize_returns: true
    winsorize_quantiles: [0.01, 0.99]
    standardize_assets: false

  clustering_config:
    linkage_method: 'single'              # single, complete, average, weighted

  allocation_config:
    risk_measure: 'variance'              # variance, mad, cvar

  constraints:
    long_only: true
    top_k_positions: 50
    max_position_weight: 0.10
    max_monthly_turnover: 0.20
    transaction_cost_bps: 10.0

# Alternative configurations for experimentation
hrp_complete_linkage:
  defaults:
    - hrp_default
  clustering_config:
    linkage_method: 'complete'

hrp_mad_risk:
  defaults:
    - hrp_default  
  allocation_config:
    risk_measure: 'mad'

hrp_short_lookback:
  defaults:
    - hrp_default
  data_config:
    lookback_days: 252                    # 1 year lookback
```

## Integration with Existing Framework

### Memory Efficiency
```python
class HRPMemoryOptimizer:
    @staticmethod
    def optimize_correlation_calculation(returns_df: pd.DataFrame) -> pd.DataFrame:
        """Optimize correlation calculation for large datasets."""
        n_assets = len(returns_df.columns)
        
        # For very large universes, use chunked correlation calculation
        if n_assets > 1000:
            return HRPMemoryOptimizer._chunked_correlation(returns_df)
        else:
            return returns_df.corr()
    
    @staticmethod  
    def _chunked_correlation(returns_df: pd.DataFrame, chunk_size: int = 200) -> pd.DataFrame:
        """Calculate correlation matrix in chunks to save memory."""
        assets = returns_df.columns
        n_assets = len(assets)
        correlation_matrix = pd.DataFrame(np.eye(n_assets), index=assets, columns=assets)
        
        for i in range(0, n_assets, chunk_size):
            end_i = min(i + chunk_size, n_assets)
            chunk_i = assets[i:end_i]
            
            for j in range(i, n_assets, chunk_size):
                end_j = min(j + chunk_size, n_assets)
                chunk_j = assets[j:end_j]
                
                # Calculate correlation for this chunk
                chunk_corr = returns_df[chunk_i].corrwith(returns_df[chunk_j])
                correlation_matrix.loc[chunk_i, chunk_j] = chunk_corr
                
                # Fill symmetric part
                if i != j:
                    correlation_matrix.loc[chunk_j, chunk_i] = chunk_corr.T
        
        return correlation_matrix
```

### Validation and Testing
```python
# tests/unit/test_models/test_hrp.py
class TestHRPModel:
    def test_correlation_distance_conversion(self):
        """Test correlation to distance matrix conversion."""
        # Create known correlation matrix
        corr_matrix = pd.DataFrame([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.5], 
            [0.3, 0.5, 1.0]
        ], columns=['A', 'B', 'C'], index=['A', 'B', 'C'])
        
        clustering = HRPClustering()
        distance_matrix = clustering.correlation_to_distance(corr_matrix)
        
        # Verify distance properties
        assert np.allclose(np.diag(distance_matrix), 0)  # Diagonal should be 0
        assert np.all(distance_matrix >= 0)              # All distances >= 0
        assert np.all(distance_matrix <= 1)              # All distances <= 1
        
        # Verify distance calculation: d = sqrt((1-ρ)/2)
        expected_d_AB = np.sqrt((1 - 0.8) / 2)
        assert np.isclose(distance_matrix[0, 1], expected_d_AB)
        
    def test_recursive_bisection(self):
        """Test recursive bisection weight allocation."""
        # Test with simple 3-asset case
        returns_df = pd.DataFrame({
            'A': np.random.normal(0.001, 0.02, 252),
            'B': np.random.normal(0.001, 0.02, 252), 
            'C': np.random.normal(0.001, 0.02, 252)
        })
        
        model = HRPPortfolioModel(
            constraints=PortfolioConstraints(),
            data_config=HRPDataConfig(lookback_days=252)
        )
        
        # Fit and predict
        model.fit(returns_df, ['A', 'B', 'C'], 
                 (returns_df.index[0], returns_df.index[-1]))
        weights = model.predict_weights(returns_df.index[-1], ['A', 'B', 'C'])
        
        # Verify weight properties
        assert np.isclose(weights.sum(), 1.0)  # Weights sum to 1
        assert np.all(weights >= 0)            # All weights non-negative
        
    def test_clustering_quality(self):
        """Test clustering quality assessment."""
        # Create assets with known correlation structure
        np.random.seed(42)
        n_assets = 20
        
        # Create two clusters with high intra-cluster correlation
        cluster1_returns = np.random.multivariate_normal(
            [0.001] * 10, 
            0.0004 * (0.8 * np.ones((10, 10)) + 0.2 * np.eye(10)), 
            252
        )
        cluster2_returns = np.random.multivariate_normal(
            [0.001] * 10,
            0.0004 * (0.7 * np.ones((10, 10)) + 0.3 * np.eye(10)),
            252
        )
        
        returns_df = pd.DataFrame(
            np.hstack([cluster1_returns, cluster2_returns]),
            columns=[f'A{i}' for i in range(10)] + [f'B{i}' for i in range(10)]
        )
        
        model = HRPPortfolioModel(
            constraints=PortfolioConstraints(),
            data_config=HRPDataConfig(lookback_days=252)
        )
        
        model.fit(returns_df, returns_df.columns.tolist(),
                 (returns_df.index[0], returns_df.index[-1]))
        
        cluster_analysis = model.get_cluster_analysis()
        
        # Clustering should identify the two groups
        assert cluster_analysis['clustering_quality']['clustering_ratio'] > 1.0
```

## Performance Optimization

### Computational Efficiency
```python
class HRPPerformanceOptimizer:
    @staticmethod
    def optimize_linkage_calculation(distance_matrix: np.ndarray) -> np.ndarray:
        """Optimize linkage calculation for large correlation matrices."""
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
        
        # Use optimized linkage methods
        condensed_distances = squareform(distance_matrix)
        
        # Single linkage is fastest for large datasets
        return linkage(condensed_distances, method='single', optimal_ordering=True)
    
    @staticmethod  
    def cache_correlation_calculation(cache_dir: str = '.hrp_cache/'):
        """Cache correlation calculations to avoid recomputation."""
        import os
        import pickle
        import hashlib
        
        os.makedirs(cache_dir, exist_ok=True)
        
        def correlation_cache_decorator(func):
            def wrapper(returns_df, *args, **kwargs):
                # Create cache key from data hash
                data_hash = hashlib.md5(
                    pd.util.hash_pandas_object(returns_df).values
                ).hexdigest()
                cache_file = os.path.join(cache_dir, f'corr_{data_hash}.pkl')
                
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                
                # Calculate and cache result
                result = func(returns_df, *args, **kwargs)
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                    
                return result
            return wrapper
        return correlation_cache_decorator
```

## Implementation Timeline

- **Week 1**: Data preprocessing and correlation matrix calculation
- **Week 2**: Hierarchical clustering implementation and validation
- **Week 3**: Recursive bisection allocation algorithm
- **Week 4**: Integration testing, performance optimization, and constraint application

This specification provides a complete implementation roadmap for the HRP module that integrates seamlessly with your existing data pipeline and framework architecture.