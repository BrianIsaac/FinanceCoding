#!/usr/bin/env python3
"""
Aggressive HRP Training Pipeline with Extended Parameter Space
Enhanced version with more comprehensive hyperparameter exploration
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from src.data.loaders.parquet_manager import ParquetManager


class AggressiveHRPTraining:
    """Aggressive HRP training with extensive parameter exploration."""

    def __init__(self):
        self.setup_logging()
        self.data_manager = ParquetManager()
        self.results = []

    def setup_logging(self):
        """Setup enhanced logging."""
        log_dir = Path("logs/training/hrp_aggressive")
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "hrp_aggressive_training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def generate_aggressive_configs(self) -> list[dict[str, Any]]:
        """
        Generate extensive hyperparameter configurations for aggressive training.

        Returns:
            List of aggressive configuration dictionaries
        """
        configurations = []

        # More comprehensive parameter space
        lookback_periods = [63, 126, 189, 252, 378, 504, 630, 756, 1008, 1260]  # 3M to 5Y
        linkage_methods = ['single', 'complete', 'average', 'ward', 'centroid', 'median']
        correlation_methods = ['pearson', 'spearman', 'kendall']
        distance_metrics = ['correlation', 'angular', 'abs_correlation']
        min_cluster_sizes = [8, 12, 15, 20, 25, 30, 35, 40]
        rebalancing_frequencies = [21, 42, 63]  # Monthly, bi-monthly, quarterly

        config_id = 0
        for lookback in lookback_periods:
            for linkage_method in linkage_methods:
                for corr_method in correlation_methods:
                    for distance_metric in distance_metrics:
                        for min_clusters in min_cluster_sizes:
                            for rebal_freq in rebalancing_frequencies:
                                # Skip invalid combinations
                                if linkage_method in ['centroid', 'median'] and distance_metric != 'correlation':
                                    continue
                                if lookback < 126 and min_clusters > 20:  # Not enough data for many clusters
                                    continue

                                config_id += 1
                                config = {
                                    'config_id': f"aggressive_hrp_{config_id:03d}",
                                    'lookback_days': lookback,
                                    'linkage_method': linkage_method,
                                    'correlation_method': corr_method,
                                    'distance_metric': distance_metric,
                                    'min_cluster_size': min_clusters,
                                    'rebalancing_frequency': rebal_freq,
                                    'max_cluster_size': min(50, lookback // 10),  # Adaptive max cluster size
                                    'risk_budget_method': 'equal_risk',
                                    'use_robust_covariance': True,
                                    'shrinkage_target': 'diagonal'
                                }
                                configurations.append(config)

        # Add some extreme configurations for stress testing
        extreme_configs = [
            {
                'config_id': 'extreme_short_term',
                'lookback_days': 21,
                'linkage_method': 'ward',
                'correlation_method': 'pearson',
                'distance_metric': 'correlation',
                'min_cluster_size': 5,
                'rebalancing_frequency': 5,  # Weekly rebalancing
                'max_cluster_size': 10,
                'risk_budget_method': 'inverse_volatility',
                'use_robust_covariance': True,
                'shrinkage_target': 'market'
            },
            {
                'config_id': 'extreme_long_term',
                'lookbook_days': 1512,  # 6 years
                'linkage_method': 'average',
                'correlation_method': 'kendall',
                'distance_metric': 'angular',
                'min_cluster_size': 50,
                'rebalancing_frequency': 126,  # Semi-annual
                'max_cluster_size': 100,
                'risk_budget_method': 'risk_parity',
                'use_robust_covariance': True,
                'shrinkage_target': 'constant_correlation'
            }
        ]

        configurations.extend(extreme_configs)

        self.logger.info(f"Generated {len(configurations)} aggressive HRP configurations")
        return configurations

    def train_hrp_configuration(self, config: dict[str, Any],
                              returns_data: pd.DataFrame) -> dict[str, Any]:
        """
        Train HRP model with aggressive configuration.

        Args:
            config: Training configuration
            returns_data: Returns data

        Returns:
            Training results dictionary
        """
        try:
            # Enhanced universe selection - use more assets
            universe = self.select_enhanced_universe(returns_data,
                                                   min_assets=600,  # Much larger universe
                                                   coverage_threshold=0.85)

            returns_filtered = returns_data[universe].dropna()

            if len(returns_filtered.columns) < config['min_cluster_size']:
                self.logger.warning(f"Not enough assets ({len(returns_filtered.columns)}) for config {config['config_id']}")
                return {'config': config, 'status': 'failed', 'error': 'insufficient_assets'}

            # Calculate correlation matrix with enhanced robustness
            correlation_matrix = self.calculate_enhanced_correlation(
                returns_filtered,
                method=config['correlation_method'],
                lookback=config['lookback_days'],
                robust=config.get('use_robust_covariance', False)
            )

            # Enhanced distance calculation
            distance_matrix = self.calculate_distance_matrix(
                correlation_matrix,
                method=config['distance_metric']
            )

            # Perform hierarchical clustering with extended validation
            clustering_result = self.perform_enhanced_clustering(
                distance_matrix,
                linkage_method=config['linkage_method'],
                min_clusters=config['min_cluster_size'],
                max_clusters=config.get('max_cluster_size', 50)
            )

            # Enhanced risk budgeting
            portfolio_weights = self.calculate_enhanced_risk_budget(
                returns_filtered,
                clustering_result['cluster_labels'],
                method=config.get('risk_budget_method', 'equal_risk')
            )

            # Comprehensive validation metrics
            validation_metrics = self.calculate_comprehensive_metrics(
                returns_filtered,
                portfolio_weights,
                clustering_result,
                correlation_matrix
            )

            result = {
                'config': config,
                'status': 'success',
                'portfolio_weights': portfolio_weights.to_dict(),
                'clustering_metrics': clustering_result,
                'validation_metrics': validation_metrics,
                'universe_size': len(universe),
                'training_timestamp': datetime.now().isoformat()
            }

            self.logger.info(f"Successfully trained {config['config_id']}: "
                           f"Silhouette={validation_metrics['silhouette_score']:.4f}, "
                           f"Clusters={clustering_result['n_clusters']}")

            return result

        except Exception as e:
            self.logger.error(f"Training failed for {config['config_id']}: {str(e)}")
            return {
                'config': config,
                'status': 'failed',
                'error': str(e),
                'training_timestamp': datetime.now().isoformat()
            }

    def select_enhanced_universe(self, returns_data: pd.DataFrame,
                               min_assets: int = 600,
                               coverage_threshold: float = 0.85) -> list[str]:
        """Select enhanced universe with more aggressive asset selection."""
        coverage = returns_data.count() / len(returns_data)
        valid_assets = coverage[coverage >= coverage_threshold].index.tolist()

        if len(valid_assets) < min_assets:
            # Gradually lower threshold to get more assets
            for threshold in [0.80, 0.75, 0.70, 0.65]:
                valid_assets = coverage[coverage >= threshold].index.tolist()
                if len(valid_assets) >= min_assets:
                    break

        # Select top assets by trading volume if we have volume data
        if len(valid_assets) > min_assets:
            # For now, just take the first min_assets
            valid_assets = valid_assets[:min_assets]

        return valid_assets

    def calculate_enhanced_correlation(self, returns_data: pd.DataFrame,
                                     method: str = 'pearson',
                                     lookback: int = 252,
                                     robust: bool = True) -> pd.DataFrame:
        """Calculate correlation with enhanced robustness."""
        # Use rolling correlation with specified lookback
        if len(returns_data) > lookback:
            returns_subset = returns_data.tail(lookback)
        else:
            returns_subset = returns_data

        if method == 'pearson':
            corr = returns_subset.corr()
        elif method == 'spearman':
            corr = returns_subset.corr(method='spearman')
        elif method == 'kendall':
            corr = returns_subset.corr(method='kendall')
        else:
            raise ValueError(f"Unknown correlation method: {method}")

        # Handle NaN values
        corr = corr.fillna(0.0)
        np.fill_diagonal(corr.values, 1.0)

        return corr

    def calculate_distance_matrix(self, correlation_matrix: pd.DataFrame,
                                method: str = 'correlation') -> np.ndarray:
        """Calculate distance matrix with multiple methods."""
        corr_array = correlation_matrix.values

        if method == 'correlation':
            distance_matrix = np.sqrt(0.5 * (1 - corr_array))
        elif method == 'angular':
            # Angular distance: arccos(correlation) / pi
            corr_clipped = np.clip(corr_array, -0.999, 0.999)
            distance_matrix = np.arccos(np.abs(corr_clipped)) / np.pi
        elif method == 'abs_correlation':
            distance_matrix = 1 - np.abs(corr_array)
        else:
            raise ValueError(f"Unknown distance method: {method}")

        # Ensure diagonal is zero
        np.fill_diagonal(distance_matrix, 0.0)

        return distance_matrix

    def perform_enhanced_clustering(self, distance_matrix: np.ndarray,
                                  linkage_method: str = 'average',
                                  min_clusters: int = 15,
                                  max_clusters: int = 50) -> dict[str, Any]:
        """Perform hierarchical clustering with enhanced validation."""

        # Convert to condensed form for linkage
        condensed_distances = squareform(distance_matrix, checks=False)

        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distances, method=linkage_method)

        # Find optimal number of clusters using silhouette analysis
        best_score = -1
        best_n_clusters = min_clusters
        best_labels = None

        for n_clusters in range(min_clusters, min(max_clusters + 1, len(distance_matrix) // 2)):
            labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

            if len(np.unique(labels)) < 2:
                continue

            try:
                score = silhouette_score(distance_matrix, labels, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
                    best_labels = labels
            except:
                continue

        if best_labels is None:
            # Fallback: use min_clusters
            best_labels = fcluster(linkage_matrix, min_clusters, criterion='maxclust')
            best_score = 0.0
            best_n_clusters = len(np.unique(best_labels))

        return {
            'cluster_labels': best_labels,
            'n_clusters': best_n_clusters,
            'silhouette_score': best_score,
            'linkage_matrix': linkage_matrix,
            'linkage_method': linkage_method
        }

    def calculate_enhanced_risk_budget(self, returns_data: pd.DataFrame,
                                     cluster_labels: np.ndarray,
                                     method: str = 'equal_risk') -> pd.Series:
        """Calculate portfolio weights using enhanced risk budgeting."""
        n_assets = len(returns_data.columns)
        weights = np.zeros(n_assets)

        # Calculate cluster weights and within-cluster weights
        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_assets = returns_data.columns[cluster_mask]
            cluster_returns = returns_data[cluster_assets]

            if len(cluster_assets) == 1:
                # Single asset cluster
                cluster_weights = np.array([1.0])
            else:
                # Calculate within-cluster weights based on method
                if method == 'equal_risk':
                    cluster_weights = np.ones(len(cluster_assets)) / len(cluster_assets)
                elif method == 'inverse_volatility':
                    volatilities = cluster_returns.std()
                    inv_vol = 1 / volatilities
                    cluster_weights = inv_vol / inv_vol.sum()
                elif method == 'risk_parity':
                    # Simplified risk parity (could be enhanced further)
                    cov_matrix = cluster_returns.cov()
                    inv_diag = 1 / np.sqrt(np.diag(cov_matrix))
                    cluster_weights = inv_diag / inv_diag.sum()
                else:
                    cluster_weights = np.ones(len(cluster_assets)) / len(cluster_assets)

            # Assign cluster weight (equal risk allocation across clusters)
            cluster_allocation = 1.0 / len(unique_clusters)
            weights[cluster_mask] = cluster_allocation * cluster_weights

        return pd.Series(weights, index=returns_data.columns)

    def calculate_comprehensive_metrics(self, returns_data: pd.DataFrame,
                                      portfolio_weights: pd.Series,
                                      clustering_result: dict,
                                      correlation_matrix: pd.DataFrame) -> dict[str, float]:
        """Calculate comprehensive validation metrics."""

        # Portfolio performance metrics
        portfolio_returns = (returns_data * portfolio_weights).sum(axis=1)
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)
        portfolio_sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)

        # Diversification metrics
        n_effective_assets = 1 / (portfolio_weights ** 2).sum()
        max_weight = portfolio_weights.max()
        weight_concentration = (portfolio_weights ** 2).sum()

        # Clustering quality metrics
        silhouette = clustering_result.get('silhouette_score', 0.0)
        n_clusters = clustering_result.get('n_clusters', 0)

        # Risk metrics
        portfolio_var = np.dot(portfolio_weights.values,
                              np.dot(returns_data.cov().values, portfolio_weights.values))
        diversification_ratio = (portfolio_weights * returns_data.std()).sum() / np.sqrt(portfolio_var)

        return {
            'portfolio_volatility': portfolio_vol,
            'portfolio_sharpe': portfolio_sharpe,
            'silhouette_score': silhouette,
            'n_clusters': n_clusters,
            'n_effective_assets': n_effective_assets,
            'max_weight': max_weight,
            'weight_concentration': weight_concentration,
            'diversification_ratio': diversification_ratio,
            'portfolio_var': portfolio_var
        }

    def run_aggressive_training(self) -> None:
        """Run aggressive HRP training across all configurations."""

        self.logger.info("Starting Aggressive HRP Training Pipeline")
        self.logger.info("="*80)

        # Load enhanced dataset
        try:
            self.logger.info("Loading production datasets with enhanced coverage...")
            returns_data = self.data_manager.load_returns()

            self.logger.info(f"Loaded returns data: {returns_data.shape}")
            self.logger.info(f"Date range: {returns_data.index[0]} to {returns_data.index[-1]}")

        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return

        # Generate aggressive configurations
        configs = self.generate_aggressive_configs()

        # Train all configurations
        self.logger.info(f"Training {len(configs)} aggressive HRP configurations...")

        successful_configs = 0
        failed_configs = 0

        for i, config in enumerate(configs, 1):
            self.logger.info(f"\nTraining configuration {i}/{len(configs)}: {config['config_id']}")

            result = self.train_hrp_configuration(config, returns_data)
            self.results.append(result)

            if result['status'] == 'success':
                successful_configs += 1
            else:
                failed_configs += 1

        # Save comprehensive results
        self.save_aggressive_results()

        # Summary
        self.logger.info("\n" + "="*80)
        self.logger.info("Aggressive HRP Training Completed!")
        self.logger.info(f"Successful configurations: {successful_configs}")
        self.logger.info(f"Failed configurations: {failed_configs}")

        if successful_configs > 0:
            successful_results = [r for r in self.results if r['status'] == 'success']
            best_result = max(successful_results,
                            key=lambda x: x['validation_metrics']['silhouette_score'])

            self.logger.info(f"Best configuration: {best_result['config']['config_id']}")
            self.logger.info(f"Best silhouette score: {best_result['validation_metrics']['silhouette_score']:.4f}")
            self.logger.info(f"Best Sharpe ratio: {best_result['validation_metrics']['portfolio_sharpe']:.4f}")

    def save_aggressive_results(self) -> None:
        """Save aggressive training results."""
        results_dir = Path("data/models/checkpoints/hrp_aggressive")
        results_dir.mkdir(parents=True, exist_ok=True)

        logs_dir = Path("logs/training/hrp_aggressive")

        # Save detailed results
        results_file = logs_dir / "hrp_aggressive_results.yaml"
        with open(results_file, 'w') as f:
            yaml.dump({
                'training_summary': {
                    'total_configs': len(self.results),
                    'successful_configs': sum(1 for r in self.results if r['status'] == 'success'),
                    'failed_configs': sum(1 for r in self.results if r['status'] == 'failed'),
                    'training_completed': datetime.now().isoformat()
                },
                'results': self.results
            }, f, default_flow_style=False)

        self.logger.info(f"Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description='Aggressive HRP Training Pipeline')
    parser.add_argument('--full-training', action='store_true',
                       help='Run full aggressive training')

    parser.parse_args()

    trainer = AggressiveHRPTraining()
    trainer.run_aggressive_training()


if __name__ == "__main__":
    main()
