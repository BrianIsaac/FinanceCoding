#!/usr/bin/env python3
"""
Improved HRP Training Pipeline with Better Clustering Parameters
Fixes the terrible silhouette scores by using more appropriate parameters
"""

import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore')

import sys

sys.path.append(str(Path(__file__).parent.parent))

import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from src.models.base.constraints import PortfolioConstraints
from src.models.hrp.model import HRPModel


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training/hrp/hrp_fixed_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_production_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load production datasets from Story 5.1"""
    logger = logging.getLogger(__name__)

    # Load production datasets
    returns_path = "data/final_new_pipeline/returns_daily_final.parquet"
    prices_path = "data/final_new_pipeline/prices_final.parquet"
    volume_path = "data/final_new_pipeline/volume_final.parquet"

    returns_data = pd.read_parquet(returns_path)
    prices_data = pd.read_parquet(prices_path)
    volume_data = pd.read_parquet(volume_path)

    logger.info(f"Loaded returns data: {returns_data.shape}")
    logger.info(f"Loaded prices data: {prices_data.shape}")
    logger.info(f"Loaded volume data: {volume_data.shape}")
    logger.info(f"Date range: {returns_data.index.min()} to {returns_data.index.max()}")

    return returns_data, prices_data, volume_data


def select_high_quality_universe(returns_data: pd.DataFrame, min_data_coverage: float = 0.80) -> list[str]:
    """Select high-quality assets with better coverage threshold"""
    logger = logging.getLogger(__name__)

    # Calculate data coverage for each asset
    coverage = returns_data.count() / len(returns_data)
    high_quality_assets = coverage[coverage >= min_data_coverage].index.tolist()

    # Additionally filter by liquidity (assets with consistent trading)
    recent_data = returns_data.tail(252)  # Last year
    recent_coverage = recent_data.count() / len(recent_data)
    liquid_assets = recent_coverage[recent_coverage >= 0.90].index.tolist()

    # Intersection of high-quality and liquid assets
    final_universe = list(set(high_quality_assets) & set(liquid_assets))

    logger.info(f"Total assets: {len(returns_data.columns)}")
    logger.info(f"High quality assets (>={min_data_coverage:.0%} coverage): {len(high_quality_assets)}")
    logger.info(f"Liquid assets (>=90% recent coverage): {len(liquid_assets)}")
    logger.info(f"Final universe: {len(final_universe)} assets")

    return final_universe


def generate_improved_hrp_configs() -> list[dict[str, Any]]:
    """Generate improved HRP configurations with better parameters"""
    configurations = []

    # Better parameter ranges
    lookback_periods = [126, 252, 504]  # 6 months, 1 year, 2 years
    linkage_methods = ['ward', 'average', 'complete']  # Ward often works better
    correlation_methods = ['pearson', 'spearman']
    min_clusters = [15, 20, 25]  # Force more granular clustering

    for lookback in lookback_periods:
        for linkage in linkage_methods:
            for corr_method in correlation_methods:
                for min_clust in min_clusters:
                    config = {
                        'name': f'hrp_lb{lookback}_{linkage}_{corr_method}_mc{min_clust}',
                        'lookback_days': lookback,
                        'linkage_method': linkage,
                        'correlation_method': corr_method,
                        'min_clusters': min_clust,
                        'min_observations': max(60, lookback // 4),  # More reasonable minimum
                        'max_clusters': 50,
                        'cluster_size_balance': True
                    }
                    configurations.append(config)

    return configurations


def validate_clustering_quality(
    returns_data: pd.DataFrame,
    config: dict[str, Any],
    train_end_date: str
) -> dict[str, Any]:
    """Validate clustering quality with improved metrics"""
    logger = logging.getLogger(__name__)

    try:
        # Get training period data
        train_data = returns_data.loc[:train_end_date].tail(config['lookback_days'])

        if len(train_data) < config['min_observations']:
            return {'error': f'Insufficient observations: {len(train_data)} < {config["min_observations"]}'}

        # Calculate correlation matrix
        if config['correlation_method'] == 'pearson':
            corr_matrix = train_data.corr()
        else:  # spearman
            corr_matrix = train_data.corr(method='spearman')

        # Remove assets with too many NaN correlations
        valid_mask = corr_matrix.count() >= len(corr_matrix) * 0.8  # 80% valid correlations
        corr_matrix = corr_matrix.loc[valid_mask, valid_mask]

        if corr_matrix.empty or len(corr_matrix) < 20:
            return {'error': f'Insufficient valid correlations: {len(corr_matrix)} assets'}

        # Fill remaining NaN values with mean correlation
        mean_corr = corr_matrix.values[~np.isnan(corr_matrix.values)].mean()
        corr_matrix = corr_matrix.fillna(mean_corr)

        # Convert correlation to distance
        distance_matrix = 1 - np.abs(corr_matrix)  # Use absolute correlation
        distance_matrix = np.clip(distance_matrix, 0, 2)  # Ensure valid distances

        # Perform hierarchical clustering with improved method
        condensed_distances = squareform(distance_matrix)

        # Use ward linkage for better cluster separation
        if config['linkage_method'] == 'ward':
            # Ward requires Euclidean distance, so we use a different approach
            linkage_matrix = sch.linkage(condensed_distances, method='ward')
        else:
            linkage_matrix = sch.linkage(condensed_distances, method=config['linkage_method'])

        # Determine optimal number of clusters
        n_assets = len(corr_matrix)
        optimal_clusters = min(
            max(config['min_clusters'], n_assets // 8),  # At least min_clusters, reasonable ratio
            config.get('max_clusters', min(50, n_assets // 2))  # Not more than half the assets
        )

        # Get cluster labels
        cluster_labels = sch.fcluster(linkage_matrix, optimal_clusters, criterion='maxclust')

        # Calculate silhouette score
        silhouette = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')

        # Calculate cluster statistics
        unique_clusters = np.unique(cluster_labels)
        cluster_sizes = [np.sum(cluster_labels == cluster) for cluster in unique_clusters]

        # Within-cluster correlation analysis
        within_cluster_corrs = []
        between_cluster_corrs = []

        for cluster in unique_clusters:
            cluster_mask = cluster_labels == cluster
            cluster_assets = corr_matrix.index[cluster_mask]

            if len(cluster_assets) > 1:
                cluster_corr_matrix = corr_matrix.loc[cluster_assets, cluster_assets]
                # Get upper triangle correlations (excluding diagonal)
                upper_indices = np.triu_indices_from(cluster_corr_matrix, k=1)
                within_correlations = cluster_corr_matrix.values[upper_indices]
                within_cluster_corrs.extend(within_correlations[~np.isnan(within_correlations)])

                # Between-cluster correlations
                other_assets = corr_matrix.index[~cluster_mask]
                if len(other_assets) > 0:
                    between_corr_matrix = corr_matrix.loc[cluster_assets, other_assets]
                    between_correlations = between_corr_matrix.values.flatten()
                    between_cluster_corrs.extend(between_correlations[~np.isnan(between_correlations)])

        # Calculate metrics
        avg_within_cluster_corr = np.mean(within_cluster_corrs) if within_cluster_corrs else 0
        avg_between_cluster_corr = np.mean(between_cluster_corrs) if between_cluster_corrs else 0
        cluster_separation = avg_within_cluster_corr - avg_between_cluster_corr

        results = {
            'config_name': config['name'],
            'n_assets': n_assets,
            'n_observations': len(train_data),
            'n_clusters': optimal_clusters,
            'silhouette_score': silhouette,
            'avg_within_cluster_correlation': avg_within_cluster_corr,
            'avg_between_cluster_correlation': avg_between_cluster_corr,
            'cluster_separation': cluster_separation,
            'cluster_sizes': cluster_sizes,
            'min_cluster_size': min(cluster_sizes),
            'max_cluster_size': max(cluster_sizes),
            'cluster_size_std': np.std(cluster_sizes),
            'assets': corr_matrix.index.tolist(),
            'linkage_method': config['linkage_method'],
            'lookback_days': config['lookback_days'],
            'correlation_method': config['correlation_method']
        }

        logger.info(f"Clustering validation for {config['name']}: "
                   f"Silhouette={silhouette:.4f}, Clusters={optimal_clusters}, "
                   f"Separation={cluster_separation:.4f}")

        return results

    except Exception as e:
        logger.error(f"Clustering validation failed for {config['name']}: {str(e)}")
        return {'error': str(e)}


def train_hrp_model(
    returns_data: pd.DataFrame,
    config: dict[str, Any],
    train_start: str,
    train_end: str,
    universe: list[str]
) -> dict[str, Any]:
    """Train HRP model with improved configuration"""
    logger = logging.getLogger(__name__)

    try:
        # Setup constraints
        constraints = PortfolioConstraints(
            long_only=True,
            max_position_weight=0.08,  # Slightly lower max weight
            min_weight_threshold=0.005,  # Lower minimum weight
            top_k_positions=min(100, len(universe) // 2),  # More positions
            max_monthly_turnover=0.25
        )

        # Create HRP model
        hrp_model = HRPModel(
            lookback_days=config['lookback_days'],
            min_observations=config['min_observations'],
            correlation_method=config['correlation_method'],
            linkage_method=config['linkage_method'],
            constraints=constraints
        )

        # Filter data to universe and training period
        filtered_data = returns_data[universe].loc[train_start:train_end]

        # Further filter to assets with sufficient data
        coverage = filtered_data.count() / len(filtered_data)
        valid_assets = coverage[coverage >= 0.7].index.tolist()  # 70% coverage minimum

        if len(valid_assets) < 50:
            return {
                'config_name': config['name'],
                'fitted_successfully': False,
                'error': f'Insufficient valid assets: {len(valid_assets)}'
            }

        filtered_data = filtered_data[valid_assets]
        logger.info(f"Training {config['name']} with {len(valid_assets)} assets")

        # Fit model
        hrp_model.fit(filtered_data)

        # Generate predictions
        latest_data = filtered_data.tail(1)
        weights = hrp_model.predict_weights(latest_data)

        # Calculate metrics
        non_zero_positions = np.sum(weights > 0)
        weights_sum = np.sum(weights)
        max_weight = np.max(weights)
        min_weight = np.min(weights[weights > 0]) if non_zero_positions > 0 else 0

        results = {
            'config_name': config['name'],
            'fitted_successfully': True,
            'universe_size': len(valid_assets),
            'training_start': train_start,
            'training_end': train_end,
            'non_zero_positions': non_zero_positions,
            'weights_sum': weights_sum,
            'max_weight': max_weight,
            'min_weight': min_weight,
            'prediction_universe_size': len(weights),
            'model_info': hrp_model.get_model_info()
        }

        # Save model checkpoint
        checkpoint_dir = Path("data/models/checkpoints/hrp_fixed")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{config['name']}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'model': hrp_model,
                'config': config,
                'results': results,
                'weights': weights,
                'timestamp': datetime.now().isoformat()
            }, f)

        logger.info(f"Successfully trained and saved {config['name']}")
        return results

    except Exception as e:
        logger.error(f"Training failed for {config['name']}: {str(e)}")
        return {
            'config_name': config['name'],
            'fitted_successfully': False,
            'error': str(e)
        }


def main():
    """Main execution function"""
    logger = setup_logging()
    logger.info("Starting improved HRP training pipeline execution")

    # Create directories
    Path("logs/training/hrp").mkdir(parents=True, exist_ok=True)
    Path("data/models/checkpoints/hrp_fixed").mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading production datasets from Story 5.1...")
    returns_data, prices_data, volume_data = load_production_data()

    # Select high-quality universe
    logger.info("Selecting high-quality asset universe...")
    universe = select_high_quality_universe(returns_data, min_data_coverage=0.80)

    if len(universe) < 100:
        logger.error(f"Universe too small: {len(universe)} assets")
        return

    # Generate improved configurations
    configs = generate_improved_hrp_configs()
    logger.info(f"Generated {len(configs)} improved HRP configurations")

    # Training parameters
    train_end_date = "2024-12-30"
    train_start_date = "2021-12-31"

    # Results storage
    training_results = {}
    clustering_diagnostics = {}
    successful_configs = 0

    # Train each configuration
    for i, config in enumerate(configs, 1):
        logger.info(f"Processing configuration {i}/{len(configs)}: {config['name']}")

        try:
            # Validate clustering first
            logger.info(f"Validating clustering for {config['name']}")
            clustering_result = validate_clustering_quality(
                returns_data[universe], config, train_end_date
            )
            clustering_diagnostics[config['name']] = clustering_result

            if 'error' in clustering_result:
                logger.warning(f"Clustering validation failed for {config['name']}: {clustering_result['error']}")
                continue

            # Train model if clustering looks good
            if clustering_result.get('silhouette_score', 0) > 0.15:  # Minimum reasonable threshold
                logger.info(f"Training HRP model: {config['name']}")
                training_result = train_hrp_model(
                    returns_data, config, train_start_date, train_end_date, universe
                )
                training_results[config['name']] = training_result

                if training_result.get('fitted_successfully', False):
                    successful_configs += 1
                    logger.info(f"Successfully trained {config['name']}")
                else:
                    logger.warning(f"Training failed for {config['name']}")
            else:
                logger.warning(f"Skipping training for {config['name']} due to poor clustering (silhouette={clustering_result.get('silhouette_score', 0):.4f})")

        except Exception as e:
            logger.error(f"Failed to process configuration {config['name']}: {str(e)}")
            continue

    # Find best configuration
    best_config = None
    best_score = -np.inf

    for config_name, diagnostic in clustering_diagnostics.items():
        if 'silhouette_score' in diagnostic:
            silhouette = diagnostic['silhouette_score']
            separation = diagnostic.get('cluster_separation', 0)
            combined_score = silhouette + 0.5 * separation  # Weight silhouette more heavily

            if combined_score > best_score:
                best_score = combined_score
                best_config = config_name

    # Analyze results by parameter
    lookback_analysis = {}
    linkage_analysis = {}

    for config_name, diagnostic in clustering_diagnostics.items():
        if 'silhouette_score' in diagnostic:
            # Extract parameters from config name
            parts = config_name.split('_')
            lookback = f"lookback_{parts[1][2:]}"  # Remove 'lb' prefix
            linkage = f"linkage_{parts[2]}"

            # Lookback analysis
            if lookback not in lookback_analysis:
                lookback_analysis[lookback] = {'configs': [], 'silhouette_scores': []}
            lookback_analysis[lookback]['configs'].append(config_name)
            lookback_analysis[lookback]['silhouette_scores'].append(diagnostic['silhouette_score'])

            # Linkage analysis
            if linkage not in linkage_analysis:
                linkage_analysis[linkage] = {'configs': [], 'silhouette_scores': []}
            linkage_analysis[linkage]['configs'].append(config_name)
            linkage_analysis[linkage]['silhouette_scores'].append(diagnostic['silhouette_score'])

    # Calculate averages
    for lookback, data in lookback_analysis.items():
        data['avg_silhouette_score'] = np.mean(data['silhouette_scores'])
        data['n_successful_configs'] = len(data['configs'])

    for linkage, data in linkage_analysis.items():
        data['avg_silhouette_score'] = np.mean(data['silhouette_scores'])
        data['n_successful_configs'] = len(data['configs'])

    # Save comprehensive results
    results_summary = {
        'best_configuration': best_config,
        'best_score': best_score,
        'successful_configs': successful_configs,
        'total_configs_tested': len(configs),
        'universe_size': len(universe),
        'training_results': training_results,
        'clustering_diagnostics': clustering_diagnostics,
        'parameter_comparison': {
            'lookback_analysis': lookback_analysis,
            'linkage_analysis': linkage_analysis
        }
    }

    results_path = "logs/training/hrp/hrp_fixed_validation_results.yaml"
    with open(results_path, 'w') as f:
        yaml.dump(results_summary, f, default_flow_style=False)

    logger.info(f"Results saved to: {results_path}")

    # Print summary

    if best_config:
        clustering_diagnostics[best_config]

    for lookback, data in sorted(lookback_analysis.items()):
        pass

    for linkage, data in sorted(linkage_analysis.items()):
        pass



if __name__ == "__main__":
    main()
