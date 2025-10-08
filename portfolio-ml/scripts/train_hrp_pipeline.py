"""
HRP Model Training Pipeline Execution.

This script implements comprehensive HRP training across multiple parameter configurations
with clustering validation, following Story 5.2 Task 1 requirements.
"""

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import silhouette_score

from src.data.processors.universe_builder import UniverseBuilder
from src.models.base.constraints import PortfolioConstraints
from src.models.hrp.clustering import ClusteringConfig, HRPClustering
from src.models.hrp.model import HRPConfig, HRPModel

warnings.filterwarnings("ignore", category=UserWarning)


class HRPTrainingPipeline:
    """
    Complete HRP model training pipeline with parameter validation and clustering metrics.

    This class implements comprehensive HRP training across multiple parameter configurations
    with clustering validation, following Google-style documentation standards. Supports
    extensive hyperparameter testing and validation across different linkage methods and
    correlation measures.

    Implements all subtasks from Story 5.2 Task 1:
    - Single-linkage hierarchical clustering on correlation distance matrices
    - Recursive bisection allocation with equal risk contribution
    - Parameter validation across multiple lookback periods and linkage methods
    - Clustering validation metrics and correlation distance matrix analysis

    Attributes:
        config_path: Path to HRP configuration YAML file
        base_config: Loaded configuration dictionary
        logger: Configured logging instance
        data_path: Path to production-ready datasets
        checkpoints_path: Path for model checkpoint storage
        training_results: Dictionary storing training results
        clustering_diagnostics: Dictionary storing clustering validation metrics

    Example:
        >>> pipeline = HRPTrainingPipeline("configs/models/hrp_default.yaml")
        >>> results = pipeline.run_full_pipeline()
        >>> print(f"Best configuration: {results['best_configuration']}")
    """

    def __init__(self, config_path: str = "configs/models/hrp_default.yaml"):
        """
        Initialize HRP training pipeline.

        Args:
            config_path: Path to HRP configuration file
        """
        self.config_path = config_path
        self.base_config = self._load_config()

        # Setup logging
        self._setup_logging()

        # Initialize data paths
        self.data_path = Path("data/final_new_pipeline")
        self.checkpoints_path = Path("data/models/checkpoints/hrp")
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)

        # Training results storage
        self.training_results: dict[str, Any] = {}
        self.clustering_diagnostics: dict[str, Any] = {}

    def _load_config(self) -> dict[str, Any]:
        """Load HRP configuration from YAML file."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_path = Path("logs/training/hrp")
        log_path.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / 'hrp_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> tuple[pd.DataFrame, UniverseBuilder]:
        """
        Load production-ready datasets from Story 5.1.

        Returns:
            Tuple of (returns_data, universe_calendar)
        """
        self.logger.info("Loading production datasets from Story 5.1...")

        # Load returns data
        returns_path = self.data_path / "returns_daily_final.parquet"
        returns_data = pd.read_parquet(returns_path)
        returns_data.index = pd.to_datetime(returns_data.index)

        # Load universe builder (from data/universe if available)
        try:
            from src.config.data import UniverseConfig
            universe_config = UniverseConfig(universe_type="midcap400")
            universe_builder = UniverseBuilder(universe_config)
        except Exception as e:
            self.logger.warning(f"Could not load universe builder: {e}")
            # Create simple universe from data columns
            universe_builder = None

        self.logger.info(f"Loaded returns data: {returns_data.shape}")
        self.logger.info(f"Date range: {returns_data.index.min()} to {returns_data.index.max()}")

        return returns_data, universe_builder

    def _generate_parameter_configurations(self) -> list[dict[str, Any]]:
        """
        Generate multiple parameter configurations for validation.

        Implements Subtask 1.3: Parameter validation across multiple lookback periods
        and linkage methods.

        Returns:
            List of parameter configuration dictionaries
        """
        configs = []

        # Lookback periods: 252 (1 year), 504 (2 years), 756 (3 years) days
        lookback_periods = [252, 504, 756]

        # Linkage methods to test
        linkage_methods = ["single", "complete", "average"]

        # Correlation methods
        correlation_methods = ["pearson", "spearman"]

        for lookback in lookback_periods:
            for linkage in linkage_methods:
                for corr_method in correlation_methods:
                    config = {
                        "lookback_days": lookback,
                        "linkage_method": linkage,
                        "correlation_method": corr_method,
                        "config_name": f"hrp_lb{lookback}_{linkage}_{corr_method}"
                    }
                    configs.append(config)

        self.logger.info(f"Generated {len(configs)} parameter configurations")
        return configs

    def validate_clustering(self, model: HRPModel, returns_data: pd.DataFrame,
                          config: dict[str, Any]) -> dict[str, Any]:
        """
        Validate clustering quality and generate correlation distance analysis.

        Implements Subtask 1.4: Clustering validation metrics and correlation
        distance matrix analysis.

        Args:
            model: Fitted HRP model
            returns_data: Historical returns data
            config: Model configuration

        Returns:
            Dictionary of clustering validation metrics
        """
        self.logger.info(f"Validating clustering for {config['config_name']}")

        # Get a representative date for analysis
        analysis_date = returns_data.index[-100]  # 100 days before end
        universe = returns_data.columns.tolist()[:100]  # Use first 100 assets

        try:
            # Get clustering diagnostics from model
            diagnostics = model.get_clustering_diagnostics(analysis_date, universe)

            if "error" in diagnostics:
                self.logger.warning(f"Clustering diagnostics failed: {diagnostics['error']}")
                return {"error": diagnostics["error"]}

            # Calculate lookback returns for additional metrics
            lookback_days = config["lookback_days"]
            start_date = analysis_date - pd.Timedelta(days=lookback_days)

            mask = (returns_data.index >= start_date) & (returns_data.index < analysis_date)
            lookback_returns = returns_data[mask]

            # Filter for available universe
            available_assets = [asset for asset in universe if asset in lookback_returns.columns]
            if len(available_assets) < 10:
                return {"error": "Insufficient assets for clustering validation"}

            filtered_returns = lookback_returns[available_assets[:50]]  # Limit to 50 assets

            # Build clustering components
            clustering_config = ClusteringConfig(
                linkage_method=config["linkage_method"],
                correlation_method=config["correlation_method"],
                min_observations=min(100, len(filtered_returns))
            )

            clustering_engine = HRPClustering(clustering_config)

            # Calculate correlation distance matrix
            distance_matrix = clustering_engine.build_correlation_distance(filtered_returns)

            # Perform hierarchical clustering
            linkage_matrix = clustering_engine.hierarchical_clustering(distance_matrix)

            # Calculate clustering quality metrics
            correlation_matrix = filtered_returns.corr()

            # Silhouette score (using distance matrix)
            n_assets = len(filtered_returns.columns)
            if n_assets > 3:
                # Create cluster labels from linkage matrix
                from scipy.cluster.hierarchy import fcluster
                cluster_labels = fcluster(linkage_matrix, t=min(10, n_assets//3), criterion='maxclust')

                # Calculate silhouette score
                if len(np.unique(cluster_labels)) > 1:
                    silhouette_avg = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
                else:
                    silhouette_avg = 0.0
            else:
                silhouette_avg = 0.0
                cluster_labels = np.ones(n_assets)

            # Calculate average correlation within and between clusters
            unique_clusters = np.unique(cluster_labels)
            within_cluster_corr = []
            between_cluster_corr = []

            for cluster_id in unique_clusters:
                cluster_assets = filtered_returns.columns[cluster_labels == cluster_id]
                if len(cluster_assets) > 1:
                    # Within cluster correlation
                    cluster_corr_matrix = correlation_matrix.loc[cluster_assets, cluster_assets]
                    within_corr = cluster_corr_matrix.values[np.triu_indices_from(cluster_corr_matrix, k=1)]
                    within_cluster_corr.extend(within_corr)

                    # Between cluster correlation
                    other_assets = filtered_returns.columns[cluster_labels != cluster_id]
                    if len(other_assets) > 0:
                        between_corr_matrix = correlation_matrix.loc[cluster_assets, other_assets]
                        between_cluster_corr.extend(between_corr_matrix.values.flatten())

            validation_metrics = {
                "config_name": config["config_name"],
                "n_assets": n_assets,
                "n_clusters": len(unique_clusters),
                "silhouette_score": float(silhouette_avg),
                "avg_within_cluster_correlation": float(np.mean(within_cluster_corr)) if within_cluster_corr else 0.0,
                "avg_between_cluster_correlation": float(np.mean(between_cluster_corr)) if between_cluster_corr else 0.0,
                "distance_matrix_stats": {
                    "mean_distance": float(np.mean(distance_matrix)),
                    "std_distance": float(np.std(distance_matrix)),
                    "min_distance": float(np.min(distance_matrix[distance_matrix > 0])),  # Exclude diagonal
                    "max_distance": float(np.max(distance_matrix))
                },
                "correlation_matrix_stats": {
                    "mean_correlation": float(correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)].mean()),
                    "std_correlation": float(correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)].std())
                },
                "linkage_method": config["linkage_method"],
                "correlation_method": config["correlation_method"],
                "lookback_days": config["lookback_days"]
            }

            validation_metrics.update(diagnostics)

            self.logger.info(f"Clustering validation completed for {config['config_name']}")
            return validation_metrics

        except Exception as e:
            self.logger.error(f"Clustering validation failed for {config['config_name']}: {str(e)}")
            return {"error": str(e), "config_name": config["config_name"]}

    def train_single_configuration(self, returns_data: pd.DataFrame,
                                 config: dict[str, Any]) -> tuple[HRPModel, dict[str, Any]]:
        """
        Train HRP model with single parameter configuration.

        Args:
            returns_data: Historical returns data
            config: Model configuration

        Returns:
            Tuple of (fitted_model, training_metrics)
        """
        self.logger.info(f"Training HRP model: {config['config_name']}")

        # Create HRP configuration
        clustering_config = ClusteringConfig(
            linkage_method=config["linkage_method"],
            correlation_method=config["correlation_method"],
            min_observations=min(252, len(returns_data) // 4)
        )

        hrp_config = HRPConfig(
            lookback_days=config["lookback_days"],
            clustering_config=clustering_config,
            min_observations=clustering_config.min_observations,
            correlation_method=config["correlation_method"]
        )

        # Create portfolio constraints from base config
        constraints = PortfolioConstraints(
            long_only=self.base_config["constraints"]["long_only"],
            max_position_weight=self.base_config["constraints"]["max_position_weight"],
            top_k_positions=self.base_config["constraints"]["top_k_positions"],
            max_monthly_turnover=self.base_config["constraints"]["max_monthly_turnover"],
            min_weight_threshold=self.base_config["constraints"]["min_weight_threshold"]
        )

        # Initialize model
        model = HRPModel(constraints=constraints, hrp_config=hrp_config)

        # Define training period (use last 3 years of data)
        end_date = returns_data.index[-1]
        start_date = end_date - pd.Timedelta(days=1095)  # 3 years

        # Filter data for training period first
        time_mask = (returns_data.index >= start_date) & (returns_data.index <= end_date)
        period_returns = returns_data[time_mask]

        # Get universe for training - filter out assets with insufficient data
        # Calculate data coverage for each asset in the training period
        data_coverage = period_returns.count() / len(period_returns)
        valid_assets = data_coverage[data_coverage >= 0.5].index.tolist()  # At least 50% coverage

        # Also filter out zero-variance assets
        asset_variances = period_returns[valid_assets].var()
        non_zero_var_assets = asset_variances[asset_variances > 1e-8].index.tolist()

        # Get top 100 assets with good coverage and non-zero variance
        universe = non_zero_var_assets[:100]

        self.logger.info(f"Filtered universe from {len(returns_data.columns)} to {len(universe)} valid assets")

        # Fit the model
        try:
            model.fit(returns_data, universe, (start_date, end_date))

            # Test prediction capability
            pred_date = end_date - pd.Timedelta(days=30)
            weights = model.predict_weights(pred_date, universe[:50])  # Test with subset

            training_metrics = {
                "config_name": config["config_name"],
                "training_start": start_date.strftime("%Y-%m-%d"),
                "training_end": end_date.strftime("%Y-%m-%d"),
                "universe_size": len(universe),
                "fitted_successfully": True,
                "prediction_universe_size": len(weights),
                "weights_sum": float(weights.sum()),
                "max_weight": float(weights.max()),
                "min_weight": float(weights.min()),
                "non_zero_positions": int((weights > 1e-6).sum()),
                "model_info": model.get_model_info()
            }

            self.logger.info(f"Successfully trained {config['config_name']}")

        except Exception as e:
            self.logger.error(f"Training failed for {config['config_name']}: {str(e)}")
            model = None
            training_metrics = {
                "config_name": config["config_name"],
                "fitted_successfully": False,
                "error": str(e)
            }

        return model, training_metrics

    def run_parameter_validation(self, returns_data: pd.DataFrame) -> dict[str, Any]:
        """
        Execute comprehensive parameter validation across all configurations.

        Implements complete Task 1 with all subtasks.

        Args:
            returns_data: Historical returns data

        Returns:
            Dictionary containing all validation results
        """
        self.logger.info("Starting comprehensive HRP parameter validation")

        configs = self._generate_parameter_configurations()

        validation_results = {
            "training_results": {},
            "clustering_diagnostics": {},
            "parameter_comparison": {},
            "best_configuration": None
        }

        best_score = -np.inf
        best_config_name = None

        for config in configs:
            self.logger.info(f"Processing configuration: {config['config_name']}")

            # Train model with this configuration
            model, training_metrics = self.train_single_configuration(returns_data, config)
            validation_results["training_results"][config["config_name"]] = training_metrics

            if model is not None and training_metrics["fitted_successfully"]:
                # Validate clustering
                clustering_metrics = self.validate_clustering(model, returns_data, config)
                validation_results["clustering_diagnostics"][config["config_name"]] = clustering_metrics

                # Score configuration based on clustering quality and training success
                if "error" not in clustering_metrics:
                    # Composite score: silhouette score + difference between within/between cluster correlation
                    within_corr = clustering_metrics.get("avg_within_cluster_correlation", 0)
                    between_corr = clustering_metrics.get("avg_between_cluster_correlation", 0)
                    silhouette = clustering_metrics.get("silhouette_score", 0)

                    score = silhouette + (within_corr - between_corr)  # Higher is better

                    if score > best_score:
                        best_score = score
                        best_config_name = config["config_name"]

                # Save model checkpoint
                checkpoint_path = self.checkpoints_path / f"{config['config_name']}.pkl"
                try:
                    import pickle
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump({
                            'model': model,
                            'config': config,
                            'training_metrics': training_metrics,
                            'clustering_metrics': clustering_metrics
                        }, f)
                    self.logger.info(f"Saved checkpoint: {checkpoint_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save checkpoint: {e}")

        # Determine best configuration
        validation_results["best_configuration"] = best_config_name
        validation_results["best_score"] = best_score

        # Generate parameter comparison summary
        comparison_summary = self._generate_parameter_comparison(validation_results)
        validation_results["parameter_comparison"] = comparison_summary

        self.logger.info(f"Parameter validation completed. Best configuration: {best_config_name}")

        return validation_results

    def _generate_parameter_comparison(self, validation_results: dict[str, Any]) -> dict[str, Any]:
        """Generate parameter comparison analysis."""
        training_results = validation_results["training_results"]
        clustering_results = validation_results["clustering_diagnostics"]

        # Analyze by lookback period
        lookback_analysis = {}
        for lookback in [252, 504, 756]:
            configs = [k for k in training_results.keys() if f"_lb{lookback}_" in k]
            successful_configs = [
                k for k in configs
                if training_results[k]["fitted_successfully"] and k in clustering_results
            ]

            if successful_configs:
                avg_silhouette = np.mean([
                    clustering_results[k].get("silhouette_score", 0)
                    for k in successful_configs
                    if "error" not in clustering_results[k]
                ])

                lookback_analysis[f"lookback_{lookback}"] = {
                    "n_successful_configs": len(successful_configs),
                    "avg_silhouette_score": float(avg_silhouette),
                    "configs": successful_configs
                }

        # Analyze by linkage method
        linkage_analysis = {}
        for linkage in ["single", "complete", "average"]:
            configs = [k for k in training_results.keys() if f"_{linkage}_" in k]
            successful_configs = [
                k for k in configs
                if training_results[k]["fitted_successfully"] and k in clustering_results
            ]

            if successful_configs:
                avg_silhouette = np.mean([
                    clustering_results[k].get("silhouette_score", 0)
                    for k in successful_configs
                    if "error" not in clustering_results[k]
                ])

                linkage_analysis[f"linkage_{linkage}"] = {
                    "n_successful_configs": len(successful_configs),
                    "avg_silhouette_score": float(avg_silhouette),
                    "configs": successful_configs
                }

        return {
            "lookback_analysis": lookback_analysis,
            "linkage_analysis": linkage_analysis,
            "total_configs_tested": len(training_results),
            "successful_configs": len([
                k for k in training_results.keys()
                if training_results[k]["fitted_successfully"]
            ])
        }

    def save_results(self, results: dict[str, Any]) -> None:
        """Save training results to disk."""
        results_path = Path("logs/training/hrp/hrp_validation_results.yaml")
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)

        self.logger.info(f"Results saved to: {results_path}")

    def run_full_pipeline(self) -> dict[str, Any]:
        """
        Execute complete HRP training pipeline.

        Implements all Story 5.2 Task 1 requirements:
        - Subtask 1.1: HRP clustering algorithm with single-linkage hierarchical clustering
        - Subtask 1.2: Recursive bisection allocation logic with equal risk contribution
        - Subtask 1.3: Parameter validation across multiple lookback periods and linkage methods
        - Subtask 1.4: Clustering validation metrics and correlation distance analysis

        Returns:
            Complete training and validation results
        """
        self.logger.info("Starting complete HRP training pipeline execution")

        # Load data
        returns_data, universe_builder = self.load_data()

        # Execute parameter validation (includes all subtasks)
        results = self.run_parameter_validation(returns_data)

        # Save results
        self.save_results(results)

        self.logger.info("HRP training pipeline completed successfully")

        return results


if __name__ == "__main__":
    """Execute HRP training pipeline."""

    # Initialize and run pipeline
    pipeline = HRPTrainingPipeline()

    # Execute complete training pipeline
    results = pipeline.run_full_pipeline()


    if results["best_configuration"]:
        pass


    for _lookback, _data in results["parameter_comparison"]["lookback_analysis"].items():
        pass

    for _linkage, _data in results["parameter_comparison"]["linkage_analysis"].items():
        pass

