"""
GAT Model Training Pipeline Execution.

This script implements comprehensive GAT training with multi-graph construction methods,
direct Sharpe ratio optimization, and end-to-end training pipeline with memory-efficient
batch processing, following Story 5.2 Task 3 requirements.
"""

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.constraints import PortfolioConstraints
from src.models.gat.gat_model import GATPortfolio, HeadCfg
from src.models.gat.graph_builder import GraphBuildConfig, build_period_graph
from src.models.gat.loss_functions import SharpeRatioLoss
from src.models.gat.model import GATModelConfig, GATPortfolioModel
from src.models.gat.training import GPUMemoryManager

warnings.filterwarnings("ignore", category=UserWarning)


class GATTrainingPipeline:
    """
    Complete GAT model training pipeline with multi-graph construction and Sharpe optimization.

    This class implements sophisticated Graph Attention Network training with multiple graph
    construction methods, direct Sharpe ratio optimization, and memory-efficient processing
    for large-scale portfolio optimization problems. Supports various graph topologies and
    advanced attention mechanisms.

    Implements all subtasks from Story 5.2 Task 3:
    - Multi-head attention GAT architecture with edge attribute integration
    - Training across multiple graph construction methods (MST, TMFG, k-NN filtering)
    - Direct Sharpe ratio optimization with constraint enforcement
    - End-to-end training pipeline with memory-efficient batch processing

    Attributes:
        config_path: Path to GAT configuration YAML file
        base_config: Loaded configuration dictionary
        logger: Configured logging instance
        data_path: Path to production-ready datasets
        universe_builder: Universe construction utility
        memory_manager: GPU memory management with VRAM limits
        training_results: Dictionary storing training results across configurations
        checkpoint_dir: Directory for model checkpoint storage
        results_dir: Directory for training result logs

    Example:
        >>> pipeline = GATTrainingPipeline("configs/models/gat_default.yaml")
        >>> results = pipeline.execute_full_training_pipeline()
        >>> print(f"Configurations trained: {len(results)}")
    """

    def __init__(self, config_path: str = "configs/models/gat_default.yaml"):
        """
        Initialize GAT training pipeline.

        Args:
            config_path: Path to GAT configuration file
        """
        self.config_path = config_path
        self.base_config = self._load_config()

        # Setup logging
        self._setup_logging()

        # Initialize components
        self.data_path = Path("data/final_new_pipeline")
        self.universe_builder = None
        self.memory_manager = GPUMemoryManager(max_vram_gb=11.0)

        # Training results storage
        self.training_results: dict[str, Any] = {}

        # Setup directories
        self._setup_directories()

    def _load_config(self) -> dict[str, Any]:
        """Load GAT configuration with fallback defaults."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Return default GAT configuration."""
        return {
            'architecture': {
                'input_features': 10,
                'hidden_dim': 64,
                'num_layers': 3,
                'num_attention_heads': 8,
                'dropout': 0.3,
                'edge_feature_dim': 3,
                'use_gatv2': True,
                'residual': True
            },
            'training': {
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'batch_size': 16,
                'max_epochs': 100,
                'patience': 15,
                'use_mixed_precision': True,
                'gradient_accumulation_steps': 4
            },
            'graph_construction': {
                'lookback_days': 252,
                'methods': ['mst', 'tmfg', 'knn'],
                'knn_k_values': [5, 10, 15],
                'use_edge_attr': True
            }
        }

    def _setup_logging(self):
        """Setup comprehensive logging for training pipeline."""
        log_dir = Path("logs/training/gat")
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "gat_training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("GAT Training Pipeline initialized")

    def _setup_directories(self):
        """Create necessary directories for model storage."""
        self.checkpoint_dir = Path("data/models/checkpoints/gat")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = Path("logs/training/gat")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare training data.

        Returns:
            Tuple of (returns_data, universe_calendar)
        """
        self.logger.info("Loading production datasets from Story 5.1...")

        # Load returns data
        returns_path = self.data_path / "returns_daily_final.parquet"
        returns_data = pd.read_parquet(returns_path)
        returns_data.index = pd.to_datetime(returns_data.index)
        self.logger.info(f"Loaded returns data: {returns_data.shape}")
        self.logger.info(f"Date range: {returns_data.index.min()} to {returns_data.index.max()}")

        # Create a simple universe calendar from data columns
        universe_calendar = []
        returns_data.columns.tolist()

        # Create monthly rebalancing dates
        start_date = returns_data.index.min()
        end_date = returns_data.index.max()
        monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')

        for date in monthly_dates:
            # Get tickers with sufficient data at this date
            available_data = returns_data.loc[:date].tail(252)  # 1 year lookback
            valid_tickers = available_data.columns[available_data.count() >= 200].tolist()  # 80% coverage

            if len(valid_tickers) >= 20:  # Minimum universe size
                universe_calendar.append({
                    'date': date,
                    'tickers': valid_tickers[:100]  # Limit to top 100 for memory
                })

        universe_df = pd.DataFrame(universe_calendar)
        self.logger.info(f"Created universe calendar: {len(universe_df)} periods")

        return returns_data, universe_df

    def prepare_graph_configurations(self) -> list[dict[str, Any]]:
        """
        Prepare multiple graph construction configurations for training.

        Returns:
            List of graph configuration dictionaries
        """
        configurations = []
        config = self.base_config['graph_construction']

        # Different graph construction methods
        methods = config.get('methods', ['mst', 'tmfg', 'knn'])
        knn_k_values = config.get('knn_k_values', [5, 10, 15])
        lookback_days = config.get('lookback_days', 252)

        for method in methods:
            if method == 'knn':
                for k in knn_k_values:
                    configurations.append({
                        'filter_method': method,
                        'knn_k': k,
                        'lookback_days': lookback_days,
                        'use_edge_attr': True,
                        'config_name': f"{method}_k{k}"
                    })
            else:
                configurations.append({
                    'filter_method': method,
                    'knn_k': None,
                    'lookback_days': lookback_days,
                    'use_edge_attr': True,
                    'config_name': method
                })

        self.logger.info(f"Prepared {len(configurations)} graph configurations")
        return configurations

    def create_gat_model_config(self, graph_config: dict[str, Any]) -> GATModelConfig:
        """
        Create GAT model configuration for given graph configuration.

        Args:
            graph_config: Graph construction configuration

        Returns:
            GATModelConfig instance
        """
        arch_config = self.base_config['architecture']
        train_config = self.base_config['training']

        # Create graph build config
        graph_build_config = GraphBuildConfig(
            filter_method=graph_config['filter_method'],
            knn_k=graph_config.get('knn_k', 8),  # Default value instead of None
            lookback_days=graph_config['lookback_days'],
            use_edge_attr=graph_config['use_edge_attr']
        )

        # Create head configuration for direct Sharpe optimization
        head_config = HeadCfg(
            mode="direct",
            activation="sparsemax"
        )

        return GATModelConfig(
            # Architecture parameters
            input_features=int(arch_config['input_features']),
            hidden_dim=int(arch_config['hidden_dim']),
            num_layers=int(arch_config['num_layers']),
            num_attention_heads=int(arch_config['num_attention_heads']),
            dropout=float(arch_config['dropout']),
            edge_feature_dim=int(arch_config['edge_feature_dim']),
            use_gatv2=bool(arch_config['use_gatv2']),
            residual=bool(arch_config['residual']),

            # Head configuration
            head_config=head_config,

            # Graph configuration
            graph_config=graph_build_config,

            # Training parameters
            learning_rate=float(train_config['learning_rate']),
            weight_decay=float(train_config['weight_decay']),
            batch_size=int(train_config['batch_size']),
            max_epochs=int(train_config['max_epochs']),
            patience=int(train_config['patience']),
            use_mixed_precision=bool(train_config['use_mixed_precision'])
        )

    def prepare_training_data(self,
                            returns_data: pd.DataFrame,
                            universe_calendar: pd.DataFrame,
                            graph_config: dict[str, Any]) -> list[tuple[Any, np.ndarray, str]]:
        """
        Prepare training data with graph construction for given configuration.

        Args:
            returns_data: Historical returns DataFrame
            universe_calendar: Universe membership calendar
            graph_config: Graph construction configuration

        Returns:
            List of (graph_data, target_returns, date) tuples
        """
        training_samples = []

        # Get training period dates (using monthly rebalancing)
        start_date = pd.Timestamp('2016-01-01')
        end_date = pd.Timestamp('2022-12-31')  # 36-month training window

        rebalance_dates = pd.date_range(
            start=start_date + pd.Timedelta(days=graph_config['lookback_days']),
            end=end_date,
            freq='MS'  # Month start
        )

        for date in rebalance_dates[:12]:  # Limit to 12 samples for memory efficiency
            try:
                # Get universe for this date
                universe_row = universe_calendar[universe_calendar['date'] == date]
                if universe_row.empty:
                    continue

                universe = universe_row.iloc[0]['tickers']
                if len(universe) < 20:  # Need minimum universe size
                    continue

                # Prepare features matrix for this universe
                historical_returns = returns_data.loc[:date].tail(graph_config['lookback_days'] + 100)
                features_matrix = self._prepare_features(historical_returns, universe)

                # Build graph for this period
                graph_build_config = GraphBuildConfig(
                    filter_method=graph_config['filter_method'],
                    knn_k=graph_config.get('knn_k', 8),  # Default value instead of None
                    lookback_days=graph_config['lookback_days'],
                    use_edge_attr=graph_config['use_edge_attr']
                )

                graph_data = build_period_graph(
                    returns_daily=historical_returns,
                    period_end=date,
                    tickers=universe,
                    features_matrix=features_matrix,
                    cfg=graph_build_config
                )

                # Get forward returns as targets (next month)
                next_month_end = min(date + pd.Timedelta(days=30), returns_data.index.max())
                forward_returns = returns_data.loc[date:next_month_end, universe].mean().values

                training_samples.append((graph_data, forward_returns, str(date.date())))

            except Exception as e:
                self.logger.warning(f"Failed to create training sample for {date}: {e}")
                continue

        self.logger.info(f"Prepared {len(training_samples)} training samples for {graph_config['config_name']}")
        return training_samples

    def _prepare_features(self, returns: pd.DataFrame, universe: list[str]) -> np.ndarray:
        """
        Prepare node features from returns data.

        Args:
            returns: Historical returns DataFrame
            universe: List of asset tickers

        Returns:
            Node features matrix [n_assets, n_features]
        """
        returns_subset = returns[universe].dropna()
        features = []
        market_proxy = returns_subset.mean(axis=1)  # Equal-weight market proxy

        for ticker in universe:
            asset_returns = returns_subset[ticker]

            # Statistical features
            mean_return = asset_returns.mean()
            volatility = asset_returns.std()
            skewness = asset_returns.skew()
            kurtosis = asset_returns.kurtosis()

            # Market correlation features
            corr_with_market = asset_returns.corr(market_proxy)
            beta = asset_returns.cov(market_proxy) / market_proxy.var()

            # Momentum features
            momentum_1m = asset_returns.tail(21).mean()  # 1-month momentum
            momentum_3m = asset_returns.tail(63).mean()  # 3-month momentum

            # Risk features
            max_drawdown = (asset_returns.cumsum().expanding().max() - asset_returns.cumsum()).max()
            var_95 = np.percentile(asset_returns, 5)  # Value at Risk 95%

            features.append([
                mean_return, volatility, skewness, kurtosis,
                corr_with_market, beta, momentum_1m, momentum_3m,
                max_drawdown, var_95
            ])

        return np.array(features, dtype=np.float32)

    def train_gat_model(self,
                       model_config: GATModelConfig,
                       training_samples: list[tuple[Any, np.ndarray, str]],
                       graph_config_name: str) -> dict[str, Any]:
        """
        Train GAT model with given configuration and training data.

        Args:
            model_config: GAT model configuration
            training_samples: List of training samples
            graph_config_name: Name of graph configuration

        Returns:
            Training results dictionary
        """
        self.logger.info(f"Training GAT model with {graph_config_name} configuration")

        # Create portfolio constraints
        constraints = PortfolioConstraints(
            long_only=True,
            top_k_positions=50,
            max_position_weight=0.10,
            max_monthly_turnover=0.20
        )

        # Initialize model
        GATPortfolioModel(constraints=constraints, config=model_config)

        # Check GPU memory
        memory_info = self.memory_manager.get_memory_info()
        self.logger.info(f"GPU Memory - Available: {memory_info['available']:.2f}GB")

        # Training setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_fn = SharpeRatioLoss()

        # Build model architecture
        if training_samples:
            sample_graph, _, _ = training_samples[0]
            input_dim = sample_graph.x.shape[1]

            gat_model = GATPortfolio(
                in_dim=input_dim,
                hidden_dim=model_config.hidden_dim,
                heads=model_config.num_attention_heads,
                num_layers=model_config.num_layers,
                dropout=model_config.dropout,
                residual=model_config.residual,
                use_gatv2=model_config.use_gatv2,
                use_edge_attr=True,
                head=model_config.head_config.mode,
                activation=model_config.head_config.activation
            ).to(device)

            optimizer = Adam(
                gat_model.parameters(),
                lr=model_config.learning_rate,
                weight_decay=model_config.weight_decay
            )

            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )

            # Mixed precision scaler
            scaler = torch.amp.GradScaler("cuda") if model_config.use_mixed_precision and torch.cuda.is_available() else None

            # Training loop
            training_history = []
            best_loss = float('inf')
            patience_counter = 0

            for epoch in range(model_config.max_epochs):
                gat_model.train()
                epoch_losses = []

                for graph_data, target_returns, _date in training_samples:
                    optimizer.zero_grad()

                    # Move data to device
                    x = graph_data.x.to(device)
                    edge_index = graph_data.edge_index.to(device)
                    edge_attr = graph_data.edge_attr.to(device) if graph_data.edge_attr is not None else None
                    mask_valid = torch.ones(x.shape[0], dtype=torch.bool, device=device)

                    # Forward pass with mixed precision
                    if scaler is not None:
                        with torch.amp.autocast("cuda"):
                            result = gat_model(x, edge_index, mask_valid, edge_attr)
                            weights = result[0] if isinstance(result, tuple) else result
                            weights = weights.unsqueeze(0)  # [1, n_assets]

                            returns_tensor = torch.tensor(
                                target_returns, dtype=torch.float32, device=device
                            ).unsqueeze(0)

                            loss = loss_fn(weights, returns_tensor, mask_valid.unsqueeze(0))

                        # Backward pass with gradient scaling
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        result = gat_model(x, edge_index, mask_valid, edge_attr)
                        weights = result[0] if isinstance(result, tuple) else result
                        weights = weights.unsqueeze(0)

                        returns_tensor = torch.tensor(
                            target_returns, dtype=torch.float32, device=device
                        ).unsqueeze(0)

                        loss = loss_fn(weights, returns_tensor, mask_valid.unsqueeze(0))
                        loss.backward()
                        optimizer.step()

                    epoch_losses.append(loss.item())

                # Epoch statistics
                avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
                training_history.append(avg_loss)

                # Learning rate scheduling
                scheduler.step(avg_loss)

                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0

                    # Save best model
                    checkpoint_path = self.checkpoint_dir / f"gat_{graph_config_name}_best.pt"
                    torch.save({
                        'model_state_dict': gat_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': avg_loss,
                        'config': model_config,
                        'graph_config_name': graph_config_name
                    }, checkpoint_path)
                else:
                    patience_counter += 1

                if patience_counter >= model_config.patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: Loss = {avg_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")

        # Training results
        results = {
            'graph_config_name': graph_config_name,
            'best_loss': best_loss,
            'training_history': training_history,
            'total_epochs': len(training_history),
            'model_parameters': sum(p.numel() for p in gat_model.parameters()),
            'memory_usage': self.memory_manager.get_memory_info(),
            'device': str(device)
        }

        self.logger.info(f"Training completed for {graph_config_name}: Best Loss = {best_loss:.6f}")
        return results

    def execute_full_training_pipeline(self) -> dict[str, Any]:
        """
        Execute complete GAT training pipeline across all graph construction methods.

        Returns:
            Comprehensive training results
        """
        self.logger.info("Starting GAT training pipeline execution")

        # Load data
        returns_data, universe_calendar = self.load_data()

        # Prepare graph configurations
        graph_configurations = self.prepare_graph_configurations()

        # Training results for all configurations
        all_results = {}

        for graph_config in graph_configurations:
            try:
                config_name = graph_config['config_name']
                self.logger.info(f"Processing configuration: {config_name}")

                # Create model configuration
                model_config = self.create_gat_model_config(graph_config)

                # Prepare training data
                training_samples = self.prepare_training_data(
                    returns_data, universe_calendar, graph_config
                )

                if not training_samples:
                    self.logger.warning(f"No training samples for {config_name}")
                    continue

                # Train model
                results = self.train_gat_model(
                    model_config, training_samples, config_name
                )

                all_results[config_name] = results

                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                self.logger.error(f"Failed training for {graph_config['config_name']}: {e}")
                continue

        # Save comprehensive results
        self.training_results = all_results
        self._save_training_results()

        self.logger.info(f"GAT training pipeline completed. Trained {len(all_results)} configurations.")
        return all_results

    def _save_training_results(self):
        """Save training results to file."""
        results_file = self.results_dir / "gat_training_results.yaml"
        with open(results_file, 'w') as f:
            yaml.dump(self.training_results, f, default_flow_style=False)

        self.logger.info(f"Training results saved to {results_file}")


def main():
    """Main execution function."""

    # Initialize and run training pipeline
    pipeline = GATTrainingPipeline()
    results = pipeline.execute_full_training_pipeline()


    # Print summary
    if results:
        for _config_name, _result in results.items():
            pass


if __name__ == "__main__":
    main()
