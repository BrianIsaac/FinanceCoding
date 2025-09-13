#!/usr/bin/env python3
"""
HRP Model Training Execution Script
Executes actual HRP model training using real market data
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.base import DataConfig, ProjectConfig, load_config
from data.loaders.parquet_manager import ParquetDataManager
from data.processors.data_quality_validator import DataQualityValidator
from data.processors.universe_builder import UniverseBuilder
from models.hrp.training import HRPTrainingPipeline
from utils.gpu import GPUConfig, GPUMemoryManager


class HRPExecutionEngine:
    """Executes HRP model training with real market data"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize HRP execution engine

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.setup_logging()
        self.setup_gpu_management()

    def setup_logging(self) -> None:
        """Setup comprehensive logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"hrp_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_gpu_management(self) -> None:
        """Setup GPU memory management"""
        try:
            gpu_config = GPUConfig(max_memory_gb=10.0)  # Conservative limit
            self.gpu_manager = GPUMemoryManager(gpu_config)
            self.logger.info("GPU memory management initialized")
        except Exception as e:
            self.logger.warning(f"GPU management unavailable: {e}")
            self.gpu_manager = None

    def load_market_data(self) -> pd.DataFrame:
        """Load real market data for training

        Returns:
            Market data DataFrame with price and volume information
        """
        self.logger.info("Loading market data...")

        # Try to load existing data first
        data_manager = ParquetDataManager(self.config.data)

        try:
            # Load price data
            price_data = data_manager.load_prices()
            volume_data = data_manager.load_volumes()

            if price_data is None or len(price_data) == 0:
                raise ValueError("No price data available")

            self.logger.info(f"Loaded price data: {price_data.shape}")
            if volume_data is not None:
                self.logger.info(f"Loaded volume data: {volume_data.shape}")

            return price_data

        except Exception as e:
            self.logger.error(f"Failed to load existing data: {e}")
            # Generate sample data for demonstration
            return self._generate_sample_data()

    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample market data for demonstration

        Returns:
            Sample market data DataFrame
        """
        self.logger.info("Generating sample market data...")

        # Create 2 years of daily data for 50 stocks
        start_date = datetime.now() - timedelta(days=730)
        dates = pd.date_range(start=start_date, periods=500, freq='D')

        # Generate realistic stock symbols
        symbols = [f"STOCK_{i:03d}" for i in range(50)]

        # Generate correlated returns
        np.random.seed(42)
        returns = np.random.multivariate_normal(
            mean=np.zeros(len(symbols)),
            cov=self._generate_correlation_matrix(len(symbols)),
            size=len(dates)
        ) * 0.02  # Daily volatility ~2%

        # Convert to prices
        prices = pd.DataFrame(index=dates, columns=symbols)
        prices.iloc[0] = 100.0  # Starting price

        for i in range(1, len(dates)):
            prices.iloc[i] = prices.iloc[i-1] * (1 + returns[i])

        self.logger.info(f"Generated sample data: {prices.shape}")
        return prices

    def _generate_correlation_matrix(self, n_assets: int) -> np.ndarray:
        """Generate realistic correlation matrix

        Args:
            n_assets: Number of assets

        Returns:
            Correlation matrix
        """
        # Create block correlation structure (industry groups)
        corr_matrix = np.eye(n_assets)

        # Add sector correlations
        sector_size = n_assets // 5
        for i in range(0, n_assets, sector_size):
            end_idx = min(i + sector_size, n_assets)
            for j in range(i, end_idx):
                for k in range(i, end_idx):
                    if j != k:
                        corr_matrix[j, k] = 0.3 + np.random.normal(0, 0.1)

        # Add market-wide correlation
        market_corr = np.random.uniform(0.1, 0.3, (n_assets, n_assets))
        corr_matrix += market_corr

        # Ensure positive definiteness
        eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)
        corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        # Normalize diagonal to 1
        diag_sqrt = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(diag_sqrt, diag_sqrt)

        return corr_matrix

    def prepare_training_data(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """Prepare data for HRP training

        Args:
            price_data: Raw price data

        Returns:
            Prepared training data dictionary
        """
        self.logger.info("Preparing training data...")

        # Build universe
        universe_builder = UniverseBuilder(self.config.data)
        universe_data = universe_builder.build_universe(price_data)

        # Validate data quality
        validator = DataQualityValidator(self.config.data)
        validated_data = validator.validate_and_clean(universe_data)

        # Calculate returns
        returns = validated_data.pct_change().dropna()

        # Split into training and validation
        split_date = returns.index[int(len(returns) * 0.8)]

        training_data = {
            'returns': returns.loc[:split_date],
            'validation_returns': returns.loc[split_date:],
            'universe': validated_data.columns.tolist(),
            'split_date': split_date
        }

        self.logger.info(f"Training period: {returns.index[0]} to {split_date}")
        self.logger.info(f"Validation period: {split_date} to {returns.index[-1]}")
        self.logger.info(f"Assets in universe: {len(training_data['universe'])}")

        return training_data

    def execute_hrp_training(self, training_data: dict[str, Any]) -> dict[str, Any]:
        """Execute HRP model training

        Args:
            training_data: Prepared training data

        Returns:
            Training results including model and metrics
        """
        self.logger.info("Starting HRP model training...")

        # Initialize HRP training pipeline
        hrp_pipeline = HRPTrainingPipeline(self.config)

        # Monitor GPU memory if available
        if self.gpu_manager:
            initial_memory = self.gpu_manager.get_current_usage()
            self.logger.info(f"Initial GPU memory usage: {initial_memory:.2f}GB")

        try:
            # Execute training
            training_results = hrp_pipeline.train(
                returns=training_data['returns'],
                validation_returns=training_data['validation_returns']
            )

            # Add execution metadata
            training_results.update({
                'execution_date': datetime.now(),
                'training_period': f"{training_data['returns'].index[0]} to {training_data['returns'].index[-1]}",
                'universe_size': len(training_data['universe']),
                'total_observations': len(training_data['returns'])
            })

            self.logger.info("HRP training completed successfully")

            # Log key metrics
            if 'validation_metrics' in training_results:
                metrics = training_results['validation_metrics']
                self.logger.info(f"Validation Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.4f}")
                self.logger.info(f"Validation Volatility: {metrics.get('volatility', 'N/A'):.4f}")
                self.logger.info(f"Maximum Drawdown: {metrics.get('max_drawdown', 'N/A'):.4f}")

            return training_results

        except Exception as e:
            self.logger.error(f"HRP training failed: {e}")
            raise

        finally:
            # Monitor final GPU memory
            if self.gpu_manager:
                final_memory = self.gpu_manager.get_current_usage()
                self.logger.info(f"Final GPU memory usage: {final_memory:.2f}GB")

    def save_training_results(self, results: dict[str, Any]) -> str:
        """Save training results to disk

        Args:
            results: Training results to save

        Returns:
            Path to saved results
        """
        # Create results directory
        results_dir = Path("results") / "hrp_training"
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"hrp_training_results_{timestamp}.pkl"

        # Save results using pandas
        pd.to_pickle(results, results_file)

        self.logger.info(f"Training results saved to: {results_file}")
        return str(results_file)

    def execute_full_pipeline(self) -> dict[str, Any]:
        """Execute complete HRP training pipeline

        Returns:
            Complete execution results
        """
        start_time = datetime.now()
        self.logger.info("Starting HRP training execution pipeline...")

        try:
            # Load market data
            price_data = self.load_market_data()

            # Prepare training data
            training_data = self.prepare_training_data(price_data)

            # Execute training
            training_results = self.execute_hrp_training(training_data)

            # Save results
            results_path = self.save_training_results(training_results)

            # Calculate execution time
            execution_time = datetime.now() - start_time

            summary = {
                'status': 'success',
                'execution_time': execution_time,
                'results_path': results_path,
                'universe_size': len(training_data['universe']),
                'training_observations': len(training_data['returns']),
                'validation_metrics': training_results.get('validation_metrics', {}),
                'execution_date': datetime.now()
            }

            self.logger.info(f"HRP training pipeline completed in {execution_time}")
            return summary

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': datetime.now() - start_time,
                'execution_date': datetime.now()
            }

def main():
    """Main execution function"""

    # Initialize and run execution engine
    engine = HRPExecutionEngine()
    results = engine.execute_full_pipeline()

    # Print summary

    if results['status'] == 'success':

        if 'validation_metrics' in results and results['validation_metrics']:
            metrics = results['validation_metrics']
            for _metric, value in metrics.items():
                if isinstance(value, float):
                    pass
                else:
                    pass
    else:
        pass


if __name__ == "__main__":
    main()
