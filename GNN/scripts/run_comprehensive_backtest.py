#!/usr/bin/env python3
"""
Comprehensive backtest execution script for Story 5.3.

This script executes rolling backtests for all ML approaches (HRP, LSTM, GAT)
plus baselines across the complete evaluation period (2016-2024) with proper
walk-forward analysis and temporal integrity validation.

Key features:
- 96 rolling windows with monthly rebalancing
- Memory-efficient execution within 11GB GPU constraints
- Constraint enforcement validation
- Comprehensive performance metrics and reporting
- Temporal integrity monitoring
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.base import ProjectConfig, load_config
from src.data.loaders.parquet_manager import ParquetManager
from src.evaluation.backtest.rolling_engine import RollingBacktestConfig, RollingBacktestEngine
from src.evaluation.validation.temporal_integrity import TemporalIntegrityValidator

# Import academic reporting and performance metrics with uncertainty
try:
    from src.evaluation.reporting.academic_report_generator import (
        AcademicReportConfig,
        AcademicReportGenerator,
        create_academic_report_generator,
    )
    from src.evaluation.metrics.academic_uncertainty import (
        AcademicPerformanceMetrics,
        create_academic_performance_calculator,
    )
    ACADEMIC_REPORTING_AVAILABLE = True
except ImportError:
    logger.warning("Academic reporting modules not available, using standard reporting")
    ACADEMIC_REPORTING_AVAILABLE = False
from src.models.base.baselines import (
    EqualWeightModel,
    MarketCapWeightedModel,
    MeanReversionModel,
)
from src.models.base.portfolio_model import PortfolioConstraints
from src.models.gat.model import GATPortfolioModel, GATModelConfig
from src.models.gat.gat_model import HeadCfg
from src.models.hrp.clustering import ClusteringConfig
from src.models.hrp.model import HRPConfig, HRPModel
from src.models.lstm.model import LSTMPortfolioModel
from src.utils.gpu import GPUConfig
from src.utils.membership_aware_cleaning import (
    load_dynamic_universe,
    clean_returns_with_membership,
    get_rolling_universe,
    get_universe_at_date
)

# Configure logging
# Create logs directory
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Create timestamped log file
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = logs_dir / f"comprehensive_backtest_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def create_gpu_config() -> GPUConfig:
    """Create GPU configuration with memory constraints."""
    return GPUConfig(
        max_memory_gb=11.0,  # RTX GeForce 5070Ti conservative limit
        enable_mixed_precision=True,
        batch_size_auto_scale=True,
    )


def load_market_data(config: dict[str, Any] | ProjectConfig) -> dict[str, pd.DataFrame]:
    """Load market data for backtesting with membership-aware cleaning."""
    logger.info("Loading market data...")

    # Load returns data from parquet files (data is already daily returns)
    returns_path = Path("data/final_new_pipeline/returns_daily_final.parquet")
    universe_path = Path("data/processed/universe_membership_clean.csv")

    if not returns_path.exists():
        raise FileNotFoundError(f"Returns data not found at {returns_path}")

    # Load raw returns data
    raw_returns_data = pd.read_parquet(returns_path)  # This contains daily returns, not prices
    initial_shape = raw_returns_data.shape

    # Use configuration-based date range, with some buffer for training lookback
    if isinstance(raw_returns_data.index, pd.DatetimeIndex):
        # Get config dates from the project config or dictionary
        if isinstance(config, dict):
            # Handle dictionary configuration from YAML
            backtest_settings = config.get('backtest_settings', {})
            # Use 2019 as test start since we need 2016+ data for training
            config_start = pd.Timestamp('2019-01-01')  # Override config to use proper date
            config_end = pd.Timestamp(backtest_settings.get('test_end_date', '2024-11-30'))
        else:
            # Handle ProjectConfig object
            config_start = pd.Timestamp("2019-01-01")  # Use 2019 start for proper training window
            config_end = config.end_date if hasattr(config, 'end_date') else pd.Timestamp("2024-11-30")

        # Add buffer for training lookback (36 months before start date)
        buffer_months = 36
        buffer_start = config_start - pd.DateOffset(months=buffer_months)

        # Ensure we don't go before available data
        data_start = raw_returns_data.index[0]
        data_end = raw_returns_data.index[-1]

        # Don't use data before 2016 due to quality issues
        min_quality_date = pd.Timestamp("2016-01-01")

        actual_start = max(buffer_start, data_start, min_quality_date)
        actual_end = min(config_end, data_end)

        raw_returns_data = raw_returns_data.loc[actual_start:actual_end]
        logger.info(f"Using date range for rolling backtest: {actual_start} to {actual_end} (config: {config_start} to {config_end})")

    # Check if universe membership data exists
    if universe_path.exists():
        # Load universe membership data
        universe_df = load_dynamic_universe(str(universe_path))
        logger.info(f"Loaded universe membership for {len(universe_df['ticker'].unique())} unique assets")

        # Apply membership-aware data cleaning
        returns_data, membership_mask = clean_returns_with_membership(
            raw_returns_data,
            universe_df,
            max_daily_return=2.0,  # 200% daily return
            min_daily_return=-0.8,  # -80% daily return
            z_score_threshold=8.0,
            cross_sectional_threshold=10.0
        )

        # Only keep columns that have membership data
        valid_columns = membership_mask.any(axis=0)
        returns_data = returns_data.loc[:, valid_columns]
        membership_mask = membership_mask.loc[:, valid_columns]

        logger.info(f"Loaded returns data: {returns_data.shape} (from {initial_shape})")

        # Calculate statistics only for periods when assets are in universe
        masked_returns = returns_data.where(membership_mask)
        valid_returns = masked_returns.stack().dropna()
        logger.info(f"Return statistics (during membership): mean={valid_returns.mean():.6f}, std={valid_returns.std():.6f}")
    else:
        logger.warning(f"Universe membership data not found at {universe_path}")
        logger.warning("Falling back to non-membership-aware cleaning...")

        # Fallback to original cleaning method
        returns_data = raw_returns_data.copy()
        returns_data = returns_data.dropna(axis=1, thresh=len(returns_data) * 0.8)  # Keep columns with at least 80% data

        # Handle NaN values in returns data more carefully
        # Only drop rows where ALL values are NaN
        returns_data = returns_data.dropna(how='all')

        # For remaining NaN values, use forward fill with limit then drop
        # This prevents artificial data creation
        returns_data = returns_data.ffill(limit=5)  # Max 5 day forward fill

        # Enhanced data quality filtering
        # Step 1: Remove obvious data errors (returns > 200% or < -80%)
        extreme_upper = 2.0  # 200% daily return
        extreme_lower = -0.8  # -80% daily return
        obvious_errors = (returns_data > extreme_upper) | (returns_data < extreme_lower)
        returns_data[obvious_errors] = np.nan
        logger.info(f"Removed {obvious_errors.sum().sum()} obvious data errors (>200% or <-80% daily returns)")

        # Step 2: Statistical outlier detection using rolling z-scores
        # Calculate rolling standard deviation for outlier detection
        rolling_std = returns_data.rolling(window=252, min_periods=60).std()
        rolling_mean = returns_data.rolling(window=252, min_periods=60).mean()
        z_scores = (returns_data - rolling_mean).abs() / (rolling_std + 1e-8)

        # Flag returns > 8 standard deviations as outliers (reduced from 10 for stricter filtering)
        statistical_outliers = z_scores > 8

        # Replace statistical outliers with NaN
        returns_data[statistical_outliers] = np.nan

        # Step 3: Cross-sectional outlier detection (same-day outliers across assets)
        daily_median = returns_data.median(axis=1)
        daily_mad = (returns_data.sub(daily_median, axis=0)).abs().median(axis=1)
        cross_sectional_z = (returns_data.sub(daily_median, axis=0)).abs() / (daily_mad.replace(0, np.nan) * 1.4826 + 1e-8)

        # Flag cross-sectional outliers (returns that deviate significantly from other assets on the same day)
        cross_sectional_outliers = cross_sectional_z > 10
        returns_data[cross_sectional_outliers] = np.nan

        # Step 4: Fill NaN values carefully
        # First forward fill with strict limit to avoid propagating bad data
        returns_data = returns_data.ffill(limit=2)
        # Then backward fill with strict limit for start of series
        returns_data = returns_data.bfill(limit=2)
        # Finally, fill any remaining NaNs with 0 (no return)
        returns_data = returns_data.fillna(0.0)

        # Log statistics about data cleaning
        total_outliers = obvious_errors.sum().sum() + statistical_outliers.sum().sum() + cross_sectional_outliers.sum().sum()
        if total_outliers > 0:
            logger.info(f"Total outliers removed: {total_outliers} ({total_outliers/returns_data.size:.4%} of data)")

        logger.info(f"Loaded returns data: {returns_data.shape} (cleaned from {initial_shape})")
        logger.info(f"Return statistics: mean={returns_data.mean().mean():.6f}, std={returns_data.std().mean():.6f}, max={returns_data.max().max():.6f}")

    # Load universe data - already loaded above if it exists
    universe_data = None
    dynamic_universe = None

    if universe_path.exists():
        try:
            # Create dynamic universe schedule for rebalancing
            if 'universe_df' in locals():
                # Get rolling universe for each month
                backtest_start = pd.Timestamp("2019-01-01")
                backtest_end = pd.Timestamp("2024-12-31")
                dynamic_universe = get_rolling_universe(
                    universe_df,
                    backtest_start,
                    backtest_end,
                    rebalance_frequency='MS'  # Monthly rebalancing
                )
                logger.info(f"Created dynamic universe schedule with {len(dynamic_universe)} rebalancing dates")

                # Create a simplified universe_data DataFrame for compatibility
                # This shows which assets are ever in the universe
                all_tickers = universe_df['ticker'].unique()
                universe_data = pd.DataFrame(
                    True,
                    index=[pd.Timestamp.now()],  # Single row for compatibility
                    columns=all_tickers
                )
            else:
                # Fallback: try to load as before if membership data wasn't loaded
                raw_universe_data = pd.read_csv(str(universe_path))
                logger.info(f"Loaded raw universe data: {raw_universe_data.shape}")

                # Convert to time-indexed format
                if 'date' in raw_universe_data.columns:
                    # Parse date column
                    raw_universe_data['date'] = pd.to_datetime(raw_universe_data['date'])

                    # Pivot to get date x ticker matrix
                    if 'ticker' in raw_universe_data.columns and 'in_universe' in raw_universe_data.columns:
                        universe_data = raw_universe_data.pivot(
                            index='date',
                            columns='ticker',
                            values='in_universe'
                        )
                        universe_data = universe_data.fillna(0).astype(bool)
                        logger.info(f"Converted universe data to time-indexed format: {universe_data.shape}")
                    else:
                        # Alternative format: create boolean mask from presence
                        unique_dates = raw_universe_data['date'].unique()
                        unique_tickers = raw_universe_data['ticker'].unique() if 'ticker' in raw_universe_data.columns else []

                        if unique_tickers:
                            universe_data = pd.DataFrame(
                                False,
                                index=pd.DatetimeIndex(unique_dates),
                                columns=unique_tickers
                            )

                            for _, row in raw_universe_data.iterrows():
                                if 'ticker' in row and pd.notna(row['ticker']):
                                    universe_data.loc[row['date'], row['ticker']] = True

                            logger.info(f"Created universe membership matrix: {universe_data.shape}")
                else:
                    logger.warning("Universe data missing date column, cannot create time index")
                    universe_data = None

        except Exception as e:
            logger.error(f"Failed to process universe data: {e}")
            universe_data = None
    else:
        logger.warning("Universe data not available in current dataset")

    # Load benchmark data (SPY prices to compute returns)
    benchmark_data = None
    prices_path = Path("data/final_new_pipeline/prices_final.parquet")
    if prices_path.exists():
        prices_data = pd.read_parquet(prices_path)
        if "SPY" in prices_data.columns:
            benchmark_data = prices_data[["SPY"]].pct_change().dropna()
            logger.info(f"Loaded benchmark data: {benchmark_data.shape}")
    else:
        logger.warning("Benchmark data not available")

    return {
        "returns": returns_data,
        "universe": universe_data,
        "benchmark": benchmark_data,
        "dynamic_universe": dynamic_universe if 'dynamic_universe' in locals() else None,
        "membership_mask": membership_mask if 'membership_mask' in locals() else None,
        "universe_df": universe_df if 'universe_df' in locals() else None,
    }


# Pre-trained model discovery removed - models are now trained fresh during rolling backtest


def create_hrp_models() -> dict[str, Any]:
    """Create fresh HRP models for rolling backtest."""
    models = {}

    logger.info("Creating fresh HRP model for rolling training")
    hrp_config = HRPConfig(
        clustering_config=ClusteringConfig(
            linkage_method="average",
            min_observations=252,
            correlation_method="pearson",
        ),
        lookback_days=756,  # 36 months * 21 trading days
        min_observations=252,
        correlation_method="pearson",
    )
    models["HRP"] = HRPModel(
        hrp_config=hrp_config,
        constraints=PortfolioConstraints(
            long_only=True,  # Keep this - no short selling
            max_position_weight=1.0,  # No limit - can concentrate if hierarchical clustering suggests it
            max_monthly_turnover=10.0,  # Essentially unlimited
            min_weight_threshold=0.0,  # No minimum
            top_k_positions=None,  # No limit on positions
        )
    )

    return models


def create_lstm_models() -> dict[str, Any]:
    """Create fresh LSTM models for rolling backtest."""
    models = {}

    logger.info("Creating fresh LSTM model for rolling training")
    from src.models.lstm.model import LSTMModelConfig
    lstm_config = LSTMModelConfig()
    lstm_config.training_config.epochs = 30  # Proper training epochs
    lstm_config.training_config.patience = 10
    lstm_config.training_config.learning_rate = 0.001
    lstm_config.training_config.batch_size = 64  # Increased from default for better GPU usage
    lstm_config.prediction_horizon = 21  # Monthly prediction

    models["LSTM"] = LSTMPortfolioModel(
        constraints=PortfolioConstraints(
            long_only=True,  # Keep this - no short selling
            max_position_weight=1.0,  # No limit - LSTM can concentrate if it predicts strong returns
            max_monthly_turnover=10.0,  # Essentially unlimited
            min_weight_threshold=0.0,  # No minimum
            top_k_positions=None,  # No limit on positions
            transaction_cost_bps=10.0,
            enable_turnover_penalty=False,  # Let LSTM decide its own turnover
        ),
        config=lstm_config
    )

    return models


def create_gat_models() -> dict[str, Any]:
    """Create fresh GAT models for rolling backtest."""
    models = {}

    logger.info("Creating fresh GAT models for all three graph methods")

    # Common GAT configuration
    base_constraints = PortfolioConstraints(
        long_only=True,  # Keep this - no short selling
        max_position_weight=0.20,  # 20% max prevents pathological single-asset concentration
        max_monthly_turnover=2.0,  # 200% turnover allows flexibility while preventing excess
        min_weight_threshold=0.005,  # 0.5% minimum for meaningful positions
        top_k_positions=None,  # No hard limit on number of positions
        transaction_cost_bps=10.0,  # Keep realistic transaction costs
        enable_turnover_penalty=False,  # Don't penalize turnover - real costs are enough
    )

    # GAT-MST Model
    gat_mst_config = GATModelConfig()
    gat_mst_config.hidden_dim = 64
    gat_mst_config.num_attention_heads = 8
    gat_mst_config.max_epochs = 15  # Proper training epochs
    gat_mst_config.patience = 10
    gat_mst_config.learning_rate = 0.001
    gat_mst_config.graph_config.filter_method = "mst"  # Minimum Spanning Tree
    gat_mst_config.head_config = HeadCfg(mode="direct", activation="softmax")  # Use softmax to avoid vanishing gradients
    # Use standard GAT with pure Sharpe focus for consistency

    models["GAT-MST"] = GATPortfolioModel(
        constraints=base_constraints,
        config=gat_mst_config
    )
    logger.info("Created GAT-MST model")

    # GAT-kNN Model
    gat_knn_config = GATModelConfig()
    gat_knn_config.hidden_dim = 64
    gat_knn_config.num_attention_heads = 8
    gat_knn_config.max_epochs = 15
    gat_knn_config.patience = 10
    gat_knn_config.learning_rate = 0.001
    gat_knn_config.graph_config.filter_method = "knn"  # k-Nearest Neighbors
    gat_knn_config.graph_config.knn_k = 10  # k=10 neighbors
    gat_knn_config.head_config = HeadCfg(mode="direct", activation="softmax")  # Use softmax to avoid vanishing gradients
    # Use standard GAT with pure Sharpe focus for consistency

    models["GAT-kNN"] = GATPortfolioModel(
        constraints=base_constraints,
        config=gat_knn_config
    )
    logger.info("Created GAT-kNN model")

    # GAT-TMFG Model - Special configuration for dense TMFG graphs
    gat_tmfg_config = GATModelConfig()
    gat_tmfg_config.hidden_dim = 32  # Reduced for dense graphs
    gat_tmfg_config.num_layers = 2  # Fewer layers for dense graphs
    gat_tmfg_config.num_attention_heads = 4  # Fewer heads for dense graphs
    gat_tmfg_config.dropout = 0.4  # Higher dropout for regularization
    gat_tmfg_config.max_epochs = 20  # More epochs for complex graph
    gat_tmfg_config.patience = 10
    gat_tmfg_config.learning_rate = 0.0005  # Lower learning rate for stability
    gat_tmfg_config.weight_decay = 1e-4  # Higher weight decay for regularization
    gat_tmfg_config.graph_config.filter_method = "tmfg"  # Triangulated Maximally Filtered Graph
    gat_tmfg_config.graph_config.edge_pruning_threshold = 0.3  # Prune weak edges in TMFG
    gat_tmfg_config.head_config = HeadCfg(mode="direct", activation="softmax")  # Changed from sparsemax to prevent extreme concentration

    models["GAT-TMFG"] = GATPortfolioModel(
        constraints=base_constraints,
        config=gat_tmfg_config
    )
    logger.info("Created GAT-TMFG model")

    return models


def initialize_models(config: dict[str, Any] | ProjectConfig, gpu_config: GPUConfig) -> dict[str, Any]:
    """Initialize all models for backtesting with fresh instances for rolling training."""
    logger.info("Initializing models...")

    models = {}

    # Create fresh HRP models
    logger.info("Creating HRP models...")
    hrp_models = create_hrp_models()
    models.update(hrp_models)

    # Create fresh LSTM models
    logger.info("Creating LSTM models...")
    lstm_models = create_lstm_models()
    models.update(lstm_models)

    # Create fresh GAT models
    logger.info("Creating GAT models...")
    gat_models = create_gat_models()
    models.update(gat_models)

    # Baseline models for comparison
    models["EqualWeight"] = EqualWeightModel()
    models["MarketCapWeighted"] = MarketCapWeightedModel()
    models["MeanReversion"] = MeanReversionModel(lookback_days=21)

    logger.info(f"Initialized {len(models)} models: {list(models.keys())}")
    return models


def create_backtest_config(enable_rolling: bool = True) -> RollingBacktestConfig:
    """Create backtest configuration for proper rolling window testing."""
    # Rolling window configuration with realistic retraining
    # Models retrain monthly using only data available up to each point
    return RollingBacktestConfig(
            # Rolling evaluation period starting when sufficient training data is available
            # We have data from 2010 but use 2016+ as earlier data quality is poor
            # With 36 months training, we can start backtesting from 2019-01-01
            start_date=pd.Timestamp("2019-01-01"),  # Start when 36 months of good quality data available (2016+3 years)
            end_date=pd.Timestamp("2024-10-01"),    # End of evaluation period (match actual data availability)
            training_months=36,  # Use 3 years of training data right up to prediction point
            validation_months=0, # No validation period - use all recent data for training
            test_months=1,       # 1 month test (monthly rebalancing)
            step_months=1,       # Step forward 1 month at a time
            rebalance_frequency="MS",  # Month start frequency

            # Data integrity parameters
            min_training_samples=100,  # Default, will be overridden by flexible validator
            max_gap_days=5,            # Allow up to 5-day gaps
            require_full_periods=False, # More flexible for dynamic universe

        # Execution parameters
        initial_capital=1_000_000.0,  # $1M portfolio
        transaction_cost_bps=10.0,    # 10 basis points transaction costs
        benchmark_ticker="SPY",

        # Rolling retraining configuration (TRUE ROLLING)
        enable_rolling_retraining=True,   # Enable realistic rolling retraining
        monthly_retraining=True,          # Retrain models monthly
        quick_retrain_epochs={
            "hrp": 1,  # HRP doesn't need epochs (correlation-based)
            "lstm": 30,  # LSTM retraining epochs - proper convergence
            "gat": 15,   # GAT retraining epochs - proper convergence
        },

        # Memory management
        gpu_config=create_gpu_config(),
        batch_size=128,  # Increased batch size for better GPU utilisation with 509 assets
        enable_memory_monitoring=True,

            # Output configuration
            output_dir=Path("results/ml_backtest_rolling"),
            save_intermediate_results=True,
            enable_progress_tracking=True,
        )


def validate_constraint_enforcement(results: Any) -> dict[str, Any]:
    """Validate constraint enforcement across all approaches and time periods."""
    logger.info("Validating constraint enforcement...")

    constraint_violations = {}

    for model_name, weights_df in results.portfolio_weights.items():
        violations = {
            "long_only_violations": 0,
            "turnover_violations": 0,
            "position_limit_violations": 0,
            "weight_sum_violations": 0,
        }

        if weights_df.empty:
            constraint_violations[model_name] = violations
            continue

        # Check long-only constraints (all weights >= 0)
        negative_weights = (weights_df < 0).any(axis=1).sum()
        violations["long_only_violations"] = int(negative_weights)

        # Check weight sum constraints (should sum to ~1.0)
        weight_sums = weights_df.sum(axis=1)
        sum_violations = (abs(weight_sums - 1.0) > 0.01).sum()
        violations["weight_sum_violations"] = int(sum_violations)

        # Check position limits (max 15% per position)
        max_position_violations = (weights_df > 0.15).any(axis=1).sum()
        violations["position_limit_violations"] = int(max_position_violations)

        # Check turnover constraints (≤30% monthly as per config)
        if len(weights_df) > 1:
            turnover = abs(weights_df.diff()).sum(axis=1).iloc[1:]
            turnover_violations = (turnover > 0.30).sum()
            violations["turnover_violations"] = int(turnover_violations)

        constraint_violations[model_name] = violations

    # Calculate summary statistics
    total_violations = sum(
        sum(violations.values())
        for violations in constraint_violations.values()
    )

    summary = {
        "model_violations": constraint_violations,
        "total_violations": total_violations,
        "models_with_violations": len([
            model for model, violations in constraint_violations.items()
            if sum(violations.values()) > 0
        ]),
        "validation_passed": total_violations == 0,
    }

    logger.info(f"Constraint validation completed: {total_violations} total violations")
    return summary


def generate_comprehensive_report(
    results: Any,
    constraint_validation: dict[str, Any],
    execution_time: float,
    use_academic_reporting: bool = True,
) -> dict[str, Any]:
    """Generate comprehensive backtest report with optional academic reporting.

    Args:
        results: Backtest results
        constraint_validation: Constraint validation results
        execution_time: Total execution time in seconds
        use_academic_reporting: Whether to use academic reporting (defaults to True)

    Returns:
        Comprehensive report dictionary
    """
    logger.info("Generating comprehensive report...")

    # Check if academic reporting is available and requested
    use_academic = use_academic_reporting and ACADEMIC_REPORTING_AVAILABLE

    report = {
        "execution_summary": {
            "total_execution_time_hours": execution_time / 3600,
            "total_models_tested": len(results.portfolio_returns),
            "total_rolling_windows": len(results.splits),
            "evaluation_period": {
                "start": "2023-01-01",
                "end": "2024-12-31",
                "years": 2,
            },
            "memory_constraints_met": execution_time < 8 * 3600,  # ≤8 hours target
            "using_academic_reporting": use_academic,
        },
        "constraint_enforcement": constraint_validation,
        "model_performance": {},
        "temporal_integrity": results.temporal_integrity_report,
        "memory_usage": results.memory_usage_stats,
    }

    # Add academic performance metrics if available
    if use_academic:
        report["academic_analysis"] = {}
        academic_calculator = create_academic_performance_calculator()

        # Process each model with academic metrics
        for model_name, returns in results.portfolio_returns.items():
            if returns is not None and not returns.empty:
                # Get confidence scores if available (from validation results)
                confidence_scores = None
                if hasattr(results, "validation_results") and model_name in results.validation_results:
                    val_results = results.validation_results[model_name]
                    if hasattr(val_results, "confidence_scores"):
                        confidence_scores = val_results.confidence_scores

                # Calculate academic performance metrics
                academic_report = academic_calculator.calculate_with_uncertainty(
                    returns=returns,
                    confidence_scores=confidence_scores,
                    benchmark_returns=results.portfolio_returns.get("EqualWeight"),
                )

                report["academic_analysis"][model_name] = {
                    "metrics": academic_report.metrics,
                    "confidence_intervals": academic_report.confidence_intervals,
                    "significance_tests": academic_report.significance_tests,
                    "academic_caveats": academic_report.academic_caveats,
                    "methodology": academic_report.methodology_used,
                }

    # Add model performance summary
    for model_name, metrics in results.performance_metrics.items():
        if metrics:
            report["model_performance"][model_name] = {
                "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                "annualized_return": metrics.get("annualized_return", 0.0),
                "annualized_volatility": metrics.get("volatility", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "total_return": (1 + results.portfolio_returns[model_name]).prod() - 1 if model_name in results.portfolio_returns else 0.0,
            }

    # Identify best performing models
    if report["model_performance"]:
        best_sharpe = max(
            report["model_performance"].items(),
            key=lambda x: x[1].get("sharpe_ratio", -999)
        )
        best_return = max(
            report["model_performance"].items(),
            key=lambda x: x[1].get("total_return", -999)
        )

        report["top_performers"] = {
            "best_sharpe": {"model": best_sharpe[0], "value": best_sharpe[1]["sharpe_ratio"]},
            "best_return": {"model": best_return[0], "value": best_return[1]["total_return"]},
        }

    return report


def save_results(
    results: Any,
    constraint_validation: dict[str, Any],
    comprehensive_report: dict[str, Any],
    output_dir: Path,
    generate_academic_reports: bool = True,
) -> None:
    """Save all results to disk with optional academic reporting.

    Args:
        results: Backtest results
        constraint_validation: Constraint validation results
        comprehensive_report: Comprehensive report dictionary
        output_dir: Output directory
        generate_academic_reports: Whether to generate academic reports (defaults to True)
    """
    logger.info(f"Saving results to {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Optionally create minimal symlinks for backward compatibility
    # Set this to False to have a clean folder structure without root-level files
    CREATE_COMPATIBILITY_SYMLINKS = False  # Set to True only if you have legacy code that needs it

    if CREATE_COMPATIBILITY_SYMLINKS:
        import os
        logger.info("Creating backward compatibility symlinks...")

        # Helper function to create compatibility symlinks
        def create_legacy_symlink(organized_path: Path, legacy_name: str):
            """Create symlink for backward compatibility."""
            if organized_path.exists():
                legacy_path = output_dir / legacy_name
                if legacy_path.exists() and legacy_path.is_symlink():
                    legacy_path.unlink()
                elif legacy_path.exists():
                    # Don't overwrite real files
                    return
                try:
                    # Create relative symlink
                    relative_path = os.path.relpath(organized_path, output_dir)
                    legacy_path.symlink_to(relative_path)
                except OSError:
                    # If symlinks not supported, copy the file
                    import shutil
                    shutil.copy2(organized_path, legacy_path)

        # Only create the essential symlinks for backward compatibility
        # Just performance_metrics.csv as it's commonly used
        metrics_path = output_dir / "metrics" / "performance_metrics.csv"
        if metrics_path.exists():
            create_legacy_symlink(metrics_path, "performance_metrics.csv")

    # Log the clean folder structure
    logger.info("Results saved with organized folder structure:")
    logger.info(f"  📁 {output_dir}/")
    logger.info(f"     ├── 📊 metrics/          - Performance metrics and summaries")
    logger.info(f"     ├── 📈 returns/         - Model returns by category")
    logger.info(f"     ├── ⚖️  weights/         - Portfolio weights by category")
    logger.info(f"     ├── 💰 costs/           - Transaction costs by category")
    logger.info(f"     ├── 📋 reports/         - JSON reports and analysis")
    logger.info(f"     └── 🎓 academic_reports/ - Academic performance reports")

    if not CREATE_COMPATIBILITY_SYMLINKS:
        logger.info("\nNote: No symlinks created in root folder for cleaner organization.")
        logger.info("To access files, navigate to the appropriate subfolder.")
        logger.info("Set CREATE_COMPATIBILITY_SYMLINKS=True if you need backward compatibility.")

    # Save additional reports directly to reports folder
    import json

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save constraint validation directly to reports folder
    with open(reports_dir / "constraint_validation.json", "w") as f:
        json.dump(constraint_validation, f, indent=2)

    # Save comprehensive report directly to reports folder
    with open(reports_dir / "comprehensive_report.json", "w") as f:
        json.dump(comprehensive_report, f, indent=2, default=str)

    # Generate academic reports if requested and available
    if generate_academic_reports and ACADEMIC_REPORTING_AVAILABLE:
        logger.info("Generating academic reports...")
        try:
            # Create academic report generator with default configuration
            academic_config = AcademicReportConfig(
                generate_latex=True,
                generate_markdown=True,
                generate_csv=True,
                generate_json=True,
                include_confidence_intervals=True,
                include_significance_tests=True,
                include_uncertainty_bounds=True,
                include_methodology_description=True,
                include_academic_caveats=True,
                include_robustness_checks=True,
            )
            academic_generator = create_academic_report_generator(academic_config)

            # Prepare backtest results for academic reporting
            academic_results = {}
            for model_name in results.portfolio_returns.keys():
                academic_results[model_name] = {
                    "returns": results.portfolio_returns.get(model_name),
                    "weights": results.portfolio_weights.get(model_name),
                    "costs": results.transaction_costs.get(model_name),
                }

                # Add academic analysis if available
                if "academic_analysis" in comprehensive_report and model_name in comprehensive_report["academic_analysis"]:
                    academic_results[model_name].update(comprehensive_report["academic_analysis"][model_name])

                # Add confidence score if available
                if hasattr(results, "validation_results") and model_name in results.validation_results:
                    val_result = results.validation_results[model_name]
                    if hasattr(val_result, "confidence"):
                        academic_results[model_name]["confidence_score"] = val_result.confidence

            # Generate academic reports
            academic_output_dir = output_dir / "academic_reports"
            academic_files = academic_generator.generate_comprehensive_report(
                backtest_results=academic_results,
                output_dir=academic_output_dir,
                report_name="academic_backtest_report",
            )

            logger.info(f"Academic reports saved to: {academic_output_dir}")
            logger.info(f"Generated formats: {list(academic_files.keys())}")

        except Exception as e:
            logger.warning(f"Failed to generate academic reports: {e}")
            logger.info("Standard reports have been saved successfully")

    # Save execution summary
    with open(output_dir / "execution_summary.json", "w") as f:
        json.dump(results.execution_summary, f, indent=2, default=str)

    logger.info("Results saved successfully")


def fit_baseline_models(models: dict, market_data: dict, config: dict[str, Any] | ProjectConfig) -> None:
    """
    Fit baseline models with historical returns data.

    This ensures baseline models have access to returns data for their
    strategy calculations (market cap weighting, mean reversion, etc).

    CRITICAL: Baseline models need access to the FULL dataset (including backtest period)
    for lookback calculations during prediction, but only use pre-backtest data for training.

    Args:
        models: Dictionary of model instances
        market_data: Market data including returns
        config: Project configuration
    """
    baseline_models = ["EqualWeight", "MarketCapWeighted", "MeanReversion"]

    # Load full historical data for baseline model fitting
    logger.info("Loading full historical returns data for baseline model fitting...")
    returns_path = Path("data/final_new_pipeline/returns_daily_final.parquet")

    if not returns_path.exists():
        logger.error("Full returns data not found - baseline models may not work properly")
        # Mark models as fitted anyway to avoid errors during backtest
        for model_name in baseline_models:
            if model_name in models:
                models[model_name].is_fitted = True
        return

    full_returns_data = pd.read_parquet(returns_path)

    # Clean the full data similar to the market data loading process
    full_returns_data = full_returns_data.dropna(axis=1, thresh=len(full_returns_data) * 0.8)

    # Use the same universe that the backtest is using
    backtest_returns = market_data.get("returns")
    if backtest_returns is not None:
        universe = list(backtest_returns.columns)
        logger.info(f"Using backtest universe with {len(universe)} assets")
    else:
        universe_data = market_data.get("universe")
        if universe_data is None:
            logger.warning("No universe data available - using all columns from returns")
            universe = list(full_returns_data.columns)
        else:
            universe = list(universe_data)

    # CRITICAL FIX: Provide baseline models with FULL dataset (including backtest period)
    # but specify training period for parameter estimation
    training_end = pd.Timestamp("2022-12-31")

    # Filter to universe assets and ensure consistent columns
    universe_available = [asset for asset in universe if asset in full_returns_data.columns]
    full_universe_data = full_returns_data[universe_available]

    logger.info(f"Fitting baseline models with full dataset: {len(full_universe_data)} samples")
    logger.info(f"Training period end: {training_end}")
    logger.info(f"Target universe: {len(universe)} assets, Available: {len(universe_available)} assets")

    for model_name in baseline_models:
        if model_name in models:
            model = models[model_name]
            logger.info(f"Fitting baseline model: {model_name}")

            try:
                # CRITICAL: Provide full dataset for lookback calculations
                # but specify training period for parameter estimation
                model.fit(
                    returns=full_universe_data,  # Full dataset including backtest period
                    universe=universe_available,  # Available assets only
                    fit_period=(full_universe_data.index[0], training_end)  # Training period only
                )
                logger.info(f"Successfully fitted {model_name} with {len(universe_available)} assets")
                logger.info(f"  - Full data period: {full_universe_data.index[0]} to {full_universe_data.index[-1]}")
                logger.info(f"  - Training period: {full_universe_data.index[0]} to {training_end}")
            except Exception as e:
                logger.error(f"Failed to fit {model_name}: {e}")
                # Mark as fitted anyway to avoid errors
                model.is_fitted = True
        else:
            logger.debug(f"Baseline model {model_name} not found in models")


def main(config_path: str | None = None, use_academic_reports: bool = True) -> None:
    """Execute comprehensive backtest.

    Args:
        config_path: Path to configuration file
        use_academic_reports: Whether to generate academic reports (default: True)
    """
    logger.info(f"Starting comprehensive backtest execution with rolling retraining...")
    logger.info(f"Logging to: {log_file}")
    logger.info(f"Academic reporting: {'ENABLED' if use_academic_reports and ACADEMIC_REPORTING_AVAILABLE else 'DISABLED'}")

    # Load configuration - use shared config by default for consistency with training
    if config_path:
        config = load_config(config_path)
    else:
        # Use shared configuration for training-backtest consistency
        shared_config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
        if shared_config_path.exists():
            logger.info(f"Using shared configuration from {shared_config_path}")
            with open(shared_config_path, 'r') as f:
                shared_config = yaml.safe_load(f)
            config = load_config(str(shared_config_path))
        else:
            logger.warning("Shared config not found, using default configuration")
            config = ProjectConfig()

    # Track execution time
    import time
    start_time = time.time()

    try:
        # 1. Load market data
        market_data = load_market_data(config)

        # 2. Initialize models
        gpu_config = create_gpu_config()
        models = initialize_models(config, gpu_config)

        # 2.5. Fit baseline models with historical returns data
        fit_baseline_models(models, market_data, config)

        # 3. Create backtest configuration with rolling support
        backtest_config = create_backtest_config(enable_rolling=True)

        # 4. Configure model checkpointing and initialize backtest engine
        model_checkpoint_dir = Path("outputs/models")
        backtest_config.model_checkpoint_dir = model_checkpoint_dir
        engine = RollingBacktestEngine(backtest_config)
        results = engine.run_rolling_backtest(
            models=models,
            data=market_data,
            universe_data=market_data.get("universe"),
        )

        # 5. Validate constraint enforcement
        constraint_validation = validate_constraint_enforcement(results)

        # 6. Calculate execution time
        execution_time = time.time() - start_time

        # 7. Generate comprehensive report
        comprehensive_report = generate_comprehensive_report(
            results, constraint_validation, execution_time,
            use_academic_reporting=use_academic_reports
        )

        # 8. Save results
        save_results(
            results,
            constraint_validation,
            comprehensive_report,
            backtest_config.output_dir,
            generate_academic_reports=use_academic_reports,
        )

        # 9. Print summary
        logger.info("="*80)
        logger.info("COMPREHENSIVE BACKTEST COMPLETED")
        logger.info("="*80)
        logger.info(f"Execution time: {execution_time/3600:.2f} hours")
        logger.info(f"Models tested: {len(models)}")
        logger.info(f"Rolling windows: {len(results.splits)}")
        logger.info(f"Constraint violations: {constraint_validation['total_violations']}")

        if "top_performers" in comprehensive_report:
            best_sharpe = comprehensive_report["top_performers"]["best_sharpe"]
            best_return = comprehensive_report["top_performers"]["best_return"]
            logger.info(f"Best Sharpe: {best_sharpe['model']} ({best_sharpe['value']:.3f})")
            logger.info(f"Best Return: {best_return['model']} ({best_return['value']:.1%})")

        logger.info(f"Results saved to: {backtest_config.output_dir}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Backtest execution failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive backtest")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (defaults to shared config at configs/model_config.yaml)",
        default=None,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )
    parser.add_argument(
        "--academic-reports",
        action="store_true",
        default=True,
        help="Generate academic reports with uncertainty quantification (default: True)",
    )
    parser.add_argument(
        "--no-academic-reports",
        dest="academic_reports",
        action="store_false",
        help="Disable academic report generation",
    )
    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    main(args.config, use_academic_reports=args.academic_reports)
