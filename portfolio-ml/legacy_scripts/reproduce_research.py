#!/usr/bin/env python3
"""
Comprehensive Research Reproduction Script

This script orchestrates the complete reproduction of all research results
for the ML-based portfolio optimization study.

Usage:
    python scripts/reproduce_research.py --experiment full_evaluation
    python scripts/reproduce_research.py --experiment baseline_comparison
    python scripts/reproduce_research.py --config custom_config.yaml
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.config_loader import ConfigLoader
from data.loaders.parquet_manager import ParquetDataLoader
from evaluation.backtest.rolling_backtest_engine import RollingBacktestEngine
from evaluation.reporting.comprehensive_report import ComprehensiveReportGenerator
from evaluation.validation.significance import StatisticalValidation
from models.model_registry import ModelRegistry


class ExperimentOrchestrator:
    """
    Orchestrates complete research reproduction experiments.

    Handles data loading, model training, evaluation, statistical testing,
    and report generation with progress tracking and error handling.
    """

    def __init__(self, config_path: str, output_dir: str = "results", log_level: str = "INFO"):
        """
        Initialize experiment orchestrator.

        Args:
            config_path: Path to experiment configuration file
            output_dir: Directory for experiment results
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.log_level = log_level

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Load configuration
        self.config = self._load_config()

        # Initialize components
        self.data_loader = ParquetDataLoader()
        self.model_registry = ModelRegistry()
        self.statistical_validator = StatisticalValidation()
        self.report_generator = ComprehensiveReportGenerator()

        # Set random seeds for reproducibility
        self._set_random_seeds(self.config.get("random_seed", 42))

        self.logger.info(f"Initialized ExperimentOrchestrator for: {self.config['name']}")

    def _setup_logging(self) -> None:
        """Setup comprehensive logging configuration."""
        # Create logs directory
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        # Configure logging
        log_file = log_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_config(self) -> dict:
        """Load experiment configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Validate required configuration keys
        required_keys = ['name', 'models', 'evaluation', 'output']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

        self.logger.info(f"Loaded configuration: {config['name']}")
        return config

    def _set_random_seeds(self, seed: int) -> None:
        """Set random seeds for reproducible results."""
        np.random.seed(seed)

        # Set PyTorch seeds if available
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            self.logger.warning("PyTorch not available - skipping PyTorch seed setting")

        # Set Python random seed
        import random
        random.seed(seed)

        self.logger.info(f"Set random seeds to: {seed}")

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all required data for experiments.

        Returns:
            Tuple of (returns, prices, universe_calendar)
        """
        self.logger.info("Loading data for experiments...")

        try:
            # Load processed data
            returns = self.data_loader.load_returns()
            prices = self.data_loader.load_prices()
            universe_calendar = self.data_loader.load_universe_calendar()

            # Validate data integrity
            self._validate_data(returns, prices, universe_calendar)

            self.logger.info(f"Loaded data: {len(returns.columns)} assets, {len(returns)} days")
            return returns, prices, universe_calendar

        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise

    def _validate_data(self, returns: pd.DataFrame, prices: pd.DataFrame,
                      universe_calendar: pd.DataFrame) -> None:
        """Validate data completeness and integrity."""
        # Check data coverage
        coverage = (1 - returns.isnull().mean()).mean()
        if coverage < 0.95:
            self.logger.warning(f"Data coverage below 95%: {coverage:.2%}")

        # Check temporal alignment
        if not returns.index.equals(prices.index):
            raise ValueError("Returns and prices indices not aligned")

        # Check universe calendar consistency
        unique_tickers = set(universe_calendar['ticker'].unique())
        data_tickers = set(returns.columns)

        missing_data = unique_tickers - data_tickers
        if missing_data:
            self.logger.warning(f"Missing data for {len(missing_data)} universe tickers")

        self.logger.info("Data validation completed successfully")

    def initialize_models(self, returns: pd.DataFrame) -> dict:
        """
        Initialize all models specified in configuration.

        Args:
            returns: Historical returns data for model initialization

        Returns:
            Dictionary of initialized models
        """
        self.logger.info("Initializing models...")

        models = {}

        for model_config in self.config['models']:
            model_name = model_config['name']
            config_path = model_config['config']

            try:
                # Load model-specific configuration
                model_cfg = ConfigLoader.load_config(config_path)

                # Override with experiment-specific parameters
                if 'parameters' in model_config:
                    model_cfg.update(model_config['parameters'])

                # Create model instance
                model = self.model_registry.create_model(model_name, model_cfg)
                models[model_name] = model

                self.logger.info(f"Initialized model: {model_name}")

            except Exception as e:
                self.logger.error(f"Failed to initialize model {model_name}: {str(e)}")
                raise

        self.logger.info(f"Initialized {len(models)} models successfully")
        return models

    def run_rolling_backtest(self, models: dict, returns: pd.DataFrame,
                           universe_calendar: pd.DataFrame) -> dict:
        """
        Execute rolling backtest for all models.

        Args:
            models: Dictionary of initialized models
            returns: Historical returns data
            universe_calendar: Dynamic universe membership data

        Returns:
            Dictionary of backtest results by model
        """
        self.logger.info("Starting rolling backtest execution...")

        # Extract backtest configuration
        backtest_config = self.config['evaluation']['backtest']

        # Initialize backtest engine
        engine = RollingBacktestEngine(
            models=list(models.values()),
            start_date=backtest_config['start_date'],
            end_date=backtest_config['end_date'],
            training_window=backtest_config.get('training_window', 756),
            rebalance_frequency=backtest_config.get('rebalance_frequency', 'monthly'),
            transaction_cost_bps=backtest_config.get('transaction_cost_bps', 10.0)
        )

        try:
            # Execute backtest
            results = engine.run_backtest(
                returns=returns,
                universe_calendar=universe_calendar
            )

            # Calculate performance metrics
            performance = engine.calculate_performance_metrics(results)

            # Save intermediate results
            self._save_backtest_results(results, performance)

            self.logger.info("Rolling backtest completed successfully")
            return {'results': results, 'performance': performance}

        except Exception as e:
            self.logger.error(f"Rolling backtest failed: {str(e)}")
            raise

    def run_statistical_analysis(self, backtest_results: dict) -> dict:
        """
        Execute comprehensive statistical analysis.

        Args:
            backtest_results: Results from rolling backtest

        Returns:
            Dictionary of statistical test results
        """
        self.logger.info("Starting statistical analysis...")

        # Extract statistical testing configuration
        stats_config = self.config['evaluation'].get('statistical_tests', {})

        try:
            performance_data = backtest_results['performance']

            # Sharpe ratio significance testing
            sharpe_tests = self.statistical_validator.test_sharpe_ratios(
                performance_data,
                significance_level=stats_config.get('significance_level', 0.05),
                method="jobson_korkie"
            )

            # Bootstrap confidence intervals
            bootstrap_results = self.statistical_validator.bootstrap_confidence_intervals(
                performance_data,
                n_bootstrap=stats_config.get('bootstrap_iterations', 10000),
                confidence_level=stats_config.get('confidence_level', 0.95)
            )

            # Multiple comparison corrections
            corrected_p_values = self.statistical_validator.apply_multiple_corrections(
                sharpe_tests['p_values'],
                method=stats_config.get('correction_method', 'holm_sidak')
            )

            statistical_results = {
                'sharpe_tests': sharpe_tests,
                'bootstrap_ci': bootstrap_results,
                'multiple_corrections': corrected_p_values
            }

            # Save statistical results
            self._save_statistical_results(statistical_results)

            self.logger.info("Statistical analysis completed successfully")
            return statistical_results

        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {str(e)}")
            raise

    def generate_reports(self, backtest_results: dict,
                        statistical_results: dict) -> None:
        """
        Generate comprehensive research reports.

        Args:
            backtest_results: Results from rolling backtest
            statistical_results: Results from statistical analysis
        """
        self.logger.info("Generating comprehensive reports...")

        try:
            # Extract output configuration
            output_config = self.config['output']

            # Generate comprehensive report
            self.report_generator.generate_comprehensive_report(
                performance_results=backtest_results['performance'],
                statistical_tests=statistical_results,
                output_path=self.output_dir / "comprehensive_report.html",
                include_interactive=True
            )

            # Generate additional formats if specified
            export_formats = output_config.get('export_formats', ['html'])

            for format_type in export_formats:
                if format_type == 'pdf':
                    self._generate_pdf_report(backtest_results, statistical_results)
                elif format_type == 'csv':
                    self._export_csv_results(backtest_results, statistical_results)
                elif format_type == 'latex':
                    self._export_latex_tables(backtest_results, statistical_results)

            self.logger.info("Report generation completed successfully")

        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            raise

    def run_complete_experiment(self) -> None:
        """Execute complete research reproduction experiment."""
        start_time = time.time()

        try:
            self.logger.info(f"Starting complete experiment: {self.config['name']}")

            # Step 1: Load data
            returns, prices, universe_calendar = self.load_data()

            # Step 2: Initialize models
            models = self.initialize_models(returns)

            # Step 3: Run rolling backtest
            backtest_results = self.run_rolling_backtest(
                models, returns, universe_calendar
            )

            # Step 4: Statistical analysis
            statistical_results = self.run_statistical_analysis(backtest_results)

            # Step 5: Generate reports
            self.generate_reports(backtest_results, statistical_results)

            # Log completion
            duration = time.time() - start_time
            self.logger.info(f"Experiment completed successfully in {duration:.1f} seconds")

            # Save experiment summary
            self._save_experiment_summary(duration, backtest_results, statistical_results)

        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            self._save_error_report(str(e))
            raise

    def _save_backtest_results(self, results: dict, performance: dict) -> None:
        """Save backtest results to disk."""
        results_dir = self.output_dir / "backtest_results"
        results_dir.mkdir(exist_ok=True)

        # Save performance metrics
        performance_df = pd.DataFrame(performance)
        performance_df.to_csv(results_dir / "performance_metrics.csv")

        # Save detailed results
        for model_name, model_results in results.items():
            model_results.to_csv(results_dir / f"{model_name}_detailed_results.csv")

    def _save_statistical_results(self, results: dict) -> None:
        """Save statistical test results to disk."""
        stats_dir = self.output_dir / "statistical_results"
        stats_dir.mkdir(exist_ok=True)

        # Save as YAML for human readability
        with open(stats_dir / "statistical_tests.yaml", 'w') as f:
            yaml.dump(results, f, default_flow_style=False)

        # Save p-values as CSV
        if 'sharpe_tests' in results:
            pd.DataFrame(results['sharpe_tests']).to_csv(
                stats_dir / "sharpe_test_results.csv"
            )

    def _generate_pdf_report(self, backtest_results: dict,
                            statistical_results: dict) -> None:
        """Generate PDF report (placeholder for PDF generation)."""
        self.logger.info("PDF report generation not yet implemented")
        # TODO: Implement PDF generation using reportlab

    def _export_csv_results(self, backtest_results: dict,
                           statistical_results: dict) -> None:
        """Export all results in CSV format."""
        csv_dir = self.output_dir / "csv_export"
        csv_dir.mkdir(exist_ok=True)

        # Export performance metrics
        performance_df = pd.DataFrame(backtest_results['performance'])
        performance_df.to_csv(csv_dir / "performance_summary.csv")

    def _export_latex_tables(self, backtest_results: dict,
                            statistical_results: dict) -> None:
        """Export publication-ready LaTeX tables."""
        latex_dir = self.output_dir / "latex_tables"
        latex_dir.mkdir(exist_ok=True)

        # Generate performance table
        performance_df = pd.DataFrame(backtest_results['performance'])
        latex_table = performance_df.to_latex(
            float_format="%.3f",
            caption="Model Performance Comparison",
            label="tab:performance"
        )

        with open(latex_dir / "performance_table.tex", 'w') as f:
            f.write(latex_table)

    def _save_experiment_summary(self, duration: float, backtest_results: dict,
                                statistical_results: dict) -> None:
        """Save experiment summary and metadata."""
        summary = {
            'experiment_name': self.config['name'],
            'completion_time': datetime.now().isoformat(),
            'duration_seconds': duration,
            'models_evaluated': len(self.config['models']),
            'statistical_significance': self._extract_key_findings(statistical_results),
            'configuration': self.config
        }

        with open(self.output_dir / "experiment_summary.yaml", 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)

    def _extract_key_findings(self, statistical_results: dict) -> dict:
        """Extract key statistical findings for summary."""
        # Placeholder for extracting key statistical findings
        return {
            'significant_differences': "Analysis pending",
            'best_performing_model': "Analysis pending"
        }

    def _save_error_report(self, error_message: str) -> None:
        """Save error report for debugging."""
        error_report = {
            'timestamp': datetime.now().isoformat(),
            'error_message': error_message,
            'configuration': self.config
        }

        with open(self.output_dir / "error_report.yaml", 'w') as f:
            yaml.dump(error_report, f, default_flow_style=False)


def main():
    """Main entry point for research reproduction script."""
    parser = argparse.ArgumentParser(
        description="Reproduce ML-based portfolio optimization research"
    )
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['full_evaluation', 'baseline_comparison', 'sensitivity_analysis'],
        help='Predefined experiment to run'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Custom configuration file path'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    args = parser.parse_args()

    # Determine configuration file
    if args.config:
        config_path = args.config
    elif args.experiment:
        config_path = f"configs/experiments/{args.experiment}.yaml"
    else:
        parser.error("Either --experiment or --config must be specified")

    # Create output directory with experiment name
    if args.experiment:
        output_dir = Path(args.output_dir) / args.experiment
    else:
        output_dir = Path(args.output_dir) / "custom_experiment"

    try:
        # Initialize and run experiment
        orchestrator = ExperimentOrchestrator(
            config_path=config_path,
            output_dir=str(output_dir),
            log_level=args.log_level
        )

        orchestrator.run_complete_experiment()


    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
