#!/usr/bin/env python3
"""
Command-line script to run portfolio optimization experiments.

This script provides a command-line interface for executing
experiments defined in YAML configuration files.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments import run_experiment_from_config


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("experiment.log")],
    )


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(description="Run portfolio optimization experiments")
    parser.add_argument("config", type=Path, help="Path to experiment configuration YAML file")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Validate config file exists
    if not args.config.exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    try:
        logger.info(f"Starting experiment from config: {args.config}")

        # Run experiment
        results = run_experiment_from_config(args.config)

        if "error" in results:
            logger.error(f"Experiment failed: {results['error']}")
            sys.exit(1)

        # Print summary

        for _model_name, _model_data in results.get("models", {}).items():
            pass

        logger.info("Experiment completed successfully")

    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
