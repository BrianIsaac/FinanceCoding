#!/usr/bin/env python3
"""
Example usage of the Unified Model Training Orchestrator.

This script demonstrates different ways to use the unified training pipeline
for training portfolio optimization models with various configurations.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.train_all_models import UnifiedModelTrainer


def example_train_all_models():
    """Example: Train all models with default configuration."""
    print("=" * 60)
    print("Example 1: Training All Models with Default Configuration")
    print("=" * 60)

    trainer = UnifiedModelTrainer(
        output_dir="outputs/examples/train_all",
        gpu_memory_limit_gb=11.0,
        quick_test=True,  # Use quick test for example
    )

    results = trainer.run_comprehensive_training()

    print(f"\nTraining completed! Results summary:")
    for model_type, model_results in results.items():
        success_count = sum(1 for r in model_results.values() if r.get("status") == "success")
        total_count = len(model_results)
        print(f"{model_type.upper()}: {success_count}/{total_count} configurations successful")


def example_train_specific_models():
    """Example: Train only specific model types."""
    print("=" * 60)
    print("Example 2: Training Only HRP and LSTM Models")
    print("=" * 60)

    trainer = UnifiedModelTrainer(
        output_dir="outputs/examples/train_specific",
        gpu_memory_limit_gb=11.0,
        quick_test=True,
    )

    # Train only HRP and LSTM models
    results = trainer.run_comprehensive_training(models=["hrp", "lstm"])

    print(f"\nSpecific model training completed!")


def example_train_with_custom_config():
    """Example: Train with custom configuration file."""
    print("=" * 60)
    print("Example 3: Training with Custom Configuration")
    print("=" * 60)

    trainer = UnifiedModelTrainer(
        config_path="configs/experiments/training_config.yaml",
        output_dir="outputs/examples/train_custom",
        gpu_memory_limit_gb=11.0,
        quick_test=True,
    )

    results = trainer.run_comprehensive_training()

    print(f"\nCustom configuration training completed!")


def example_production_training():
    """Example: Full production training (commented out - takes longer)."""
    print("=" * 60)
    print("Example 4: Production Training Setup (Not Executed)")
    print("=" * 60)

    print("""
    # For production training, use:

    trainer = UnifiedModelTrainer(
        config_path="configs/experiments/training_config.yaml",
        output_dir="outputs/production/training",
        gpu_memory_limit_gb=11.0,
        quick_test=False,  # Full dataset training
    )

    results = trainer.run_comprehensive_training()

    # This would train all models with full datasets and take several hours
    """)


def main():
    """Run all training examples."""
    print("Unified Model Training Orchestrator - Examples")
    print("=" * 60)

    try:
        # Run examples (using quick_test=True for demonstration)
        example_train_all_models()
        print("\n" + "=" * 60 + "\n")

        example_train_specific_models()
        print("\n" + "=" * 60 + "\n")

        example_train_with_custom_config()
        print("\n" + "=" * 60 + "\n")

        example_production_training()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("Check outputs/examples/ for training results")
        print("=" * 60)

    except Exception as e:
        print(f"Example execution failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())