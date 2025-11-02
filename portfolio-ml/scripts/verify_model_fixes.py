#!/usr/bin/env python3
"""
Verification script for LSTM and GAT model fixes.
Tests that gradient flow works and NaN losses are resolved.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_lstm_gradients():
    """Test that LSTM model can compute gradients properly."""
    print("\n" + "="*60)
    print("Testing LSTM Gradient Flow")
    print("="*60)

    from src.models.lstm.architecture import SharpeRatioLoss
    import torch.nn.functional as F

    # Create dummy data
    batch_size = 32
    n_assets = 100

    # Simulate model predictions and actual returns
    predicted_returns = torch.randn(batch_size, n_assets, requires_grad=True)
    actual_returns = torch.randn(batch_size, n_assets)

    # Create loss function
    criterion = SharpeRatioLoss()

    # Forward pass
    loss = criterion(predicted_returns, actual_returns)

    # Check if loss is valid
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    print(f"✅ Loss computed successfully: {loss.item():.4f}")

    # Backward pass
    try:
        loss.backward()

        # Check if gradients exist
        if predicted_returns.grad is not None:
            grad_norm = predicted_returns.grad.norm().item()
            print(f"✅ Gradients computed successfully!")
            print(f"   Gradient norm: {grad_norm:.6f}")
            print(f"   Gradient mean: {predicted_returns.grad.mean().item():.6f}")
            print(f"   Gradient std: {predicted_returns.grad.std().item():.6f}")

            # Check gradients are reasonable
            assert grad_norm > 0, "Gradients are zero!"
            assert grad_norm < 1000, f"Gradients are exploding: {grad_norm}"
            print(f"✅ Gradient magnitudes are reasonable")

            return True
        else:
            print("❌ No gradients computed!")
            return False

    except Exception as e:
        print(f"❌ Backward pass failed: {str(e)}")
        return False


def test_lstm_model_training():
    """Test that LSTM model can perform a training step."""
    print("\n" + "="*60)
    print("Testing LSTM Model Training")
    print("="*60)

    from src.models.lstm.model import LSTMPortfolioModel, LSTMConfig
    from src.models.base.constraints import PortfolioConstraints

    # Create constraints
    constraints = PortfolioConstraints(
        max_position_weight=0.20,
        min_weight_threshold=0.001,
        long_only=True
    )

    # Create model
    config = LSTMConfig()
    model = LSTMPortfolioModel(constraints=constraints, config=config)

    # Create dummy data
    n_days = 100
    n_assets = 50
    universe = [f"ASSET_{i}" for i in range(n_assets)]

    # Create returns DataFrame
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    returns_data = pd.DataFrame(
        np.random.randn(n_days, n_assets) * 0.01,  # 1% daily returns
        index=dates,
        columns=universe
    )

    # Set up training period
    fit_period = {
        'start': dates[0],
        'end': dates[-1]
    }

    try:
        # Attempt to fit model
        print("Attempting to train LSTM model...")
        model.fit(returns_data, universe, fit_period)

        # Check if model trained
        if hasattr(model, 'model') and model.model is not None:
            # Check for non-zero parameters
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

            print(f"✅ Model created successfully")
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")

            # Make a prediction to verify forward pass
            test_date = dates[-1]
            weights = model.predict_weights(test_date, universe)

            if weights is not None and len(weights) > 0:
                print(f"✅ Model can make predictions")
                print(f"   Weight sum: {weights.sum():.4f}")
                print(f"   Max weight: {weights.max():.4f}")
                print(f"   Non-zero weights: {(weights > 1e-6).sum()}/{len(weights)}")
                return True
            else:
                print("❌ Model prediction failed")
                return False
        else:
            print("❌ Model not created properly")
            return False

    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_gat_nan_free():
    """Test that GAT model doesn't produce NaN losses."""
    print("\n" + "="*60)
    print("Testing GAT NaN-Free Training")
    print("="*60)

    from src.models.gat.model import GATPortfolioModel, GATModelConfig
    from src.models.base.constraints import PortfolioConstraints

    # Create constraints
    constraints = PortfolioConstraints(
        max_position_weight=0.20,
        min_weight_threshold=0.001,
        long_only=True
    )

    # Create model with enhanced preset
    config = GATModelConfig(preset="enhanced")
    model = GATPortfolioModel(constraints=constraints, config=config)

    # Create dummy data
    n_days = 300  # Need enough for GAT's lookback
    n_assets = 100
    universe = [f"ASSET_{i}" for i in range(n_assets)]

    # Create returns DataFrame
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
    returns_data = pd.DataFrame(
        np.random.randn(n_days, n_assets) * 0.01,
        index=dates,
        columns=universe
    )

    # Set up training period
    fit_period = {
        'start': dates[0],
        'end': dates[-1]
    }

    try:
        print("Attempting to train GAT model...")

        # Mock a quick training iteration to check for NaN
        model.fit(returns_data, universe, fit_period)

        # Check if model trained without NaN
        if hasattr(model, 'model') and model.model is not None:
            print(f"✅ GAT model created successfully")

            # Make a prediction
            test_date = dates[-1]
            weights = model.predict_weights(test_date, universe)

            # Check for NaN in weights
            if weights is not None:
                has_nan = np.isnan(weights.values).any() if hasattr(weights, 'values') else np.isnan(weights).any()

                if not has_nan:
                    print(f"✅ No NaN values in predictions!")
                    print(f"   Weight sum: {weights.sum():.4f}")
                    print(f"   Max weight: {weights.max():.4f}")
                    return True
                else:
                    print("❌ NaN values detected in predictions!")
                    return False
            else:
                print("❌ No weights returned")
                return False
        else:
            print("❌ Model not created")
            return False

    except Exception as e:
        print(f"❌ GAT training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_gat_loss_stability():
    """Test GAT loss computation stability."""
    print("\n" + "="*60)
    print("Testing GAT Loss Stability")
    print("="*60)

    from src.models.gat.loss_functions import SharpeRatioLoss

    # Test with various edge cases
    test_cases = [
        ("Normal case", torch.randn(10, 50), torch.randn(10, 20, 50)),
        ("Small returns", torch.randn(10, 50) * 0.0001, torch.randn(10, 20, 50) * 0.0001),
        ("Zero std", torch.ones(10, 50) / 50, torch.ones(10, 20, 50) * 0.001),
        ("Large returns", torch.randn(10, 50) * 10, torch.randn(10, 20, 50) * 10),
    ]

    criterion = SharpeRatioLoss(formulation="standard")

    all_passed = True
    for name, weights, returns in test_cases:
        try:
            # Normalise weights
            weights = torch.softmax(weights, dim=-1)

            loss = criterion(weights, returns)

            is_finite = torch.isfinite(loss).item()
            if is_finite:
                print(f"✅ {name}: Loss = {loss.item():.4f}")
            else:
                print(f"❌ {name}: Loss is not finite (NaN or Inf)")
                all_passed = False

        except Exception as e:
            print(f"❌ {name}: Failed with error: {str(e)}")
            all_passed = False

    return all_passed


def main():
    """Run all verification tests."""
    print("\n" + "="*60)
    print("MODEL FIX VERIFICATION SUITE")
    print("="*60)

    tests = [
        ("LSTM Gradient Flow", test_lstm_gradients),
        ("LSTM Model Training", test_lstm_model_training),
        ("GAT Loss Stability", test_gat_loss_stability),
        ("GAT NaN-Free Training", test_gat_nan_free),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {str(e)}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print("="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Models are ready for training.")
    else:
        print("⚠️ Some tests failed. Please review the issues above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())