#!/usr/bin/env python3
"""Manual verification for Phase 8: GAT Time-Series Node Features - Configuration.

This script verifies:
1. Paper preset creates correct time-series configuration
2. Enhanced preset still uses static features
3. Configuration validation logic works correctly
4. Time-series GAT can be instantiated with paper preset
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models.gat.model import GATModelConfig, GATPortfolioModel
from src.models.gat.gat_model import GATPortfolio


def test_paper_preset_configuration():
    """Test that paper preset creates correct time-series configuration."""
    print("\n" + "=" * 80)
    print("TEST 1: Paper Preset Configuration")
    print("=" * 80)

    config = GATModelConfig(preset="paper_reproduction")

    # Check all time-series related parameters
    checks = {
        "node_feature_type": ("timeseries", config.node_feature_type),
        "timeseries_length": (756, config.timeseries_length),
        "timeseries_features": (["volatility"], config.timeseries_features),
        "use_temporal_encoder": (True, config.use_temporal_encoder),
        "temporal_encoder_type": ("lstm", config.temporal_encoder_type),
        "temporal_encoder_hidden": (64, config.temporal_encoder_hidden),
        "temporal_encoder_layers": (1, config.temporal_encoder_layers),
    }

    all_passed = True
    for param_name, (expected, actual) in checks.items():
        passed = expected == actual
        status = "✓" if passed else "✗"
        print(f"{status} {param_name}: expected={expected}, actual={actual}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✅ TEST 1 PASSED: Paper preset correctly configured for time-series")
    else:
        print("\n❌ TEST 1 FAILED: Paper preset configuration mismatch")

    return all_passed


def test_enhanced_preset_static():
    """Test that enhanced preset still uses static features."""
    print("\n" + "=" * 80)
    print("TEST 2: Enhanced Preset Uses Static Features")
    print("=" * 80)

    config = GATModelConfig(preset="enhanced")

    expected = "static"
    actual = config.node_feature_type
    passed = expected == actual

    status = "✓" if passed else "✗"
    print(f"{status} node_feature_type: expected={expected}, actual={actual}")

    if passed:
        print("\n✅ TEST 2 PASSED: Enhanced preset uses static features")
    else:
        print("\n❌ TEST 2 FAILED: Enhanced preset configuration changed")

    return passed


def test_configuration_validation():
    """Test configuration validation logic."""
    print("\n" + "=" * 80)
    print("TEST 3: Configuration Validation Logic")
    print("=" * 80)

    all_passed = True

    # Test 3.1: Invalid temporal encoder type (only validated when node_feature_type="timeseries")
    print("\nTest 3.1: Invalid temporal encoder type should raise ValueError when using timeseries")
    try:
        config = GATModelConfig(
            node_feature_type="timeseries",  # Must enable timeseries for validation
            temporal_encoder_type="invalid"
        )
        print("✗ Should have raised ValueError for invalid encoder type")
        all_passed = False
    except ValueError as e:
        if "temporal_encoder_type must be one of" in str(e):
            print(f"✓ Correctly raised ValueError: {e}")
        else:
            print(f"✗ Raised ValueError but with wrong message: {e}")
            all_passed = False

    # Test 3.2: Invalid timeseries_length (only validated when node_feature_type="timeseries")
    print("\nTest 3.2: Invalid timeseries_length should raise ValueError when using timeseries")
    try:
        config = GATModelConfig(
            node_feature_type="timeseries",  # Must enable timeseries for validation
            timeseries_length=-10
        )
        print("✗ Should have raised ValueError for negative timeseries_length")
        all_passed = False
    except ValueError as e:
        if "timeseries_length must be positive" in str(e):
            print(f"✓ Correctly raised ValueError: {e}")
        else:
            print(f"✗ Raised ValueError but with wrong message: {e}")
            all_passed = False

    # Test 3.3: Valid configuration with timeseries
    print("\nTest 3.3: Valid time-series configuration should not raise errors")
    try:
        config = GATModelConfig(
            node_feature_type="timeseries",
            timeseries_length=60,
            temporal_encoder_type="conv1d",
        )
        print("✓ Valid time-series configuration accepted")
    except Exception as e:
        print(f"✗ Valid configuration raised error: {e}")
        all_passed = False

    # Test 3.4: Auto-enable temporal encoder
    print("\nTest 3.4: Temporal encoder should auto-enable for timeseries type")
    config = GATModelConfig(
        node_feature_type="timeseries",
        use_temporal_encoder=False,  # Should be overridden
    )
    if config.use_temporal_encoder:
        print("✓ Temporal encoder auto-enabled for timeseries type")
    else:
        print("✗ Temporal encoder not auto-enabled")
        all_passed = False

    if all_passed:
        print("\n✅ TEST 3 PASSED: Configuration validation logic works correctly")
    else:
        print("\n❌ TEST 3 FAILED: Configuration validation issues found")

    return all_passed


def test_gat_instantiation():
    """Test that time-series GAT can be instantiated with paper preset.

    Note: This test only verifies instantiation and basic model structure.
    Full end-to-end testing with forward passes is covered in Phase 9.
    """
    print("\n" + "=" * 80)
    print("TEST 4: GAT Instantiation with Paper Preset")
    print("=" * 80)

    try:
        # Create configuration
        config = GATModelConfig(preset="paper_reproduction")
        print(f"✓ Paper preset configuration created")

        # Create GATPortfolio model (low-level)
        print("\nAttempting to create GATPortfolio...")
        gat_model = GATPortfolio(
            in_dim=1,  # Number of input features per timestep
            hidden_dim=config.hidden_dim,
            heads=config.num_attention_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_gatv2=config.use_gatv2,
            residual=config.residual,
            use_temporal_encoder=config.use_temporal_encoder,
            temporal_encoder_type=config.temporal_encoder_type,
            timeseries_length=config.timeseries_length,
        )
        print(f"✓ GATPortfolio created successfully")

        # Check model has temporal encoder
        if hasattr(gat_model, "temporal_encoder") and gat_model.temporal_encoder is not None:
            print(f"✓ Model has temporal encoder: {type(gat_model.temporal_encoder).__name__}")
        else:
            print("✗ Model missing temporal encoder")
            return False

        # Verify temporal encoder can process time-series data
        print("\nTesting temporal encoder with dummy time-series data...")
        batch_size = 5
        seq_length = 60  # Using 60 instead of 756 for quick test
        num_features = 1

        # Create dummy time-series node features
        x_timeseries = torch.randn(batch_size, seq_length, num_features)

        # Test temporal encoder directly (not full forward pass)
        with torch.no_grad():
            encoded = gat_model.temporal_encoder(x_timeseries)

        print(f"✓ Temporal encoder forward pass successful")
        print(f"  Input shape: {x_timeseries.shape}")
        print(f"  Encoded shape: {encoded.shape}")
        print(f"  Expected encoded dim: {config.hidden_dim}")

        # Verify output properties
        if encoded.shape[0] == batch_size:
            print(f"✓ Output batch size correct: {encoded.shape[0]}")
        else:
            print(f"✗ Output batch size incorrect: expected {batch_size}, got {encoded.shape[0]}")
            return False

        if encoded.shape[1] == config.hidden_dim:
            print(f"✓ Output hidden dim correct: {encoded.shape[1]}")
        else:
            print(f"✗ Output hidden dim incorrect: expected {config.hidden_dim}, got {encoded.shape[1]}")
            return False

        if torch.isfinite(encoded).all():
            print(f"✓ Encoded output contains no NaN/Inf")
        else:
            print(f"✗ Encoded output contains NaN/Inf")
            return False

        print("\n✅ TEST 4 PASSED: Time-series GAT instantiation successful")
        print("Note: Full end-to-end forward pass testing is covered in Phase 9")
        return True

    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all manual verification tests."""
    print("=" * 80)
    print("PHASE 8 MANUAL VERIFICATION")
    print("GAT Time-Series Node Features - Configuration")
    print("=" * 80)

    results = {
        "test_paper_preset_configuration": test_paper_preset_configuration(),
        "test_enhanced_preset_static": test_enhanced_preset_static(),
        "test_configuration_validation": test_configuration_validation(),
        "test_gat_instantiation": test_gat_instantiation(),
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL PHASE 8 MANUAL VERIFICATION TESTS PASSED")
        print("=" * 80)
        print("\nPhase 8 is complete. Ready to proceed to Phase 9.")
        return 0
    else:
        print("❌ SOME PHASE 8 MANUAL VERIFICATION TESTS FAILED")
        print("=" * 80)
        print("\nPlease review failures before proceeding to Phase 9.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
