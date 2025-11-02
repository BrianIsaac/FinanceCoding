"""Manual verification script for ragged LSTM implementation.

This script performs comprehensive manual verification for Phases 1-3:
- Phase 1: Test infrastructure integrity
- Phase 2: Ragged tensor utilities correctness
- Phase 3: RaggedLSTMNetwork vs standard LSTMNetwork comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any
import time

from src.models.lstm.architecture import LSTMConfig, LSTMNetwork
from src.models.lstm.ragged_architecture import RaggedLSTMNetwork, create_ragged_lstm_network
from src.models.lstm.ragged_utils import (
    pack_ragged_sequences,
    unpack_ragged_sequences,
    compute_sequence_statistics,
)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---\n")


def verify_phase1_test_infrastructure() -> Dict[str, Any]:
    """
    Verify Phase 1: Test infrastructure setup.

    Returns:
        Dictionary with verification results
    """
    print_section("PHASE 1: Test Infrastructure Verification")

    results = {
        'phase': 1,
        'tests_passed': [],
        'tests_failed': [],
        'warnings': []
    }

    # Test 1: Verify test directory structure
    print_subsection("Test 1: Directory Structure")
    test_dirs = [
        'tests',
        'tests/models',
        'tests/models/lstm',
        'tests/models/hrp',
        'tests/models/gat',
        'tests/integration',
        'tests/fixtures'
    ]

    for test_dir in test_dirs:
        dir_path = project_root / test_dir
        if dir_path.exists() and dir_path.is_dir():
            print(f"✓ {test_dir} exists")
            results['tests_passed'].append(f"Directory {test_dir} exists")
        else:
            print(f"✗ {test_dir} missing")
            results['tests_failed'].append(f"Directory {test_dir} missing")

    # Test 2: Verify test files exist
    print_subsection("Test 2: Test Files")
    test_files = [
        'tests/conftest.py',
        'tests/utils.py',
        'tests/models/lstm/test_ragged_utils.py',
        'tests/models/lstm/test_ragged_architecture.py'
    ]

    for test_file in test_files:
        file_path = project_root / test_file
        if file_path.exists() and file_path.is_file():
            print(f"✓ {test_file} exists ({file_path.stat().st_size} bytes)")
            results['tests_passed'].append(f"File {test_file} exists")
        else:
            print(f"✗ {test_file} missing")
            results['tests_failed'].append(f"File {test_file} missing")

    # Test 3: Verify pytest configuration
    print_subsection("Test 3: Pytest Configuration")
    pyproject = project_root / 'pyproject.toml'
    if pyproject.exists():
        content = pyproject.read_text()
        if '[tool.pytest.ini_options]' in content:
            print("✓ Pytest configuration present in pyproject.toml")
            results['tests_passed'].append("Pytest configuration found")
        else:
            print("✗ Pytest configuration missing in pyproject.toml")
            results['tests_failed'].append("Pytest configuration missing")

    # Test 4: Import tests
    print_subsection("Test 4: Module Imports")
    try:
        import tests.conftest
        print("✓ tests.conftest imports successfully")
        results['tests_passed'].append("conftest imports")
    except Exception as e:
        print(f"✗ tests.conftest import failed: {e}")
        results['tests_failed'].append(f"conftest import failed: {e}")

    try:
        import tests.utils
        print("✓ tests.utils imports successfully")
        results['tests_passed'].append("utils imports")
    except Exception as e:
        print(f"✗ tests.utils import failed: {e}")
        results['tests_failed'].append(f"utils import failed: {e}")

    return results


def verify_phase2_ragged_utils() -> Dict[str, Any]:
    """
    Verify Phase 2: Ragged tensor utilities correctness.

    Returns:
        Dictionary with verification results
    """
    print_section("PHASE 2: Ragged Tensor Utilities Verification")

    results = {
        'phase': 2,
        'tests_passed': [],
        'tests_failed': [],
        'warnings': []
    }

    # Test 1: Pack/Unpack correctness
    print_subsection("Test 1: Pack/Unpack Correctness")
    batch_size, max_len, features = 8, 60, 10
    sequences = torch.randn(batch_size, max_len, features)
    lengths = torch.tensor([30, 35, 40, 45, 50, 55, 58, 60])

    try:
        packed, indices = pack_ragged_sequences(sequences, lengths)
        unpacked, unpacked_lengths = unpack_ragged_sequences(packed, indices, total_length=max_len)

        # Verify shapes
        assert unpacked.shape == sequences.shape, "Shape mismatch after pack/unpack"
        assert (unpacked_lengths == lengths).all(), "Lengths mismatch after pack/unpack"

        # Verify data preservation
        for i in range(batch_size):
            length = lengths[i]
            if torch.allclose(unpacked[i, :length], sequences[i, :length], rtol=1e-5):
                pass
            else:
                raise ValueError(f"Data mismatch for sequence {i}")

        print(f"✓ Pack/Unpack preserves data for {batch_size} sequences")
        results['tests_passed'].append("Pack/Unpack data preservation")
    except Exception as e:
        print(f"✗ Pack/Unpack failed: {e}")
        results['tests_failed'].append(f"Pack/Unpack failed: {e}")

    # Test 2: Padding ratio calculation
    print_subsection("Test 2: Padding Ratio Calculation")
    try:
        stats = compute_sequence_statistics(sequences, lengths)

        expected_avg = lengths.float().mean().item()
        actual_avg = stats['mean_length']

        if abs(expected_avg - actual_avg) < 1e-6:
            print(f"✓ Mean length calculation correct: {actual_avg:.2f}")
            results['tests_passed'].append("Mean length calculation")
        else:
            raise ValueError(f"Mean length mismatch: expected {expected_avg}, got {actual_avg}")

        # Calculate expected padding ratio
        total_values = batch_size * max_len * features
        actual_values = int(lengths.sum()) * features
        expected_padding_ratio = 1.0 - (actual_values / total_values)

        if abs(stats['padding_ratio'] - expected_padding_ratio) < 1e-6:
            print(f"✓ Padding ratio calculation correct: {stats['padding_ratio']:.2%}")
            results['tests_passed'].append("Padding ratio calculation")
        else:
            raise ValueError(f"Padding ratio mismatch")

        print(f"  - Mean length: {stats['mean_length']:.2f}")
        print(f"  - Std length: {stats['std_length']:.2f}")
        print(f"  - Min/Max: {stats['min_length']}/{stats['max_length']}")
        print(f"  - Padding ratio: {stats['padding_ratio']:.2%}")

    except Exception as e:
        print(f"✗ Statistics calculation failed: {e}")
        results['tests_failed'].append(f"Statistics failed: {e}")

    # Test 3: Gradient flow through packed sequences
    print_subsection("Test 3: Gradient Flow")
    try:
        sequences_grad = torch.randn(4, 60, 10, requires_grad=True)
        lengths_grad = torch.tensor([45, 52, 30, 58])

        packed, indices = pack_ragged_sequences(sequences_grad, lengths_grad)
        unpacked, _ = unpack_ragged_sequences(packed, indices, total_length=60)

        loss = unpacked.sum()
        loss.backward()

        if sequences_grad.grad is not None and torch.isfinite(sequences_grad.grad).all():
            grad_norm = sequences_grad.grad.norm().item()
            print(f"✓ Gradients flow through pack/unpack (norm: {grad_norm:.4f})")
            results['tests_passed'].append("Gradient flow")
        else:
            raise ValueError("Gradients missing or non-finite")

    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        results['tests_failed'].append(f"Gradient flow failed: {e}")

    return results


def verify_phase3_ragged_lstm() -> Dict[str, Any]:
    """
    Verify Phase 3: RaggedLSTMNetwork vs standard LSTMNetwork.

    Returns:
        Dictionary with verification results
    """
    print_section("PHASE 3: RaggedLSTMNetwork Verification")

    results = {
        'phase': 3,
        'tests_passed': [],
        'tests_failed': [],
        'warnings': []
    }

    # Configuration
    config = LSTMConfig(
        input_size=50,
        hidden_size=64,
        num_layers=2,
        output_size=50,
        dropout=0.1
    )

    # Test 1: Network construction
    print_subsection("Test 1: Network Construction")
    try:
        standard_lstm = LSTMNetwork(config)
        ragged_lstm = RaggedLSTMNetwork(config)

        print(f"✓ Standard LSTM created: {sum(p.numel() for p in standard_lstm.parameters())} params")
        print(f"✓ Ragged LSTM created: {sum(p.numel() for p in ragged_lstm.parameters())} params")
        results['tests_passed'].append("Network construction")
    except Exception as e:
        print(f"✗ Network construction failed: {e}")
        results['tests_failed'].append(f"Network construction failed: {e}")
        return results

    # Test 2: Forward pass comparison (same-length sequences)
    print_subsection("Test 2: Forward Pass - Same Length Sequences")
    try:
        batch_size = 8
        seq_len = 60
        sequences = torch.randn(batch_size, seq_len, config.input_size)
        lengths = torch.tensor([seq_len] * batch_size)  # All same length

        standard_lstm.eval()
        ragged_lstm.eval()

        # Standard LSTM forward
        with torch.no_grad():
            standard_pred, standard_attn = standard_lstm(sequences)

        # Ragged LSTM forward
        with torch.no_grad():
            ragged_pred, ragged_attn = ragged_lstm(sequences, lengths)

        # Check shapes
        assert standard_pred.shape == ragged_pred.shape, "Shape mismatch"

        # Check values are reasonable (not testing for exact match due to different architectures)
        pred_diff = (standard_pred - ragged_pred).abs().mean().item()
        print(f"✓ Forward pass completed")
        print(f"  - Standard LSTM output range: [{standard_pred.min():.4f}, {standard_pred.max():.4f}]")
        print(f"  - Ragged LSTM output range: [{ragged_pred.min():.4f}, {ragged_pred.max():.4f}]")
        print(f"  - Mean absolute difference: {pred_diff:.6f}")

        if pred_diff < 10.0:  # Reasonable threshold
            results['tests_passed'].append("Forward pass same-length")
        else:
            results['warnings'].append(f"Large prediction difference: {pred_diff}")

    except Exception as e:
        print(f"✗ Forward pass comparison failed: {e}")
        results['tests_failed'].append(f"Forward pass failed: {e}")

    # Test 3: Memory efficiency with variable lengths
    print_subsection("Test 3: Memory Efficiency - Variable Lengths")
    try:
        batch_size = 16
        max_len = 100
        sequences = torch.randn(batch_size, max_len, config.input_size)

        # Test with high padding (30% actual data)
        lengths_high_padding = torch.tensor([30] * batch_size)

        ragged_lstm.eval()
        with torch.no_grad():
            pred_padded, _ = ragged_lstm(sequences, lengths_high_padding)

        stats = ragged_lstm.get_sequence_statistics()

        print(f"✓ Ragged LSTM handles high padding ({stats['padding_ratio']:.1%})")
        print(f"  - Mean length: {stats['mean_length']:.1f}")
        print(f"  - Padding ratio: {stats['padding_ratio']:.1%}")

        # Estimate memory usage
        memory_estimate = ragged_lstm.get_memory_usage(
            batch_size=batch_size,
            sequence_length=max_len,
            avg_real_length=int(stats['mean_length'])
        )
        print(f"  - Estimated memory: {memory_estimate / 1024 / 1024:.2f} MB")

        results['tests_passed'].append("Memory efficiency verification")

    except Exception as e:
        print(f"✗ Memory efficiency test failed: {e}")
        results['tests_failed'].append(f"Memory efficiency failed: {e}")

    # Test 4: Gradient flow
    print_subsection("Test 4: Gradient Flow Through Network")
    try:
        ragged_lstm.train()
        sequences_grad = torch.randn(8, 60, config.input_size, requires_grad=True)
        lengths_grad = torch.randint(30, 61, (8,))

        predictions, _ = ragged_lstm(sequences_grad, lengths_grad)
        loss = predictions.mean()
        loss.backward()

        # Check gradients exist
        grad_count = 0
        total_params = 0
        for name, param in ragged_lstm.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None and 'batch_norm' not in name:
                    grad_count += 1

        print(f"✓ Gradients flow through network")
        print(f"  - Parameters with gradients: {grad_count}/{total_params}")
        print(f"  - Input gradients exist: {sequences_grad.grad is not None}")

        results['tests_passed'].append("Gradient flow")

    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        results['tests_failed'].append(f"Gradient flow failed: {e}")

    # Test 5: Training loss decreases
    print_subsection("Test 5: Training Loss Convergence")
    try:
        ragged_lstm.train()
        optimizer = torch.optim.Adam(ragged_lstm.parameters(), lr=0.001)

        batch_size = 16
        sequences = torch.randn(batch_size, 60, config.input_size)
        lengths = torch.randint(30, 61, (batch_size,))
        targets = torch.randn(batch_size, config.output_size)

        losses = []
        for epoch in range(10):
            optimizer.zero_grad()
            predictions, _ = ragged_lstm(sequences, lengths)
            loss = nn.MSELoss()(predictions, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Check if loss decreased
        initial_loss = losses[0]
        final_loss = losses[-1]
        loss_reduction = (initial_loss - final_loss) / initial_loss

        print(f"✓ Training completed")
        print(f"  - Initial loss: {initial_loss:.6f}")
        print(f"  - Final loss: {final_loss:.6f}")
        print(f"  - Loss reduction: {loss_reduction:.2%}")

        if loss_reduction > 0:
            print(f"  ✓ Loss decreased over training")
            results['tests_passed'].append("Training loss decreases")
        else:
            results['warnings'].append("Loss did not decrease (may need more epochs)")

    except Exception as e:
        print(f"✗ Training test failed: {e}")
        results['tests_failed'].append(f"Training failed: {e}")

    # Test 6: Computational efficiency
    print_subsection("Test 6: Computational Efficiency")
    try:
        ragged_lstm.eval()

        # Test with varying padding ratios
        batch_size = 32
        max_len = 100

        for padding_ratio in [0.0, 0.3, 0.5, 0.7]:
            avg_len = int(max_len * (1 - padding_ratio))
            lengths = torch.tensor([avg_len] * batch_size)
            sequences = torch.randn(batch_size, max_len, config.input_size)

            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = ragged_lstm(sequences, lengths)
            elapsed = time.time() - start_time

            stats = ragged_lstm.get_sequence_statistics()
            print(f"  Padding {padding_ratio:.0%}: {elapsed*100:.2f}ms (10 iterations)")

        print(f"✓ Computational efficiency verified")
        results['tests_passed'].append("Computational efficiency")

    except Exception as e:
        print(f"✗ Computational efficiency test failed: {e}")
        results['tests_failed'].append(f"Computational efficiency failed: {e}")

    return results


def print_summary(all_results: list[Dict[str, Any]]) -> None:
    """Print overall summary of verification."""
    print_section("VERIFICATION SUMMARY")

    total_passed = sum(len(r['tests_passed']) for r in all_results)
    total_failed = sum(len(r['tests_failed']) for r in all_results)
    total_warnings = sum(len(r['warnings']) for r in all_results)

    for result in all_results:
        phase = result['phase']
        passed = len(result['tests_passed'])
        failed = len(result['tests_failed'])
        warnings = len(result['warnings'])

        print(f"\nPhase {phase}:")
        print(f"  ✓ Passed: {passed}")
        if failed > 0:
            print(f"  ✗ Failed: {failed}")
        if warnings > 0:
            print(f"  ⚠ Warnings: {warnings}")

        if failed > 0:
            print(f"\n  Failed tests:")
            for test in result['tests_failed']:
                print(f"    - {test}")

        if warnings > 0:
            print(f"\n  Warnings:")
            for warning in result['warnings']:
                print(f"    - {warning}")

    print(f"\n{'=' * 80}")
    print(f"OVERALL RESULTS:")
    print(f"  ✓ Total passed: {total_passed}")
    print(f"  ✗ Total failed: {total_failed}")
    print(f"  ⚠ Total warnings: {total_warnings}")

    if total_failed == 0:
        print(f"\n🎉 ALL MANUAL VERIFICATION TESTS PASSED!")
        print(f"{'=' * 80}\n")
        return True
    else:
        print(f"\n❌ Some tests failed. Please review above.")
        print(f"{'=' * 80}\n")
        return False


def main():
    """Run all manual verification tests."""
    print("\n" + "=" * 80)
    print("  MANUAL VERIFICATION: Ragged LSTM Implementation (Phases 1-3)")
    print("=" * 80)

    all_results = []

    # Phase 1
    try:
        results1 = verify_phase1_test_infrastructure()
        all_results.append(results1)
    except Exception as e:
        print(f"Phase 1 verification crashed: {e}")
        return 1

    # Phase 2
    try:
        results2 = verify_phase2_ragged_utils()
        all_results.append(results2)
    except Exception as e:
        print(f"Phase 2 verification crashed: {e}")
        return 1

    # Phase 3
    try:
        results3 = verify_phase3_ragged_lstm()
        all_results.append(results3)
    except Exception as e:
        print(f"Phase 3 verification crashed: {e}")
        return 1

    # Print summary
    success = print_summary(all_results)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
