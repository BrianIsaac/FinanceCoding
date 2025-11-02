#!/usr/bin/env python3
"""
Validate PyTorch packed sequence support for ragged LSTM implementation.

This script verifies that the current PyTorch installation supports all required
features for implementing ragged tensors with pack_padded_sequence.

References:
    - PyTorch Documentation: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
    - Implementation Plan: thoughts/shared/plans/2025-10-29-unified-ragged-lstm-forward-fill-removal.md
"""

from __future__ import annotations

import sys
import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_pytorch_version() -> Tuple[bool, str]:
    """
    Check if PyTorch version supports packed sequences.

    Returns:
        Tuple of (is_supported, message)
    """
    version = torch.__version__
    # Handle versions like "2.8.0+cu128"
    version_clean = version.split('+')[0]
    major, minor = map(int, version_clean.split('.')[:2])

    # pack_padded_sequence has been available since PyTorch 0.4
    # But we recommend >= 1.9.0 for stability
    is_supported = (major > 1) or (major == 1 and minor >= 9)

    if is_supported:
        message = f"PyTorch version {version} supports packed sequences ✓"
    else:
        message = f"PyTorch version {version} is too old (recommend >= 1.9.0) ✗"

    return is_supported, message


def test_basic_packing_unpacking() -> Tuple[bool, str]:
    """
    Test basic pack_padded_sequence and pad_packed_sequence functionality.

    Returns:
        Tuple of (success, message)
    """
    try:
        # Create test data: 4 sequences with varying lengths
        batch_size = 4
        max_seq_len = 60
        feature_dim = 10

        # Simulate variable-length sequences (like financial data with missing values)
        sequences = torch.randn(batch_size, max_seq_len, feature_dim)
        lengths = torch.tensor([45, 52, 30, 58])  # Variable lengths

        # Sort by length (descending) - required for pack_padded_sequence
        sorted_lengths, sort_idx = lengths.sort(descending=True)
        sequences_sorted = sequences[sort_idx]

        # Pack sequences
        packed = pack_padded_sequence(
            sequences_sorted,
            sorted_lengths.cpu(),  # Must be on CPU
            batch_first=True,
            enforce_sorted=True
        )

        # Unpack sequences
        unpacked, lengths_out = pad_packed_sequence(
            packed,
            batch_first=True,
            total_length=max_seq_len  # Important for DataParallel consistency
        )

        # Verify shapes
        assert unpacked.shape == sequences_sorted.shape, \
            f"Shape mismatch: {unpacked.shape} != {sequences_sorted.shape}"

        # Verify lengths
        assert torch.all(lengths_out == sorted_lengths), \
            f"Length mismatch: {lengths_out} != {sorted_lengths}"

        # Verify data integrity for non-padded portions
        for i, length in enumerate(sorted_lengths):
            original = sequences_sorted[i, :length, :]
            restored = unpacked[i, :length, :]
            diff = torch.abs(original - restored).max().item()
            assert diff < 1e-5, f"Data corruption detected: max diff = {diff}"

        return True, "Basic packing/unpacking test passed ✓"

    except Exception as e:
        return False, f"Basic packing/unpacking test failed: {e} ✗"


def test_lstm_with_packed_sequences() -> Tuple[bool, str]:
    """
    Test LSTM processing with packed sequences.

    Returns:
        Tuple of (success, message)
    """
    try:
        # Create LSTM
        hidden_size = 128
        num_layers = 2
        input_size = 10

        lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        # Create test data
        batch_size = 4
        max_seq_len = 60
        sequences = torch.randn(batch_size, max_seq_len, input_size)
        lengths = torch.tensor([45, 52, 30, 58])

        # Sort sequences
        sorted_lengths, sort_idx = lengths.sort(descending=True)
        sequences_sorted = sequences[sort_idx]

        # Pack sequences
        packed_input = pack_padded_sequence(
            sequences_sorted,
            sorted_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True
        )

        # Process through LSTM
        packed_output, (hidden, cell) = lstm(packed_input)

        # Unpack output
        lstm_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=max_seq_len
        )

        # Verify output shape
        expected_shape = (batch_size, max_seq_len, hidden_size)
        assert lstm_output.shape == expected_shape, \
            f"LSTM output shape mismatch: {lstm_output.shape} != {expected_shape}"

        # Verify hidden state shape
        expected_hidden = (num_layers, batch_size, hidden_size)
        assert hidden.shape == expected_hidden, \
            f"Hidden state shape mismatch: {hidden.shape} != {expected_hidden}"

        # Verify no NaN or Inf values
        assert torch.isfinite(lstm_output).all(), "LSTM output contains NaN/Inf"
        assert torch.isfinite(hidden).all(), "Hidden state contains NaN/Inf"
        assert torch.isfinite(cell).all(), "Cell state contains NaN/Inf"

        return True, "LSTM with packed sequences test passed ✓"

    except Exception as e:
        return False, f"LSTM with packed sequences test failed: {e} ✗"


def test_gradient_flow_through_packed_lstm() -> Tuple[bool, str]:
    """
    Test that gradients flow correctly through packed LSTM.

    Returns:
        Tuple of (success, message)
    """
    try:
        # Create LSTM
        lstm = nn.LSTM(
            input_size=10,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        # Create test data with gradient tracking
        batch_size = 4
        max_seq_len = 60
        sequences = torch.randn(batch_size, max_seq_len, 10, requires_grad=True)
        lengths = torch.tensor([45, 52, 30, 58])

        # Sort sequences
        sorted_lengths, sort_idx = lengths.sort(descending=True)
        sequences_sorted = sequences[sort_idx]

        # Pack and process
        packed_input = pack_padded_sequence(
            sequences_sorted,
            sorted_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True
        )

        packed_output, (hidden, cell) = lstm(packed_input)

        # Unpack
        lstm_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=max_seq_len
        )

        # Create a simple loss (sum of outputs)
        loss = lstm_output.sum()

        # Backward pass
        loss.backward()

        # Verify gradients exist and are finite
        assert sequences.grad is not None, "No gradient for input sequences"
        assert torch.isfinite(sequences.grad).all(), "Input gradients contain NaN/Inf"

        # Verify LSTM parameters have gradients
        for name, param in lstm.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN/Inf"

        return True, "Gradient flow test passed ✓"

    except Exception as e:
        return False, f"Gradient flow test failed: {e} ✗"


def test_enforce_sorted_parameter() -> Tuple[bool, str]:
    """
    Test the enforce_sorted parameter for pack_padded_sequence.

    Returns:
        Tuple of (success, message)
    """
    try:
        # Create test data
        sequences = torch.randn(4, 60, 10)

        # Test with sorted lengths (should work with enforce_sorted=True)
        sorted_lengths = torch.tensor([58, 52, 45, 30])  # Descending
        packed_sorted = pack_padded_sequence(
            sequences,
            sorted_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True
        )
        assert packed_sorted is not None

        # Test with unsorted lengths (should work with enforce_sorted=False)
        unsorted_lengths = torch.tensor([45, 52, 30, 58])  # Not sorted
        packed_unsorted = pack_padded_sequence(
            sequences,
            unsorted_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        assert packed_unsorted is not None

        # Verify that enforce_sorted=True with unsorted data raises error
        try:
            pack_padded_sequence(
                sequences,
                unsorted_lengths.cpu(),
                batch_first=True,
                enforce_sorted=True
            )
            return False, "enforce_sorted=True should raise error with unsorted lengths ✗"
        except RuntimeError:
            pass  # Expected

        return True, "enforce_sorted parameter test passed ✓"

    except Exception as e:
        return False, f"enforce_sorted parameter test failed: {e} ✗"


def test_batch_first_consistency() -> Tuple[bool, str]:
    """
    Test batch_first=True consistency (required for our implementation).

    Returns:
        Tuple of (success, message)
    """
    try:
        # Create test data in batch-first format
        batch_size = 4
        max_seq_len = 60
        feature_dim = 10

        sequences_batch_first = torch.randn(batch_size, max_seq_len, feature_dim)
        lengths = torch.tensor([45, 52, 30, 58])

        sorted_lengths, sort_idx = lengths.sort(descending=True)
        sequences_sorted = sequences_batch_first[sort_idx]

        # Pack with batch_first=True
        packed = pack_padded_sequence(
            sequences_sorted,
            sorted_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True
        )

        # Unpack with batch_first=True
        unpacked, _ = pad_packed_sequence(
            packed,
            batch_first=True,
            total_length=max_seq_len
        )

        # Verify shape is batch-first
        assert unpacked.shape[0] == batch_size, "First dimension should be batch"
        assert unpacked.shape[1] == max_seq_len, "Second dimension should be sequence"
        assert unpacked.shape[2] == feature_dim, "Third dimension should be features"

        return True, "batch_first consistency test passed ✓"

    except Exception as e:
        return False, f"batch_first consistency test failed: {e} ✗"


def test_computational_efficiency() -> Tuple[bool, str]:
    """
    Verify that packed sequences work correctly (efficiency varies by hardware).

    Note: Computational savings from packed sequences depend on:
    - GPU vs CPU (GPU shows more benefit)
    - Batch size (larger batches show more benefit)
    - Padding ratio (higher padding shows more benefit)
    - cuDNN availability

    This test verifies correctness rather than strict performance gains.

    Returns:
        Tuple of (success, message)
    """
    try:
        import time

        # Create LSTM
        lstm = nn.LSTM(input_size=10, hidden_size=128, num_layers=2, batch_first=True)
        lstm.eval()

        # Create test data with significant padding
        batch_size = 32
        max_seq_len = 60
        sequences = torch.randn(batch_size, max_seq_len, 10)

        # Simulate high padding ratio (30-40% padding)
        lengths = torch.randint(35, max_seq_len + 1, (batch_size,))
        sorted_lengths, sort_idx = lengths.sort(descending=True)
        sequences_sorted = sequences[sort_idx]

        # Benchmark with padding
        start_padded = time.time()
        for _ in range(100):
            with torch.no_grad():
                output_padded, _ = lstm(sequences_sorted)
        time_padded = time.time() - start_padded

        # Benchmark with packed sequences
        packed_input = pack_padded_sequence(
            sequences_sorted,
            sorted_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True
        )

        start_packed = time.time()
        for _ in range(100):
            with torch.no_grad():
                output_packed, _ = lstm(packed_input)
                # Unpack to compare
                output_unpacked, _ = pad_packed_sequence(
                    output_packed,
                    batch_first=True,
                    total_length=max_seq_len
                )
        time_packed = time.time() - start_packed

        speedup = (time_padded - time_packed) / time_padded * 100

        # Verify outputs are identical (within numerical precision)
        # Only compare non-padded regions
        for i, length in enumerate(sorted_lengths):
            padded_valid = output_padded[i, :length, :]
            unpacked_valid = output_unpacked[i, :length, :]
            diff = torch.abs(padded_valid - unpacked_valid).max().item()
            if diff > 1e-4:
                return False, f"Packed sequence output differs from padded at seq {i}: max diff = {diff:.2e} ✗"

        # Report performance (informational only)
        device = "GPU" if torch.cuda.is_available() else "CPU"
        message = f"Computational efficiency on {device}: {speedup:+.1f}% speedup with packed sequences"

        # On CPU, packing overhead may outweigh benefits
        # This is expected and documented in PyTorch
        if speedup < 0:
            message += " (expected on CPU; GPU shows greater benefit)"

        return True, f"{message} ✓"

    except Exception as e:
        return False, f"Computational efficiency test failed: {e} ✗"


def main() -> int:
    """
    Run all validation tests.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("=" * 70)
    logger.info("PyTorch Packed Sequence Compatibility Validation")
    logger.info("=" * 70)
    logger.info("")

    tests = [
        ("PyTorch Version", check_pytorch_version),
        ("Basic Packing/Unpacking", test_basic_packing_unpacking),
        ("LSTM with Packed Sequences", test_lstm_with_packed_sequences),
        ("Gradient Flow", test_gradient_flow_through_packed_lstm),
        ("enforce_sorted Parameter", test_enforce_sorted_parameter),
        ("batch_first Consistency", test_batch_first_consistency),
        ("Computational Efficiency", test_computational_efficiency),
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"Running: {test_name}")
        success, message = test_func()
        results.append((test_name, success, message))
        logger.info(f"  {message}")
        logger.info("")

    # Summary
    logger.info("=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for test_name, success, message in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"  [{status}] {test_name}")

    logger.info("")
    logger.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("")
        logger.info("✓ All tests passed! PyTorch installation supports packed sequences.")
        logger.info("✓ Ready to implement ragged LSTM architecture.")
        logger.info("")
        logger.info("Recommended next steps:")
        logger.info("  1. Capture baseline metrics (Phase 0.5)")
        logger.info("  2. Create test infrastructure (Phase 0)")
        logger.info("  3. Implement ragged tensor utilities (Phase 1)")
        logger.info("")
        return 0
    else:
        logger.error("")
        logger.error(f"✗ {total - passed} test(s) failed!")
        logger.error("✗ PyTorch installation may not fully support packed sequences.")
        logger.error("")
        logger.error("Recommended actions:")
        logger.error(f"  1. Upgrade PyTorch: uv pip install --upgrade 'torch>=1.9.0'")
        logger.error("  2. Rerun this validation script")
        logger.error("")
        return 1


if __name__ == "__main__":
    sys.exit(main())
