"""Comprehensive unit tests for ragged tensor utilities."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

from src.models.lstm.ragged_utils import (
    validate_sequence_batch,
    pack_ragged_sequences,
    unpack_ragged_sequences,
    compute_sequence_statistics,
    create_length_mask,
)


class TestValidateSequenceBatch:
    """Test validation of sequence batches."""

    def test_valid_sequences(self):
        """Test validation passes for valid sequences."""
        sequences = torch.randn(32, 60, 10)
        lengths = torch.randint(30, 61, (32,))

        is_valid, msg = validate_sequence_batch(sequences, lengths, strict=False)
        assert is_valid
        assert msg is None

    def test_valid_sequences_strict(self):
        """Test validation passes in strict mode."""
        sequences = torch.randn(16, 50, 8)
        lengths = torch.randint(20, 51, (16,))

        # Should not raise
        validate_sequence_batch(sequences, lengths, strict=True)

    def test_wrong_lengths_shape(self):
        """Test validation fails for wrong lengths shape."""
        sequences = torch.randn(32, 60, 10)
        lengths = torch.randint(30, 61, (32, 1))  # Wrong shape

        is_valid, msg = validate_sequence_batch(sequences, lengths, strict=False)
        assert not is_valid
        assert "must be 1D tensor" in msg

    def test_wrong_lengths_shape_strict(self):
        """Test validation raises in strict mode for wrong shape."""
        sequences = torch.randn(32, 60, 10)
        lengths = torch.randint(30, 61, (32, 1))

        with pytest.raises(ValueError, match="must be 1D tensor"):
            validate_sequence_batch(sequences, lengths, strict=True)

    def test_lengths_batch_mismatch(self):
        """Test validation fails when lengths size doesn't match batch."""
        sequences = torch.randn(32, 60, 10)
        lengths = torch.randint(30, 61, (16,))  # Wrong size

        is_valid, msg = validate_sequence_batch(sequences, lengths, strict=False)
        assert not is_valid
        assert "doesn't match batch size" in msg

    def test_negative_length(self):
        """Test validation fails for negative lengths."""
        sequences = torch.randn(32, 60, 10)
        lengths = torch.randint(30, 61, (32,))
        lengths[5] = -1  # Invalid

        is_valid, msg = validate_sequence_batch(sequences, lengths, strict=False)
        assert not is_valid
        assert "must be positive" in msg

    def test_zero_length(self):
        """Test validation fails for zero length."""
        sequences = torch.randn(32, 60, 10)
        lengths = torch.randint(30, 61, (32,))
        lengths[10] = 0  # Invalid

        is_valid, msg = validate_sequence_batch(sequences, lengths, strict=False)
        assert not is_valid
        assert "must be positive" in msg

    def test_length_exceeds_max(self):
        """Test validation fails when length exceeds max_seq_len."""
        sequences = torch.randn(32, 60, 10)
        lengths = torch.randint(30, 61, (32,))
        lengths[0] = 61  # Exceeds max

        is_valid, msg = validate_sequence_batch(sequences, lengths, strict=False)
        assert not is_valid
        assert "exceeds max_seq_len" in msg

    def test_nan_in_sequences(self):
        """Test validation fails for NaN in sequences."""
        sequences = torch.randn(32, 60, 10)
        sequences[5, 10, 3] = float('nan')
        lengths = torch.randint(30, 61, (32,))

        is_valid, msg = validate_sequence_batch(sequences, lengths, strict=False)
        assert not is_valid
        assert "NaN" in msg

    def test_inf_in_sequences(self):
        """Test validation fails for Inf in sequences."""
        sequences = torch.randn(32, 60, 10)
        sequences[8, 20, 5] = float('inf')
        lengths = torch.randint(30, 61, (32,))

        is_valid, msg = validate_sequence_batch(sequences, lengths, strict=False)
        assert not is_valid
        assert "Inf" in msg


class TestPackUnpackRaggedSequences:
    """Test packing and unpacking of ragged sequences."""

    def test_pack_basic(self):
        """Test basic packing functionality."""
        sequences = torch.randn(4, 60, 10)
        lengths = torch.tensor([45, 52, 30, 58])

        packed, indices = pack_ragged_sequences(sequences, lengths)

        assert isinstance(packed, PackedSequence)
        assert indices is not None  # Should have sorting indices

    def test_pack_sorted_sequences(self):
        """Test packing with pre-sorted sequences."""
        sequences = torch.randn(4, 60, 10)
        lengths = torch.tensor([60, 55, 50, 45])  # Already sorted descending

        packed, indices = pack_ragged_sequences(sequences, lengths, enforce_sorted=False)

        assert isinstance(packed, PackedSequence)
        # Should still return indices even if already sorted
        assert indices is not None

    def test_pack_enforce_sorted_valid(self):
        """Test enforce_sorted with valid sorted sequences."""
        sequences = torch.randn(4, 60, 10)
        lengths = torch.tensor([60, 55, 50, 45])  # Sorted descending

        packed, indices = pack_ragged_sequences(sequences, lengths, enforce_sorted=True)

        assert isinstance(packed, PackedSequence)

    def test_pack_enforce_sorted_invalid(self):
        """Test enforce_sorted fails with unsorted sequences."""
        sequences = torch.randn(4, 60, 10)
        lengths = torch.tensor([45, 52, 30, 58])  # Not sorted

        with pytest.raises(ValueError, match="not sorted by length"):
            pack_ragged_sequences(sequences, lengths, enforce_sorted=True)

    def test_pack_unpack_roundtrip(self):
        """Test that pack and unpack preserve data."""
        sequences = torch.randn(8, 50, 12)
        lengths = torch.randint(20, 51, (8,))

        # Pack
        packed, indices = pack_ragged_sequences(sequences, lengths)

        # Unpack
        unpacked, unpacked_lengths = unpack_ragged_sequences(packed, indices, total_length=50)

        # Check shapes match
        assert unpacked.shape == sequences.shape
        assert (unpacked_lengths == lengths).all()

        # Check actual data matches (within valid positions)
        for i in range(len(lengths)):
            length = lengths[i]
            torch.testing.assert_close(
                unpacked[i, :length, :],
                sequences[i, :length, :],
                rtol=1e-5,
                atol=1e-7
            )

    def test_pack_unpack_preserves_order(self):
        """Test that unpacking restores original order."""
        batch_size = 16
        max_len = 60
        features = 10

        sequences = torch.randn(batch_size, max_len, features)
        lengths = torch.randint(30, max_len + 1, (batch_size,))

        # Pack and unpack
        packed, indices = pack_ragged_sequences(sequences, lengths)
        unpacked, unpacked_lengths = unpack_ragged_sequences(packed, indices)

        # Should match original order
        assert (unpacked_lengths == lengths).all()

    def test_unpack_without_indices(self):
        """Test unpacking without restoring order."""
        sequences = torch.randn(4, 60, 10)
        lengths = torch.tensor([60, 55, 50, 45])  # Pre-sorted

        packed, _ = pack_ragged_sequences(sequences, lengths)
        unpacked, unpacked_lengths = unpack_ragged_sequences(packed, sorted_indices=None)

        # Should remain sorted
        assert torch.all(unpacked_lengths[:-1] >= unpacked_lengths[1:])

    def test_unpack_with_total_length(self):
        """Test unpacking with specified total_length."""
        sequences = torch.randn(4, 60, 10)
        lengths = torch.tensor([45, 52, 30, 58])

        packed, indices = pack_ragged_sequences(sequences, lengths)
        unpacked, _ = unpack_ragged_sequences(packed, indices, total_length=80)

        # Should pad to total_length
        assert unpacked.shape[1] == 80


class TestComputeSequenceStatistics:
    """Test sequence statistics computation."""

    def test_statistics_basic(self):
        """Test basic statistics computation."""
        sequences = torch.randn(32, 60, 10)
        lengths = torch.tensor([45] * 32)  # All same length

        stats = compute_sequence_statistics(sequences, lengths)

        assert stats['mean_length'] == 45.0
        assert stats['std_length'] == 0.0
        assert stats['min_length'] == 45
        assert stats['max_length'] == 45
        assert stats['batch_size'] == 32
        assert stats['max_seq_len'] == 60
        assert stats['n_features'] == 10

    def test_statistics_variable_lengths(self):
        """Test statistics with variable lengths."""
        sequences = torch.randn(16, 100, 8)
        lengths = torch.randint(30, 101, (16,))

        stats = compute_sequence_statistics(sequences, lengths)

        assert 30 <= stats['mean_length'] <= 100
        assert stats['std_length'] >= 0
        assert stats['min_length'] == lengths.min().item()
        assert stats['max_length'] == lengths.max().item()

    def test_padding_ratio_calculation(self):
        """Test padding ratio is calculated correctly."""
        sequences = torch.randn(4, 100, 10)
        lengths = torch.tensor([50, 50, 50, 50])  # 50% of max

        stats = compute_sequence_statistics(sequences, lengths)

        # Expected padding ratio: 1 - (4*50*10)/(4*100*10) = 0.5
        assert abs(stats['padding_ratio'] - 0.5) < 1e-6

    def test_padding_ratio_no_padding(self):
        """Test padding ratio is zero when no padding."""
        sequences = torch.randn(8, 60, 12)
        lengths = torch.tensor([60] * 8)  # All full length

        stats = compute_sequence_statistics(sequences, lengths)

        assert stats['padding_ratio'] == 0.0

    def test_padding_ratio_high_padding(self):
        """Test padding ratio with high padding."""
        sequences = torch.randn(10, 100, 5)
        lengths = torch.tensor([10] * 10)  # Only 10% of max

        stats = compute_sequence_statistics(sequences, lengths)

        # Expected: 1 - (10*10*5)/(10*100*5) = 0.9
        assert abs(stats['padding_ratio'] - 0.9) < 1e-6


class TestCreateLengthMask:
    """Test length mask creation."""

    def test_mask_basic(self):
        """Test basic mask creation."""
        sequences = torch.randn(4, 60, 10)
        lengths = torch.tensor([30, 40, 50, 60])

        mask = create_length_mask(sequences, lengths)

        assert mask.shape == (4, 60)
        assert mask.dtype == torch.bool

        # Check first sequence (length 30)
        assert mask[0, :30].all()
        assert not mask[0, 30:].any()

        # Check last sequence (length 60)
        assert mask[3, :60].all()

    def test_mask_correctness(self):
        """Test mask correctly identifies valid positions."""
        sequences = torch.randn(8, 50, 12)
        lengths = torch.randint(20, 51, (8,))

        mask = create_length_mask(sequences, lengths)

        for i in range(len(lengths)):
            length = lengths[i]
            # Valid positions should be True
            assert mask[i, :length].all()
            # Padding positions should be False
            if length < 50:
                assert not mask[i, length:].any()

    def test_mask_edge_case_zero_length(self):
        """Test mask with zero length (edge case)."""
        sequences = torch.randn(2, 60, 10)
        lengths = torch.tensor([0, 60])

        mask = create_length_mask(sequences, lengths)

        # First sequence: all False (length 0)
        assert not mask[0].any()
        # Second sequence: all True (length 60)
        assert mask[1].all()

    def test_mask_edge_case_full_length(self):
        """Test mask when all sequences are full length."""
        sequences = torch.randn(10, 40, 8)
        lengths = torch.tensor([40] * 10)

        mask = create_length_mask(sequences, lengths)

        # All positions should be valid
        assert mask.all()


class TestGradientFlow:
    """Test gradient propagation through ragged sequences."""

    def test_gradients_through_packed_sequence(self):
        """Test gradients flow through pack/unpack operations."""
        sequences = torch.randn(4, 60, 10, requires_grad=True)
        lengths = torch.tensor([45, 52, 30, 58])

        # Pack and unpack
        packed, indices = pack_ragged_sequences(sequences, lengths)
        unpacked, _ = unpack_ragged_sequences(packed, indices)

        # Compute loss and backprop
        loss = unpacked.sum()
        loss.backward()

        # Check gradients exist and are finite
        assert sequences.grad is not None
        assert torch.isfinite(sequences.grad).all()

    def test_gradients_preserved_in_roundtrip(self):
        """Test gradients are preserved through pack/unpack."""
        sequences = torch.randn(8, 50, 12, requires_grad=True)
        lengths = torch.randint(20, 51, (8,))

        # Pack, unpack, and compute loss
        packed, indices = pack_ragged_sequences(sequences, lengths)
        unpacked, _ = unpack_ragged_sequences(packed, indices, total_length=50)

        # Loss only on valid positions
        mask = create_length_mask(sequences, lengths)
        loss = (unpacked * mask.unsqueeze(-1)).sum()
        loss.backward()

        # Gradients should exist
        assert sequences.grad is not None
        assert torch.isfinite(sequences.grad).all()

        # Gradients should be non-zero in valid positions
        for i in range(len(lengths)):
            length = lengths[i]
            assert (sequences.grad[i, :length, :] != 0).any()

    def test_backward_pass_completes(self):
        """Test backward pass completes without errors."""
        batch_size = 16
        max_len = 60
        features = 10

        sequences = torch.randn(batch_size, max_len, features, requires_grad=True)
        lengths = torch.randint(30, max_len + 1, (batch_size,))

        # Forward pass
        packed, indices = pack_ragged_sequences(sequences, lengths)
        unpacked, _ = unpack_ragged_sequences(packed, indices)

        # Backward pass
        loss = unpacked.mean()
        loss.backward()

        # Should complete without errors
        assert sequences.grad is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_sequence(self):
        """Test with batch size of 1."""
        sequences = torch.randn(1, 60, 10)
        lengths = torch.tensor([45])

        packed, indices = pack_ragged_sequences(sequences, lengths)
        unpacked, unpacked_lengths = unpack_ragged_sequences(packed, indices, total_length=60)

        assert unpacked.shape == sequences.shape
        assert unpacked_lengths == lengths

    def test_large_batch(self):
        """Test with large batch size."""
        sequences = torch.randn(128, 50, 8)
        lengths = torch.randint(20, 51, (128,))

        packed, indices = pack_ragged_sequences(sequences, lengths)
        unpacked, unpacked_lengths = unpack_ragged_sequences(packed, indices, total_length=50)

        assert unpacked.shape == sequences.shape
        assert (unpacked_lengths == lengths).all()

    def test_minimal_sequence_length(self):
        """Test with minimal sequence length (1)."""
        sequences = torch.randn(4, 60, 10)
        lengths = torch.tensor([1, 2, 3, 4])

        packed, indices = pack_ragged_sequences(sequences, lengths)
        unpacked, unpacked_lengths = unpack_ragged_sequences(packed, indices, total_length=60)

        assert (unpacked_lengths == lengths).all()

    def test_all_same_length(self):
        """Test when all sequences have same length."""
        sequences = torch.randn(16, 60, 10)
        lengths = torch.tensor([45] * 16)

        stats = compute_sequence_statistics(sequences, lengths)

        assert stats['std_length'] == 0.0
        assert stats['min_length'] == stats['max_length']
