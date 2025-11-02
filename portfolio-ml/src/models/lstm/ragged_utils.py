"""Utilities for handling ragged (variable-length) tensor sequences."""

from __future__ import annotations

from typing import Tuple, List, Optional
import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logger = logging.getLogger(__name__)


def validate_sequence_batch(
    sequences: torch.Tensor,
    lengths: torch.Tensor,
    strict: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Validate a batch of ragged sequences.

    Args:
        sequences: Tensor of shape [batch, max_seq_len, features]
        lengths: Tensor of shape [batch] with actual sequence lengths
        strict: If True, raises errors; if False, returns (False, error_msg)

    Returns:
        Tuple of (is_valid, error_message)

    Raises:
        ValueError: If strict=True and validation fails

    Examples:
        >>> sequences = torch.randn(32, 60, 10)
        >>> lengths = torch.tensor([45, 52, 30, ...])  # 32 lengths
        >>> is_valid, msg = validate_sequence_batch(sequences, lengths)
        >>> assert is_valid
    """
    batch_size, max_seq_len, n_features = sequences.shape

    # Check 1: Lengths tensor shape
    if lengths.dim() != 1:
        msg = f"Lengths must be 1D tensor, got shape {lengths.shape}"
        if strict:
            raise ValueError(msg)
        return False, msg

    if len(lengths) != batch_size:
        msg = f"Lengths size {len(lengths)} doesn't match batch size {batch_size}"
        if strict:
            raise ValueError(msg)
        return False, msg

    # Check 2: All lengths are positive
    if (lengths <= 0).any():
        msg = f"All lengths must be positive, got min={lengths.min().item()}"
        if strict:
            raise ValueError(msg)
        return False, msg

    # Check 3: No length exceeds max_seq_len
    if (lengths > max_seq_len).any():
        msg = f"Length {lengths.max().item()} exceeds max_seq_len {max_seq_len}"
        if strict:
            raise ValueError(msg)
        return False, msg

    # Check 4: No NaN or Inf in sequences
    if torch.isnan(sequences).any():
        msg = f"Sequences contain NaN values"
        if strict:
            raise ValueError(msg)
        return False, msg

    if torch.isinf(sequences).any():
        msg = f"Sequences contain Inf values"
        if strict:
            raise ValueError(msg)
        return False, msg

    return True, None


def pack_ragged_sequences(
    sequences: torch.Tensor,
    lengths: torch.Tensor,
    enforce_sorted: bool = False
) -> Tuple[nn.utils.rnn.PackedSequence, Optional[torch.Tensor]]:
    """
    Pack variable-length sequences into PackedSequence format.

    Args:
        sequences: Tensor of shape [batch, max_seq_len, features]
        lengths: Tensor of shape [batch] with actual sequence lengths
        enforce_sorted: If True, requires lengths to be sorted descending

    Returns:
        Tuple of (packed_sequences, sorted_indices) where sorted_indices
        is None if enforce_sorted=False or sequences were already sorted

    Examples:
        >>> sequences = torch.randn(4, 60, 10)
        >>> lengths = torch.tensor([45, 52, 30, 58])
        >>> packed, indices = pack_ragged_sequences(sequences, lengths)
        >>> # packed contains only valid timesteps (no padding computation)
    """
    # Validate inputs
    validate_sequence_batch(sequences, lengths, strict=True)

    # Handle sorting if needed
    sorted_indices = None
    if not enforce_sorted:
        # Sort by length descending (required by pack_padded_sequence)
        lengths_sorted, sorted_indices = torch.sort(lengths, descending=True)
        sequences_sorted = sequences[sorted_indices]
    else:
        # Verify already sorted
        if not torch.all(lengths[:-1] >= lengths[1:]):
            raise ValueError("enforce_sorted=True but sequences not sorted by length")
        lengths_sorted = lengths
        sequences_sorted = sequences

    # Pack sequences (this is where the magic happens - no computation on padding!)
    packed = pack_padded_sequence(
        sequences_sorted,
        lengths_sorted.cpu(),  # pack_padded_sequence requires CPU lengths
        batch_first=True,
        enforce_sorted=True  # Already sorted above
    )

    return packed, sorted_indices


def unpack_ragged_sequences(
    packed: nn.utils.rnn.PackedSequence,
    sorted_indices: Optional[torch.Tensor] = None,
    total_length: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unpack PackedSequence back to padded tensor format.

    Args:
        packed: PackedSequence from pack_ragged_sequences
        sorted_indices: Indices from packing (to restore original order)
        total_length: Optional maximum sequence length (for padding)

    Returns:
        Tuple of (sequences, lengths) where:
        - sequences: [batch, max_seq_len, features] padded tensor
        - lengths: [batch] actual sequence lengths

    Examples:
        >>> packed, indices = pack_ragged_sequences(sequences, lengths)
        >>> # ... LSTM processing ...
        >>> unpacked, unpacked_lengths = unpack_ragged_sequences(packed, indices)
        >>> # unpacked has same shape as original sequences
    """
    # Unpack sequences
    sequences, lengths = pad_packed_sequence(
        packed,
        batch_first=True,
        total_length=total_length
    )

    # Restore original order if needed
    if sorted_indices is not None:
        # Create inverse permutation
        inverse_indices = torch.argsort(sorted_indices)
        # Ensure indices are on the same device as the tensors
        inverse_indices = inverse_indices.to(sequences.device)
        sequences = sequences[inverse_indices]
        # Move lengths to same device before indexing
        lengths = lengths.to(sequences.device)
        lengths = lengths[inverse_indices]

    return sequences, lengths


def compute_sequence_statistics(
    sequences: torch.Tensor,
    lengths: torch.Tensor
) -> dict[str, float]:
    """
    Compute statistics about sequence lengths for monitoring.

    Args:
        sequences: Tensor of shape [batch, max_seq_len, features]
        lengths: Tensor of shape [batch] with actual sequence lengths

    Returns:
        Dictionary with statistics:
        - mean_length: Average sequence length
        - std_length: Standard deviation of lengths
        - min_length: Minimum length
        - max_length: Maximum length
        - padding_ratio: Fraction of values that are padding

    Examples:
        >>> stats = compute_sequence_statistics(sequences, lengths)
        >>> print(f"Avg length: {stats['mean_length']:.1f}")
        >>> print(f"Padding: {stats['padding_ratio']:.2%}")
    """
    batch_size, max_seq_len, n_features = sequences.shape

    mean_len = float(lengths.float().mean())
    std_len = float(lengths.float().std())
    min_len = int(lengths.min())
    max_len = int(lengths.max())

    # Calculate padding ratio
    total_values = batch_size * max_seq_len * n_features
    actual_values = int(lengths.sum()) * n_features
    padding_ratio = 1.0 - (actual_values / total_values)

    return {
        'mean_length': mean_len,
        'std_length': std_len,
        'min_length': min_len,
        'max_length': max_len,
        'padding_ratio': padding_ratio,
        'batch_size': batch_size,
        'max_seq_len': max_seq_len,
        'n_features': n_features,
    }


def create_length_mask(
    sequences: torch.Tensor,
    lengths: torch.Tensor
) -> torch.Tensor:
    """
    Create boolean mask indicating valid (non-padding) positions.

    Args:
        sequences: Tensor of shape [batch, max_seq_len, features]
        lengths: Tensor of shape [batch] with actual sequence lengths

    Returns:
        Boolean mask of shape [batch, max_seq_len] where True = valid

    Examples:
        >>> mask = create_length_mask(sequences, lengths)
        >>> # Use mask for attention, loss computation, etc.
        >>> valid_values = sequences[mask.unsqueeze(-1).expand_as(sequences)]
    """
    batch_size, max_seq_len = sequences.shape[:2]

    # Create range tensor [0, 1, 2, ..., max_seq_len-1]
    range_tensor = torch.arange(max_seq_len, device=sequences.device)
    range_tensor = range_tensor.unsqueeze(0).expand(batch_size, -1)

    # Create mask: position < length
    mask = range_tensor < lengths.unsqueeze(1)

    return mask
