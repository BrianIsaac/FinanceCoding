"""
LSTM Temporal Pattern Analysis Framework.

This module provides interpretability tools for LSTM models, including
gradient-based attribution methods, temporal importance analysis,
and hidden state pattern extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

try:
    from captum.attr import IntegratedGradients, LayerConductance
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    IntegratedGradients = None
    LayerConductance = None

from ...models.lstm.model import LSTMPortfolioModel


@dataclass
class TemporalAttributionConfig:
    """Configuration for temporal attribution analysis."""

    attribution_method: str = "gradient_shap"  # "integrated_gradients", "layer_conductance", "gradient_shap"
    baseline_strategy: str = "zeros"  # "zeros", "mean", "random"
    n_integration_steps: int = 50  # For integrated gradients
    temporal_smoothing: bool = True  # Apply smoothing to temporal importance
    smoothing_window: int = 5  # Size of smoothing window
    significance_threshold: float = 0.01  # Minimum attribution to consider significant


class LSTMAttributor:
    """
    LSTM temporal pattern analyzer.

    Provides tools for analyzing which historical periods most influence
    LSTM predictions using gradient-based attribution methods.
    """

    def __init__(
        self,
        model: LSTMPortfolioModel,
        config: TemporalAttributionConfig | None = None
    ):
        """
        Initialize LSTM attributor.

        Args:
            model: Trained LSTM portfolio model
            config: Attribution analysis configuration
        """
        self.model = model
        self.network = model.network
        self.config = config or TemporalAttributionConfig()

        if not model.is_fitted or self.network is None:
            raise ValueError("Model must be fitted before analysis")

        # Initialize attribution methods
        self._setup_attribution_methods()

    def _setup_attribution_methods(self) -> None:
        """Initialize gradient-based attribution methods."""
        if self.config.attribution_method in ["integrated_gradients", "layer_conductance"]:
            if not CAPTUM_AVAILABLE:
                raise ImportError(
                    "captum is required for advanced attribution methods. "
                    "Install it with: pip install captum or use 'gradient_shap' method"
                )

            if self.config.attribution_method == "integrated_gradients":
                self.ig_attributor = IntegratedGradients(self.network)
            elif self.config.attribution_method == "layer_conductance":
                # Use LSTM layer for conductance analysis
                self.conductance_attributor = LayerConductance(self.network, self.network.lstm)
        elif self.config.attribution_method == "gradient_shap":
            # Simple gradient-based method that doesn't require captum
            pass  # Will use custom implementation
        else:
            raise ValueError(f"Unknown attribution method: {self.config.attribution_method}")

    def analyze_temporal_importance(
        self,
        input_sequence: torch.Tensor,
        target_asset_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """
        Analyze temporal importance for LSTM predictions.

        Args:
            input_sequence: Input sequence tensor [batch_size, seq_len, n_features]
            target_asset_idx: Index of target asset to analyze

        Returns:
            Dictionary containing temporal attribution analysis
        """
        self.network.eval()

        # Generate baseline for attribution
        baseline = self._generate_baseline(input_sequence)

        # Compute attributions
        if self.config.attribution_method == "integrated_gradients":
            attributions = self._compute_integrated_gradients(
                input_sequence, baseline, target_asset_idx
            )
        elif self.config.attribution_method == "layer_conductance":
            attributions = self._compute_layer_conductance(
                input_sequence, target_asset_idx
            )
        elif self.config.attribution_method == "gradient_shap":
            attributions = self._compute_gradient_shap(
                input_sequence, baseline, target_asset_idx
            )
        else:
            raise ValueError(f"Unknown attribution method: {self.config.attribution_method}")

        # Process attributions
        processed_attributions = self._process_attributions(attributions, input_sequence.shape)

        return processed_attributions

    def _generate_baseline(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Generate baseline for attribution analysis.

        Args:
            input_sequence: Input sequence tensor

        Returns:
            Baseline tensor
        """
        if self.config.baseline_strategy == "zeros":
            return torch.zeros_like(input_sequence)
        elif self.config.baseline_strategy == "mean":
            return torch.full_like(input_sequence, input_sequence.mean())
        elif self.config.baseline_strategy == "random":
            return torch.randn_like(input_sequence) * input_sequence.std() + input_sequence.mean()
        else:
            raise ValueError(f"Unknown baseline strategy: {self.config.baseline_strategy}")

    def _compute_integrated_gradients(
        self,
        input_sequence: torch.Tensor,
        baseline: torch.Tensor,
        target_asset_idx: int,
    ) -> torch.Tensor:
        """
        Compute integrated gradients attribution.

        Args:
            input_sequence: Input sequence
            baseline: Baseline for integration
            target_asset_idx: Target asset index

        Returns:
            Attribution tensor
        """
        input_sequence.requires_grad_(True)

        # Define forward function for specific asset
        def forward_func(x):
            predictions, _ = self.network(x)
            if predictions.dim() == 2 and predictions.size(1) > target_asset_idx:
                return predictions[:, target_asset_idx]
            else:
                return predictions.squeeze()

        # Compute attributions
        attributions = self.ig_attributor.attribute(
            input_sequence,
            baseline,
            target=target_asset_idx,
            n_steps=self.config.n_integration_steps,
        )

        return attributions

    def _compute_layer_conductance(
        self,
        input_sequence: torch.Tensor,
        target_asset_idx: int,
    ) -> torch.Tensor:
        """
        Compute layer conductance attribution.

        Args:
            input_sequence: Input sequence
            target_asset_idx: Target asset index

        Returns:
            Attribution tensor
        """
        input_sequence.requires_grad_(True)

        # Compute conductance
        conductance = self.conductance_attributor.attribute(
            input_sequence,
            target=target_asset_idx,
        )

        return conductance

    def _compute_gradient_shap(
        self,
        input_sequence: torch.Tensor,
        baseline: torch.Tensor,
        target_asset_idx: int,
    ) -> torch.Tensor:
        """
        Compute gradient-based SHAP approximation attribution.

        Args:
            input_sequence: Input sequence
            baseline: Baseline for attribution
            target_asset_idx: Target asset index

        Returns:
            Attribution tensor
        """
        input_sequence.requires_grad_(True)

        # Forward pass
        predictions, _ = self.network(input_sequence)

        # Get target output
        if predictions.dim() == 2 and predictions.size(1) > target_asset_idx:
            target_output = predictions[:, target_asset_idx]
        else:
            target_output = predictions.squeeze()

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=target_output.sum(),
            inputs=input_sequence,
            create_graph=False,
            retain_graph=False,
        )[0]

        # Multiply gradients by (input - baseline) for SHAP approximation
        attributions = gradients * (input_sequence - baseline)

        return attributions

    def _process_attributions(
        self,
        attributions: torch.Tensor,
        input_shape: torch.Size
    ) -> dict[str, torch.Tensor]:
        """
        Process raw attributions into temporal importance scores.

        Args:
            attributions: Raw attribution tensor
            input_shape: Original input shape

        Returns:
            Processed attribution results
        """
        # Sum attributions across feature dimension to get temporal importance
        if len(attributions.shape) == 3:  # [batch, seq_len, features]
            temporal_importance = attributions.abs().sum(dim=2)  # [batch, seq_len]
        else:
            temporal_importance = attributions.abs()

        # Apply smoothing if requested
        if self.config.temporal_smoothing:
            temporal_importance = self._apply_temporal_smoothing(temporal_importance)

        # Normalize to sum to 1
        temporal_importance_norm = temporal_importance / temporal_importance.sum(dim=1, keepdim=True)

        # Identify significant time periods
        significant_mask = temporal_importance_norm > self.config.significance_threshold

        return {
            "temporal_importance": temporal_importance,
            "temporal_importance_normalized": temporal_importance_norm,
            "significant_periods": significant_mask,
            "max_importance_idx": temporal_importance_norm.argmax(dim=1),
            "attribution_entropy": self._compute_attribution_entropy(temporal_importance_norm),
        }

    def _apply_temporal_smoothing(self, temporal_importance: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal smoothing to importance scores.

        Args:
            temporal_importance: Raw temporal importance scores

        Returns:
            Smoothed importance scores
        """
        window_size = self.config.smoothing_window
        if window_size <= 1:
            return temporal_importance

        # Apply 1D convolution for smoothing
        batch_size, seq_len = temporal_importance.shape

        # Create uniform smoothing kernel
        kernel = torch.ones(1, 1, window_size) / window_size

        # Pad input for boundary handling
        padded_input = F.pad(
            temporal_importance.unsqueeze(1),
            (window_size // 2, window_size // 2),
            mode='reflect'
        )

        # Apply convolution
        smoothed = F.conv1d(padded_input, kernel, padding=0)

        return smoothed.squeeze(1)

    def _compute_attribution_entropy(self, normalized_attributions: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of attribution distribution.

        Args:
            normalized_attributions: Normalized attribution scores

        Returns:
            Attribution entropy values
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-12
        log_attr = torch.log(normalized_attributions + epsilon)
        entropy = -(normalized_attributions * log_attr).sum(dim=1)

        return entropy

    def analyze_attention_patterns(
        self,
        input_sequence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Analyze LSTM attention patterns.

        Args:
            input_sequence: Input sequence tensor

        Returns:
            Attention analysis results
        """
        self.network.eval()

        with torch.no_grad():
            # Forward pass to get attention weights
            predictions, attention_weights = self.network(input_sequence)

        # Analyze attention patterns
        attention_analysis = {
            "attention_weights": attention_weights,
            "attention_entropy": self._compute_attention_entropy(attention_weights),
            "peak_attention_indices": attention_weights.argmax(dim=1),
            "attention_concentration": self._compute_attention_concentration(attention_weights),
        }

        return attention_analysis

    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of attention distribution.

        Args:
            attention_weights: Attention weight tensor [batch_size, seq_len]

        Returns:
            Attention entropy values
        """
        epsilon = 1e-12
        log_attn = torch.log(attention_weights + epsilon)
        entropy = -(attention_weights * log_attn).sum(dim=1)

        return entropy

    def _compute_attention_concentration(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute attention concentration (Gini coefficient).

        Args:
            attention_weights: Attention weight tensor

        Returns:
            Attention concentration values
        """
        # Sort attention weights
        sorted_weights, _ = torch.sort(attention_weights, dim=1)

        # Compute Gini coefficient
        n = sorted_weights.size(1)
        index = torch.arange(1, n + 1, dtype=torch.float32).unsqueeze(0)
        gini = (2 * index * sorted_weights).sum(dim=1) / (n * sorted_weights.sum(dim=1)) - (n + 1) / n

        return gini

    def generate_temporal_heatmap_data(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        dates: list[pd.Timestamp],
        sequence_length: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate temporal heatmap data for multiple dates.

        Args:
            returns: Historical returns DataFrame
            universe: Asset universe
            dates: Analysis dates
            sequence_length: Sequence length (uses model config if None)

        Returns:
            DataFrame with temporal importance data
        """
        if sequence_length is None:
            sequence_length = self.network.config.sequence_length

        heatmap_data = []

        for date in dates:
            # Get historical sequence ending at this date
            end_idx = returns.index.get_loc(date)
            start_idx = max(0, end_idx - sequence_length + 1)

            sequence_data = returns.iloc[start_idx:end_idx + 1][universe].values
            sequence_dates = returns.index[start_idx:end_idx + 1]

            if len(sequence_data) < sequence_length:
                continue  # Skip if insufficient data

            # Convert to tensor
            input_tensor = torch.FloatTensor(sequence_data).unsqueeze(0).unsqueeze(-1)

            # Analyze temporal importance for each asset
            for asset_idx, asset in enumerate(universe):
                try:
                    attribution_results = self.analyze_temporal_importance(
                        input_tensor, target_asset_idx=asset_idx
                    )

                    temporal_importance = attribution_results["temporal_importance_normalized"][0]

                    # Create heatmap entries
                    for _i, (seq_date, importance) in enumerate(zip(sequence_dates, temporal_importance)):
                        heatmap_data.append({
                            "prediction_date": date,
                            "historical_date": seq_date,
                            "asset": asset,
                            "temporal_importance": float(importance),
                            "days_back": (date - seq_date).days,
                        })

                except Exception:
                    # Skip problematic cases
                    continue

        return pd.DataFrame(heatmap_data)

    def analyze_hidden_state_patterns(
        self,
        input_sequence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Analyze LSTM hidden state evolution patterns.

        Args:
            input_sequence: Input sequence tensor

        Returns:
            Hidden state analysis results
        """
        self.network.eval()

        # Hook to capture LSTM hidden states
        hidden_states = []

        def capture_hidden_states(module, input, output):
            # output is (output, (h_n, c_n))
            if isinstance(output, tuple) and len(output) == 2:
                hidden_states.append(output[0].detach())  # LSTM output

        # Register hook
        handle = self.network.lstm.register_forward_hook(capture_hidden_states)

        try:
            with torch.no_grad():
                _ = self.network(input_sequence)
        finally:
            handle.remove()

        if not hidden_states:
            raise RuntimeError("Failed to capture LSTM hidden states")

        lstm_output = hidden_states[0]  # [batch_size, seq_len, hidden_size]

        # Analyze hidden state patterns
        analysis_results = {
            "hidden_states": lstm_output,
            "hidden_state_norms": torch.norm(lstm_output, dim=2),  # L2 norm per timestep
            "hidden_state_changes": self._compute_hidden_state_changes(lstm_output),
            "activation_patterns": self._analyze_activation_patterns(lstm_output),
        }

        return analysis_results

    def _compute_hidden_state_changes(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute changes in hidden states over time.

        Args:
            hidden_states: LSTM hidden states [batch_size, seq_len, hidden_size]

        Returns:
            Hidden state change magnitudes
        """
        # Compute differences between consecutive timesteps
        changes = torch.diff(hidden_states, dim=1)  # [batch_size, seq_len-1, hidden_size]
        change_magnitudes = torch.norm(changes, dim=2)  # [batch_size, seq_len-1]

        return change_magnitudes

    def _analyze_activation_patterns(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Analyze activation patterns in hidden states.

        Args:
            hidden_states: LSTM hidden states

        Returns:
            Activation pattern analysis
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Compute statistics over time dimension
        mean_activation = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        std_activation = hidden_states.std(dim=1)
        max_activation = hidden_states.max(dim=1)[0]
        min_activation = hidden_states.min(dim=1)[0]

        # Compute temporal correlation
        temporal_correlation = self._compute_temporal_correlation(hidden_states)

        return {
            "mean_activation": mean_activation,
            "std_activation": std_activation,
            "max_activation": max_activation,
            "min_activation": min_activation,
            "temporal_correlation": temporal_correlation,
        }

    def _compute_temporal_correlation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal correlation of hidden units.

        Args:
            hidden_states: Hidden states tensor

        Returns:
            Temporal correlation matrix
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Compute correlation for each batch
        correlations = []

        for b in range(batch_size):
            states = hidden_states[b]  # [seq_len, hidden_size]

            # Center the data
            states_centered = states - states.mean(dim=0, keepdim=True)

            # Compute correlation matrix
            cov = torch.mm(states_centered.T, states_centered) / (seq_len - 1)
            std = torch.sqrt(torch.diag(cov))
            correlation = cov / torch.outer(std, std)

            correlations.append(correlation)

        return torch.stack(correlations)

    def get_model_interpretation_summary(
        self,
        input_sequence: torch.Tensor,
        sequence_dates: pd.DatetimeIndex,
    ) -> dict[str, Any]:
        """
        Generate comprehensive interpretation summary.

        Args:
            input_sequence: Input sequence tensor
            sequence_dates: Corresponding dates for sequence

        Returns:
            Comprehensive interpretation summary
        """
        # Temporal attribution analysis
        attribution_results = self.analyze_temporal_importance(input_sequence)

        # Attention pattern analysis
        attention_results = self.analyze_attention_patterns(input_sequence)

        # Hidden state analysis
        hidden_state_results = self.analyze_hidden_state_patterns(input_sequence)

        # Extract key insights
        temporal_importance = attribution_results["temporal_importance_normalized"][0]
        attention_weights = attention_results["attention_weights"][0]

        # Find most important periods
        top_k = min(5, len(temporal_importance))
        top_attribution_indices = temporal_importance.topk(top_k)[1]
        top_attention_indices = attention_weights.topk(top_k)[1]

        summary = {
            "interpretation_method": self.config.attribution_method,
            "sequence_length": len(temporal_importance),
            "temporal_importance_stats": {
                "mean": float(temporal_importance.mean()),
                "std": float(temporal_importance.std()),
                "entropy": float(attribution_results["attribution_entropy"][0]),
            },
            "attention_stats": {
                "mean": float(attention_weights.mean()),
                "std": float(attention_weights.std()),
                "entropy": float(attention_results["attention_entropy"][0]),
                "concentration": float(attention_results["attention_concentration"][0]),
            },
            "most_important_periods": {
                "attribution_based": [
                    {
                        "date": sequence_dates[idx].strftime("%Y-%m-%d"),
                        "days_back": len(temporal_importance) - idx - 1,
                        "importance": float(temporal_importance[idx]),
                    }
                    for idx in top_attribution_indices
                ],
                "attention_based": [
                    {
                        "date": sequence_dates[idx].strftime("%Y-%m-%d"),
                        "days_back": len(attention_weights) - idx - 1,
                        "weight": float(attention_weights[idx]),
                    }
                    for idx in top_attention_indices
                ],
            },
            "hidden_state_insights": {
                "avg_norm": float(hidden_state_results["hidden_state_norms"].mean()),
                "max_change": float(hidden_state_results["hidden_state_changes"].max()),
            },
        }

        return summary
