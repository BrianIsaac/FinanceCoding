"""
Market regime analysis framework for portfolio visualization.

This module provides comprehensive market regime analysis including automatic regime detection,
regime-specific performance comparisons, regime transition analysis, and statistical
significance testing integration.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib/Seaborn not available. Static plotting disabled.", stacklevel=2)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("Plotly not available. Interactive plotting disabled.", stacklevel=2)

from src.evaluation.metrics.returns import PerformanceAnalytics
from src.evaluation.validation.significance import StatisticalValidation


@dataclass
class RegimeAnalysisConfig:
    """Configuration for market regime analysis."""

    figsize: tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = "whitegrid"
    color_palette: str = "husl"
    regime_detection_method: str = "hmm"  # 'hmm', 'kmeans', 'threshold'
    n_regimes: int = 3
    lookback_window: int = 252  # Trading days for regime detection
    regime_colors: dict[str, str] = None

    def __post_init__(self):
        if self.regime_colors is None:
            self.regime_colors = {
                "bull": "#2E7D32",  # Green
                "bear": "#C62828",  # Red
                "sideways": "#F57C00",  # Orange
                "neutral": "#6A6B6A",  # Gray
            }


class MarketRegimeAnalysis:
    """
    Market regime analysis framework for portfolio visualization.

    Provides comprehensive regime analysis including automatic detection,
    performance comparisons, transition analysis, and statistical integration.
    """

    def __init__(self, config: RegimeAnalysisConfig = None):
        """
        Initialize market regime analysis framework.

        Args:
            config: Configuration for regime analysis behavior
        """
        self.config = config or RegimeAnalysisConfig()

        if HAS_MATPLOTLIB:
            sns.set_style(self.config.style)
            plt.rcParams["figure.dpi"] = self.config.dpi

        self.statistical_validator = StatisticalValidation()

    def detect_market_regimes(
        self, market_data: pd.Series, method: str | None = None, n_regimes: int | None = None
    ) -> pd.Series:
        """
        Implement automatic regime detection (bull, bear, sideways markets).

        Args:
            market_data: Market index or benchmark return series
            method: Detection method ('hmm', 'kmeans', 'threshold')
            n_regimes: Number of regimes to detect

        Returns:
            Series with regime labels indexed by date
        """
        method = method or self.config.regime_detection_method
        n_regimes = n_regimes or self.config.n_regimes

        if method == "threshold":
            return self._detect_regimes_threshold(market_data)
        elif method == "kmeans":
            return self._detect_regimes_kmeans(market_data, n_regimes)
        elif method == "hmm":
            return self._detect_regimes_hmm(market_data, n_regimes)
        else:
            raise ValueError(f"Unknown regime detection method: {method}")

    def _detect_regimes_threshold(self, market_data: pd.Series) -> pd.Series:
        """Detect regimes using simple threshold-based approach."""
        # Calculate rolling statistics
        window = self.config.lookback_window
        rolling_return = market_data.rolling(window=window).mean() * 252  # Annualized
        rolling_volatility = market_data.rolling(window=window).std() * np.sqrt(252)

        # Define regime thresholds
        regimes = pd.Series(index=market_data.index, dtype="object")

        for i in range(len(market_data)):
            if pd.isna(rolling_return.iloc[i]) or pd.isna(rolling_volatility.iloc[i]):
                regimes.iloc[i] = "neutral"
                continue

            ret = rolling_return.iloc[i]
            vol = rolling_volatility.iloc[i]

            if ret > 0.05 and vol < 0.25:  # High return, low volatility
                regimes.iloc[i] = "bull"
            elif ret < -0.05 or vol > 0.35:  # Negative return or high volatility
                regimes.iloc[i] = "bear"
            else:
                regimes.iloc[i] = "sideways"

        return regimes

    def _detect_regimes_kmeans(self, market_data: pd.Series, n_regimes: int) -> pd.Series:
        """Detect regimes using K-means clustering."""
        # Prepare features
        features = self._prepare_regime_features(market_data)

        # Apply K-means
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features.dropna())

        # Map clusters to regime names
        regime_mapping = self._map_clusters_to_regimes(
            features.dropna(), cluster_labels, kmeans.cluster_centers_
        )

        # Create regime series
        regimes = pd.Series(index=market_data.index, dtype="object")
        valid_indices = features.dropna().index

        for i, idx in enumerate(valid_indices):
            regimes[idx] = regime_mapping[cluster_labels[i]]

        # Forward fill for missing values
        regimes = regimes.fillna(method="ffill").fillna("neutral")

        return regimes

    def _detect_regimes_hmm(self, market_data: pd.Series, n_regimes: int) -> pd.Series:
        """Detect regimes using Hidden Markov Model approach (simplified)."""
        # For simplicity, use Gaussian Mixture Model as proxy for HMM
        features = self._prepare_regime_features(market_data)

        # Apply Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_regimes, random_state=42, covariance_type="full")
        cluster_labels = gmm.fit_predict(features.dropna())

        # Map clusters to regime names
        regime_mapping = self._map_clusters_to_regimes(
            features.dropna(), cluster_labels, gmm.means_
        )

        # Create regime series
        regimes = pd.Series(index=market_data.index, dtype="object")
        valid_indices = features.dropna().index

        for i, idx in enumerate(valid_indices):
            regimes[idx] = regime_mapping[cluster_labels[i]]

        # Forward fill for missing values
        regimes = regimes.fillna(method="ffill").fillna("neutral")

        return regimes

    def _prepare_regime_features(self, market_data: pd.Series) -> pd.DataFrame:
        """Prepare features for regime detection."""
        window = min(self.config.lookback_window, len(market_data) // 4)

        features = pd.DataFrame(index=market_data.index)

        # Rolling statistics
        features["return"] = market_data.rolling(window=window).mean() * 252
        features["volatility"] = market_data.rolling(window=window).std() * np.sqrt(252)
        features["skewness"] = market_data.rolling(window=window).skew()
        features["kurtosis"] = market_data.rolling(window=window).kurt()

        # Momentum indicators
        features["momentum_short"] = market_data.rolling(window=window // 4).mean()
        features["momentum_long"] = market_data.rolling(window=window).mean()
        features["momentum_ratio"] = features["momentum_short"] / features["momentum_long"]

        # Drawdown
        cum_returns = (1 + market_data).cumprod()
        running_max = cum_returns.expanding(min_periods=1).max()
        features["drawdown"] = (cum_returns - running_max) / running_max

        return features

    def _map_clusters_to_regimes(
        self, features: pd.DataFrame, labels: np.ndarray, centers: np.ndarray
    ) -> dict[int, str]:
        """Map cluster labels to regime names based on characteristics."""
        regime_mapping = {}

        for cluster_id in range(len(centers)):
            center = centers[cluster_id]

            if len(center) >= 2:  # Ensure we have return and volatility features
                avg_return = center[0]  # Assuming return is first feature
                avg_volatility = center[1]  # Assuming volatility is second feature

                if avg_return > 0.05 and avg_volatility < 0.25:
                    regime_mapping[cluster_id] = "bull"
                elif avg_return < -0.05 or avg_volatility > 0.35:
                    regime_mapping[cluster_id] = "bear"
                else:
                    regime_mapping[cluster_id] = "sideways"
            else:
                regime_mapping[cluster_id] = "neutral"

        # Ensure we have all regime types
        regime_types = list(regime_mapping.values())
        if len(set(regime_types)) < len(self.config.regime_colors):
            # Fill in missing regimes
            missing_regimes = set(self.config.regime_colors.keys()) - set(regime_types)
            for i, regime in enumerate(missing_regimes):
                if i < len(regime_mapping):
                    # Replace least confident assignment
                    regime_mapping[i] = regime

        return regime_mapping

    def create_regime_specific_performance_table(
        self,
        returns_data: dict[str, pd.Series],
        regime_labels: pd.Series,
        statistical_results: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Create regime-specific performance comparison tables.

        Args:
            returns_data: Dictionary mapping approach names to return series
            regime_labels: Series with regime labels
            statistical_results: Optional statistical significance results

        Returns:
            Dictionary mapping regime names to performance DataFrames
        """
        regime_performance = {}
        unique_regimes = regime_labels.dropna().unique()

        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            regime_performance_data = {}

            for approach, returns_series in returns_data.items():
                # Filter returns for this regime
                regime_returns = returns_series[regime_mask]

                if len(regime_returns) > 10:  # Need sufficient data
                    # Calculate performance metrics
                    perf_analytics = PerformanceAnalytics()

                    metrics = {
                        "total_return": (1 + regime_returns).prod() - 1,
                        "annualized_return": (1 + regime_returns.mean()) ** 252 - 1,
                        "volatility": regime_returns.std() * np.sqrt(252),
                        "sharpe_ratio": perf_analytics.sharpe_ratio(regime_returns),
                        "max_drawdown": perf_analytics.maximum_drawdown(regime_returns)[0],
                        "win_rate": (regime_returns > 0).mean(),
                        "periods": len(regime_returns),
                    }

                    # Add statistical significance if available
                    if statistical_results and approach in statistical_results:
                        regime_stats = statistical_results[approach].get(f"regime_{regime}", {})
                        metrics["p_value"] = regime_stats.get("p_value", 1.0)
                        metrics["significance"] = self._get_significance_symbol(metrics["p_value"])

                    regime_performance_data[approach] = metrics

            if regime_performance_data:
                regime_df = pd.DataFrame(regime_performance_data).T
                regime_performance[regime] = regime_df

        return regime_performance

    def create_regime_transition_analysis(
        self,
        returns_data: dict[str, pd.Series],
        regime_labels: pd.Series,
        save_path: str | Path | None = None,
        interactive: bool = True,
    ) -> plt.Figure | go.Figure:
        """
        Build regime transition analysis showing model adaptation.

        Args:
            returns_data: Dictionary mapping approach names to return series
            regime_labels: Series with regime labels
            save_path: Path to save the analysis
            interactive: Whether to create interactive plot

        Returns:
            Matplotlib Figure or Plotly Figure
        """
        # Calculate transition probabilities and performance
        transition_data = self._analyze_regime_transitions(returns_data, regime_labels)

        if interactive and HAS_PLOTLY:
            return self._create_transition_analysis_interactive(transition_data, save_path)
        elif HAS_MATPLOTLIB:
            return self._create_transition_analysis_static(transition_data, save_path)
        else:
            raise ImportError("Neither Plotly nor Matplotlib available for plotting")

    def _analyze_regime_transitions(
        self, returns_data: dict[str, pd.Series], regime_labels: pd.Series
    ) -> dict[str, Any]:
        """Analyze regime transitions and performance adaptation."""
        # Calculate transition matrix
        unique_regimes = regime_labels.dropna().unique()
        n_regimes = len(unique_regimes)

        transition_matrix = pd.DataFrame(
            np.zeros((n_regimes, n_regimes)), index=unique_regimes, columns=unique_regimes
        )

        # Count transitions
        for i in range(len(regime_labels) - 1):
            if pd.notna(regime_labels.iloc[i]) and pd.notna(regime_labels.iloc[i + 1]):
                current_regime = regime_labels.iloc[i]
                next_regime = regime_labels.iloc[i + 1]
                transition_matrix.loc[current_regime, next_regime] += 1

        # Normalize to probabilities
        transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)

        # Calculate performance adaptation
        adaptation_performance = {}
        for approach, returns_series in returns_data.items():
            approach_adaptation = {}

            # Find regime change points
            regime_changes = regime_labels != regime_labels.shift(1)
            change_points = regime_labels[regime_changes].index

            # Analyze performance around transitions
            pre_transition_performance = []
            post_transition_performance = []

            for change_point in change_points:
                if change_point in returns_series.index:
                    change_idx = returns_series.index.get_loc(change_point)

                    # Pre-transition window (10 days before)
                    pre_start = max(0, change_idx - 10)
                    pre_returns = returns_series.iloc[pre_start:change_idx]
                    if len(pre_returns) > 0:
                        pre_transition_performance.append(pre_returns.mean())

                    # Post-transition window (10 days after)
                    post_end = min(len(returns_series), change_idx + 10)
                    post_returns = returns_series.iloc[change_idx:post_end]
                    if len(post_returns) > 0:
                        post_transition_performance.append(post_returns.mean())

            approach_adaptation["pre_transition_mean"] = (
                np.mean(pre_transition_performance) if pre_transition_performance else 0
            )
            approach_adaptation["post_transition_mean"] = (
                np.mean(post_transition_performance) if post_transition_performance else 0
            )
            approach_adaptation["adaptation_score"] = (
                approach_adaptation["post_transition_mean"]
                - approach_adaptation["pre_transition_mean"]
            )

            adaptation_performance[approach] = approach_adaptation

        return {
            "transition_matrix": transition_matrix,
            "adaptation_performance": adaptation_performance,
            "regime_labels": regime_labels,
        }

    def _create_transition_analysis_interactive(
        self, transition_data: dict[str, Any], save_path: str | Path | None
    ) -> go.Figure:
        """Create interactive regime transition analysis using Plotly."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Regime Transition Matrix",
                "Regime Timeline",
                "Adaptation Performance",
                "Transition Statistics",
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "table"}],
            ],
            vertical_spacing=0.1,
        )

        transition_matrix = transition_data["transition_matrix"]
        adaptation_performance = transition_data["adaptation_performance"]
        regime_labels = transition_data["regime_labels"]

        # 1. Transition matrix heatmap (top left)
        fig.add_trace(
            go.Heatmap(
                z=transition_matrix.values,
                x=list(transition_matrix.columns),
                y=list(transition_matrix.index),
                colorscale="Blues",
                showscale=True,
                colorbar={"title": "Probability"},
                hovertemplate="From %{y} to %{x}<br>Probability: %{z:.2%}<extra></extra>",
                text=[[f"{val:.2%}" for val in row] for row in transition_matrix.values],
                texttemplate="%{text}",
                textfont={"size": 10},
            ),
            row=1,
            col=1,
        )

        # 2. Regime timeline (top right)
        regime_numeric = regime_labels.map(
            {regime: idx for idx, regime in enumerate(transition_matrix.index)}
        )

        fig.add_trace(
            go.Scatter(
                x=regime_labels.index,
                y=regime_numeric.values,
                mode="lines",
                name="Market Regime",
                line={"width": 2},
                hovertemplate="Date: %{x}<br>Regime: %{text}<extra></extra>",
                text=regime_labels.values,
            ),
            row=1,
            col=2,
        )

        # 3. Adaptation performance (bottom left)
        approaches = list(adaptation_performance.keys())
        # Convert to bps
        adaptation_scores = [
            data["adaptation_score"] * 10000 for data in adaptation_performance.values()
        ]

        fig.add_trace(
            go.Bar(
                x=approaches,
                y=adaptation_scores,
                name="Adaptation Score",
                marker_color="lightcoral",
                hovertemplate="<b>%{x}</b><br>Adaptation Score: %{y:.1f} bps<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # 4. Summary statistics table (bottom right)
        regime_counts = regime_labels.value_counts()
        regime_durations = self._calculate_regime_durations(regime_labels)

        table_data = []
        for regime in transition_matrix.index:
            count = regime_counts.get(regime, 0)
            avg_duration = regime_durations.get(regime, 0)
            table_data.append([regime, count, f"{avg_duration:.1f}"])

        fig.add_trace(
            go.Table(
                header={
                    "values": ["Regime", "Periods", "Avg Duration"],
                    "fill_color": "lightblue",
                    "align": "left",
                },
                cells={"values": list(zip(*table_data)), "fill_color": "white", "align": "left"},
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(title="Market Regime Transition Analysis", height=800, showlegend=False)

        # Update y-axis for regime timeline
        fig.update_yaxes(
            tickvals=list(range(len(transition_matrix.index))),
            ticktext=list(transition_matrix.index),
            row=1,
            col=2,
        )

        if save_path:
            fig.write_html(str(save_path).replace(".png", ".html"))

        return fig

    def _create_transition_analysis_static(
        self, transition_data: dict[str, Any], save_path: str | Path | None
    ) -> plt.Figure:
        """Create static regime transition analysis using Matplotlib."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Market Regime Transition Analysis", fontsize=16, fontweight="bold")

        transition_matrix = transition_data["transition_matrix"]
        adaptation_performance = transition_data["adaptation_performance"]
        regime_labels = transition_data["regime_labels"]

        # 1. Transition matrix heatmap (top left)
        ax1 = axes[0, 0]
        sns.heatmap(
            transition_matrix,
            ax=ax1,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            cbar_kws={"format": "%.0%%"},
        )
        ax1.set_title("Regime Transition Matrix")
        ax1.set_xlabel("To Regime")
        ax1.set_ylabel("From Regime")

        # 2. Regime timeline (top right)
        ax2 = axes[0, 1]
        [self.config.regime_colors.get(regime, "gray") for regime in regime_labels.values]

        # Create regime blocks
        unique_regimes = regime_labels.dropna().unique()
        y_positions = {regime: idx for idx, regime in enumerate(unique_regimes)}

        prev_regime = None
        start_idx = None

        for idx, (_date, regime) in enumerate(regime_labels.items()):
            if regime != prev_regime:
                if prev_regime is not None and start_idx is not None:
                    # Draw previous regime block
                    width = idx - start_idx
                    rect = Rectangle(
                        (start_idx, y_positions[prev_regime] - 0.4),
                        width,
                        0.8,
                        facecolor=self.config.regime_colors.get(prev_regime, "gray"),
                        alpha=0.7,
                    )
                    ax2.add_patch(rect)

                start_idx = idx
                prev_regime = regime

        # Draw last regime block
        if prev_regime is not None and start_idx is not None:
            width = len(regime_labels) - start_idx
            rect = Rectangle(
                (start_idx, y_positions[prev_regime] - 0.4),
                width,
                0.8,
                facecolor=self.config.regime_colors.get(prev_regime, "gray"),
                alpha=0.7,
            )
            ax2.add_patch(rect)

        ax2.set_title("Market Regime Timeline")
        ax2.set_xlabel("Time Period")
        ax2.set_ylabel("Regime")
        ax2.set_yticks(list(y_positions.values()))
        ax2.set_yticklabels(list(y_positions.keys()))
        ax2.set_xlim(0, len(regime_labels))

        # 3. Adaptation performance (bottom left)
        ax3 = axes[1, 0]
        approaches = list(adaptation_performance.keys())
        adaptation_scores = [
            data["adaptation_score"] * 10000 for data in adaptation_performance.values()
        ]  # Convert to bps

        colors = sns.color_palette(self.config.color_palette, len(approaches))
        bars = ax3.bar(approaches, adaptation_scores, color=colors, alpha=0.7)

        ax3.set_title("Regime Adaptation Performance")
        ax3.set_xlabel("Approach")
        ax3.set_ylabel("Adaptation Score (bps)")
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color="black", linestyle="-", alpha=0.5)

        # Add value labels on bars
        for bar, score in zip(bars, adaptation_scores):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{score:.1f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
            )

        # Rotate x-axis labels if needed
        if len(approaches) > 3:
            ax3.tick_params(axis="x", rotation=45)

        # 4. Regime statistics (bottom right)
        ax4 = axes[1, 1]
        regime_counts = regime_labels.value_counts()
        regime_durations = self._calculate_regime_durations(regime_labels)

        # Create grouped bar chart
        regimes = list(regime_counts.index)
        counts = list(regime_counts.values)
        durations = [regime_durations.get(regime, 0) for regime in regimes]

        x = np.arange(len(regimes))
        width = 0.35

        ax4.bar(x - width / 2, counts, width, label="Period Count", alpha=0.8)
        ax4_twin = ax4.twinx()
        ax4_twin.bar(
            x + width / 2, durations, width, label="Avg Duration", alpha=0.8, color="orange"
        )

        ax4.set_title("Regime Statistics")
        ax4.set_xlabel("Regime")
        ax4.set_ylabel("Period Count")
        ax4_twin.set_ylabel("Average Duration (days)")
        ax4.set_xticks(x)
        ax4.set_xticklabels(regimes)

        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        return fig

    def _calculate_regime_durations(self, regime_labels: pd.Series) -> dict[str, float]:
        """Calculate average duration for each regime."""
        durations = {}
        current_regime = None
        current_duration = 0
        regime_durations = {}

        for regime in regime_labels.values:
            if regime != current_regime:
                if current_regime is not None:
                    if current_regime not in regime_durations:
                        regime_durations[current_regime] = []
                    regime_durations[current_regime].append(current_duration)

                current_regime = regime
                current_duration = 1
            else:
                current_duration += 1

        # Add last regime
        if current_regime is not None:
            if current_regime not in regime_durations:
                regime_durations[current_regime] = []
            regime_durations[current_regime].append(current_duration)

        # Calculate averages
        for regime, duration_list in regime_durations.items():
            durations[regime] = np.mean(duration_list)

        return durations

    def add_regime_based_statistical_testing(
        self,
        returns_data: dict[str, pd.Series],
        regime_labels: pd.Series,
        baseline_returns: dict[str, pd.Series] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Add regime-based statistical significance testing integration.

        Args:
            returns_data: Dictionary mapping approach names to return series
            regime_labels: Series with regime labels
            baseline_returns: Optional baseline returns for comparison

        Returns:
            Dictionary with statistical test results by approach and regime
        """
        statistical_results = {}
        unique_regimes = regime_labels.dropna().unique()

        for approach, returns_series in returns_data.items():
            approach_results = {}

            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                regime_returns = returns_series[regime_mask]

                if len(regime_returns) > 10:  # Need sufficient data
                    regime_stats = {}

                    # Test against zero (market neutral performance)
                    t_stat, p_value = stats.ttest_1samp(regime_returns, 0)
                    regime_stats["t_stat_vs_zero"] = t_stat
                    regime_stats["p_value_vs_zero"] = p_value

                    # Test against baseline if available
                    if baseline_returns:
                        for baseline_name, baseline_series in baseline_returns.items():
                            baseline_regime_returns = baseline_series[regime_mask]

                            if len(baseline_regime_returns) > 10:
                                # Align series
                                aligned_approach, aligned_baseline = regime_returns.align(
                                    baseline_regime_returns, join="inner"
                                )

                                if len(aligned_approach) > 10:
                                    t_stat, p_value = stats.ttest_rel(
                                        aligned_approach, aligned_baseline
                                    )
                                    regime_stats[f"t_stat_vs_{baseline_name}"] = t_stat
                                    regime_stats[f"p_value_vs_{baseline_name}"] = p_value

                    # Calculate effect size (Cohen's d)
                    if len(regime_returns) > 1:
                        cohen_d = regime_returns.mean() / regime_returns.std()
                        regime_stats["cohens_d"] = cohen_d

                    approach_results[f"regime_{regime}"] = regime_stats

            statistical_results[approach] = approach_results

        return statistical_results

    def _get_significance_symbol(self, p_value: float) -> str:
        """Get significance symbol based on p-value."""
        if p_value <= 0.001:
            return "***"
        elif p_value <= 0.01:
            return "**"
        elif p_value <= 0.05:
            return "*"
        else:
            return ""

    def export_regime_analysis(
        self, figure: plt.Figure | go.Figure, filename: str, formats: list[str] = None
    ) -> dict[str, str]:
        """
        Export regime analysis in multiple formats.

        Args:
            figure: Matplotlib or Plotly figure to export
            filename: Base filename for export
            formats: Export formats (png, pdf, html, svg)

        Returns:
            Dictionary mapping format names to file paths
        """
        if formats is None:
            formats = ["png", "html"] if HAS_PLOTLY else ["png"]

        exported_files = {}

        for format_type in formats:
            filepath = f"{filename}.{format_type}"

            try:
                if isinstance(figure, go.Figure):
                    # Plotly figure
                    if format_type in ["png", "pdf", "svg"]:
                        figure.write_image(filepath)
                    elif format_type == "html":
                        figure.write_html(filepath)
                    exported_files[format_type] = filepath

                elif HAS_MATPLOTLIB and hasattr(figure, "savefig"):
                    # Matplotlib figure
                    if format_type in ["png", "pdf", "svg"]:
                        figure.savefig(filepath, dpi=self.config.dpi, bbox_inches="tight")
                        exported_files[format_type] = filepath

            except Exception as e:
                warnings.warn(f"Failed to export {format_type}: {e}", stacklevel=2)

        return exported_files
