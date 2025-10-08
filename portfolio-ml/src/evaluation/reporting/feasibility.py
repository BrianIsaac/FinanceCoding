"""
Implementation feasibility assessment framework for comprehensive model evaluation.

This module provides detailed feasibility analysis including computational requirements,
operational complexity, total cost of ownership, and implementation timeline assessment
for institutional deployment decisions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Matplotlib/Seaborn support can be added when needed
HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("Plotly not available. Interactive plotting disabled.", stacklevel=2)


@dataclass
class FeasibilityConfig:
    """Configuration for feasibility assessment framework."""

    # Cost benchmarks (annual USD)
    developer_cost_per_hour: float = 150.0
    gpu_cost_per_hour: float = 2.5
    infrastructure_base_cost: float = 50000.0
    monitoring_cost_multiplier: float = 0.1

    # Complexity scoring weights
    computational_weight: float = 0.35
    operational_weight: float = 0.30
    maintenance_weight: float = 0.20
    regulatory_weight: float = 0.15

    # Implementation timeline factors (months)
    base_implementation_time: float = 3.0
    complexity_multiplier: float = 1.5
    testing_validation_time: float = 2.0


class ImplementationFeasibilityAssessor:
    """
    Comprehensive implementation feasibility assessor for institutional deployment.

    Evaluates computational requirements, operational complexity, total cost of ownership,
    and implementation timelines to guide strategic deployment decisions.
    """

    def __init__(self, config: FeasibilityConfig = None):
        """
        Initialize implementation feasibility assessor.

        Args:
            config: Feasibility assessment configuration
        """
        self.config = config or FeasibilityConfig()

    def assess_computational_requirements(
        self,
        model_specifications: dict[str, dict[str, Any]],
        operational_metrics: dict[str, dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Assess computational requirements for each model approach.

        Args:
            model_specifications: Technical specifications for each model
            operational_metrics: Runtime operational metrics

        Returns:
            DataFrame with computational requirements analysis
        """
        computational_data = []

        for model_name, specs in model_specifications.items():
            # GPU memory requirements
            gpu_memory_gb = specs.get("gpu_memory_required", 4.0)
            gpu_utilization = specs.get("gpu_utilization_percent", 70.0)

            # Training computational requirements
            training_time_hours = specs.get("training_time_hours", 2.0)
            training_frequency_days = specs.get("retraining_frequency_days", 30)
            training_cost_per_cycle = training_time_hours * self.config.gpu_cost_per_hour

            # Inference computational requirements
            inference_time_ms = specs.get("inference_time_ms", 10.0)
            daily_predictions = specs.get("daily_predictions", 500)
            inference_cost_per_day = (
                daily_predictions * inference_time_ms / 1000 / 3600
            ) * self.config.gpu_cost_per_hour

            # Annual computational costs
            annual_training_cost = training_cost_per_cycle * (365 / training_frequency_days)
            annual_inference_cost = inference_cost_per_day * 365
            total_annual_compute_cost = annual_training_cost + annual_inference_cost

            # Complexity scoring (0-100, higher is more complex)
            gpu_complexity = min(100, (gpu_memory_gb / 12.0) * 100)  # 12GB baseline
            time_complexity = min(100, (training_time_hours / 24.0) * 100)  # 24 hours baseline
            frequency_complexity = max(
                0, 100 - (training_frequency_days / 30.0) * 100
            )  # 30 days baseline

            computational_complexity = (gpu_complexity + time_complexity + frequency_complexity) / 3

            # Integration with operational metrics if available
            actual_performance = {}
            if operational_metrics and model_name in operational_metrics:
                ops = operational_metrics[model_name]
                actual_performance = {
                    "Actual GPU Usage (GB)": ops.get("actual_gpu_memory", gpu_memory_gb),
                    "Actual Training Time (h)": ops.get(
                        "actual_training_time", training_time_hours
                    ),
                    "Actual Inference Time (ms)": ops.get(
                        "actual_inference_time", inference_time_ms
                    ),
                }

            computational_data.append(
                {
                    "Model": model_name,
                    "GPU Memory Required (GB)": gpu_memory_gb,
                    "GPU Utilization (%)": gpu_utilization,
                    "Training Time (hours)": training_time_hours,
                    "Retraining Frequency (days)": training_frequency_days,
                    "Inference Time (ms)": inference_time_ms,
                    "Daily Predictions": daily_predictions,
                    "Annual Training Cost ($)": f"{annual_training_cost:,.0f}",
                    "Annual Inference Cost ($)": f"{annual_inference_cost:,.0f}",
                    "Total Annual Compute Cost ($)": f"{total_annual_compute_cost:,.0f}",
                    "Computational Complexity (0-100)": f"{computational_complexity:.1f}",
                    **actual_performance,
                }
            )

        return pd.DataFrame(computational_data)

    def assess_operational_complexity(
        self,
        model_specifications: dict[str, dict[str, Any]],
        operational_metrics: dict[str, dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Assess operational complexity and maintenance requirements.

        Args:
            model_specifications: Technical specifications for each model
            operational_metrics: Runtime operational metrics

        Returns:
            DataFrame with operational complexity analysis
        """
        operational_data = []

        for model_name, specs in model_specifications.items():
            # Data requirements
            data_sources = specs.get("data_sources", 1)
            data_preprocessing_steps = specs.get("preprocessing_steps", 3)
            data_update_frequency = specs.get("data_update_frequency_hours", 24)

            # Model complexity
            model_parameters = specs.get("model_parameters", 100000)
            hyperparameters = specs.get("hyperparameters_count", 5)
            feature_engineering_steps = specs.get("feature_engineering_steps", 2)

            # Monitoring requirements
            performance_metrics_tracked = specs.get("performance_metrics", 10)
            alert_thresholds = specs.get("alert_thresholds", 5)
            model_drift_monitoring = specs.get("drift_monitoring_enabled", True)

            # Operational scoring (0-100, higher is more complex)
            data_complexity = min(100, (data_sources * 20 + data_preprocessing_steps * 10))
            model_complexity = min(100, np.log10(model_parameters) * 10 + hyperparameters * 5)
            monitoring_complexity = performance_metrics_tracked * 5 + alert_thresholds * 10

            overall_complexity = (data_complexity + model_complexity + monitoring_complexity) / 3

            # Maintenance effort estimation (hours per month)
            data_maintenance_hours = data_sources * 8 + data_preprocessing_steps * 2
            model_maintenance_hours = hyperparameters * 1 + (10 if model_drift_monitoring else 5)
            monitoring_maintenance_hours = performance_metrics_tracked * 1 + alert_thresholds * 2

            total_maintenance_hours = (
                data_maintenance_hours + model_maintenance_hours + monitoring_maintenance_hours
            )
            monthly_maintenance_cost = total_maintenance_hours * self.config.developer_cost_per_hour

            # Integration with operational metrics
            actual_operations = {}
            if operational_metrics and model_name in operational_metrics:
                ops = operational_metrics[model_name]
                actual_operations = {
                    "Actual Downtime (%)": ops.get("downtime_percentage", 0.0),
                    "Alert Frequency (per week)": ops.get("alerts_per_week", 2),
                    "Manual Interventions (per month)": ops.get("manual_interventions", 1),
                }

            operational_data.append(
                {
                    "Model": model_name,
                    "Data Sources": data_sources,
                    "Preprocessing Steps": data_preprocessing_steps,
                    "Data Update Frequency (hours)": data_update_frequency,
                    "Model Parameters": f"{model_parameters:,}",
                    "Hyperparameters": hyperparameters,
                    "Feature Engineering Steps": feature_engineering_steps,
                    "Performance Metrics Tracked": performance_metrics_tracked,
                    "Alert Thresholds": alert_thresholds,
                    "Drift Monitoring": "Yes" if model_drift_monitoring else "No",
                    "Data Complexity (0-100)": f"{data_complexity:.1f}",
                    "Model Complexity (0-100)": f"{model_complexity:.1f}",
                    "Monitoring Complexity (0-100)": f"{monitoring_complexity:.1f}",
                    "Overall Complexity (0-100)": f"{overall_complexity:.1f}",
                    "Monthly Maintenance Hours": f"{total_maintenance_hours:.1f}",
                    "Monthly Maintenance Cost ($)": f"{monthly_maintenance_cost:,.0f}",
                    **actual_operations,
                }
            )

        return pd.DataFrame(operational_data)

    def calculate_total_cost_ownership(
        self,
        computational_requirements: pd.DataFrame,
        operational_complexity: pd.DataFrame,
        implementation_specifications: dict[str, dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Calculate total cost of ownership for each model approach.

        Args:
            computational_requirements: Computational requirements analysis
            operational_complexity: Operational complexity analysis
            implementation_specifications: Implementation-specific cost factors

        Returns:
            DataFrame with total cost of ownership analysis
        """
        tco_data = []

        for _, comp_row in computational_requirements.iterrows():
            model_name = comp_row["Model"]

            # Find corresponding operational row
            ops_row = operational_complexity[operational_complexity["Model"] == model_name].iloc[0]

            # Extract costs
            annual_compute_cost = float(comp_row["Total Annual Compute Cost ($)"].replace(",", ""))
            monthly_maintenance_cost = float(
                ops_row["Monthly Maintenance Cost ($)"].replace(",", "")
            )
            annual_maintenance_cost = monthly_maintenance_cost * 12

            # Implementation costs
            impl_specs = (
                implementation_specifications.get(model_name, {})
                if implementation_specifications
                else {}
            )

            # One-time implementation costs
            development_months = impl_specs.get("development_months", 4.0)
            development_team_size = impl_specs.get("team_size", 2.0)
            development_cost = (
                development_months
                * development_team_size
                * 160
                * self.config.developer_cost_per_hour
            )  # 160 hours per month

            infrastructure_setup_cost = impl_specs.get(
                "infrastructure_setup", self.config.infrastructure_base_cost
            )
            testing_validation_cost = impl_specs.get("testing_cost", development_cost * 0.3)
            training_cost = impl_specs.get("training_cost", 20000.0)

            total_implementation_cost = (
                development_cost
                + infrastructure_setup_cost
                + testing_validation_cost
                + training_cost
            )

            # Ongoing annual costs
            infrastructure_annual_cost = impl_specs.get(
                "annual_infrastructure", self.config.infrastructure_base_cost * 0.3
            )
            licensing_annual_cost = impl_specs.get("annual_licensing", 10000.0)
            compliance_annual_cost = impl_specs.get("annual_compliance", 15000.0)

            total_annual_operating_cost = (
                annual_compute_cost
                + annual_maintenance_cost
                + infrastructure_annual_cost
                + licensing_annual_cost
                + compliance_annual_cost
            )

            # 3-year TCO calculation
            three_year_tco = total_implementation_cost + (total_annual_operating_cost * 3)

            # Cost per prediction (approximation)
            daily_predictions = comp_row["Daily Predictions"]
            annual_predictions = daily_predictions * 365
            cost_per_prediction = (
                total_annual_operating_cost / annual_predictions if annual_predictions > 0 else 0
            )

            # Risk-adjusted costs (implementation risk premium)
            overall_complexity = float(ops_row["Overall Complexity (0-100)"])
            risk_premium = (
                1 + (overall_complexity / 100) * 0.5
            )  # Up to 50% premium for high complexity
            risk_adjusted_tco = three_year_tco * risk_premium

            tco_data.append(
                {
                    "Model": model_name,
                    "Development Cost ($)": f"{development_cost:,.0f}",
                    "Infrastructure Setup ($)": f"{infrastructure_setup_cost:,.0f}",
                    "Testing & Validation ($)": f"{testing_validation_cost:,.0f}",
                    "Training Cost ($)": f"{training_cost:,.0f}",
                    "Total Implementation ($)": f"{total_implementation_cost:,.0f}",
                    "Annual Compute ($)": f"{annual_compute_cost:,.0f}",
                    "Annual Maintenance ($)": f"{annual_maintenance_cost:,.0f}",
                    "Annual Infrastructure ($)": f"{infrastructure_annual_cost:,.0f}",
                    "Annual Licensing ($)": f"{licensing_annual_cost:,.0f}",
                    "Annual Compliance ($)": f"{compliance_annual_cost:,.0f}",
                    "Total Annual Operating ($)": f"{total_annual_operating_cost:,.0f}",
                    "3-Year TCO ($)": f"{three_year_tco:,.0f}",
                    "Risk-Adjusted TCO ($)": f"{risk_adjusted_tco:,.0f}",
                    "Cost per Prediction ($)": f"{cost_per_prediction:.4f}",
                    "Complexity Risk Premium": f"{(risk_premium-1)*100:.1f}%",
                }
            )

        return pd.DataFrame(tco_data)

    def assess_implementation_timeline(
        self,
        model_specifications: dict[str, dict[str, Any]],
        organizational_factors: dict[str, Any] = None,
    ) -> pd.DataFrame:
        """
        Assess implementation timeline for each model approach.

        Args:
            model_specifications: Technical specifications for each model
            organizational_factors: Organization-specific implementation factors

        Returns:
            DataFrame with implementation timeline analysis
        """
        timeline_data = []

        # Default organizational factors
        if organizational_factors is None:
            organizational_factors = {
                "team_experience_level": "medium",  # low, medium, high
                "existing_infrastructure": "partial",  # none, partial, complete
                "regulatory_complexity": "standard",  # low, standard, high
                "change_management_maturity": "medium",  # low, medium, high
            }

        for model_name, specs in model_specifications.items():
            # Base timeline components (months)
            base_development = self.config.base_implementation_time

            # Complexity adjustments
            complexity_score = specs.get("complexity_score", 50) / 100  # 0-1 scale
            complexity_adjustment = (
                base_development * complexity_score * self.config.complexity_multiplier
            )

            # Technical implementation phases
            data_pipeline_setup = specs.get("data_pipeline_complexity", 1.0)  # months
            model_development = base_development + complexity_adjustment
            integration_testing = specs.get("integration_complexity", 1.5)  # months
            validation_testing = self.config.testing_validation_time
            deployment_setup = specs.get("deployment_complexity", 1.0)  # months

            # Organizational adjustments
            org_multiplier = self._calculate_organizational_multiplier(organizational_factors)

            # Adjusted timeline
            adjusted_data_pipeline = data_pipeline_setup * org_multiplier
            adjusted_development = model_development * org_multiplier
            adjusted_integration = integration_testing * org_multiplier
            adjusted_validation = validation_testing * org_multiplier
            adjusted_deployment = deployment_setup * org_multiplier

            # Total timeline
            total_months = (
                adjusted_data_pipeline
                + adjusted_development
                + adjusted_integration
                + adjusted_validation
                + adjusted_deployment
            )

            # Risk adjustments
            technical_risk = specs.get("technical_risk_score", 50) / 100  # 0-1 scale
            risk_buffer = total_months * technical_risk * 0.3  # Up to 30% buffer

            final_timeline = total_months + risk_buffer

            # Resource requirements
            peak_team_size = specs.get("peak_team_size", 3)
            total_person_months = final_timeline * peak_team_size * 0.7  # 70% utilization

            timeline_data.append(
                {
                    "Model": model_name,
                    "Data Pipeline Setup (months)": f"{adjusted_data_pipeline:.1f}",
                    "Model Development (months)": f"{adjusted_development:.1f}",
                    "Integration Testing (months)": f"{adjusted_integration:.1f}",
                    "Validation Testing (months)": f"{adjusted_validation:.1f}",
                    "Deployment Setup (months)": f"{adjusted_deployment:.1f}",
                    "Base Timeline (months)": f"{total_months:.1f}",
                    "Risk Buffer (months)": f"{risk_buffer:.1f}",
                    "Total Timeline (months)": f"{final_timeline:.1f}",
                    "Peak Team Size": peak_team_size,
                    "Total Person-Months": f"{total_person_months:.1f}",
                    "Organizational Multiplier": f"{org_multiplier:.2f}",
                    "Technical Risk Score": f"{technical_risk*100:.1f}%",
                    "Complexity Score": f"{complexity_score*100:.1f}%",
                }
            )

        return pd.DataFrame(timeline_data)

    def _calculate_organizational_multiplier(self, org_factors: dict[str, str]) -> float:
        """Calculate organizational multiplier based on implementation factors."""
        multiplier = 1.0

        # Team experience adjustment
        experience_adjustments = {"low": 1.5, "medium": 1.0, "high": 0.8}
        multiplier *= experience_adjustments.get(org_factors["team_experience_level"], 1.0)

        # Infrastructure readiness adjustment
        infrastructure_adjustments = {"none": 1.4, "partial": 1.1, "complete": 0.9}
        multiplier *= infrastructure_adjustments.get(org_factors["existing_infrastructure"], 1.0)

        # Regulatory complexity adjustment
        regulatory_adjustments = {"low": 0.9, "standard": 1.0, "high": 1.3}
        multiplier *= regulatory_adjustments.get(org_factors["regulatory_complexity"], 1.0)

        # Change management maturity adjustment
        change_adjustments = {"low": 1.3, "medium": 1.0, "high": 0.9}
        multiplier *= change_adjustments.get(org_factors["change_management_maturity"], 1.0)

        return multiplier

    def create_feasibility_summary_report(
        self,
        computational_requirements: pd.DataFrame,
        operational_complexity: pd.DataFrame,
        tco_analysis: pd.DataFrame,
        timeline_analysis: pd.DataFrame,
        output_path: Path = None,
    ) -> str:
        """
        Create comprehensive feasibility summary report.

        Args:
            computational_requirements: Computational requirements analysis
            operational_complexity: Operational complexity analysis
            tco_analysis: Total cost of ownership analysis
            timeline_analysis: Implementation timeline analysis
            output_path: Optional path to save report

        Returns:
            Formatted feasibility summary report
        """
        if any(
            df.empty
            for df in [
                computational_requirements,
                operational_complexity,
                tco_analysis,
                timeline_analysis,
            ]
        ):
            return "Insufficient data for feasibility summary generation."

        # Extract key insights
        models = computational_requirements["Model"].tolist()

        # Find most feasible model (lowest complexity, cost, and timeline)
        complexity_scores = []
        for model in models:
            comp_score = float(
                computational_requirements[computational_requirements["Model"] == model][
                    "Computational Complexity (0-100)"
                ].iloc[0]
            )
            ops_score = float(
                operational_complexity[operational_complexity["Model"] == model][
                    "Overall Complexity (0-100)"
                ].iloc[0]
            )
            complexity_scores.append((comp_score + ops_score) / 2)

        most_feasible_idx = np.argmin(complexity_scores)
        most_feasible = models[most_feasible_idx]

        # Find most cost-effective model
        tco_values = []
        for model in models:
            tco_str = tco_analysis[tco_analysis["Model"] == model]["Risk-Adjusted TCO ($)"].iloc[0]
            tco_values.append(float(tco_str.replace(",", "")))

        most_cost_effective_idx = np.argmin(tco_values)
        most_cost_effective = models[most_cost_effective_idx]

        # Find fastest to implement
        timeline_values = []
        for model in models:
            timeline_str = timeline_analysis[timeline_analysis["Model"] == model][
                "Total Timeline (months)"
            ].iloc[0]
            timeline_values.append(float(timeline_str))

        fastest_implementation_idx = np.argmin(timeline_values)
        fastest_implementation = models[fastest_implementation_idx]

        report = (
            f"""
IMPLEMENTATION FEASIBILITY ASSESSMENT REPORT

EXECUTIVE SUMMARY
================
Total Models Evaluated: {len(models)}
Analysis Components: Computational Requirements, Operational Complexity, TCO, Timeline

KEY FINDINGS
============
• Most Feasible Overall: {most_feasible} (complexity score: {complexity_scores[most_feasible_idx]:.1f}/100)
• Most Cost-Effective: {most_cost_effective} (3-year TCO: ${tco_values[most_cost_effective_idx]:,.0f})
• Fastest Implementation: {fastest_implementation} ({timeline_values[fastest_implementation_idx]:.1f} months)

COMPUTATIONAL REQUIREMENTS SUMMARY
=================================
"""
            + self._format_dataframe_summary(
                computational_requirements,
                [
                    "Model",
                    "GPU Memory Required (GB)",
                    "Total Annual Compute Cost ($)",
                    "Computational Complexity (0-100)",
                ],
            )
            + """

OPERATIONAL COMPLEXITY SUMMARY
=============================
"""
            + self._format_dataframe_summary(
                operational_complexity,
                ["Model", "Overall Complexity (0-100)", "Monthly Maintenance Cost ($)"],
            )
            + """

TOTAL COST OF OWNERSHIP SUMMARY
==============================
"""
            + self._format_dataframe_summary(
                tco_analysis,
                [
                    "Model",
                    "Total Implementation ($)",
                    "Total Annual Operating ($)",
                    "Risk-Adjusted TCO ($)",
                ],
            )
            + """

IMPLEMENTATION TIMELINE SUMMARY
==============================
"""
            + self._format_dataframe_summary(
                timeline_analysis, ["Model", "Total Timeline (months)", "Total Person-Months"]
            )
            + f"""

STRATEGIC RECOMMENDATIONS
========================
1. BALANCED APPROACH: {most_feasible} offers optimal feasibility vs performance trade-off
2. COST OPTIMIZATION: {most_cost_effective} provides best TCO for cost-conscious deployment
3. RAPID DEPLOYMENT: {fastest_implementation} enables fastest time-to-value

IMPLEMENTATION PRIORITIES
========================
1. IMMEDIATE: Begin planning for {most_feasible} implementation
2. BUDGET PLANNING: Allocate ${min(tco_values):,.0f} - ${max(tco_values):,.0f} for 3-year deployment
3. RESOURCE ALLOCATION: Plan for {min(timeline_values):.1f} - {max(timeline_values):.1f} month implementation

RISK CONSIDERATIONS
==================
• Complexity ranges from {min(complexity_scores):.1f} to {max(complexity_scores):.1f} (0-100 scale)
• Timeline risks include technical complexity and organizational factors
• Cost risks include computational scaling and maintenance requirements

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        )

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)

        return report

    def _format_dataframe_summary(self, df: pd.DataFrame, columns: list[str]) -> str:
        """Format dataframe summary for report inclusion."""
        if df.empty:
            return "No data available."

        summary_df = df[columns] if all(col in df.columns for col in columns) else df
        return summary_df.to_string(index=False, max_rows=20)

    def create_feasibility_visualization(
        self,
        computational_requirements: pd.DataFrame,
        operational_complexity: pd.DataFrame,
        tco_analysis: pd.DataFrame,
        output_dir: Path = None,
    ) -> dict[str, Any]:
        """
        Create feasibility assessment visualizations.

        Args:
            computational_requirements: Computational requirements analysis
            operational_complexity: Operational complexity analysis
            tco_analysis: TCO analysis
            output_dir: Optional output directory for saving visualizations

        Returns:
            Dictionary containing feasibility visualizations
        """
        if not HAS_PLOTLY or any(
            df.empty for df in [computational_requirements, operational_complexity, tco_analysis]
        ):
            return {}

        visualizations = {}

        # Complexity vs Cost scatter plot
        merged_data = computational_requirements.merge(
            operational_complexity[["Model", "Overall Complexity (0-100)"]], on="Model"
        )
        merged_data = merged_data.merge(
            tco_analysis[["Model", "Risk-Adjusted TCO ($)"]], on="Model"
        )

        # Clean numeric data
        merged_data["Complexity"] = pd.to_numeric(
            merged_data["Overall Complexity (0-100)"], errors="coerce"
        )
        merged_data["TCO_Clean"] = (
            merged_data["Risk-Adjusted TCO ($)"].str.replace(",", "").astype(float)
        )

        fig_complexity_cost = go.Figure()

        fig_complexity_cost.add_trace(
            go.Scatter(
                x=merged_data["Complexity"],
                y=merged_data["TCO_Clean"],
                mode="markers+text",
                text=merged_data["Model"],
                textposition="top center",
                marker={"size": 12, "opacity": 0.7},
                name="Models",
            )
        )

        fig_complexity_cost.update_layout(
            title="Implementation Feasibility: Complexity vs Total Cost of Ownership",
            xaxis_title="Operational Complexity (0-100)",
            yaxis_title="3-Year Risk-Adjusted TCO ($)",
            template="plotly_white",
        )

        visualizations["complexity_vs_cost"] = fig_complexity_cost

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            fig_complexity_cost.write_html(output_dir / "feasibility_complexity_vs_cost.html")

        return visualizations
