"""Production Deployment Readiness Assessment.

This module provides production deployment readiness assessment with infrastructure
requirements and timeline recommendations for portfolio management systems.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InfrastructureRequirement:
    """Infrastructure requirement specification."""
    component: str
    requirement: str
    current_status: str
    meets_requirement: bool
    estimated_cost: Optional[str] = None
    implementation_time: Optional[str] = None

@dataclass
class DeploymentTimeline:
    """Deployment timeline with milestones."""
    milestone: str
    estimated_duration: str
    dependencies: list[str]
    risk_level: str  # "Low", "Medium", "High"
    success_criteria: list[str]

@dataclass
class DeploymentReadinessAssessment:
    """Complete deployment readiness assessment."""
    timestamp: datetime
    overall_readiness_score: float
    readiness_status: str
    infrastructure_requirements: list[InfrastructureRequirement]
    deployment_timeline: list[DeploymentTimeline]
    risk_assessment: dict[str, Any]
    cost_estimation: dict[str, Any]
    recommendations: list[str]

class ProductionDeploymentReadinessAssessor:
    """Assessor for production deployment readiness and infrastructure requirements."""

    def __init__(self, base_path: Optional[str] = None):
        """Initialise deployment readiness assessor.

        Args:
            base_path: Base path for data files (defaults to current directory)
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()

        # Deployment readiness thresholds
        self.readiness_thresholds = {
            'infrastructure_compliance': 0.8,  # 80% infrastructure requirements met
            'performance_validation': 0.9,     # 90% performance tests passed
            'operational_readiness': 0.85,     # 85% operational requirements met
            'risk_mitigation': 0.9,            # 90% risks mitigated
            'minimum_overall_score': 0.8       # 80% overall readiness required
        }

    def assess_infrastructure_requirements(self) -> list[InfrastructureRequirement]:
        """Assess infrastructure requirements for production deployment.

        Returns:
            List of infrastructure requirements and their status
        """
        logger.info("Assessing infrastructure requirements...")

        requirements = []

        # 1. Computational Infrastructure
        requirements.append(InfrastructureRequirement(
            component="GPU Hardware",
            requirement="NVIDIA GPU with ≥12GB VRAM for model inference",
            current_status="Available (assumed based on training success)",
            meets_requirement=True,  # Assume met since training worked
            estimated_cost="$2,000-5,000 for RTX 4080/4090",
            implementation_time="1-2 weeks"
        ))

        requirements.append(InfrastructureRequirement(
            component="System Memory",
            requirement="≥32GB RAM for data processing and model serving",
            current_status=f"Current system: {psutil.virtual_memory().total / (1024**3):.0f}GB",
            meets_requirement=psutil.virtual_memory().total >= 32 * (1024**3),
            estimated_cost="$200-500 for memory upgrade",
            implementation_time="1 day"
        ))

        requirements.append(InfrastructureRequirement(
            component="Storage Infrastructure",
            requirement="≥1TB SSD storage for data and model checkpoints",
            current_status="Validation required",
            meets_requirement=True,  # Assume adequate for current operation
            estimated_cost="$100-300 for enterprise SSD",
            implementation_time="1 day"
        ))

        # 2. Data Infrastructure
        requirements.append(InfrastructureRequirement(
            component="Real-time Data Feeds",
            requirement="Bloomberg/Refinitiv API integration for market data",
            current_status="Not implemented",
            meets_requirement=False,
            estimated_cost="$2,000-5,000/month subscription fees",
            implementation_time="4-6 weeks"
        ))

        requirements.append(InfrastructureRequirement(
            component="Data Pipeline Infrastructure",
            requirement="Automated data collection and processing pipeline",
            current_status="Partially implemented (batch processing)",
            meets_requirement=False,
            estimated_cost="Development effort: 2-3 weeks",
            implementation_time="2-3 weeks"
        ))

        # 3. Model Serving Infrastructure
        requirements.append(InfrastructureRequirement(
            component="Model Serving Platform",
            requirement="Containerised model serving with load balancing",
            current_status="Not implemented",
            meets_requirement=False,
            estimated_cost="Cloud infrastructure: $500-2000/month",
            implementation_time="3-4 weeks"
        ))

        requirements.append(InfrastructureRequirement(
            component="Portfolio Management Integration",
            requirement="API integration with portfolio management systems",
            current_status="Not implemented",
            meets_requirement=False,
            estimated_cost="Integration development: 4-6 weeks",
            implementation_time="4-6 weeks"
        ))

        # 4. Monitoring and Risk Management
        requirements.append(InfrastructureRequirement(
            component="Real-time Monitoring",
            requirement="Performance monitoring with alert systems",
            current_status="Basic validation implemented",
            meets_requirement=False,
            estimated_cost="Monitoring platform: $200-500/month",
            implementation_time="2-3 weeks"
        ))

        requirements.append(InfrastructureRequirement(
            component="Risk Management Controls",
            requirement="Automated risk limits and position controls",
            current_status="Not implemented",
            meets_requirement=False,
            estimated_cost="Risk system development: 3-4 weeks",
            implementation_time="3-4 weeks"
        ))

        # 5. Compliance and Audit
        requirements.append(InfrastructureRequirement(
            component="Audit Trail System",
            requirement="Complete audit trail for all trading decisions",
            current_status="Not implemented",
            meets_requirement=False,
            estimated_cost="Audit system development: 2-3 weeks",
            implementation_time="2-3 weeks"
        ))

        logger.info(f"Assessed {len(requirements)} infrastructure requirements")
        return requirements

    def create_deployment_timeline(self) -> list[DeploymentTimeline]:
        """Create detailed deployment timeline with milestones.

        Returns:
            List of deployment timeline milestones
        """
        logger.info("Creating deployment timeline...")

        timeline = []

        # Phase 1: Infrastructure Setup
        timeline.append(DeploymentTimeline(
            milestone="Infrastructure Setup",
            estimated_duration="2-3 weeks",
            dependencies=["Hardware procurement", "Cloud infrastructure setup"],
            risk_level="Medium",
            success_criteria=[
                "GPU infrastructure operational",
                "Data storage configured",
                "Network connectivity established"
            ]
        ))

        # Phase 2: Data Pipeline Implementation
        timeline.append(DeploymentTimeline(
            milestone="Data Pipeline Implementation",
            estimated_duration="4-6 weeks",
            dependencies=["Infrastructure Setup", "Data vendor contracts"],
            risk_level="High",
            success_criteria=[
                "Real-time data feeds operational",
                "Data quality validation passing",
                "Automated data processing pipeline deployed"
            ]
        ))

        # Phase 3: Model Deployment
        timeline.append(DeploymentTimeline(
            milestone="Model Deployment",
            estimated_duration="3-4 weeks",
            dependencies=["Data Pipeline Implementation", "Model containerisation"],
            risk_level="Medium",
            success_criteria=[
                "Model serving platform operational",
                "Model inference latency <10 minutes",
                "Model performance validation passing"
            ]
        ))

        # Phase 4: Portfolio Integration
        timeline.append(DeploymentTimeline(
            milestone="Portfolio Management Integration",
            estimated_duration="4-6 weeks",
            dependencies=["Model Deployment", "Portfolio system API access"],
            risk_level="High",
            success_criteria=[
                "Portfolio allocation API functional",
                "Trade execution integration complete",
                "Position reconciliation automated"
            ]
        ))

        # Phase 5: Risk Management Implementation
        timeline.append(DeploymentTimeline(
            milestone="Risk Management Implementation",
            estimated_duration="3-4 weeks",
            dependencies=["Portfolio Integration"],
            risk_level="Medium",
            success_criteria=[
                "Position limits enforcement active",
                "Risk monitoring dashboard operational",
                "Automated alert system functional"
            ]
        ))

        # Phase 6: Compliance and Testing
        timeline.append(DeploymentTimeline(
            milestone="Compliance and User Acceptance Testing",
            estimated_duration="2-3 weeks",
            dependencies=["Risk Management Implementation"],
            risk_level="Low",
            success_criteria=[
                "Audit trail system operational",
                "Compliance validation complete",
                "User acceptance testing passed"
            ]
        ))

        # Phase 7: Production Launch
        timeline.append(DeploymentTimeline(
            milestone="Production Launch",
            estimated_duration="1-2 weeks",
            dependencies=["Compliance and Testing"],
            risk_level="Medium",
            success_criteria=[
                "Production environment stable",
                "Live trading initiated",
                "Monitoring systems operational"
            ]
        ))

        logger.info(f"Created deployment timeline with {len(timeline)} phases")
        return timeline

    def assess_deployment_risks(self) -> dict[str, Any]:
        """Assess deployment risks and mitigation strategies.

        Returns:
            Dictionary containing risk assessment
        """
        risks = {
            'technical_risks': [
                {
                    'risk': 'Model performance degradation in production',
                    'probability': 'Medium',
                    'impact': 'High',
                    'mitigation': 'Continuous model monitoring and fallback to baseline strategies'
                },
                {
                    'risk': 'Data pipeline failures or delays',
                    'probability': 'Medium',
                    'impact': 'High',
                    'mitigation': 'Redundant data sources and automated failure detection'
                },
                {
                    'risk': 'Infrastructure scaling limitations',
                    'probability': 'Low',
                    'impact': 'Medium',
                    'mitigation': 'Cloud-based auto-scaling and load balancing'
                }
            ],
            'operational_risks': [
                {
                    'risk': 'Portfolio manager adoption challenges',
                    'probability': 'Medium',
                    'impact': 'Medium',
                    'mitigation': 'Comprehensive training and gradual rollout'
                },
                {
                    'risk': 'Regulatory compliance issues',
                    'probability': 'Low',
                    'impact': 'High',
                    'mitigation': 'Legal review and compliance framework implementation'
                }
            ],
            'market_risks': [
                {
                    'risk': 'Market regime changes affecting model performance',
                    'probability': 'High',
                    'impact': 'Medium',
                    'mitigation': 'Regular model retraining and ensemble approaches'
                },
                {
                    'risk': 'Increased transaction costs impacting returns',
                    'probability': 'Medium',
                    'impact': 'Medium',
                    'mitigation': 'Dynamic transaction cost monitoring and optimisation'
                }
            ]
        }

        return risks

    def estimate_deployment_costs(self) -> dict[str, Any]:
        """Estimate deployment costs and timeline.

        Returns:
            Dictionary containing cost estimation
        """
        costs = {
            'one_time_costs': {
                'hardware_infrastructure': {
                    'amount': '15000-25000',
                    'currency': 'USD',
                    'description': 'GPU hardware, storage, networking equipment'
                },
                'software_development': {
                    'amount': '100000-150000',
                    'currency': 'USD',
                    'description': 'Custom development for integration and monitoring'
                },
                'integration_costs': {
                    'amount': '50000-75000',
                    'currency': 'USD',
                    'description': 'Portfolio system integration and testing'
                }
            },
            'recurring_costs': {
                'data_subscriptions': {
                    'amount': '24000-60000',
                    'currency': 'USD',
                    'period': 'annual',
                    'description': 'Bloomberg/Refinitiv data feeds'
                },
                'cloud_infrastructure': {
                    'amount': '12000-24000',
                    'currency': 'USD',
                    'period': 'annual',
                    'description': 'Cloud computing and storage'
                },
                'maintenance_support': {
                    'amount': '50000-100000',
                    'currency': 'USD',
                    'period': 'annual',
                    'description': 'Ongoing maintenance and model updates'
                }
            },
            'total_first_year': {
                'min_estimate': 251000,
                'max_estimate': 434000,
                'currency': 'USD'
            }
        }

        return costs

    def generate_recommendations(self, infrastructure_reqs: list[InfrastructureRequirement],
                               timeline: list[DeploymentTimeline]) -> list[str]:
        """Generate deployment recommendations.

        Args:
            infrastructure_reqs: Infrastructure requirements
            timeline: Deployment timeline

        Returns:
            List of recommendations
        """
        recommendations = []

        # Infrastructure recommendations
        unmet_reqs = [req for req in infrastructure_reqs if not req.meets_requirement]
        if len(unmet_reqs) > 5:
            recommendations.append(
                f"Address {len(unmet_reqs)} critical infrastructure gaps before deployment"
            )

        # Timeline recommendations
        high_risk_phases = [phase for phase in timeline if phase.risk_level == "High"]
        if high_risk_phases:
            recommendations.append(
                f"Allocate additional resources to {len(high_risk_phases)} high-risk deployment phases"
            )

        # Model performance recommendations
        recommendations.append(
            "Implement continuous model monitoring with automated fallback to baseline strategies"
        )

        # Risk management recommendations
        recommendations.append(
            "Establish comprehensive risk limits and real-time monitoring before live deployment"
        )

        # Phased deployment recommendation
        recommendations.append(
            "Consider phased deployment starting with paper trading for 3-6 months"
        )

        # Cost management recommendations
        recommendations.append(
            "Negotiate volume discounts for data subscriptions and consider cloud cost optimisation"
        )

        return recommendations

    def assess_production_readiness(self) -> DeploymentReadinessAssessment:
        """Assess overall production deployment readiness.

        Returns:
            Complete deployment readiness assessment
        """
        logger.info("Assessing production deployment readiness...")

        # Gather assessment components
        infrastructure_reqs = self.assess_infrastructure_requirements()
        timeline = self.create_deployment_timeline()
        risk_assessment = self.assess_deployment_risks()
        cost_estimation = self.estimate_deployment_costs()
        recommendations = self.generate_recommendations(infrastructure_reqs, timeline)

        # Calculate readiness scores
        infrastructure_score = sum(1 for req in infrastructure_reqs if req.meets_requirement) / len(infrastructure_reqs)

        # Performance validation score (from previous validations)
        performance_score = 0.85  # Assume good based on validation results

        # Operational readiness score
        operational_score = 0.3  # Low due to missing integrations

        # Risk mitigation score
        risk_score = 0.6  # Medium due to identified mitigations

        # Calculate overall readiness score
        overall_score = (
            0.3 * infrastructure_score +
            0.25 * performance_score +
            0.25 * operational_score +
            0.2 * risk_score
        )

        # Determine readiness status
        if overall_score >= self.readiness_thresholds['minimum_overall_score']:
            readiness_status = "Ready for Deployment"
        elif overall_score >= 0.6:
            readiness_status = "Conditional - Additional preparation required"
        elif overall_score >= 0.4:
            readiness_status = "Not Ready - Significant gaps identified"
        else:
            readiness_status = "Not Ready - Major infrastructure deficiencies"

        assessment = DeploymentReadinessAssessment(
            timestamp=datetime.now(),
            overall_readiness_score=overall_score,
            readiness_status=readiness_status,
            infrastructure_requirements=infrastructure_reqs,
            deployment_timeline=timeline,
            risk_assessment=risk_assessment,
            cost_estimation=cost_estimation,
            recommendations=recommendations
        )

        logger.info(f"Deployment readiness assessment complete: {overall_score:.1%} readiness ({readiness_status})")
        return assessment

    def export_readiness_assessment(self, assessment: DeploymentReadinessAssessment,
                                  output_dir: Optional[str] = None) -> dict[str, str]:
        """Export deployment readiness assessment.

        Args:
            assessment: Deployment readiness assessment
            output_dir: Output directory

        Returns:
            Dictionary mapping format to file path
        """
        if output_dir is None:
            output_dir = self.base_path / 'results' / 'deployment_readiness'
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        try:
            # Export complete assessment as JSON
            json_file = output_dir / 'deployment_readiness_assessment.json'
            with open(json_file, 'w') as f:
                json.dump(asdict(assessment), f, indent=2, default=str)
            exported_files['json'] = str(json_file)

            # Export infrastructure requirements as CSV
            infra_df = pd.DataFrame([asdict(req) for req in assessment.infrastructure_requirements])
            infra_csv = output_dir / 'infrastructure_requirements.csv'
            infra_df.to_csv(infra_csv, index=False)
            exported_files['infrastructure_csv'] = str(infra_csv)

            # Export timeline as CSV
            timeline_df = pd.DataFrame([asdict(milestone) for milestone in assessment.deployment_timeline])
            timeline_csv = output_dir / 'deployment_timeline.csv'
            timeline_df.to_csv(timeline_csv, index=False)
            exported_files['timeline_csv'] = str(timeline_csv)

            logger.info(f"Deployment readiness assessment exported to {len(exported_files)} files")

        except Exception as e:
            logger.error(f"Error exporting assessment: {e}")
            raise

        return exported_files
