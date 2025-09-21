"""
Comprehensive evaluation reporting framework.

This module provides institutional-grade reporting capabilities including executive
summaries, detailed performance analysis, feasibility assessments, strategic
recommendations, and multi-format export functionality.
"""

from src.evaluation.reporting.charts import ChartConfig, TimeSeriesCharts
from src.evaluation.reporting.comprehensive_report import ComprehensiveReportGenerator, ReportConfig
from src.evaluation.reporting.executive import ExecutiveConfig, ExecutiveSummaryGenerator
from src.evaluation.reporting.export import ExportConfig, PortfolioExporter
from src.evaluation.reporting.feasibility import (
    FeasibilityConfig,
    ImplementationFeasibilityAssessor,
)
from src.evaluation.reporting.heatmaps import HeatmapConfig, PerformanceHeatmaps
from src.evaluation.reporting.interactive import DashboardConfig, InteractiveDashboard
from src.evaluation.reporting.operational_analysis import (
    OperationalConfig,
    OperationalEfficiencyAnalysis,
)
from src.evaluation.reporting.performance_analysis import (
    ComprehensivePerformanceAnalyzer,
    PerformanceAnalysisConfig,
)
from src.evaluation.reporting.regime_analysis import MarketRegimeAnalysis, RegimeAnalysisConfig
from src.evaluation.reporting.risk_return import RiskReturnAnalysis, RiskReturnConfig
from src.evaluation.reporting.strategy import StrategyConfig, StrategyRecommendationEngine

# Academic reporting with flexible framework
from src.evaluation.reporting.academic_report_generator import (
    AcademicReportConfig,
    AcademicReportGenerator,
    create_academic_report_generator,
)

# Legacy components
from src.evaluation.reporting.tables import PerformanceComparisonTables, TableConfig

__all__ = [
    # Academic reporting with flexible framework
    "AcademicReportConfig",
    "AcademicReportGenerator",
    "create_academic_report_generator",
    # New comprehensive reporting framework
    "ComprehensiveReportGenerator",
    "ReportConfig",
    "ExecutiveSummaryGenerator",
    "ExecutiveConfig",
    "ComprehensivePerformanceAnalyzer",
    "PerformanceAnalysisConfig",
    "ImplementationFeasibilityAssessor",
    "FeasibilityConfig",
    "StrategyRecommendationEngine",
    "StrategyConfig",
    # Legacy specialized components
    "PerformanceComparisonTables",
    "TableConfig",
    "TimeSeriesCharts",
    "ChartConfig",
    "PerformanceHeatmaps",
    "HeatmapConfig",
    "RiskReturnAnalysis",
    "RiskReturnConfig",
    "MarketRegimeAnalysis",
    "RegimeAnalysisConfig",
    "OperationalEfficiencyAnalysis",
    "OperationalConfig",
    "InteractiveDashboard",
    "DashboardConfig",
    "PortfolioExporter",
    "ExportConfig",
]
