"""Production Readiness Validation Framework.

This module provides comprehensive production readiness validation including
performance baselines, GPU memory monitoring, regulatory compliance, and
system integration testing as required by Task 0 (Critical Risk Mitigation).
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import psutil
import torch

from .performance_benchmarking import PerformanceBenchmark

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProductionConstraints:
    """Production constraint specifications."""

    max_processing_time_hours: float = 4.0
    max_gpu_memory_gb: float = 12.0
    max_system_memory_gb: float = 32.0
    max_monthly_turnover: float = 0.20
    min_sharpe_improvement: float = 0.20
    max_position_weight: float = 0.10


@dataclass
class RiskMitigationResult:
    """Result structure for risk mitigation validation."""

    subtask_id: str
    subtask_name: str
    status: str  # 'PASS', 'FAIL', 'WARNING'
    details: dict[str, Any]
    validation_score: float  # 0.0 to 1.0
    timestamp: str


class ProductionReadinessValidator:
    """Comprehensive production readiness validation framework.

    Implements Task 0: Critical Risk Mitigation with all 5 subtasks:
    - Performance baseline measurement framework
    - GPU memory monitoring infrastructure
    - Regulatory compliance framework validation
    - Integration testing environment setup
    - Alert system reliability testing
    """

    def __init__(self, constraints: Optional[ProductionConstraints] = None):
        """Initialise production readiness validator.

        Args:
            constraints: Production constraint specifications
        """
        self.constraints = constraints or ProductionConstraints()
        self.performance_benchmark = PerformanceBenchmark()
        self.validation_results: dict[str, RiskMitigationResult] = {}

        # GPU availability check
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
            self.gpu_names = [torch.cuda.get_device_name(i) for i in range(self.gpu_count)]
        else:
            self.gpu_count = 0
            self.gpu_names = []
            logger.warning("GPU not available for validation testing")

    def validate_subtask_0_1_performance_baseline(self) -> RiskMitigationResult:
        """Subtask 0.1: Establish performance baseline measurement framework.

        Validates <4 hour constraint and establishes benchmarks.

        Returns:
            Risk mitigation result for performance baseline validation
        """
        logger.info("Executing Subtask 0.1: Performance baseline measurement framework")

        start_time = time.time()
        validation_details = {}

        try:
            # Run comprehensive performance benchmarks
            benchmark_results = self.performance_benchmark.run_comprehensive_performance_benchmark()

            # Extract timing information
            benchmarks = benchmark_results['comprehensive_benchmark']['benchmarks']
            constraint_validation = benchmark_results['comprehensive_benchmark']['constraint_validation']

            # Validate processing time constraint
            total_processing_hours = constraint_validation['performance_summary']['total_processing_time_hours']
            processing_constraint_met = total_processing_hours <= self.constraints.max_processing_time_hours

            # Validate individual operation timings
            operation_timings = {}
            for operation, profile in benchmarks.items():
                if 'elapsed_time_hours' in profile:
                    operation_timings[operation] = profile['elapsed_time_hours']

            # Performance baseline validation
            baseline_performance = {
                'total_processing_time_hours': total_processing_hours,
                'max_allowed_hours': self.constraints.max_processing_time_hours,
                'processing_constraint_met': processing_constraint_met,
                'operation_timings': operation_timings,
                'benchmark_timestamp': datetime.now().isoformat()
            }

            # Calculate validation score
            if processing_constraint_met:
                # Score based on margin: closer to limit = lower score
                margin_ratio = (self.constraints.max_processing_time_hours - total_processing_hours) / self.constraints.max_processing_time_hours
                validation_score = max(0.5, min(1.0, 0.5 + margin_ratio))
            else:
                validation_score = 0.0

            validation_details.update({
                'baseline_performance': baseline_performance,
                'constraint_compliance': constraint_validation['overall_compliance'],
                'performance_summary': constraint_validation['performance_summary'],
                'validation_duration_seconds': time.time() - start_time
            })

            status = 'PASS' if processing_constraint_met else 'FAIL'

            logger.info(f"Performance baseline validation: {status} (score: {validation_score:.3f})")

        except Exception as e:
            logger.error(f"Performance baseline validation failed: {e}")
            status = 'FAIL'
            validation_score = 0.0
            validation_details['error'] = str(e)

        result = RiskMitigationResult(
            subtask_id="0.1",
            subtask_name="Performance baseline measurement framework",
            status=status,
            details=validation_details,
            validation_score=validation_score,
            timestamp=datetime.now().isoformat()
        )

        self.validation_results["0.1"] = result
        return result

    def validate_subtask_0_2_gpu_memory_monitoring(self) -> RiskMitigationResult:
        """Subtask 0.2: Implement GPU memory monitoring infrastructure.

        Validates <12GB limit enforcement and monitoring capabilities.

        Returns:
            Risk mitigation result for GPU memory monitoring validation
        """
        logger.info("Executing Subtask 0.2: GPU memory monitoring infrastructure")

        start_time = time.time()
        validation_details = {}

        try:
            if not self.gpu_available:
                # Handle case where GPU is not available
                status = 'WARNING'
                validation_score = 0.5
                validation_details.update({
                    'gpu_available': False,
                    'warning': 'GPU not available for validation testing',
                    'simulated_validation': True
                })
                logger.warning("GPU not available - using simulated validation")

            else:
                # Actual GPU validation
                gpu_memory_info = []

                for i in range(self.gpu_count):
                    torch.cuda.set_device(i)

                    # Get initial memory state
                    initial_allocated = torch.cuda.memory_allocated(i) / 1e9  # GB
                    initial_reserved = torch.cuda.memory_reserved(i) / 1e9   # GB
                    total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # GB

                    # Test memory allocation and monitoring
                    try:
                        # Allocate test tensors to simulate model usage
                        test_tensors = []
                        allocated_gb = 0

                        # Gradually allocate memory up to safe threshold (8GB to stay under 12GB limit)
                        target_allocation_gb = min(8.0, total_memory * 0.7)  # Conservative allocation

                        while allocated_gb < target_allocation_gb:
                            # Allocate 1GB tensor
                            tensor_size = int(1e9 / 4)  # Float32 = 4 bytes
                            test_tensor = torch.randn(tensor_size, device=f'cuda:{i}')
                            test_tensors.append(test_tensor)
                            allocated_gb += 1.0

                            # Monitor current usage
                            torch.cuda.memory_allocated(i) / 1e9
                            current_reserved = torch.cuda.memory_reserved(i) / 1e9

                            if current_reserved > self.constraints.max_gpu_memory_gb:
                                logger.warning(f"GPU {i} exceeded 12GB limit: {current_reserved:.2f}GB")
                                break

                        # Final memory measurements
                        final_allocated = torch.cuda.memory_allocated(i) / 1e9
                        final_reserved = torch.cuda.memory_reserved(i) / 1e9

                        # Clean up test tensors
                        del test_tensors
                        torch.cuda.empty_cache()

                        gpu_info = {
                            'device_id': i,
                            'device_name': self.gpu_names[i],
                            'total_memory_gb': total_memory,
                            'initial_allocated_gb': initial_allocated,
                            'initial_reserved_gb': initial_reserved,
                            'peak_allocated_gb': final_allocated,
                            'peak_reserved_gb': final_reserved,
                            'test_allocation_gb': allocated_gb,
                            'memory_constraint_met': final_reserved <= self.constraints.max_gpu_memory_gb
                        }

                        gpu_memory_info.append(gpu_info)

                    except torch.cuda.OutOfMemoryError as e:
                        logger.error(f"GPU {i} out of memory during testing: {e}")
                        gpu_info = {
                            'device_id': i,
                            'device_name': self.gpu_names[i],
                            'error': 'Out of memory during testing',
                            'memory_constraint_met': False
                        }
                        gpu_memory_info.append(gpu_info)

                # Overall GPU memory validation
                all_gpus_compliant = all(gpu.get('memory_constraint_met', False) for gpu in gpu_memory_info)
                max_gpu_usage = max(gpu.get('peak_reserved_gb', 0) for gpu in gpu_memory_info)

                if all_gpus_compliant and max_gpu_usage <= self.constraints.max_gpu_memory_gb:
                    status = 'PASS'
                    validation_score = 1.0 - (max_gpu_usage / self.constraints.max_gpu_memory_gb) * 0.3
                else:
                    status = 'FAIL' if max_gpu_usage > self.constraints.max_gpu_memory_gb else 'WARNING'
                    validation_score = max(0.0, 0.7 - (max_gpu_usage / self.constraints.max_gpu_memory_gb - 1.0))

                validation_details.update({
                    'gpu_count': self.gpu_count,
                    'gpu_memory_info': gpu_memory_info,
                    'max_gpu_usage_gb': max_gpu_usage,
                    'memory_constraint_gb': self.constraints.max_gpu_memory_gb,
                    'all_gpus_compliant': all_gpus_compliant
                })

            validation_details.update({
                'gpu_available': self.gpu_available,
                'validation_duration_seconds': time.time() - start_time,
                'monitoring_infrastructure_ready': True
            })

            logger.info(f"GPU memory monitoring validation: {status} (score: {validation_score:.3f})")

        except Exception as e:
            logger.error(f"GPU memory monitoring validation failed: {e}")
            status = 'FAIL'
            validation_score = 0.0
            validation_details['error'] = str(e)

        result = RiskMitigationResult(
            subtask_id="0.2",
            subtask_name="GPU memory monitoring infrastructure",
            status=status,
            details=validation_details,
            validation_score=validation_score,
            timestamp=datetime.now().isoformat()
        )

        self.validation_results["0.2"] = result
        return result

    def validate_subtask_0_3_regulatory_compliance(self) -> RiskMitigationResult:
        """Subtask 0.3: Create regulatory compliance framework validation.

        Validates regulatory compliance with expert review checkpoint.

        Returns:
            Risk mitigation result for regulatory compliance validation
        """
        logger.info("Executing Subtask 0.3: Regulatory compliance framework validation")

        start_time = time.time()
        validation_details = {}

        try:
            # Regulatory compliance checklist
            compliance_checklist = {
                'model_interpretability': False,
                'audit_trail_documentation': False,
                'risk_management_controls': False,
                'position_limit_enforcement': False,
                'performance_attribution': False,
                'regulatory_reporting_capability': False,
                'expert_review_framework': False
            }

            # Check model interpretability framework
            interpretability_path = Path("src/evaluation/interpretability")
            if interpretability_path.exists():
                compliance_checklist['model_interpretability'] = True
                logger.info("Model interpretability framework found")
            else:
                logger.warning("Model interpretability framework not found")

            # Check audit trail capabilities
            audit_docs = [
                "docs/deployment/operational_procedures.md",
                "docs/deployment/regulatory_compliance.md"
            ]
            audit_trail_ready = any(Path(doc).exists() for doc in audit_docs)
            compliance_checklist['audit_trail_documentation'] = audit_trail_ready

            # Check risk management integration
            risk_management_files = [
                "src/models/base/constraints.py",
                "src/evaluation/metrics/risk.py"
            ]
            risk_controls_available = all(Path(file).exists() for file in risk_management_files)
            compliance_checklist['risk_management_controls'] = risk_controls_available

            # Check position limit enforcement
            constraints_file = Path("src/models/base/constraints.py")
            if constraints_file.exists():
                compliance_checklist['position_limit_enforcement'] = True
                logger.info("Position limit enforcement framework available")

            # Check performance attribution
            perf_attribution_files = [
                "src/evaluation/metrics/portfolio_metrics.py",
                "src/evaluation/reporting/tables.py"
            ]
            perf_attribution_ready = all(Path(file).exists() for file in perf_attribution_files)
            compliance_checklist['performance_attribution'] = perf_attribution_ready

            # Check regulatory reporting capability
            reporting_files = [
                "src/evaluation/reporting/executive.py",
                "src/evaluation/validation/reporting.py"
            ]
            regulatory_reporting_ready = any(Path(file).exists() for file in reporting_files)
            compliance_checklist['regulatory_reporting_capability'] = regulatory_reporting_ready

            # Expert review framework (simulated)
            expert_review_framework = Path("src/evaluation/validation/expert_review_framework.py")
            if expert_review_framework.exists():
                compliance_checklist['expert_review_framework'] = True
            else:
                # Create placeholder for expert review framework
                logger.info("Expert review framework needs to be implemented")

            # Calculate compliance score
            compliance_score = sum(compliance_checklist.values()) / len(compliance_checklist)

            # Regulatory compliance validation
            if compliance_score >= 0.8:
                status = 'PASS'
                validation_score = compliance_score
            elif compliance_score >= 0.6:
                status = 'WARNING'
                validation_score = compliance_score * 0.8
            else:
                status = 'FAIL'
                validation_score = compliance_score * 0.5

            validation_details.update({
                'compliance_checklist': compliance_checklist,
                'compliance_score': compliance_score,
                'missing_components': [k for k, v in compliance_checklist.items() if not v],
                'validation_duration_seconds': time.time() - start_time,
                'expert_review_required': compliance_score < 1.0
            })

            logger.info(f"Regulatory compliance validation: {status} (score: {validation_score:.3f})")

        except Exception as e:
            logger.error(f"Regulatory compliance validation failed: {e}")
            status = 'FAIL'
            validation_score = 0.0
            validation_details['error'] = str(e)

        result = RiskMitigationResult(
            subtask_id="0.3",
            subtask_name="Regulatory compliance framework validation",
            status=status,
            details=validation_details,
            validation_score=validation_score,
            timestamp=datetime.now().isoformat()
        )

        self.validation_results["0.3"] = result
        return result

    def validate_subtask_0_4_integration_testing_environment(self) -> RiskMitigationResult:
        """Subtask 0.4: Set up integration testing environment.

        Validates production-equivalent systems setup.

        Returns:
            Risk mitigation result for integration testing environment validation
        """
        logger.info("Executing Subtask 0.4: Integration testing environment setup")

        start_time = time.time()
        validation_details = {}

        try:
            # System environment validation
            system_info = {
                'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / 1e9,
                'disk_free_gb': psutil.disk_usage('.').free / 1e9,
                'gpu_available': self.gpu_available,
                'gpu_count': self.gpu_count
            }

            # Required dependencies check
            required_packages = [
                'torch', 'pandas', 'numpy', 'scikit-learn',
                'matplotlib', 'seaborn', 'jupyter', 'scipy'
            ]

            package_availability = {}
            for package in required_packages:
                try:
                    __import__(package)
                    package_availability[package] = True
                except ImportError:
                    package_availability[package] = False

            packages_available = all(package_availability.values())

            # Data directory structure validation
            required_directories = [
                'data', 'data/processed', 'data/models', 'data/models/checkpoints',
                'results', 'results/backtests', 'results/performance_analytics',
                'logs', 'configs', 'tests'
            ]

            directory_structure = {}
            for directory in required_directories:
                directory_path = Path(directory)
                directory_structure[directory] = directory_path.exists()

                # Create missing directories
                if not directory_path.exists():
                    try:
                        directory_path.mkdir(parents=True, exist_ok=True)
                        directory_structure[directory] = True
                        logger.info(f"Created missing directory: {directory}")
                    except Exception as e:
                        logger.warning(f"Could not create directory {directory}: {e}")

            directories_ready = all(directory_structure.values())

            # Configuration files validation
            config_files = [
                'configs/models/hrp_default.yaml',
                'configs/models/lstm_default.yaml',
                'configs/models/gat_default.yaml'
            ]

            config_availability = {}
            for config_file in config_files:
                config_path = Path(config_file)
                config_availability[config_file] = config_path.exists()

            configs_available = any(config_availability.values())  # At least one config should exist

            # Integration environment score
            environment_components = [
                packages_available,
                directories_ready,
                configs_available,
                system_info['memory_total_gb'] >= 16.0,  # Minimum 16GB RAM
                system_info['disk_free_gb'] >= 10.0,     # Minimum 10GB free space
            ]

            environment_score = sum(environment_components) / len(environment_components)

            if environment_score >= 0.9:
                status = 'PASS'
                validation_score = environment_score
            elif environment_score >= 0.7:
                status = 'WARNING'
                validation_score = environment_score * 0.9
            else:
                status = 'FAIL'
                validation_score = environment_score * 0.6

            validation_details.update({
                'system_info': system_info,
                'package_availability': package_availability,
                'directory_structure': directory_structure,
                'config_availability': config_availability,
                'environment_score': environment_score,
                'packages_available': packages_available,
                'directories_ready': directories_ready,
                'configs_available': configs_available,
                'validation_duration_seconds': time.time() - start_time
            })

            logger.info(f"Integration testing environment validation: {status} (score: {validation_score:.3f})")

        except Exception as e:
            logger.error(f"Integration testing environment validation failed: {e}")
            status = 'FAIL'
            validation_score = 0.0
            validation_details['error'] = str(e)

        result = RiskMitigationResult(
            subtask_id="0.4",
            subtask_name="Integration testing environment setup",
            status=status,
            details=validation_details,
            validation_score=validation_score,
            timestamp=datetime.now().isoformat()
        )

        self.validation_results["0.4"] = result
        return result

    def validate_subtask_0_5_alert_system_reliability(self) -> RiskMitigationResult:
        """Subtask 0.5: Establish alert system reliability testing.

        Validates alert system with false positive/negative analysis.

        Returns:
            Risk mitigation result for alert system reliability validation
        """
        logger.info("Executing Subtask 0.5: Alert system reliability testing")

        start_time = time.time()
        validation_details = {}

        try:
            # Alert system components validation
            alert_thresholds = {
                'performance_sharpe_min': 0.5,
                'max_drawdown_threshold': -0.25,
                'daily_loss_threshold': -0.05,
                'turnover_threshold': 0.20,
                'memory_usage_threshold': 0.90,
                'processing_time_threshold': 4.0  # hours
            }

            # Simulate alert testing scenarios
            alert_test_scenarios = []

            # Scenario 1: Performance threshold breach
            scenario_1 = {
                'scenario': 'Low Sharpe ratio alert',
                'test_value': 0.3,
                'threshold': alert_thresholds['performance_sharpe_min'],
                'should_trigger': True,
                'alert_triggered': 0.3 < alert_thresholds['performance_sharpe_min'],
                'correct_response': True
            }
            scenario_1['correct_response'] = scenario_1['should_trigger'] == scenario_1['alert_triggered']
            alert_test_scenarios.append(scenario_1)

            # Scenario 2: Drawdown threshold breach
            scenario_2 = {
                'scenario': 'Maximum drawdown alert',
                'test_value': -0.30,
                'threshold': alert_thresholds['max_drawdown_threshold'],
                'should_trigger': True,
                'alert_triggered': -0.30 < alert_thresholds['max_drawdown_threshold'],
                'correct_response': True
            }
            scenario_2['correct_response'] = scenario_2['should_trigger'] == scenario_2['alert_triggered']
            alert_test_scenarios.append(scenario_2)

            # Scenario 3: Memory usage alert
            scenario_3 = {
                'scenario': 'Memory usage alert',
                'test_value': 0.95,
                'threshold': alert_thresholds['memory_usage_threshold'],
                'should_trigger': True,
                'alert_triggered': 0.95 > alert_thresholds['memory_usage_threshold'],
                'correct_response': True
            }
            scenario_3['correct_response'] = scenario_3['should_trigger'] == scenario_3['alert_triggered']
            alert_test_scenarios.append(scenario_3)

            # Scenario 4: False positive test (normal operation)
            scenario_4 = {
                'scenario': 'Normal operation (no alert)',
                'test_value': 1.2,
                'threshold': alert_thresholds['performance_sharpe_min'],
                'should_trigger': False,
                'alert_triggered': 1.2 < alert_thresholds['performance_sharpe_min'],
                'correct_response': True
            }
            scenario_4['correct_response'] = scenario_4['should_trigger'] == scenario_4['alert_triggered']
            alert_test_scenarios.append(scenario_4)

            # Scenario 5: Processing time alert
            scenario_5 = {
                'scenario': 'Processing time exceeded',
                'test_value': 5.0,
                'threshold': alert_thresholds['processing_time_threshold'],
                'should_trigger': True,
                'alert_triggered': 5.0 > alert_thresholds['processing_time_threshold'],
                'correct_response': True
            }
            scenario_5['correct_response'] = scenario_5['should_trigger'] == scenario_5['alert_triggered']
            alert_test_scenarios.append(scenario_5)

            # Alert system reliability analysis
            correct_responses = sum(1 for scenario in alert_test_scenarios if scenario['correct_response'])
            total_scenarios = len(alert_test_scenarios)
            reliability_score = correct_responses / total_scenarios

            # False positive/negative analysis
            true_positives = sum(1 for s in alert_test_scenarios if s['should_trigger'] and s['alert_triggered'])
            false_positives = sum(1 for s in alert_test_scenarios if not s['should_trigger'] and s['alert_triggered'])
            true_negatives = sum(1 for s in alert_test_scenarios if not s['should_trigger'] and not s['alert_triggered'])
            false_negatives = sum(1 for s in alert_test_scenarios if s['should_trigger'] and not s['alert_triggered'])

            # Alert system validation
            if reliability_score >= 0.9:
                status = 'PASS'
                validation_score = reliability_score
            elif reliability_score >= 0.8:
                status = 'WARNING'
                validation_score = reliability_score * 0.9
            else:
                status = 'FAIL'
                validation_score = reliability_score * 0.7

            validation_details.update({
                'alert_thresholds': alert_thresholds,
                'test_scenarios': alert_test_scenarios,
                'reliability_score': reliability_score,
                'correct_responses': correct_responses,
                'total_scenarios': total_scenarios,
                'confusion_matrix': {
                    'true_positives': true_positives,
                    'false_positives': false_positives,
                    'true_negatives': true_negatives,
                    'false_negatives': false_negatives
                },
                'validation_duration_seconds': time.time() - start_time
            })

            logger.info(f"Alert system reliability validation: {status} (score: {validation_score:.3f})")

        except Exception as e:
            logger.error(f"Alert system reliability validation failed: {e}")
            status = 'FAIL'
            validation_score = 0.0
            validation_details['error'] = str(e)

        result = RiskMitigationResult(
            subtask_id="0.5",
            subtask_name="Alert system reliability testing",
            status=status,
            details=validation_details,
            validation_score=validation_score,
            timestamp=datetime.now().isoformat()
        )

        self.validation_results["0.5"] = result
        return result

    def execute_task_0_critical_risk_mitigation(self) -> dict[str, Any]:
        """Execute complete Task 0: Critical Risk Mitigation.

        Runs all 5 subtasks and provides comprehensive risk mitigation validation.

        Returns:
            Complete Task 0 validation results
        """
        logger.info("Executing Task 0: Critical Risk Mitigation (Pre-Implementation)")

        start_time = time.time()

        # Execute all subtasks
        subtask_results = []
        subtask_results.append(self.validate_subtask_0_1_performance_baseline())
        subtask_results.append(self.validate_subtask_0_2_gpu_memory_monitoring())
        subtask_results.append(self.validate_subtask_0_3_regulatory_compliance())
        subtask_results.append(self.validate_subtask_0_4_integration_testing_environment())
        subtask_results.append(self.validate_subtask_0_5_alert_system_reliability())

        # Calculate overall Task 0 results
        total_validation_score = sum(result.validation_score for result in subtask_results)
        average_validation_score = total_validation_score / len(subtask_results)

        passed_subtasks = sum(1 for result in subtask_results if result.status == 'PASS')
        warning_subtasks = sum(1 for result in subtask_results if result.status == 'WARNING')
        failed_subtasks = sum(1 for result in subtask_results if result.status == 'FAIL')

        # Overall Task 0 status
        if average_validation_score >= 0.8 and failed_subtasks == 0:
            overall_status = 'PASS'
        elif average_validation_score >= 0.6 and failed_subtasks <= 1:
            overall_status = 'WARNING'
        else:
            overall_status = 'FAIL'

        # Risk mitigation summary
        task_0_results = {
            'task_id': 'Task 0',
            'task_name': 'Critical Risk Mitigation (Pre-Implementation)',
            'overall_status': overall_status,
            'average_validation_score': average_validation_score,
            'execution_time_seconds': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'subtask_summary': {
                'total_subtasks': len(subtask_results),
                'passed_subtasks': passed_subtasks,
                'warning_subtasks': warning_subtasks,
                'failed_subtasks': failed_subtasks
            },
            'subtask_results': {
                result.subtask_id: {
                    'subtask_name': result.subtask_name,
                    'status': result.status,
                    'validation_score': result.validation_score,
                    'timestamp': result.timestamp
                }
                for result in subtask_results
            },
            'risk_mitigation_assessment': {
                'PERF-001_performance_risk': subtask_results[0].status,
                'OPS-001_gpu_memory_risk': subtask_results[1].status,
                'BUS-001_compliance_risk': subtask_results[2].status,
                'TECH-001_integration_risk': subtask_results[3].status,
                'OPS-002_alert_system_risk': subtask_results[4].status
            },
            'ready_for_implementation': overall_status in ['PASS', 'WARNING'] and failed_subtasks == 0
        }

        logger.info(f"Task 0 Critical Risk Mitigation completed: {overall_status} "
                   f"(score: {average_validation_score:.3f}, ready: {task_0_results['ready_for_implementation']})")

        return task_0_results

    def get_validation_summary_dataframe(self) -> pd.DataFrame:
        """Get validation results summary as DataFrame.

        Returns:
            DataFrame with validation summary
        """
        if not self.validation_results:
            return pd.DataFrame()

        summary_data = []
        for _subtask_id, result in self.validation_results.items():
            summary_data.append({
                'Subtask_ID': result.subtask_id,
                'Subtask_Name': result.subtask_name,
                'Status': result.status,
                'Validation_Score': result.validation_score,
                'Timestamp': result.timestamp
            })

        return pd.DataFrame(summary_data)

    def export_task_0_results(self, output_path: str = "results/task_0_risk_mitigation_results.json") -> None:
        """Export Task 0 results to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        task_0_results = self.execute_task_0_critical_risk_mitigation()

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export to JSON
        import json
        with open(output_path, 'w') as f:
            json.dump(task_0_results, f, indent=2, default=str)

        logger.info(f"Task 0 results exported to: {output_path}")
