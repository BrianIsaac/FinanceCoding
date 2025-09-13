"""Test suite for production validator.

Tests for production readiness validation framework including
all Task 0 critical risk mitigation subtasks.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.evaluation.validation.production_validator import (
    ProductionConstraints,
    ProductionReadinessValidator,
    RiskMitigationResult,
)


class TestProductionConstraints(unittest.TestCase):
    """Test ProductionConstraints dataclass."""

    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = ProductionConstraints()

        self.assertEqual(constraints.max_processing_time_hours, 4.0)
        self.assertEqual(constraints.max_gpu_memory_gb, 12.0)
        self.assertEqual(constraints.max_system_memory_gb, 32.0)
        self.assertEqual(constraints.max_monthly_turnover, 0.20)
        self.assertEqual(constraints.min_sharpe_improvement, 0.20)
        self.assertEqual(constraints.max_position_weight, 0.10)

    def test_custom_constraints(self):
        """Test custom constraint values."""
        constraints = ProductionConstraints(
            max_processing_time_hours=2.0,
            max_gpu_memory_gb=8.0,
            max_system_memory_gb=16.0
        )

        self.assertEqual(constraints.max_processing_time_hours, 2.0)
        self.assertEqual(constraints.max_gpu_memory_gb, 8.0)
        self.assertEqual(constraints.max_system_memory_gb, 16.0)


class TestRiskMitigationResult(unittest.TestCase):
    """Test RiskMitigationResult dataclass."""

    def test_risk_mitigation_result_creation(self):
        """Test creating risk mitigation result."""
        result = RiskMitigationResult(
            subtask_id="0.1",
            subtask_name="Test subtask",
            status="PASS",
            details={"test": "data"},
            validation_score=0.95,
            timestamp="2025-09-13T12:00:00"
        )

        self.assertEqual(result.subtask_id, "0.1")
        self.assertEqual(result.subtask_name, "Test subtask")
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.details, {"test": "data"})
        self.assertEqual(result.validation_score, 0.95)
        self.assertEqual(result.timestamp, "2025-09-13T12:00:00")


class TestProductionReadinessValidator(unittest.TestCase):
    """Test ProductionReadinessValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = ProductionReadinessValidator()

    def test_validator_initialization(self):
        """Test validator initialization."""
        self.assertIsInstance(self.validator.constraints, ProductionConstraints)
        self.assertIsNotNone(self.validator.performance_benchmark)
        self.assertEqual(self.validator.validation_results, {})

    def test_validator_with_custom_constraints(self):
        """Test validator with custom constraints."""
        custom_constraints = ProductionConstraints(max_processing_time_hours=2.0)
        validator = ProductionReadinessValidator(constraints=custom_constraints)

        self.assertEqual(validator.constraints.max_processing_time_hours, 2.0)

    @patch('torch.cuda.is_available')
    def test_gpu_availability_detection(self, mock_cuda_available):
        """Test GPU availability detection."""
        mock_cuda_available.return_value = True

        with patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_name', return_value='Test GPU'):
            validator = ProductionReadinessValidator()

            self.assertTrue(validator.gpu_available)
            self.assertEqual(validator.gpu_count, 1)
            self.assertEqual(validator.gpu_names, ['Test GPU'])

    @patch('torch.cuda.is_available')
    def test_no_gpu_available(self, mock_cuda_available):
        """Test behaviour when GPU is not available."""
        mock_cuda_available.return_value = False

        validator = ProductionReadinessValidator()

        self.assertFalse(validator.gpu_available)
        self.assertEqual(validator.gpu_count, 0)
        self.assertEqual(validator.gpu_names, [])

    def test_validate_subtask_0_1_performance_baseline(self):
        """Test subtask 0.1: Performance baseline validation."""
        with patch.object(self.validator.performance_benchmark, 'run_comprehensive_performance_benchmark') as mock_benchmark:
            mock_benchmark.return_value = {
                'comprehensive_benchmark': {
                    'benchmarks': {
                        'results_validation': {'elapsed_time_hours': 0.5},
                        'executive_reporting': {'elapsed_time_hours': 0.2},
                        'statistical_testing': {'elapsed_time_hours': 0.3}
                    },
                    'constraint_validation': {
                        'overall_compliance': True,
                        'performance_summary': {
                            'total_processing_time_hours': 1.0,
                            'constraints_met': 3,
                            'total_constraints': 3
                        }
                    }
                }
            }

            result = self.validator.validate_subtask_0_1_performance_baseline()

            self.assertEqual(result.subtask_id, "0.1")
            self.assertEqual(result.status, "PASS")
            self.assertGreater(result.validation_score, 0.0)
            self.assertIn('baseline_performance', result.details)

    @patch('torch.cuda.is_available')
    def test_validate_subtask_0_2_gpu_memory_monitoring_no_gpu(self, mock_cuda_available):
        """Test subtask 0.2: GPU memory monitoring without GPU."""
        mock_cuda_available.return_value = False
        validator = ProductionReadinessValidator()

        result = validator.validate_subtask_0_2_gpu_memory_monitoring()

        self.assertEqual(result.subtask_id, "0.2")
        self.assertEqual(result.status, "WARNING")
        self.assertEqual(result.validation_score, 0.5)
        self.assertFalse(result.details['gpu_available'])

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_name')
    def test_validate_subtask_0_2_gpu_memory_monitoring_with_gpu(self, mock_get_name, mock_device_count, mock_cuda_available):
        """Test subtask 0.2: GPU memory monitoring with GPU."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        mock_get_name.return_value = 'Test GPU'

        with patch('torch.cuda.set_device'), \
             patch('torch.cuda.memory_allocated', return_value=1e9), \
             patch('torch.cuda.memory_reserved', return_value=2e9), \
             patch('torch.cuda.get_device_properties') as mock_props, \
             patch('torch.randn') as mock_randn, \
             patch('torch.cuda.empty_cache'):

            # Mock device properties
            mock_device_props = Mock()
            mock_device_props.total_memory = 12e9  # 12GB
            mock_props.return_value = mock_device_props

            # Mock tensor creation to avoid actual GPU allocation
            mock_tensor = Mock()
            mock_randn.return_value = mock_tensor

            validator = ProductionReadinessValidator()
            result = validator.validate_subtask_0_2_gpu_memory_monitoring()

            self.assertEqual(result.subtask_id, "0.2")
            self.assertIn(result.status, ["PASS", "WARNING", "FAIL"])
            self.assertTrue(result.details['gpu_available'])

    def test_validate_subtask_0_3_regulatory_compliance(self):
        """Test subtask 0.3: Regulatory compliance validation."""
        with patch('pathlib.Path.exists') as mock_exists:
            # Mock some compliance components as existing
            mock_exists.return_value = True

            result = self.validator.validate_subtask_0_3_regulatory_compliance()

            self.assertEqual(result.subtask_id, "0.3")
            self.assertIn(result.status, ["PASS", "WARNING", "FAIL"])
            self.assertIn('compliance_checklist', result.details)
            self.assertIn('compliance_score', result.details)

    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_validate_subtask_0_4_integration_testing_environment(self, mock_disk, mock_memory, mock_cpu):
        """Test subtask 0.4: Integration testing environment validation."""
        # Mock system information
        mock_cpu.return_value = 8
        mock_memory.return_value = Mock(total=32e9)  # 32GB
        mock_disk.return_value = Mock(free=100e9)    # 100GB free

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.mkdir'), \
             patch('builtins.__import__'):

            result = self.validator.validate_subtask_0_4_integration_testing_environment()

            self.assertEqual(result.subtask_id, "0.4")
            self.assertIn(result.status, ["PASS", "WARNING", "FAIL"])
            self.assertIn('system_info', result.details)
            self.assertIn('environment_score', result.details)

    def test_validate_subtask_0_5_alert_system_reliability(self):
        """Test subtask 0.5: Alert system reliability validation."""
        result = self.validator.validate_subtask_0_5_alert_system_reliability()

        self.assertEqual(result.subtask_id, "0.5")
        self.assertIn(result.status, ["PASS", "WARNING", "FAIL"])
        self.assertIn('alert_thresholds', result.details)
        self.assertIn('test_scenarios', result.details)
        self.assertIn('reliability_score', result.details)
        self.assertIn('confusion_matrix', result.details)

    def test_execute_task_0_critical_risk_mitigation(self):
        """Test complete Task 0 execution."""
        with patch.object(self.validator, 'validate_subtask_0_1_performance_baseline') as mock_0_1, \
             patch.object(self.validator, 'validate_subtask_0_2_gpu_memory_monitoring') as mock_0_2, \
             patch.object(self.validator, 'validate_subtask_0_3_regulatory_compliance') as mock_0_3, \
             patch.object(self.validator, 'validate_subtask_0_4_integration_testing_environment') as mock_0_4, \
             patch.object(self.validator, 'validate_subtask_0_5_alert_system_reliability') as mock_0_5:

            # Mock all subtasks to return PASS
            mock_results = []
            for i, mock_subtask in enumerate([mock_0_1, mock_0_2, mock_0_3, mock_0_4, mock_0_5], 1):
                mock_result = RiskMitigationResult(
                    subtask_id=f"0.{i}",
                    subtask_name=f"Test subtask 0.{i}",
                    status="PASS",
                    details={},
                    validation_score=0.9,
                    timestamp="2025-09-13T12:00:00"
                )
                mock_subtask.return_value = mock_result
                mock_results.append(mock_result)

            result = self.validator.execute_task_0_critical_risk_mitigation()

            self.assertEqual(result['task_id'], 'Task 0')
            self.assertEqual(result['overall_status'], 'PASS')
            self.assertEqual(result['subtask_summary']['total_subtasks'], 5)
            self.assertEqual(result['subtask_summary']['passed_subtasks'], 5)
            self.assertTrue(result['ready_for_implementation'])

    def test_execute_task_0_with_failures(self):
        """Test Task 0 execution with some failures."""
        with patch.object(self.validator, 'validate_subtask_0_1_performance_baseline') as mock_0_1, \
             patch.object(self.validator, 'validate_subtask_0_2_gpu_memory_monitoring') as mock_0_2, \
             patch.object(self.validator, 'validate_subtask_0_3_regulatory_compliance') as mock_0_3, \
             patch.object(self.validator, 'validate_subtask_0_4_integration_testing_environment') as mock_0_4, \
             patch.object(self.validator, 'validate_subtask_0_5_alert_system_reliability') as mock_0_5:

            # Mock some subtasks to fail
            statuses = ["PASS", "FAIL", "WARNING", "PASS", "PASS"]
            scores = [0.9, 0.0, 0.6, 0.8, 0.9]

            for i, (mock_subtask, status, score) in enumerate(zip([mock_0_1, mock_0_2, mock_0_3, mock_0_4, mock_0_5], statuses, scores), 1):
                mock_result = RiskMitigationResult(
                    subtask_id=f"0.{i}",
                    subtask_name=f"Test subtask 0.{i}",
                    status=status,
                    details={},
                    validation_score=score,
                    timestamp="2025-09-13T12:00:00"
                )
                mock_subtask.return_value = mock_result

            result = self.validator.execute_task_0_critical_risk_mitigation()

            self.assertEqual(result['subtask_summary']['passed_subtasks'], 3)
            self.assertEqual(result['subtask_summary']['failed_subtasks'], 1)
            self.assertEqual(result['subtask_summary']['warning_subtasks'], 1)
            self.assertFalse(result['ready_for_implementation'])  # Has failures

    def test_get_validation_summary_dataframe_empty(self):
        """Test getting validation summary DataFrame when no results."""
        df = self.validator.get_validation_summary_dataframe()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_get_validation_summary_dataframe_with_results(self):
        """Test getting validation summary DataFrame with results."""
        # Add mock validation results
        self.validator.validation_results["0.1"] = RiskMitigationResult(
            subtask_id="0.1",
            subtask_name="Test subtask",
            status="PASS",
            details={},
            validation_score=0.9,
            timestamp="2025-09-13T12:00:00"
        )

        df = self.validator.get_validation_summary_dataframe()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['Subtask_ID'], "0.1")
        self.assertEqual(df.iloc[0]['Status'], "PASS")

    def test_export_task_0_results(self):
        """Test exporting Task 0 results to JSON."""
        with patch.object(self.validator, 'execute_task_0_critical_risk_mitigation') as mock_execute:
            mock_execute.return_value = {
                'task_id': 'Task 0',
                'overall_status': 'PASS',
                'test': 'data'
            }

            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "test_results.json"

                self.validator.export_task_0_results(str(output_path))

                self.assertTrue(output_path.exists())

                # Verify JSON content
                with open(output_path) as f:
                    data = json.load(f)

                self.assertEqual(data['task_id'], 'Task 0')
                self.assertEqual(data['overall_status'], 'PASS')


class TestProductionValidatorIntegration(unittest.TestCase):
    """Integration tests for production validator."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.validator = ProductionReadinessValidator()

    def test_full_task_0_execution_integration(self):
        """Test full Task 0 execution in integration mode."""
        # This is a more comprehensive test that actually runs the validation
        # with minimal mocking to test real integration

        result = self.validator.execute_task_0_critical_risk_mitigation()

        # Verify result structure
        self.assertIn('task_id', result)
        self.assertIn('overall_status', result)
        self.assertIn('average_validation_score', result)
        self.assertIn('subtask_summary', result)
        self.assertIn('subtask_results', result)
        self.assertIn('risk_mitigation_assessment', result)
        self.assertIn('ready_for_implementation', result)

        # Verify all subtasks were executed
        self.assertEqual(result['subtask_summary']['total_subtasks'], 5)
        self.assertEqual(len(result['subtask_results']), 5)

        # Verify risk mitigation assessment covers all risks
        risk_assessment = result['risk_mitigation_assessment']
        expected_risks = ['PERF-001_performance_risk', 'OPS-001_gpu_memory_risk',
                         'BUS-001_compliance_risk', 'TECH-001_integration_risk',
                         'OPS-002_alert_system_risk']

        for risk in expected_risks:
            self.assertIn(risk, risk_assessment)
            self.assertIn(risk_assessment[risk], ['PASS', 'WARNING', 'FAIL'])

    def test_performance_baseline_real_execution(self):
        """Test performance baseline with real benchmark execution."""
        result = self.validator.validate_subtask_0_1_performance_baseline()

        # Should complete without errors
        self.assertIn(result.status, ['PASS', 'WARNING', 'FAIL'])
        self.assertIsInstance(result.validation_score, float)
        self.assertGreaterEqual(result.validation_score, 0.0)
        self.assertLessEqual(result.validation_score, 1.0)

    def test_regulatory_compliance_real_check(self):
        """Test regulatory compliance with real file system checks."""
        result = self.validator.validate_subtask_0_3_regulatory_compliance()

        # Should complete without errors
        self.assertIn(result.status, ['PASS', 'WARNING', 'FAIL'])
        self.assertIn('compliance_checklist', result.details)
        self.assertIn('compliance_score', result.details)

        # Compliance score should be between 0 and 1
        compliance_score = result.details['compliance_score']
        self.assertGreaterEqual(compliance_score, 0.0)
        self.assertLessEqual(compliance_score, 1.0)


if __name__ == '__main__':
    unittest.main()
