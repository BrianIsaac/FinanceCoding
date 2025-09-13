"""
Comprehensive system resource monitoring for data pipeline operations.

Provides real-time monitoring of CPU, memory, disk I/O, and network usage
during large-scale data processing with alerting and reporting capabilities.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import psutil

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Snapshot of system resource usage at a point in time."""

    timestamp: pd.Timestamp
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_read_mb: float
    disk_write_mb: float
    disk_usage_percent: float
    network_sent_mb: float
    network_recv_mb: float
    process_cpu_percent: float
    process_memory_mb: float
    process_threads: int
    load_average: list[float] | None = None


@dataclass
class ResourceAlert:
    """Resource usage alert."""

    alert_type: str
    severity: str  # "warning", "critical"
    message: str
    timestamp: pd.Timestamp
    current_value: float
    threshold_value: float
    resource_name: str


@dataclass
class MonitoringSession:
    """Resource monitoring session with metadata."""

    session_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp | None = None
    operation_name: str = "unknown"
    snapshots: list[ResourceSnapshot] = field(default_factory=list)
    alerts: list[ResourceAlert] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SystemResourceMonitor:
    """
    Comprehensive system resource monitor for data pipeline operations.

    Provides continuous monitoring, alerting, and reporting of system resources
    during data processing operations with configurable thresholds and intervals.
    """

    def __init__(
        self,
        monitoring_interval: float = 5.0,
        cpu_warning_threshold: float = 80.0,
        cpu_critical_threshold: float = 95.0,
        memory_warning_threshold: float = 85.0,
        memory_critical_threshold: float = 95.0,
        disk_warning_threshold: float = 85.0,
        disk_critical_threshold: float = 95.0,
        auto_export_reports: bool = True,
        reports_dir: str = "data/monitoring_reports",
    ):
        """
        Initialize system resource monitor.

        Args:
            monitoring_interval: Seconds between resource snapshots
            cpu_warning_threshold: CPU usage warning threshold (%)
            cpu_critical_threshold: CPU usage critical threshold (%)
            memory_warning_threshold: Memory usage warning threshold (%)
            memory_critical_threshold: Memory usage critical threshold (%)
            disk_warning_threshold: Disk usage warning threshold (%)
            disk_critical_threshold: Disk usage critical threshold (%)
            auto_export_reports: Automatically export monitoring reports
            reports_dir: Directory for monitoring reports
        """
        self.monitoring_interval = monitoring_interval
        self.thresholds = {
            "cpu_warning": cpu_warning_threshold,
            "cpu_critical": cpu_critical_threshold,
            "memory_warning": memory_warning_threshold,
            "memory_critical": memory_critical_threshold,
            "disk_warning": disk_warning_threshold,
            "disk_critical": disk_critical_threshold,
        }

        self.auto_export_reports = auto_export_reports
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: threading.Thread | None = None
        self.current_session: MonitoringSession | None = None
        self.session_history: list[MonitoringSession] = []

        # System process handles
        self.process = psutil.Process()
        self.disk_io_start = None
        self.network_io_start = None

        # Alert handlers
        self.alert_handlers: list[callable] = []

        logger.info("SystemResourceMonitor initialized")

    def start_monitoring(self, operation_name: str = "pipeline_operation") -> str:
        """
        Start resource monitoring for an operation.

        Args:
            operation_name: Name of the operation being monitored

        Returns:
            Session ID for the monitoring session
        """
        if self.monitoring_active:
            logger.warning("Monitoring already active, stopping previous session")
            self.stop_monitoring()

        session_id = f"{operation_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_session = MonitoringSession(
            session_id=session_id,
            start_time=pd.Timestamp.now(),
            operation_name=operation_name,
        )

        # Reset I/O counters
        self.disk_io_start = psutil.disk_io_counters()
        self.network_io_start = psutil.net_io_counters()

        # Start monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name=f"ResourceMonitor-{session_id}",
            daemon=True
        )
        self.monitoring_thread.start()

        logger.info(f"Started resource monitoring session: {session_id}")
        return session_id

    def stop_monitoring(self) -> MonitoringSession | None:
        """
        Stop resource monitoring and return session data.

        Returns:
            Completed monitoring session
        """
        if not self.monitoring_active:
            logger.warning("No active monitoring session to stop")
            return None

        self.monitoring_active = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)

        if self.current_session:
            self.current_session.end_time = pd.Timestamp.now()
            session = self.current_session
            self.session_history.append(session)

            # Generate final report
            if self.auto_export_reports:
                self._export_session_report(session)

            logger.info(
                f"Stopped monitoring session {session.session_id} - "
                f"Duration: {session.end_time - session.start_time}, "
                f"Snapshots: {len(session.snapshots)}, "
                f"Alerts: {len(session.alerts)}"
            )

            self.current_session = None
            return session

        return None

    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring an operation."""
        session_id = self.start_monitoring(operation_name)
        try:
            yield session_id
        finally:
            self.stop_monitoring()

    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in separate thread."""
        logger.debug("Resource monitoring loop started")

        while self.monitoring_active:
            try:
                # Take resource snapshot
                snapshot = self._take_resource_snapshot()

                if self.current_session:
                    self.current_session.snapshots.append(snapshot)

                    # Check for threshold violations
                    alerts = self._check_thresholds(snapshot)
                    if alerts:
                        self.current_session.alerts.extend(alerts)
                        for alert in alerts:
                            self._handle_alert(alert)

                # Sleep until next snapshot
                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def _take_resource_snapshot(self) -> ResourceSnapshot:
        """Take snapshot of current resource usage."""
        # System CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()

        # Disk usage and I/O
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()

        # Calculate disk I/O rates
        disk_read_mb = 0.0
        disk_write_mb = 0.0
        if self.disk_io_start and disk_io:
            disk_read_mb = (disk_io.read_bytes - self.disk_io_start.read_bytes) / (1024 * 1024)
            disk_write_mb = (disk_io.write_bytes - self.disk_io_start.write_bytes) / (1024 * 1024)

        # Network I/O
        network_io = psutil.net_io_counters()
        network_sent_mb = 0.0
        network_recv_mb = 0.0
        if self.network_io_start and network_io:
            network_sent_mb = (network_io.bytes_sent - self.network_io_start.bytes_sent) / (1024 * 1024)
            network_recv_mb = (network_io.bytes_recv - self.network_io_start.bytes_recv) / (1024 * 1024)

        # Process-specific metrics
        process_memory = self.process.memory_info()

        # Load average (Unix-like systems only)
        load_average = None
        if hasattr(psutil, "getloadavg"):
            try:
                load_average = list(psutil.getloadavg())
            except Exception:
                pass

        return ResourceSnapshot(
            timestamp=pd.Timestamp.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb,
            disk_usage_percent=disk_usage.percent,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            process_cpu_percent=self.process.cpu_percent(),
            process_memory_mb=process_memory.rss / (1024 * 1024),
            process_threads=self.process.num_threads(),
            load_average=load_average,
        )

    def _check_thresholds(self, snapshot: ResourceSnapshot) -> list[ResourceAlert]:
        """Check resource thresholds and generate alerts."""
        alerts = []

        # Check CPU thresholds
        if snapshot.cpu_percent >= self.thresholds["cpu_critical"]:
            alerts.append(ResourceAlert(
                alert_type="cpu_usage",
                severity="critical",
                message=f"Critical CPU usage: {snapshot.cpu_percent:.1f}%",
                timestamp=snapshot.timestamp,
                current_value=snapshot.cpu_percent,
                threshold_value=self.thresholds["cpu_critical"],
                resource_name="cpu",
            ))
        elif snapshot.cpu_percent >= self.thresholds["cpu_warning"]:
            alerts.append(ResourceAlert(
                alert_type="cpu_usage",
                severity="warning",
                message=f"High CPU usage: {snapshot.cpu_percent:.1f}%",
                timestamp=snapshot.timestamp,
                current_value=snapshot.cpu_percent,
                threshold_value=self.thresholds["cpu_warning"],
                resource_name="cpu",
            ))

        # Check memory thresholds
        if snapshot.memory_percent >= self.thresholds["memory_critical"]:
            alerts.append(ResourceAlert(
                alert_type="memory_usage",
                severity="critical",
                message=f"Critical memory usage: {snapshot.memory_percent:.1f}%",
                timestamp=snapshot.timestamp,
                current_value=snapshot.memory_percent,
                threshold_value=self.thresholds["memory_critical"],
                resource_name="memory",
            ))
        elif snapshot.memory_percent >= self.thresholds["memory_warning"]:
            alerts.append(ResourceAlert(
                alert_type="memory_usage",
                severity="warning",
                message=f"High memory usage: {snapshot.memory_percent:.1f}%",
                timestamp=snapshot.timestamp,
                current_value=snapshot.memory_percent,
                threshold_value=self.thresholds["memory_warning"],
                resource_name="memory",
            ))

        # Check disk thresholds
        if snapshot.disk_usage_percent >= self.thresholds["disk_critical"]:
            alerts.append(ResourceAlert(
                alert_type="disk_usage",
                severity="critical",
                message=f"Critical disk usage: {snapshot.disk_usage_percent:.1f}%",
                timestamp=snapshot.timestamp,
                current_value=snapshot.disk_usage_percent,
                threshold_value=self.thresholds["disk_critical"],
                resource_name="disk",
            ))
        elif snapshot.disk_usage_percent >= self.thresholds["disk_warning"]:
            alerts.append(ResourceAlert(
                alert_type="disk_usage",
                severity="warning",
                message=f"High disk usage: {snapshot.disk_usage_percent:.1f}%",
                timestamp=snapshot.timestamp,
                current_value=snapshot.disk_usage_percent,
                threshold_value=self.thresholds["disk_warning"],
                resource_name="disk",
            ))

        return alerts

    def _handle_alert(self, alert: ResourceAlert) -> None:
        """Handle resource alert by logging and calling registered handlers."""
        if alert.severity == "critical":
            logger.critical(f"CRITICAL RESOURCE ALERT: {alert.message}")
        else:
            logger.warning(f"Resource Alert: {alert.message}")

        # Call registered alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def add_alert_handler(self, handler: callable) -> None:
        """Add custom alert handler function."""
        self.alert_handlers.append(handler)

    def get_current_stats(self) -> ResourceSnapshot:
        """Get current resource statistics."""
        return self._take_resource_snapshot()

    def get_session_summary(self, session: MonitoringSession = None) -> dict[str, Any]:
        """Get summary statistics for a monitoring session."""
        if session is None:
            session = self.current_session

        if not session or not session.snapshots:
            return {"error": "No session data available"}

        snapshots = session.snapshots

        # Calculate statistics
        cpu_values = [s.cpu_percent for s in snapshots]
        memory_values = [s.memory_percent for s in snapshots]
        process_memory_values = [s.process_memory_mb for s in snapshots]

        summary = {
            "session_id": session.session_id,
            "operation_name": session.operation_name,
            "duration_seconds": (
                (session.end_time or pd.Timestamp.now()) - session.start_time
            ).total_seconds(),
            "total_snapshots": len(snapshots),
            "total_alerts": len(session.alerts),
            "critical_alerts": sum(1 for a in session.alerts if a.severity == "critical"),
            "cpu_stats": {
                "average": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
            },
            "memory_stats": {
                "average": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
            },
            "process_memory_stats": {
                "average": sum(process_memory_values) / len(process_memory_values),
                "max": max(process_memory_values),
                "min": min(process_memory_values),
                "peak_mb": max(process_memory_values),
            },
            "alerts_by_type": {},
        }

        # Group alerts by type
        for alert in session.alerts:
            alert_type = alert.alert_type
            if alert_type not in summary["alerts_by_type"]:
                summary["alerts_by_type"][alert_type] = {"warning": 0, "critical": 0}
            summary["alerts_by_type"][alert_type][alert.severity] += 1

        return summary

    def export_session_data(self, session: MonitoringSession, output_path: Path) -> None:
        """Export session data to CSV format for analysis."""
        if not session.snapshots:
            logger.warning("No snapshot data to export")
            return

        # Convert snapshots to DataFrame
        snapshot_data = []
        for snapshot in session.snapshots:
            snapshot_dict = {
                "timestamp": snapshot.timestamp,
                "cpu_percent": snapshot.cpu_percent,
                "memory_percent": snapshot.memory_percent,
                "memory_used_mb": snapshot.memory_used_mb,
                "memory_available_mb": snapshot.memory_available_mb,
                "disk_read_mb": snapshot.disk_read_mb,
                "disk_write_mb": snapshot.disk_write_mb,
                "disk_usage_percent": snapshot.disk_usage_percent,
                "network_sent_mb": snapshot.network_sent_mb,
                "network_recv_mb": snapshot.network_recv_mb,
                "process_cpu_percent": snapshot.process_cpu_percent,
                "process_memory_mb": snapshot.process_memory_mb,
                "process_threads": snapshot.process_threads,
            }

            if snapshot.load_average:
                snapshot_dict.update({
                    "load_1min": snapshot.load_average[0],
                    "load_5min": snapshot.load_average[1],
                    "load_15min": snapshot.load_average[2],
                })

            snapshot_data.append(snapshot_dict)

        df = pd.DataFrame(snapshot_data)
        df.to_csv(output_path, index=False)

        logger.info(f"Session data exported to {output_path}")

    def _export_session_report(self, session: MonitoringSession) -> None:
        """Export comprehensive session report."""
        report_path = self.reports_dir / f"monitoring_report_{session.session_id}.json"

        summary = self.get_session_summary(session)

        report_data = {
            "report_generated_at": pd.Timestamp.now().isoformat(),
            "session_summary": summary,
            "session_metadata": session.metadata,
            "alerts": [
                {
                    "alert_type": alert.alert_type,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "resource_name": alert.resource_name,
                }
                for alert in session.alerts
            ],
            "monitoring_config": {
                "monitoring_interval": self.monitoring_interval,
                "thresholds": self.thresholds,
            },
        }

        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        # Also export CSV data
        csv_path = self.reports_dir / f"monitoring_data_{session.session_id}.csv"
        self.export_session_data(session, csv_path)

    def cleanup_old_reports(self, days_to_keep: int = 30) -> None:
        """Clean up old monitoring reports."""
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_to_keep)

        removed_count = 0
        for report_file in self.reports_dir.glob("monitoring_*"):
            if report_file.stat().st_mtime < cutoff_date.timestamp():
                report_file.unlink()
                removed_count += 1

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old monitoring reports")

    def get_system_health_check(self) -> dict[str, Any]:
        """Get comprehensive system health check."""
        current_stats = self.get_current_stats()

        health_status = "healthy"
        issues = []

        # Check current resource usage
        if current_stats.cpu_percent > self.thresholds["cpu_critical"]:
            health_status = "critical"
            issues.append(f"Critical CPU usage: {current_stats.cpu_percent:.1f}%")
        elif current_stats.cpu_percent > self.thresholds["cpu_warning"]:
            health_status = "warning" if health_status == "healthy" else health_status
            issues.append(f"High CPU usage: {current_stats.cpu_percent:.1f}%")

        if current_stats.memory_percent > self.thresholds["memory_critical"]:
            health_status = "critical"
            issues.append(f"Critical memory usage: {current_stats.memory_percent:.1f}%")
        elif current_stats.memory_percent > self.thresholds["memory_warning"]:
            health_status = "warning" if health_status == "healthy" else health_status
            issues.append(f"High memory usage: {current_stats.memory_percent:.1f}%")

        if current_stats.disk_usage_percent > self.thresholds["disk_critical"]:
            health_status = "critical"
            issues.append(f"Critical disk usage: {current_stats.disk_usage_percent:.1f}%")
        elif current_stats.disk_usage_percent > self.thresholds["disk_warning"]:
            health_status = "warning" if health_status == "healthy" else health_status
            issues.append(f"High disk usage: {current_stats.disk_usage_percent:.1f}%")

        return {
            "health_status": health_status,
            "issues": issues,
            "current_stats": {
                "cpu_percent": current_stats.cpu_percent,
                "memory_percent": current_stats.memory_percent,
                "memory_available_gb": current_stats.memory_available_mb / 1024,
                "disk_usage_percent": current_stats.disk_usage_percent,
                "process_memory_mb": current_stats.process_memory_mb,
            },
            "recommendations": self._get_health_recommendations(current_stats),
            "timestamp": current_stats.timestamp.isoformat(),
        }

    def _get_health_recommendations(self, stats: ResourceSnapshot) -> list[str]:
        """Get health recommendations based on current stats."""
        recommendations = []

        if stats.memory_percent > 80:
            recommendations.append("Consider reducing data chunk sizes to lower memory usage")
            recommendations.append("Enable memory optimization for DataFrames")

        if stats.cpu_percent > 80:
            recommendations.append("Reduce parallelism or add delays between processing batches")

        if stats.disk_usage_percent > 80:
            recommendations.append("Clean up temporary files and consider disk space expansion")

        if stats.process_memory_mb > 8000:  # 8GB
            recommendations.append("Process memory usage is high - consider chunked processing")

        return recommendations
