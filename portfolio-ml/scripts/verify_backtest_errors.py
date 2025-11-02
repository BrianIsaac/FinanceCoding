#!/usr/bin/env python3
"""
Verification script to diagnose and validate backtest error issues.

This script checks for the critical issues identified in the backtest error analysis:
1. GAT Features Dimension Mismatch (CRITICAL)
2. LSTM Shape Mismatch (HIGH)
3. Extreme Concentration (HIGH)
4. GAT Training KeyError (MEDIUM)

Usage:
    uv run python scripts/verify_backtest_errors.py [--log-file PATH]
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def count_error_patterns(log_file: Path) -> dict[str, Any]:
    """
    Count occurrences of each error pattern in the log file.

    Args:
        log_file: Path to backtest log file

    Returns:
        Dictionary with error counts and sample messages
    """
    error_patterns = {
        "gat_features_mismatch": {
            "pattern": r"Features matrix dimension mismatch: features_shape=\((\d+),",
            "count": 0,
            "samples": [],
            "severity": "CRITICAL",
            "description": "GAT feature-to-asset mapping corruption"
        },
        "lstm_shape_mismatch": {
            "pattern": r"mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)",
            "count": 0,
            "samples": [],
            "severity": "HIGH",
            "description": "LSTM stale trainer model reference"
        },
        "extreme_concentration": {
            "pattern": r"CRITICAL: Extreme concentration at ([\d-]+.*?): max weight = ([\d.]+)%",
            "count": 0,
            "samples": [],
            "severity": "CRITICAL",
            "description": "HRP constraint enforcement failure"
        },
        "gat_training_keyerror": {
            "pattern": r"GAT \w+ training error at ([\d-]+).*?not in index",
            "count": 0,
            "samples": [],
            "severity": "MEDIUM",
            "description": "GAT training with missing tickers"
        },
        "position_limit_violation": {
            "pattern": r"Position limit violation at ([\d-]+.*?): max weight = ([\d.]+)%",
            "count": 0,
            "samples": [],
            "severity": "MEDIUM",
            "description": "Model exceeds 20% position limit"
        },
        "gat_missing_assets": {
            "pattern": r"Asset \w+ not in returns, using zeros",
            "count": 0,
            "samples": [],
            "severity": "LOW",
            "description": "GAT missing asset warnings (log pollution)"
        },
        "temporal_integrity_failure": {
            "pattern": r"Temporal integrity check failed.*?Confidence: ([\d.]+)",
            "count": 0,
            "samples": [],
            "severity": "LOW",
            "description": "Validation with dummy data"
        }
    }

    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            for error_type, info in error_patterns.items():
                match = re.search(info["pattern"], line)
                if match:
                    info["count"] += 1
                    if len(info["samples"]) < 5:  # Keep first 5 samples
                        info["samples"].append({
                            "line": line_num,
                            "message": line.strip(),
                            "match_groups": match.groups()
                        })

    return error_patterns


def analyze_gat_feature_mismatch(log_file: Path) -> dict[str, Any]:
    """
    Detailed analysis of GAT features dimension mismatch.

    Checks:
    - Feature matrix sizes vs expected nodes
    - Frequency by date/window
    - Impact on model performance
    """
    mismatches = []

    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(
                r"Features matrix dimension mismatch: features_shape=\((\d+), (\d+)\), expected_nodes=(\d+)",
                line
            )
            if match:
                feature_rows, feature_cols, expected_nodes = match.groups()
                mismatches.append({
                    "feature_rows": int(feature_rows),
                    "feature_cols": int(feature_cols),
                    "expected_nodes": int(expected_nodes),
                    "discrepancy": int(feature_rows) - int(expected_nodes)
                })

    if not mismatches:
        return {"status": "OK", "count": 0}

    df = pd.DataFrame(mismatches)

    return {
        "status": "ERROR",
        "count": len(mismatches),
        "unique_discrepancies": df["discrepancy"].unique().tolist(),
        "avg_discrepancy": df["discrepancy"].mean(),
        "max_discrepancy": df["discrepancy"].max(),
        "analysis": (
            f"Found {len(mismatches)} feature dimension mismatches. "
            f"Features consistently have {df['feature_rows'].mode()[0] if not df.empty else 'N/A'} rows "
            f"but expected {df['expected_nodes'].mode()[0] if not df.empty else 'N/A'} nodes. "
            "This indicates features are created for full universe before filtering."
        )
    }


def analyze_lstm_shape_mismatch(log_file: Path) -> dict[str, Any]:
    """
    Detailed analysis of LSTM shape mismatch errors.

    Checks:
    - Input vs expected dimensions
    - Batch size optimisation failures
    - Network recreation events
    """
    shape_errors = []
    network_recreations = []

    with open(log_file, 'r') as f:
        for line in f:
            # Shape mismatch errors
            match = re.search(
                r"mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)",
                line
            )
            if match:
                m1_rows, m1_cols, m2_rows, m2_cols = match.groups()
                shape_errors.append({
                    "input_features": int(m1_cols),
                    "network_expects": int(m2_rows),
                    "mismatch": int(m1_cols) - int(m2_rows)
                })

            # Network recreation events
            match = re.search(r"Created LSTM network with input_size=(\d+) for universe_size=(\d+)", line)
            if match:
                input_size, universe_size = match.groups()
                network_recreations.append({
                    "input_size": int(input_size),
                    "universe_size": int(universe_size)
                })

    if not shape_errors:
        return {"status": "OK", "count": 0}

    return {
        "status": "ERROR",
        "count": len(shape_errors),
        "network_recreations": len(network_recreations),
        "unique_mismatches": len(set(e["mismatch"] for e in shape_errors)),
        "analysis": (
            f"Found {len(shape_errors)} shape mismatches. "
            f"Network recreated {len(network_recreations)} times but trainer model reference not updated. "
            "This causes batch size optimisation to fail."
        )
    }


def analyze_extreme_concentration(log_file: Path) -> dict[str, Any]:
    """
    Detailed analysis of extreme concentration errors.

    Checks:
    - Dates with >50% concentration
    - Models affected
    - Relationship to market events
    """
    extreme_events = []

    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(
                r"CRITICAL: Extreme concentration at ([\d-]+\s+[\d:]+): max weight = ([\d.]+)%",
                line
            )
            if match:
                date_str, weight_pct = match.groups()
                extreme_events.append({
                    "date": pd.to_datetime(date_str),
                    "max_weight": float(weight_pct)
                })

    if not extreme_events:
        return {"status": "OK", "count": 0}

    df = pd.DataFrame(extreme_events)

    return {
        "status": "CRITICAL",
        "count": len(extreme_events),
        "dates": [d.strftime("%Y-%m-%d") for d in df["date"]],
        "max_concentration": df["max_weight"].max(),
        "avg_concentration": df["max_weight"].mean(),
        "analysis": (
            f"Found {len(extreme_events)} dates with >50% concentration. "
            f"Maximum concentration: {df['max_weight'].max():.1f}%. "
            "Indicates complete constraint enforcement failure in HRP model."
        )
    }


def generate_report(results: dict[str, Any], output_file: Path | None = None) -> None:
    """Generate comprehensive verification report."""

    report_lines = [
        "=" * 80,
        "BACKTEST ERROR VERIFICATION REPORT",
        "=" * 80,
        "",
        "SUMMARY",
        "-" * 80,
    ]

    # Summary table
    total_errors = sum(info["count"] for info in results["patterns"].values())
    critical_errors = sum(
        info["count"] for info in results["patterns"].values()
        if info["severity"] == "CRITICAL"
    )
    high_errors = sum(
        info["count"] for info in results["patterns"].values()
        if info["severity"] == "HIGH"
    )

    report_lines.extend([
        f"Total error messages: {total_errors:,}",
        f"  CRITICAL severity: {critical_errors:,}",
        f"  HIGH severity: {high_errors:,}",
        "",
        "ERROR BREAKDOWN BY CATEGORY",
        "-" * 80,
    ])

    # Sort by severity and count
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    sorted_patterns = sorted(
        results["patterns"].items(),
        key=lambda x: (severity_order[x[1]["severity"]], -x[1]["count"])
    )

    for error_type, info in sorted_patterns:
        report_lines.extend([
            f"\n{error_type.upper().replace('_', ' ')}",
            f"  Severity: {info['severity']}",
            f"  Count: {info['count']:,}",
            f"  Description: {info['description']}",
        ])

        if info["samples"]:
            report_lines.append("  Sample message:")
            report_lines.append(f"    {info['samples'][0]['message'][:120]}...")

    # Detailed analyses
    report_lines.extend([
        "",
        "",
        "DETAILED ANALYSES",
        "=" * 80,
    ])

    # GAT Features Analysis
    report_lines.extend([
        "",
        "1. GAT FEATURES DIMENSION MISMATCH",
        "-" * 80,
    ])
    gat_analysis = results["detailed"]["gat_features"]
    if gat_analysis["status"] == "OK":
        report_lines.append("  Status: ✓ No issues detected")
    else:
        report_lines.extend([
            f"  Status: ✗ {gat_analysis['count']} mismatches detected",
            f"  Average discrepancy: {gat_analysis.get('avg_discrepancy', 0):.1f} rows",
            f"  Max discrepancy: {gat_analysis.get('max_discrepancy', 0)} rows",
            f"  Analysis: {gat_analysis['analysis']}",
            "",
            "  ROOT CAUSE:",
            "    Features created for full universe (759 assets) BEFORE filtering,",
            "    then truncated to match filtered assets (399). This causes feature-to-asset",
            "    misalignment because truncation assumes ordering matches, but filtering",
            "    changes the order.",
            "",
            "  FIX REQUIRED:",
            "    Filter universe BEFORE calling _get_node_features() in model.py:1564"
        ])

    # LSTM Shape Analysis
    report_lines.extend([
        "",
        "2. LSTM SHAPE MISMATCH",
        "-" * 80,
    ])
    lstm_analysis = results["detailed"]["lstm_shapes"]
    if lstm_analysis["status"] == "OK":
        report_lines.append("  Status: ✓ No issues detected")
    else:
        report_lines.extend([
            f"  Status: ✗ {lstm_analysis['count']} shape errors detected",
            f"  Network recreations: {lstm_analysis.get('network_recreations', 0)}",
            f"  Analysis: {lstm_analysis['analysis']}",
            "",
            "  ROOT CAUSE:",
            "    When network is recreated with new input_size, trainer's model reference",
            "    is not updated (model.py:511). Trainer continues using old network during",
            "    batch size optimisation.",
            "",
            "  FIX REQUIRED:",
            "    Add 'self.trainer.model = self.network' after line 510 in model.py"
        ])

    # Extreme Concentration Analysis
    report_lines.extend([
        "",
        "3. EXTREME CONCENTRATION",
        "-" * 80,
    ])
    conc_analysis = results["detailed"]["extreme_concentration"]
    if conc_analysis["status"] == "OK":
        report_lines.append("  Status: ✓ No issues detected")
    else:
        report_lines.extend([
            f"  Status: ✗ {conc_analysis['count']} critical events",
            f"  Affected dates: {', '.join(conc_analysis['dates'])}",
            f"  Max concentration: {conc_analysis.get('max_concentration', 0):.1f}%",
            f"  Analysis: {conc_analysis['analysis']}",
            "",
            "  ROOT CAUSE:",
            "    HRP and LSTM models configured with max_position_weight=1.0 (100%),",
            "    but rolling engine expects 20% limit. HRP recursive bisection can",
            "    generate extreme concentrations without constraint enforcement.",
            "",
            "  FIX REQUIRED:",
            "    1. Set max_position_weight=0.20 in backtest script (lines 391, 417)",
            "    2. Add position limit checks in HRP recursive_bisection"
        ])

    # Recommendations
    report_lines.extend([
        "",
        "",
        "RECOMMENDED ACTIONS",
        "=" * 80,
        "",
        "PRIORITY 1 (CRITICAL - Fix Immediately):",
        "  1. GAT Features Mismatch - Prevents GAT from learning properly",
        "     File: src/models/gat/model.py:1564",
        "     Action: Filter universe before feature creation",
        "",
        "PRIORITY 2 (HIGH - Fix Soon):",
        "  2. LSTM Shape Mismatch - Prevents batch size optimisation",
        "     File: src/models/lstm/model.py:511",
        "     Action: Update trainer.model reference after network recreation",
        "",
        "  3. Extreme Concentration - Causes rebalance skipping",
        "     File: scripts/run_comprehensive_backtest.py:391, 417",
        "     Action: Change max_position_weight from 1.0 to 0.20",
        "",
        "PRIORITY 3 (MEDIUM - Address When Time Permits):",
        "  4. GAT Training KeyError - Reduces training quality",
        "     File: src/models/gat/model.py:595, 619, 660",
        "     Action: Filter universe to available tickers before training",
        "",
        "  5. GAT Missing Assets - Log pollution (143K warnings)",
        "     File: src/models/gat/model.py:998",
        "     Action: Change warning to debug level or batch logging",
        "",
        "=" * 80,
    ])

    report_text = "\n".join(report_lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report written to: {output_file}")
    else:
        print(report_text)


def main():
    """Main verification workflow."""
    parser = argparse.ArgumentParser(
        description="Verify and diagnose backtest error issues"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to backtest log file (auto-detects latest if not provided)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save verification report"
    )

    args = parser.parse_args()

    # Auto-detect latest log file if not provided
    if args.log_file is None:
        outputs_dir = Path("outputs")
        if not outputs_dir.exists():
            print("Error: outputs/ directory not found")
            return 1

        log_files = list(outputs_dir.rglob("run_comprehensive_backtest.log"))
        if not log_files:
            print("Error: No log files found in outputs/")
            return 1

        # Get most recent
        args.log_file = max(log_files, key=lambda p: p.stat().st_mtime)
        print(f"Using log file: {args.log_file}")

    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}")
        return 1

    print("Analyzing log file...")
    print()

    # Run analyses
    results = {
        "patterns": count_error_patterns(args.log_file),
        "detailed": {
            "gat_features": analyze_gat_feature_mismatch(args.log_file),
            "lstm_shapes": analyze_lstm_shape_mismatch(args.log_file),
            "extreme_concentration": analyze_extreme_concentration(args.log_file),
        }
    }

    # Generate report
    generate_report(results, args.output)

    # Return non-zero if critical issues found
    critical_count = sum(
        info["count"] for info in results["patterns"].values()
        if info["severity"] == "CRITICAL"
    )
    return 1 if critical_count > 0 else 0


if __name__ == "__main__":
    exit(main())
