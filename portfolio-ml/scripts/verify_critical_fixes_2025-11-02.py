#!/usr/bin/env python3
"""
Verification script to confirm all critical fixes are active.

This script checks:
1. GAT Multiplier Fix (0.1 not 2.0)
2. Config Period Fix (2021-01-01 evaluation end date)
3. Constraint Standardization (all models at 0.15 max position weight)
4. Additional checks (no other 2.0 multipliers, turnover penalties enabled)
"""

from pathlib import Path
import re
import sys


def check_gat_multiplier():
    """Verify GAT multiplier fix (0.1 not 2.0)."""
    try:
        file_path = Path("src/models/gat/diversification_loss.py")

        if not file_path.exists():
            return False, "File not found"

        with open(file_path, 'r') as f:
            content = f.read()

        # Check line 260 area for the correct multiplier
        lines = content.split('\n')

        # Look for the min_assets_loss assignment
        found = False
        for i, line in enumerate(lines, 1):
            if 'min_assets_loss' in line and '*' in line:
                if '0.1 * min_assets_penalty' in line:
                    found = True
                    return True, f"Verified at line {i}: {line.strip()}"
                elif '2.0 * min_assets_penalty' in line:
                    return False, f"Found old 2.0 multiplier at line {i}: {line.strip()}"

        if not found:
            return False, "Could not find min_assets_loss assignment"

    except Exception as e:
        return False, str(e)


def check_config_period():
    """Verify config period fix (2021-01-01 evaluation end date)."""
    try:
        file_path = Path("configs/backtest/config.yaml")

        if not file_path.exists():
            return False, "File not found"

        with open(file_path, 'r') as f:
            content = f.read()

        lines = content.split('\n')

        # Look for evaluation_end_date
        for i, line in enumerate(lines, 1):
            if 'evaluation_end_date' in line:
                if '2021-01-01' in line:
                    return True, f"Verified at line {i}: {line.strip()}"
                elif '2019-05-01' in line:
                    return False, f"Found old date at line {i}: {line.strip()}"
                else:
                    # Extract the date value
                    match = re.search(r'"([^"]+)"', line)
                    if match:
                        date_val = match.group(1)
                        return False, f"Found unexpected date at line {i}: {date_val}"

        return False, "Could not find evaluation_end_date in config"

    except Exception as e:
        return False, str(e)


def check_constraint_standardization():
    """Verify constraint standardization (all models at 0.15 max weight)."""
    try:
        file_path = Path("scripts/run_comprehensive_backtest.py")

        if not file_path.exists():
            return False, "File not found"

        with open(file_path, 'r') as f:
            content = f.read()

        issues = []

        # Check for HRP constraints
        if 'max_position_weight=0.15' not in content:
            issues.append("Missing 0.15 max_position_weight in constraints")

        # Count occurrences of 0.15 max_position_weight (should be 3 for HRP, LSTM, GAT)
        matches = re.findall(r'max_position_weight=0\.15', content)
        if len(matches) < 3:
            issues.append(f"Expected 3+ instances of max_position_weight=0.15, found {len(matches)}")

        # Check for max_monthly_turnover=0.30
        turnover_matches = re.findall(r'max_monthly_turnover=0\.30', content)
        if len(turnover_matches) < 3:
            issues.append(f"Expected 3+ instances of max_monthly_turnover=0.30, found {len(turnover_matches)}")

        # Check for enable_turnover_penalty=True
        penalty_matches = re.findall(r'enable_turnover_penalty=True', content)
        if len(penalty_matches) < 3:
            issues.append(f"Expected 3+ instances of enable_turnover_penalty=True, found {len(penalty_matches)}")

        if issues:
            return False, "; ".join(issues)

        return True, f"All 3 models verified: max_position_weight=0.15, max_monthly_turnover=0.30, enable_turnover_penalty=True"

    except Exception as e:
        return False, str(e)


def check_no_other_multipliers():
    """Verify no other hardcoded 2.0 multipliers exist in loss calculations."""
    try:
        file_path = Path("src/models/gat/diversification_loss.py")

        if not file_path.exists():
            return False, "File not found"

        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Specific check: look for the min_assets_loss line to ensure it uses 0.1
        min_assets_loss_found = False
        for i, line in enumerate(lines, 1):
            if 'min_assets_loss' in line and '=' in line and '*' in line:
                min_assets_loss_found = True
                if '0.1 *' in line:
                    # Correct value
                    continue
                elif '2.0 *' in line or '2 *' in line:
                    return False, f"Found old 2.0 multiplier in min_assets_loss at line {i}"

        if not min_assets_loss_found:
            return False, "Could not verify min_assets_loss line"

        # Also check for any suspicious loss computations (not hyperparameter definitions)
        # that use 2.0 multipliers inappropriately
        issues = []
        in_loss_calculation = False
        for i, line in enumerate(lines, 1):
            # Skip parameter definitions (has : float =)
            if ': float =' in line or ': int =' in line:
                continue

            # Look for actual loss computation lines
            if '_loss' in line and '=' in line and 'torch' in line:
                # This is a loss computation
                if '2.0 *' in line and 'concentration' not in line.lower():
                    issues.append(f"Line {i}: {line.strip()}")

        if issues:
            return False, "Found suspicious loss calculations: " + "; ".join(issues[:2])

        return True, "No problematic 2.0 multipliers in loss calculations"

    except Exception as e:
        return False, str(e)


def check_turnover_penalties_enabled():
    """Verify all models have enable_turnover_penalty=True."""
    try:
        file_path = Path("scripts/run_comprehensive_backtest.py")

        if not file_path.exists():
            return False, "File not found"

        with open(file_path, 'r') as f:
            content = f.read()

        # Extract model initialization sections
        models = {
            'HRP': r'def create_hrp_models.*?return models',
            'LSTM': r'def create_lstm_models.*?return models',
            'GAT': r'def create_gat_models.*?return models',
        }

        missing_turnover = []

        for model_name, pattern in models.items():
            match = re.search(pattern, content, re.DOTALL)
            if match:
                section = match.group(0)
                if 'enable_turnover_penalty=True' not in section:
                    missing_turnover.append(model_name)

        if missing_turnover:
            return False, f"Missing enable_turnover_penalty=True in: {', '.join(missing_turnover)}"

        return True, "All models have enable_turnover_penalty=True"

    except Exception as e:
        return False, str(e)


def main():
    """Run all verification checks and report results."""
    print("\n" + "="*80)
    print("CRITICAL FIXES VERIFICATION - 2025-11-02")
    print("="*80 + "\n")

    checks = [
        ("Fix #1: GAT Multiplier", check_gat_multiplier),
        ("Fix #2: Config Period", check_config_period),
        ("Fix #3: Constraint Standardization", check_constraint_standardization),
        ("Check #4: No Other 2.0 Multipliers", check_no_other_multipliers),
        ("Check #5: Turnover Penalties Enabled", check_turnover_penalties_enabled),
    ]

    results = []
    all_passed = True

    for check_name, check_func in checks:
        passed, message = check_func()
        results.append((check_name, passed, message))

        status_symbol = "✓" if passed else "✗"
        status_text = "VERIFIED" if passed else "FAILED"

        print(f"{status_symbol} {check_name} - {status_text}")
        print(f"  └─ {message}\n")

        if not passed:
            all_passed = False

    # Summary
    print("="*80)
    if all_passed:
        print("OVERALL: PASS")
        print("="*80 + "\n")
        print("All critical fixes are active and verified.")
        return 0
    else:
        print("OVERALL: FAIL")
        print("="*80 + "\n")
        print("Some checks failed. See details above.")

        # Print failed checks
        failed = [r for r in results if not r[1]]
        if failed:
            print("\nFailed checks:")
            for check_name, _, message in failed:
                print(f"  - {check_name}: {message}")

        return 1


if __name__ == "__main__":
    sys.exit(main())
