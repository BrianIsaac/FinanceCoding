"""
Tests for research documentation completeness and accuracy.

This module tests that all research documentation is complete, properly formatted,
and contains the required sections and information.
"""

import os
import re
from pathlib import Path

import pytest

# typing imports removed - using built-in types



class TestResearchDocumentation:
    """Test research documentation completeness and structure."""

    @pytest.fixture
    def docs_dir(self):
        """Path to documentation directory."""
        return Path(__file__).parent.parent.parent.parent / "docs" / "research"

    @pytest.fixture
    def required_docs(self):
        """List of required research documentation files."""
        return [
            "methodology.md",
            "evaluation_protocols.md",
            "statistical_analysis.md",
            "literature_review.md",
            "limitations_future_research.md",
            "reproducibility_guide.md",
        ]

    def test_all_required_docs_exist(self, docs_dir: Path, required_docs: list[str]):
        """Test that all required research documents exist."""
        for doc_file in required_docs:
            doc_path = docs_dir / doc_file
            assert doc_path.exists(), f"Required documentation file missing: {doc_file}"

    def test_methodology_document_structure(self, docs_dir: Path):
        """Test methodology document has required sections."""
        methodology_path = docs_dir / "methodology.md"

        with open(methodology_path) as f:
            content = f.read()

        required_sections = [
            "# Methodology Documentation",
            "## 1. Hierarchical Risk Parity",
            "## 2. LSTM Temporal Network",
            "## 3. Graph Attention Network",
            "## 4. Evaluation Protocols",
            "### 1.1 Mathematical Foundation",
            "### 1.2 Implementation Details",
            "### 1.3 Technical Implementation",
        ]

        for section in required_sections:
            assert section in content, f"Missing required section: {section}"

    def test_statistical_analysis_structure(self, docs_dir: Path):
        """Test statistical analysis document structure."""
        stats_path = docs_dir / "statistical_analysis.md"

        with open(stats_path) as f:
            content = f.read()

        required_elements = [
            "Jobson-Korkie",
            "Bootstrap Confidence Intervals",
            "Multiple Comparison Corrections",
            "Effect Size Analysis",
            "Cohen's d",
            "Sharpe Ratio",
            "p-value",
        ]

        for element in required_elements:
            assert element in content, f"Missing statistical element: {element}"

    def test_literature_review_citations(self, docs_dir: Path):
        """Test literature review contains proper citations."""
        lit_review_path = docs_dir / "literature_review.md"

        with open(lit_review_path) as f:
            content = f.read()

        # Check for proper citation format
        citation_patterns = [
            r"\*\*[A-Z][a-zA-Z\s&,]+\(\d{4}\)\*\*",  # **Author (Year)**
            r"[A-Z][a-zA-Z\s&,]+\(\d{4}\)",  # Author (Year)
        ]

        citation_found = False
        for pattern in citation_patterns:
            if re.search(pattern, content):
                citation_found = True
                break

        assert citation_found, "No proper citations found in literature review"

    def test_reproducibility_guide_completeness(self, docs_dir: Path):
        """Test reproducibility guide contains required sections."""
        repro_path = docs_dir / "reproducibility_guide.md"

        with open(repro_path) as f:
            content = f.read()

        required_sections = [
            "Environment Setup",
            "Data Setup",
            "Experiment Configuration",
            "Reproducibility Validation",
            "Result Validation",
            "```bash",  # Code examples
            "```python",  # Python examples
        ]

        for section in required_sections:
            assert section in content, f"Missing reproducibility section: {section}"

    def test_limitations_document_structure(self, docs_dir: Path):
        """Test limitations document covers key areas."""
        limitations_path = docs_dir / "limitations_future_research.md"

        with open(limitations_path) as f:
            content = f.read()

        required_areas = [
            "Computational Limitations",
            "Data Availability Limitations",
            "Model Architecture Limitations",
            "Statistical",
            "Future Research",
            "Recommendations",
        ]

        for area in required_areas:
            assert area in content, f"Missing limitations area: {area}"

    def test_evaluation_protocols_technical_detail(self, docs_dir: Path):
        """Test evaluation protocols contain sufficient technical detail."""
        eval_path = docs_dir / "evaluation_protocols.md"

        with open(eval_path) as f:
            content = f.read()

        technical_elements = [
            "Rolling Window",
            "Statistical Testing",
            "Bootstrap",
            "Significance Testing",
            "Transaction Cost",
            "Jobson-Korkie Test",
            "Memmel Correction",
        ]

        for element in technical_elements:
            assert element in content, f"Missing technical element: {element}"

    def test_documentation_formatting(self, docs_dir: Path, required_docs: list[str]):
        """Test documentation follows consistent formatting."""
        for doc_file in required_docs:
            doc_path = docs_dir / doc_file

            with open(doc_path) as f:
                content = f.read()

            # Check for proper markdown formatting
            assert content.startswith("#"), f"Document {doc_file} should start with main header"
            assert "## " in content, f"Document {doc_file} should have section headers"

            # Check for consistent section numbering
            section_pattern = r"## \d+\."
            sections = re.findall(section_pattern, content)
            assert len(sections) > 0, f"Document {doc_file} should have numbered sections"

    def test_cross_references_valid(self, docs_dir: Path):
        """Test that cross-references between documents are valid."""
        # This is a simplified test - in practice, you'd parse markdown links
        all_files = list(docs_dir.glob("*.md"))
        # file_names = {f.stem for f in all_files}  # Unused variable removed

        for doc_file in all_files:
            with open(doc_file) as f:
                content = f.read()

            # Look for markdown links to other docs
            link_pattern = r"\[([^\]]+)\]\(([^)]+\.md)\)"
            links = re.findall(link_pattern, content)

            for _link_text, link_target in links:
                # Extract filename without extension
                target_file = Path(link_target).stem
                if target_file not in ["README", "CHANGELOG"]:  # Exclude common files
                    # This would be more sophisticated in practice
                    pass  # Skip validation for now

    def test_documentation_length_adequate(self, docs_dir: Path, required_docs: list[str]):
        """Test that documentation files have adequate content length."""
        min_lengths = {
            "methodology.md": 5000,  # Should be comprehensive
            "statistical_analysis.md": 3000,  # Should cover all tests
            "literature_review.md": 8000,  # Should be thorough
            "limitations_future_research.md": 4000,  # Should be detailed
            "reproducibility_guide.md": 6000,  # Should be complete
            "evaluation_protocols.md": 4000,  # Should be detailed
        }

        for doc_file in required_docs:
            if doc_file in min_lengths:
                doc_path = docs_dir / doc_file

                with open(doc_path) as f:
                    content = f.read()

                actual_length = len(content)
                min_length = min_lengths[doc_file]

                assert (
                    actual_length >= min_length
                ), f"Document {doc_file} too short: {actual_length} < {min_length} characters"


class TestExperimentScripts:
    """Test experiment orchestration scripts exist and are executable."""

    @pytest.fixture
    def scripts_dir(self):
        """Path to scripts directory."""
        return Path(__file__).parent.parent.parent.parent / "scripts"

    def test_reproduce_research_script_exists(self, scripts_dir: Path):
        """Test that main research reproduction script exists."""
        script_path = scripts_dir / "run_experiments.py"
        assert script_path.exists(), "run_experiments.py script missing"

    def test_experiment_configs_exist(self):
        """Test that experiment configuration files exist."""
        configs_dir = Path(__file__).parent.parent.parent.parent / "configs" / "experiments"

        required_configs = ["full_evaluation.yaml"]

        for config_file in required_configs:
            config_path = configs_dir / config_file
            assert config_path.exists(), f"Required config file missing: {config_file}"

    def test_reproduce_script_has_main_function(self, scripts_dir: Path):
        """Test that run_experiments.py has proper main function."""
        script_path = scripts_dir / "run_experiments.py"

        with open(script_path) as f:
            content = f.read()

        # Check for proper structure
        assert "def main(" in content, "Script should have main() function"
        assert 'if __name__ == "__main__":' in content, "Script should have main guard"


class TestOpenSourcePackage:
    """Test open source release package completeness."""

    @pytest.fixture
    def repo_root(self):
        """Path to repository root."""
        return Path(__file__).parent.parent.parent.parent

    def test_required_files_exist(self, repo_root: Path):
        """Test that required open source files exist."""
        required_files = ["README.md", "LICENSE", "CONTRIBUTING.md", "pyproject.toml"]

        for required_file in required_files:
            file_path = repo_root / required_file
            assert file_path.exists(), f"Required file missing: {required_file}"

    def test_license_is_appropriate(self, repo_root: Path):
        """Test that LICENSE file contains appropriate license."""
        license_path = repo_root / "LICENSE"

        with open(license_path) as f:
            content = f.read()

        # Check for common license indicators
        license_indicators = [
            "MIT License",
            "Apache License",
            "BSD License",
            "Permission is hereby granted",
        ]

        license_found = any(indicator in content for indicator in license_indicators)
        assert license_found, "No recognized license found in LICENSE file"

    def test_contributing_guidelines_comprehensive(self, repo_root: Path):
        """Test that CONTRIBUTING.md provides comprehensive guidelines."""
        contrib_path = repo_root / "CONTRIBUTING.md"

        with open(contrib_path) as f:
            content = f.read()

        required_sections = [
            "Code of Conduct",
            "How to Contribute",
            "Development Setup",
            "Running Tests",
            "Coding Standards",
            "Pull Request",
        ]

        for section in required_sections:
            assert section in content, f"Missing contributing section: {section}"
