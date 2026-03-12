"""Tests for the profile-report skill."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add skill directory to path (same pattern as other ClawBio skill tests)
SKILL_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = SKILL_DIR.parent.parent
sys.path.insert(0, str(SKILL_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import profile_report as pr

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def mock_profile() -> dict:
    """Load the mock profile fixture."""
    return json.loads((FIXTURES_DIR / "mock_profile.json").read_text())


@pytest.fixture
def empty_profile() -> dict:
    """Profile with no skill results."""
    return {
        "metadata": {
            "patient_id": "EMPTY001",
            "input_file": "empty.txt",
            "checksum": "000",
            "upload_date": "2026-01-01T00:00:00+00:00",
        },
        "genotypes": {"rs1": {"chrom": "1", "pos": 100, "genotype": "AG"}},
        "ancestry": None,
        "skill_results": {},
    }


@pytest.fixture
def full_profile(mock_profile: dict) -> dict:
    """Profile with all 4 skills completed."""
    profile = mock_profile.copy()
    profile["skill_results"] = dict(mock_profile["skill_results"])
    # Add synthetic PRS
    profile["skill_results"]["prs"] = {
        "run_at": "2026-01-01T00:00:00+00:00",
        "data": {
            "skill": "prs",
            "version": "0.2.0",
            "completed_at": "2026-01-01T00:00:00+00:00",
            "input_checksum": "sha256:abc123",
            "summary": {"scores_computed": 2, "traits_assessed": 2},
            "data": {
                "scores": [
                    {
                        "pgs_id": "PGS000013",
                        "trait": "Type 2 diabetes",
                        "raw_score": 0.83,
                        "z_score": -0.97,
                        "percentile": 14.0,
                        "risk_category": "Low",
                        "variants_used": 6,
                        "variants_total": 8,
                    },
                    {
                        "pgs_id": "PGS000011",
                        "trait": "Atrial fibrillation",
                        "raw_score": 1.12,
                        "z_score": 2.04,
                        "percentile": 98.0,
                        "risk_category": "Elevated",
                        "variants_used": 9,
                        "variants_total": 12,
                    },
                ],
            },
        },
    }
    # Add synthetic compare
    profile["skill_results"]["compare"] = {
        "run_at": "2026-01-01T00:00:00+00:00",
        "data": {
            "skill": "compare",
            "version": "0.2.0",
            "completed_at": "2026-01-01T00:00:00+00:00",
            "input_checksum": "sha256:abc123",
            "summary": {"total_snps_compared": 100000, "ibs2_proportion": 0.71},
            "data": {
                "ibs_summary": {
                    "total_snps_compared": 100000,
                    "ibs2_count": 71000,
                    "ibs2_proportion": 0.71,
                    "ibs1_count": 25000,
                    "ibs1_proportion": 0.25,
                    "ibs0_count": 4000,
                    "ibs0_proportion": 0.04,
                },
                "ancestry_estimation": {
                    "European": 0.85,
                    "South Asian": 0.10,
                    "East Asian": 0.05,
                },
            },
        },
    }
    return profile


# ---------------------------------------------------------------------------
# Profile loading tests
# ---------------------------------------------------------------------------


class TestLoadProfile:
    def test_load_valid_profile(self):
        """Test loading a valid profile JSON."""
        profile = pr.load_profile(FIXTURES_DIR / "mock_profile.json")
        assert profile["metadata"]["patient_id"] == "TEST001"
        assert "genotypes" in profile
        assert "skill_results" in profile

    def test_load_nonexistent_profile(self):
        """Test that loading a missing file raises ValueError."""
        with pytest.raises(ValueError, match="Profile not found"):
            pr.load_profile("/nonexistent/path.json")


# ---------------------------------------------------------------------------
# Section renderer tests
# ---------------------------------------------------------------------------


class TestRenderPharmgx:
    def test_with_data(self, mock_profile: dict):
        """Test PGx section renders gene table and drug recommendations."""
        result = pr.render_pharmgx_section(mock_profile)
        assert "## Pharmacogenomics" in result
        assert "CYP2D6" in result
        assert "Poor Metabolizer" in result
        assert "Codeine" in result
        assert "Drugs to Avoid" in result
        assert "Drugs Requiring Caution" in result

    def test_without_data(self, empty_profile: dict):
        """Test PGx section shows placeholder when no data."""
        result = pr.render_pharmgx_section(empty_profile)
        assert "Not yet assessed" in result
        assert "clawbio.py run pharmgx" in result


class TestRenderPrs:
    def test_with_data(self, full_profile: dict):
        """Test PRS section renders score table."""
        result = pr.render_prs_section(full_profile)
        assert "## Polygenic Risk Scores" in result
        assert "PGS000013" in result
        assert "Type 2 diabetes" in result
        assert "Elevated risk" in result  # 98th percentile AF

    def test_without_data(self, empty_profile: dict):
        """Test PRS section shows placeholder when no data."""
        result = pr.render_prs_section(empty_profile)
        assert "Not yet assessed" in result


class TestRenderNutrigx:
    def test_with_data(self, mock_profile: dict):
        """Test NutriGx section renders domain risk table."""
        result = pr.render_nutrigx_section(mock_profile)
        assert "## Nutrigenomics" in result
        assert "Caffeine" in result
        assert "Moderate" in result
        assert "coverage" in result.lower()

    def test_without_data(self, empty_profile: dict):
        """Test NutriGx section shows placeholder when no data."""
        result = pr.render_nutrigx_section(empty_profile)
        assert "Not yet assessed" in result


class TestRenderAncestry:
    def test_with_data(self, full_profile: dict):
        """Test ancestry section renders IBS + estimation."""
        result = pr.render_ancestry_section(full_profile)
        assert "## Ancestry" in result
        assert "European" in result
        assert "IBS" in result

    def test_without_data(self, empty_profile: dict):
        """Test ancestry section shows placeholder when no data."""
        result = pr.render_ancestry_section(empty_profile)
        assert "Not yet assessed" in result


# ---------------------------------------------------------------------------
# Cross-domain insights
# ---------------------------------------------------------------------------


class TestCrossDomainInsights:
    def test_finds_cyp1a2_overlap(self, mock_profile: dict):
        """CYP1A2 in both PGx gene profiles and NutriGx caffeine SNP."""
        result = pr.find_cross_domain_insights(mock_profile)
        assert "CYP1A2" in result
        assert "pharmgx" in result
        assert "nutrigx" in result

    def test_no_insights_empty(self, empty_profile: dict):
        """No insights when no skills completed."""
        result = pr.find_cross_domain_insights(empty_profile)
        assert "No cross-domain" in result


# ---------------------------------------------------------------------------
# Full report generation
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_full_report_all_skills(self, full_profile: dict):
        """Full report with all 4 skills produces complete document."""
        report = pr.generate_profile_report(full_profile)
        assert "# Your Genomic Profile" in report
        assert "Executive Summary" in report
        assert "Pharmacogenomics" in report
        assert "Polygenic Risk Scores" in report
        assert "Nutrigenomics" in report
        assert "Ancestry" in report
        assert "Disclaimer" in report
        assert "4 of 4" in report

    def test_partial_report(self, mock_profile: dict):
        """Report with only pharmgx + nutrigx shows placeholders for missing."""
        report = pr.generate_profile_report(mock_profile)
        assert "2 of 4" in report
        assert "Not yet assessed" in report  # PRS and compare missing


# ---------------------------------------------------------------------------
# CLI demo mode
# ---------------------------------------------------------------------------


class TestDemoMode:
    def test_build_demo_profile(self):
        """Demo profile builder returns a valid profile dict."""
        profile = pr.build_demo_profile()
        assert "metadata" in profile
        assert "genotypes" in profile
        assert "skill_results" in profile
        # Should have at least PRS and compare from synthetic data
        skills = profile.get("skill_results", {})
        assert "prs" in skills or "pharmgx" in skills
