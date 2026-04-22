"""Validation rules for simulation results before central DB upload.

Used by the staging pipeline to automatically validate results before
they can be promoted from ``staged`` → ``validated``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import ResultRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Range constraints
# ---------------------------------------------------------------------------

_VALID_RANGES: Dict[str, tuple] = {
    'mean_cof': (0.0, 5.0),
    'std_cof': (0.0, 5.0),
    'temperature': (0.0, 10_000.0),
    'force_nN': (0.0, 100_000.0),
    'pressure_gpa': (0.0, 10_000.0),
    'scan_angle': (0.0, 360.0),
    'layers': (1, 20),
    'tip_radius': (0.0, 10_000.0),
}

_REQUIRED_FIELDS = ['material', 'simulation_type']
_REQUIRED_RESULT_FIELDS = ['mean_cof']

_VALID_SIMULATION_TYPES = {'afm', 'sheetonsheet'}


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Container for validation outcomes."""

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def __bool__(self) -> bool:
        return self.is_valid

    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        if self.is_valid:
            lines.append("PASSED")
        else:
            lines.append("FAILED")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_completeness(record: ResultRecord) -> ValidationResult:
    """Check that required fields are present and non-empty."""
    result = ValidationResult()

    for field_name in _REQUIRED_FIELDS:
        val = getattr(record, field_name, None)
        if val is None or (isinstance(val, str) and not val.strip()):
            result.add_error(f"Required field '{field_name}' is missing or empty")

    has_any_result = any(
        getattr(record, f, None) is not None for f in _REQUIRED_RESULT_FIELDS
    )
    if not has_any_result:
        result.add_error(
            f"At least one result field required: {_REQUIRED_RESULT_FIELDS}"
        )

    return result


def validate_ranges(record: ResultRecord) -> ValidationResult:
    """Check that numeric fields fall within physically reasonable ranges."""
    result = ValidationResult()

    for field_name, (lo, hi) in _VALID_RANGES.items():
        val = getattr(record, field_name, None)
        if val is None:
            continue
        if val < lo or val > hi:
            result.add_error(
                f"Field '{field_name}' = {val} is outside valid range [{lo}, {hi}]"
            )

    if record.simulation_type and record.simulation_type not in _VALID_SIMULATION_TYPES:
        result.add_error(
            f"simulation_type '{record.simulation_type}' not in {_VALID_SIMULATION_TYPES}"
        )

    return result


def validate_consistency(record: ResultRecord) -> ValidationResult:
    """Check internal consistency of the record."""
    result = ValidationResult()

    # AFM should have force, sheet-on-sheet should have pressure
    if record.simulation_type == 'afm':
        if record.pressure_gpa is not None and record.force_nN is None:
            result.add_warning(
                "AFM simulation has pressure_gpa but no force_nN"
            )
    elif record.simulation_type == 'sheetonsheet':
        if record.force_nN is not None and record.pressure_gpa is None:
            result.add_warning(
                "Sheet-on-sheet simulation has force_nN but no pressure_gpa"
            )

    # COF sanity: if both mean_lf and mean_nf are present, check COF matches
    if (
        record.mean_cof is not None
        and record.mean_lf is not None
        and record.mean_nf is not None
        and record.mean_nf > 0
    ):
        expected_cof = record.mean_lf / record.mean_nf
        if abs(record.mean_cof - expected_cof) > 0.01:
            result.add_warning(
                f"mean_cof ({record.mean_cof:.4f}) does not match "
                f"mean_lf/mean_nf ({expected_cof:.4f}); difference > 0.01"
            )

    return result


def validate_no_duplicate(
    record: ResultRecord,
    existing_hashes: Optional[List[str]] = None,
) -> ValidationResult:
    """Check for duplicates based on time_series_hash."""
    result = ValidationResult()

    if record.time_series_hash and existing_hashes:
        if record.time_series_hash in existing_hashes:
            result.add_error(
                f"Duplicate time_series_hash: {record.time_series_hash[:16]}..."
            )

    return result


def validate_record(
    record: ResultRecord,
    existing_hashes: Optional[List[str]] = None,
) -> ValidationResult:
    """Run all validation checks on a ResultRecord.

    Args:
        record: The record to validate.
        existing_hashes: Optional list of existing time_series_hash values
            for duplicate detection.

    Returns:
        Combined :class:`ValidationResult`.
    """
    combined = ValidationResult()

    for check in (
        lambda: validate_completeness(record),
        lambda: validate_ranges(record),
        lambda: validate_consistency(record),
        lambda: validate_no_duplicate(record, existing_hashes),
    ):
        sub = check()
        combined.errors.extend(sub.errors)
        combined.warnings.extend(sub.warnings)
        if not sub.is_valid:
            combined.is_valid = False

    return combined
