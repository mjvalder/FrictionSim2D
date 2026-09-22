"""Validation rules for simulation results before central DB upload.

Used by the staging pipeline to automatically validate results before
they can be promoted from ``staged`` → ``validated``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
    'scan_angle': (-360.0, 360.0),      # the scan angle is signed; -15 is a real run
    'layers': (1, 20),
    'tip_radius': (0.0, 10_000.0),
    'size_x': (0.0, 100_000.0),
    'size_y': (0.0, 100_000.0),
}

_REQUIRED_FIELDS = ['material', 'family', 'model']

#: `to_record` (`src/aiida/data/results.py::_extras_for_record`) unconditionally
#: writes these into every row's `extras`: `skip_fraction` from `RunSummary`,
#: `twist_angle` and `potentials` for hetero findability. None of the three is
#: a measurement -- an extras dict containing only these is indistinguishable
#: from an empty one, so a bare truthiness check on `extras` (a non-empty dict
#: is always truthy, even `{'twist_angle': None}`) can never fail for a real
#: row, COF or not. Treat them as bookkeeping, not evidence of a result.
_BOOKKEEPING_EXTRAS_KEYS = frozenset({'skip_fraction', 'twist_angle', 'potentials'})

#: The load column each family is expected to use, for the consistency warning.
#: Families absent from this map (the PES scans) carry neither.
_LOAD_COLUMN_BY_FAMILY = {
    'afm': 'force_nN', 'indent': 'force_nN',
    'sheetonsheet': 'pressure_gpa', 'hetero': 'pressure_gpa',
    'flake': 'pressure_gpa',
}


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


def _has_measured_extra(extras: Any) -> bool:
    """True if `extras` carries at least one value beyond the bookkeeping keys.

    `mean_cof == 0.0` is a real, in-range value (`is not None`) and must not
    route through here -- see `validate_completeness`'s caller, which checks
    `mean_cof is not None` directly and only falls back to this for the extras
    half of the "carries something measured" requirement.
    """
    if not isinstance(extras, dict):
        return False
    return any(v is not None for k, v in extras.items()
               if k not in _BOOKKEEPING_EXTRAS_KEYS)


def validate_completeness(record: ResultRecord) -> ValidationResult:
    """Check the record carries an identity and something measured."""
    result = ValidationResult()

    for field_name in _REQUIRED_FIELDS:
        val = getattr(record, field_name, None)
        if val is None or (isinstance(val, str) and not val.strip()):
            result.add_error(f"Required field '{field_name}' is missing or empty")

    # A row has a result if it has a COF (0.0 counts -- that is a real value,
    # not a missing one; validate_ranges flags exact 0.0 as a warning
    # separately) OR at least one extras value that isn't pure bookkeeping (a
    # PES grid's descriptors, an indent's retract features, ...).
    if record.mean_cof is None and not _has_measured_extra(record.extras):
        result.add_error(
            "At least one result field required: mean_cof or a non-bookkeeping "
            "extras value"
        )

    # `is_complete=False` means summarise_run could not compute a COF. Storing
    # such a row locally is right; PUBLISHING it centrally is not, because
    # nothing downstream distinguishes it from a measured one.
    if not record.is_complete:
        result.add_error(
            "is_complete is False: no COF could be computed for this load "
            "point, so it must not be published"
        )

    return result


def validate_ranges(record: ResultRecord) -> ValidationResult:
    """Check that numeric fields fall within physically reasonable ranges."""
    from ..core.observables import FAMILIES  # pylint: disable=import-outside-toplevel
    from ..run.naming import MODELS  # pylint: disable=import-outside-toplevel

    result = ValidationResult()

    for field_name, (lo, hi) in _VALID_RANGES.items():
        val = getattr(record, field_name, None)
        if val is None:
            continue
        if val < lo or val > hi:
            result.add_error(
                f"Field '{field_name}' = {val} is outside valid range [{lo}, {hi}]"
            )

    if record.family and record.family not in FAMILIES:
        result.add_error(f"family '{record.family}' not in {FAMILIES}")
    if record.model and record.model not in MODELS:
        result.add_error(f"model '{record.model}' not in {MODELS}")

    # An exact zero used to pass silently as "in range". It is either real
    # superlubricity or a missing measurement, and only the uploader knows
    # which -- so say so rather than publishing it unremarked.
    if record.mean_cof == 0.0:
        result.add_warning(
            "mean_cof is exactly 0.0. If the COF could not be measured this "
            "should be None with is_complete=False; if it is real "
            "superlubricity, add a note."
        )

    return result


def validate_consistency(record: ResultRecord) -> ValidationResult:
    """Check internal consistency of the record."""
    result = ValidationResult()

    expected = _LOAD_COLUMN_BY_FAMILY.get(record.family or '')
    if expected == 'force_nN':
        if record.pressure_gpa is not None and record.force_nN is None:
            result.add_warning(
                f"{record.family} result has pressure_gpa but no force_nN")
    elif expected == 'pressure_gpa':
        if record.force_nN is not None and record.pressure_gpa is None:
            result.add_warning(
                f"{record.family} result has force_nN but no pressure_gpa")

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
