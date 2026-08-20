"""Exception hierarchy (all layers).

One base class so api/main.py can translate any expected failure into a response
without letting a raw traceback reach a findings file or the examination record.
"""

from __future__ import annotations


class PeriError(Exception):
    """Base class for every expected failure in this system."""


class IntakeError(PeriError):
    """L0: the exhibit could not be received, hashed, sealed, or probed."""


class CalibrationError(PeriError):
    """L4: calibration data is missing, malformed, or too small to fit."""


class ExaminationError(PeriError):
    """L1-L5: a stage failed while examining a sealed exhibit."""


class CorpusError(PeriError):
    """Training: the corpus is missing, malformed, or improperly split."""
