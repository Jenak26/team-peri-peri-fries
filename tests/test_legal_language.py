"""CLAUDE.md section 8: forbidden legal language, enforced by CI.

The system assists forensic examination. It does not replace judicial determination
of admissibility or weight, and it signs nothing. A phrase that claims otherwise in a
code comment, a UI string, or a report template is a defect at the same severity as a
wrong number, because it is the sentence opposing counsel will read aloud.

Matching is word-bounded so that legitimate words containing a forbidden token --
`improves`, `approves`, `disproves` -- do not trip the gate.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

FORBIDDEN = (
    "court-admissible",
    "legally valid",
    "legally admissible",
    "certified evidence",
    "meets Section 63",
    "proves",
    "guaranteed authentic",
)

_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(phrase) for phrase in FORBIDDEN) + r")\b",
    re.IGNORECASE,
)

# Files that state the rule rather than break it: the specification that defines the
# forbidden list, the build plans that quote it, and this gate itself.
ALLOWED_PREFIXES = (
    "CLAUDE.md",
    "docs/superpowers/",
    "tests/test_legal_language.py",
)

# Anything that is not source or prose. Binary blobs would only produce noise.
SCANNED_SUFFIXES = {
    ".py",
    ".md",
    ".html",
    ".js",
    ".css",
    ".json",
    ".yml",
    ".yaml",
    ".txt",
    ".toml",
    ".cff",
    ".ini",
}


def _tracked_files() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.splitlines()


def test_no_forbidden_legal_language_in_tracked_files():
    offences: list[str] = []

    for name in _tracked_files():
        if name.startswith(ALLOWED_PREFIXES):
            continue
        path = Path(name)
        if path.suffix.lower() not in SCANNED_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            match = _PATTERN.search(line)
            if match:
                offences.append(f"{name}:{lineno}: {match.group(0)!r} in {line.strip()!r}")

    assert not offences, (
        "Forbidden legal language found. This system assists forensic examination; "
        "it does not determine admissibility or weight.\n" + "\n".join(offences)
    )


def test_gate_actually_matches_each_forbidden_phrase():
    """A gate that cannot fail is not a gate."""
    for phrase in FORBIDDEN:
        assert _PATTERN.search(f"this text says {phrase} somewhere"), phrase


def test_gate_does_not_trip_on_legitimate_substrings():
    for benign in ("the loss improves", "the reviewer approves", "this disproves it"):
        assert _PATTERN.search(benign) is None, benign
