# Phase 2 - Custody Spine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build L0 and L6 - receive an exhibit, hash it, seal it read-only, probe it,
make a working copy, and record every one of those acts in an append-only SHA-256 hash
chain that a third party can verify offline.

**Architecture:** Three modules. `ledger.py` is a pure append-only chain over a JSONL
file with no knowledge of what an exhibit is. `intake.py` performs the six custody acts
and calls the ledger after each one. `manifest.py` snapshots everything that would have
to be identical for a replay to be meaningful - model checksums, config, seeds, library
versions - and hashes that snapshot.

**The working copy is a byte-identical copy of the original.** Not a transcode. A
transcode would make the working-copy hash depend on the muxer's clock and would
destroy replay determinism, and it would also mean our findings describe a file the
court never received. The original is opened `rb` exactly once, for hashing, and is
never opened for writing at any point in the program's life.

**Tech Stack:** Python 3.12, hashlib, shutil, subprocess (ffprobe), Pillow (EXIF for
image exhibits). No torch, no network.

**Spec:** `CLAUDE.md` sections 2 (L0, L6), 3 (CPU-track step 2), 7 (report page 6).
Roadmap: `docs/superpowers/plans/2026-08-20-ppf-00-ROADMAP.md`.

## Global Constraints

- Six ledger events per intake, in this exact order and with these exact names:
  `INTAKE_RECEIVED`, `ORIGINAL_HASHED`, `QUARANTINE_SEALED`, `CONTAINER_PROBED`,
  `WORKING_COPY_CREATED`, `WORKING_COPY_HASHED`.
- Genesis `prev_hash` is 64 zeros.
- Ledger lines are canonical JSON (`sort_keys`, tight separators), one per line, and
  the file is opened in append mode only.
- The original is chmod'ed read-only and that fact is asserted, not assumed.
- Evidence IDs: `EVD_<UTC compact timestamp>_<first 8 hex of the original SHA-256>`.
- All floats through `peri.core.canon.q`.
- Forbidden strings (CLAUDE.md section 8) must not appear.

---

### Task 1: Demo clip generator (needed by every later phase's tests)

**Files:**
- Create: `tools/make_demo_clip.py`
- Test: `tests/test_demo_clip.py`

**Interfaces:**
- Consumes: ffmpeg on PATH.
- Produces: `make_demo_clip(path, seconds=4, fps=25, width=320, height=240) -> Path`
  - a deterministic H.264 MP4 built from ffmpeg's `testsrc2` source with `-bitexact`
  so two invocations produce byte-identical files. Used as the fixture exhibit by
  Phases 2, 3, 4, 5, 7, 9, 10.

- [ ] **Step 1: Write the failing test**

Create `tests/test_demo_clip.py`:

```python
from peri.core.canon import sha256_file
from tools.make_demo_clip import make_demo_clip


def test_clip_is_created_and_non_trivial(tmp_path):
    out = make_demo_clip(tmp_path / "demo.mp4", seconds=2)
    assert out.is_file()
    assert out.stat().st_size > 5_000


def test_two_invocations_are_byte_identical(tmp_path):
    a = make_demo_clip(tmp_path / "a.mp4", seconds=2)
    b = make_demo_clip(tmp_path / "b.mp4", seconds=2)
    assert sha256_file(a) == sha256_file(b)


def test_clip_is_readable_by_opencv(tmp_path):
    import cv2

    out = make_demo_clip(tmp_path / "demo.mp4", seconds=2, fps=25)
    cap = cv2.VideoCapture(str(out))
    try:
        assert cap.isOpened()
        ok, frame = cap.read()
        assert ok and frame.shape[:2] == (240, 320)
    finally:
        cap.release()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_demo_clip.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'tools.make_demo_clip'`

- [ ] **Step 3: Write the implementation**

Create `tools/make_demo_clip.py`:

```python
"""Deterministic fixture exhibit generator.

Every phase needs a video file to examine. Generating one with -bitexact means the
fixture has a stable SHA-256, so tests can assert on hashes without shipping a
binary blob in the repository.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from peri.core.errors import IntakeError


def make_demo_clip(
    path: str | Path,
    seconds: int = 4,
    fps: int = 25,
    width: int = 320,
    height: int = 240,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={width}x{height}:rate={fps}:duration={seconds}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-bitexact",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-movflags",
        "+faststart",
        str(out),
    ]
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not out.is_file():
        raise IntakeError(f"ffmpeg failed to build the demo clip: {completed.stderr.strip()}")
    return out


def main() -> None:
    out = make_demo_clip("evidence/_fixtures/demo.mp4")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_demo_clip.py -v`
Expected: PASS, 3 passed.

If `test_two_invocations_are_byte_identical` fails, the installed libx264 is writing
an encoder version string despite `-bitexact`. Fix by adding `"-metadata", "encoder="`
before the output path; do not weaken the test.

- [ ] **Step 5: Generate the shared fixture and commit**

```bash
.venv/Scripts/python.exe -m tools.make_demo_clip
git add tools/make_demo_clip.py tests/test_demo_clip.py
git commit -m "feat(tools): deterministic demo clip fixture generator"
```

---

### Task 2: Append-only ledger (`peri/core/ledger.py`)

**Files:**
- Create: `peri/core/ledger.py`
- Test: `tests/test_ledger.py`

**Interfaces:**
- Consumes: `peri.core.canon.{hash_obj, canonical_json, utc_now_iso}`.
- Produces:
  - `GENESIS_HASH: str` - 64 zeros
  - `class LedgerEvent` - frozen dataclass:
    `seq: int`, `ts_utc: str`, `event: str`, `evidence_id: str`, `payload: dict`,
    `prev_hash: str`, `hash: str`; plus `to_dict()`
  - `compute_event_hash(seq, ts_utc, event, evidence_id, payload, prev_hash) -> str`
  - `class Ledger` - constructor `Ledger(path: str | Path)`:
    - `append(event: str, evidence_id: str, payload: dict) -> LedgerEvent`
    - `events() -> list[LedgerEvent]`
    - `head_hash() -> str`
    - `__len__() -> int`
  - `class LedgerVerification` - frozen dataclass:
    `ok: bool`, `count: int`, `broken_at: int | None`, `reason: str | None`
  - `verify_ledger(path: str | Path) -> LedgerVerification`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ledger.py`:

```python
import json

import pytest

from peri.core.ledger import GENESIS_HASH, Ledger, verify_ledger


def build(tmp_path, n=3):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    for i in range(n):
        ledger.append("TEST_EVENT", "EVD_X", {"i": i})
    return path, ledger


def test_first_event_links_to_genesis(tmp_path):
    _, ledger = build(tmp_path, n=1)
    assert ledger.events()[0].prev_hash == GENESIS_HASH
    assert GENESIS_HASH == "0" * 64


def test_sequence_numbers_start_at_zero_and_increment(tmp_path):
    _, ledger = build(tmp_path, n=4)
    assert [e.seq for e in ledger.events()] == [0, 1, 2, 3]


def test_each_event_links_to_its_predecessor(tmp_path):
    _, ledger = build(tmp_path, n=4)
    events = ledger.events()
    for prev, current in zip(events, events[1:]):
        assert current.prev_hash == prev.hash


def test_head_hash_is_the_last_event_hash(tmp_path):
    _, ledger = build(tmp_path, n=3)
    assert ledger.head_hash() == ledger.events()[-1].hash


def test_head_hash_of_an_empty_ledger_is_genesis(tmp_path):
    ledger = Ledger(tmp_path / "empty.jsonl")
    assert ledger.head_hash() == GENESIS_HASH
    assert len(ledger) == 0


def test_a_fresh_chain_verifies(tmp_path):
    path, _ = build(tmp_path, n=5)
    report = verify_ledger(path)
    assert report.ok is True
    assert report.count == 5
    assert report.broken_at is None


def test_tampering_with_a_payload_breaks_verification(tmp_path):
    path, _ = build(tmp_path, n=5)
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[2])
    record["payload"]["i"] = 999
    lines[2] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = verify_ledger(path)
    assert report.ok is False
    assert report.broken_at == 2
    assert "hash" in (report.reason or "")


def test_deleting_a_line_breaks_the_chain(tmp_path):
    path, _ = build(tmp_path, n=5)
    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[2]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = verify_ledger(path)
    assert report.ok is False
    assert report.broken_at == 2


def test_reopening_a_ledger_continues_the_chain(tmp_path):
    path = tmp_path / "ledger.jsonl"
    first = Ledger(path)
    first.append("A", "EVD_X", {})
    second = Ledger(path)
    event = second.append("B", "EVD_X", {})
    assert event.seq == 1
    assert event.prev_hash == first.events()[0].hash
    assert verify_ledger(path).ok is True


def test_lines_are_canonical_json(tmp_path):
    path, _ = build(tmp_path, n=2)
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert line == json.dumps(record, sort_keys=True, separators=(",", ":"))


def test_payload_must_be_json_serialisable(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    with pytest.raises(TypeError):
        ledger.append("BAD", "EVD_X", {"blob": object()})


def test_verify_of_a_missing_file_is_not_ok(tmp_path):
    report = verify_ledger(tmp_path / "nope.jsonl")
    assert report.ok is False
    assert report.count == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ledger.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'peri.core.ledger'`

- [ ] **Step 3: Write the implementation**

Create `peri/core/ledger.py`:

```python
"""L6: append-only SHA-256 hash chain of every examination event.

Tamper-evidence, not tamper-proofing. Anyone who alters or removes a recorded event
invalidates every hash after it, and verify_ledger() names the first broken link.
The laboratory is the trust anchor; this file is the audit trail it produces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from peri.core.canon import canonical_json, hash_obj, utc_now_iso

GENESIS_HASH = "0" * 64


def compute_event_hash(
    seq: int,
    ts_utc: str,
    event: str,
    evidence_id: str,
    payload: dict,
    prev_hash: str,
) -> str:
    return hash_obj(
        {
            "seq": seq,
            "ts_utc": ts_utc,
            "event": event,
            "evidence_id": evidence_id,
            "payload": payload,
            "prev_hash": prev_hash,
        }
    )


@dataclass(frozen=True)
class LedgerEvent:
    seq: int
    ts_utc: str
    event: str
    evidence_id: str
    payload: dict
    prev_hash: str
    hash: str

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "ts_utc": self.ts_utc,
            "event": self.event,
            "evidence_id": self.evidence_id,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LedgerEvent":
        return cls(
            seq=int(data["seq"]),
            ts_utc=str(data["ts_utc"]),
            event=str(data["event"]),
            evidence_id=str(data["evidence_id"]),
            payload=dict(data["payload"]),
            prev_hash=str(data["prev_hash"]),
            hash=str(data["hash"]),
        )


@dataclass(frozen=True)
class LedgerVerification:
    ok: bool
    count: int
    broken_at: int | None
    reason: str | None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "count": self.count,
            "broken_at": self.broken_at,
            "reason": self.reason,
        }


class Ledger:
    """Append-only chain backed by a JSONL file. Opened for append, never rewritten."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def events(self) -> list[LedgerEvent]:
        if not self.path.is_file():
            return []
        out: list[LedgerEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(LedgerEvent.from_dict(json.loads(line)))
        return out

    def __len__(self) -> int:
        return len(self.events())

    def head_hash(self) -> str:
        events = self.events()
        return events[-1].hash if events else GENESIS_HASH

    def append(self, event: str, evidence_id: str, payload: dict) -> LedgerEvent:
        existing = self.events()
        seq = len(existing)
        prev_hash = existing[-1].hash if existing else GENESIS_HASH
        ts_utc = utc_now_iso()
        # canonical_json raises TypeError on anything unserialisable, before we
        # write. A half-written ledger line is worse than a refused append.
        canonical_json(payload)
        digest = compute_event_hash(seq, ts_utc, event, evidence_id, payload, prev_hash)
        record = LedgerEvent(seq, ts_utc, event, evidence_id, payload, prev_hash, digest)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(record.to_dict()) + "\n")
        return record


def verify_ledger(path: str | Path) -> LedgerVerification:
    """Recompute every link. Returns the index of the first break, if any."""
    target = Path(path)
    if not target.is_file():
        return LedgerVerification(False, 0, None, "ledger file not found")

    events = Ledger(target).events()
    expected_prev = GENESIS_HASH
    for index, event in enumerate(events):
        if event.seq != index:
            return LedgerVerification(False, index, len(events), None) and LedgerVerification(
                False, index, len(events), f"sequence number {event.seq} at position {index}"
            )
        if event.prev_hash != expected_prev:
            return LedgerVerification(
                False, len(events), index, f"prev_hash mismatch at position {index}"
            )
        recomputed = compute_event_hash(
            event.seq, event.ts_utc, event.event, event.evidence_id, event.payload,
            event.prev_hash,
        )
        if recomputed != event.hash:
            return LedgerVerification(
                False, len(events), index, f"hash mismatch at position {index}"
            )
        expected_prev = event.hash

    return LedgerVerification(True, len(events), None, None)
```

**Implementer note:** the `verify_ledger` return signature is
`LedgerVerification(ok, count, broken_at, reason)`. Write every return with keyword
arguments to avoid transposing `count` and `broken_at` - the tests check both.
Rewrite the three failure returns above as:

```python
        if event.seq != index:
            return LedgerVerification(
                ok=False, count=len(events), broken_at=index,
                reason=f"sequence number {event.seq} at position {index}",
            )
        if event.prev_hash != expected_prev:
            return LedgerVerification(
                ok=False, count=len(events), broken_at=index,
                reason=f"prev_hash mismatch at position {index}",
            )
        ...
        if recomputed != event.hash:
            return LedgerVerification(
                ok=False, count=len(events), broken_at=index,
                reason=f"hash mismatch at position {index}",
            )
```

and the success return as
`LedgerVerification(ok=True, count=len(events), broken_at=None, reason=None)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ledger.py -v`
Expected: PASS, 12 passed.

- [ ] **Step 5: Commit**

```bash
git add peri/core/ledger.py tests/test_ledger.py
git commit -m "feat(ledger): append-only SHA-256 hash chain with offline verification"
```

---

### Task 3: Intake (`peri/core/intake.py`)

**Files:**
- Create: `peri/core/intake.py`
- Test: `tests/test_intake.py`

**Interfaces:**
- Consumes: `canon`, `errors.IntakeError`, `ledger.Ledger`.
- Produces:
  - `INTAKE_EVENTS: tuple[str, ...]` - the six event names in order
  - `class ExhibitRecord` - frozen dataclass:
    `evidence_id: str`, `evidence_dir: Path`, `original_path: Path`,
    `working_path: Path`, `original_sha256: str`, `working_sha256: str`,
    `original_filename: str`, `size_bytes: int`, `container: dict`,
    `read_only: bool`, `ledger_path: Path`; plus `to_dict()`
  - `probe_container(path) -> dict` - ffprobe JSON reduced to
    `{format_name, format_long_name, duration_s, bit_rate, size_bytes,
      video: {codec, width, height, fps, nb_frames, pix_fmt}, audio: {...} | None,
      tags: {...}, stream_count}`
  - `make_read_only(path) -> bool`
  - `new_evidence_id(original_sha256, now=None) -> str`
  - `intake(src_path, evidence_root="evidence", original_filename=None) -> ExhibitRecord`

- [ ] **Step 1: Write the failing test**

Create `tests/test_intake.py`:

```python
import os
import stat

import pytest

from peri.core.errors import IntakeError
from peri.core.intake import INTAKE_EVENTS, intake, probe_container
from peri.core.ledger import verify_ledger
from tools.make_demo_clip import make_demo_clip


@pytest.fixture()
def clip(tmp_path):
    return make_demo_clip(tmp_path / "src" / "exhibit.mp4", seconds=2)


def test_intake_returns_an_evidence_id_with_the_expected_shape(tmp_path, clip):
    record = intake(clip, evidence_root=tmp_path / "evidence")
    assert record.evidence_id.startswith("EVD_")
    parts = record.evidence_id.split("_")
    assert len(parts) == 3
    assert parts[2] == record.original_sha256[:8]


def test_original_is_quarantined_and_hashed(tmp_path, clip):
    from peri.core.canon import sha256_file

    record = intake(clip, evidence_root=tmp_path / "evidence")
    assert record.original_path.is_file()
    assert record.original_sha256 == sha256_file(clip)


def test_original_is_read_only(tmp_path, clip):
    record = intake(clip, evidence_root=tmp_path / "evidence")
    assert record.read_only is True
    assert not os.access(record.original_path, os.W_OK)
    mode = stat.S_IMODE(record.original_path.stat().st_mode)
    assert not (mode & stat.S_IWRITE)


def test_writing_to_the_original_raises(tmp_path, clip):
    record = intake(clip, evidence_root=tmp_path / "evidence")
    with pytest.raises(PermissionError):
        with open(record.original_path, "ab") as handle:
            handle.write(b"x")


def test_working_copy_is_byte_identical_to_the_original(tmp_path, clip):
    record = intake(clip, evidence_root=tmp_path / "evidence")
    assert record.working_path.is_file()
    assert record.working_sha256 == record.original_sha256
    assert record.working_path.read_bytes() == record.original_path.read_bytes()


def test_working_copy_is_writable(tmp_path, clip):
    record = intake(clip, evidence_root=tmp_path / "evidence")
    assert os.access(record.working_path, os.W_OK)


def test_exactly_six_chained_ledger_events_are_recorded(tmp_path, clip):
    record = intake(clip, evidence_root=tmp_path / "evidence")
    from peri.core.ledger import Ledger

    events = Ledger(record.ledger_path).events()
    assert [e.event for e in events] == list(INTAKE_EVENTS)
    assert len(events) == 6


def test_the_intake_ledger_verifies(tmp_path, clip):
    record = intake(clip, evidence_root=tmp_path / "evidence")
    report = verify_ledger(record.ledger_path)
    assert report.ok is True
    assert report.count == 6


def test_container_probe_reports_codec_fps_and_resolution(tmp_path, clip):
    probe = probe_container(clip)
    assert probe["video"]["codec"] == "h264"
    assert probe["video"]["width"] == 320
    assert probe["video"]["height"] == 240
    assert probe["video"]["fps"] == pytest.approx(25.0, abs=0.01)
    assert probe["duration_s"] == pytest.approx(2.0, abs=0.2)


def test_probe_of_a_non_media_file_raises_intake_error(tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("this is not a video", encoding="utf-8")
    with pytest.raises(IntakeError):
        probe_container(junk)


def test_intake_of_a_missing_file_raises_intake_error(tmp_path):
    with pytest.raises(IntakeError):
        intake(tmp_path / "absent.mp4", evidence_root=tmp_path / "evidence")


def test_two_intakes_of_the_same_bytes_share_the_id_suffix(tmp_path, clip):
    a = intake(clip, evidence_root=tmp_path / "evidence")
    b = intake(clip, evidence_root=tmp_path / "evidence")
    assert a.evidence_id.split("_")[2] == b.evidence_id.split("_")[2]
    assert a.evidence_dir != b.evidence_dir or a.evidence_id == b.evidence_id


def test_record_dict_carries_both_hashes_for_the_report(tmp_path, clip):
    payload = intake(clip, evidence_root=tmp_path / "evidence").to_dict()
    assert len(payload["original_sha256"]) == 64
    assert len(payload["working_sha256"]) == 64
    assert payload["container"]["video"]["codec"] == "h264"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_intake.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'peri.core.intake'`

- [ ] **Step 3: Write the implementation**

Create `peri/core/intake.py`:

```python
"""L0: receive, hash, quarantine, probe, and copy an exhibit.

Custody rule enforced here and stated on the report's chain-of-custody page: the
original is opened for reading exactly once, to hash it, and is then marked
read-only. Every subsequent stage operates on the working copy, which is a
bit-identical duplicate. No stage in this system ever opens the original for
writing.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from peri.core.canon import q, sha256_file
from peri.core.errors import IntakeError
from peri.core.ledger import Ledger

INTAKE_EVENTS = (
    "INTAKE_RECEIVED",
    "ORIGINAL_HASHED",
    "QUARANTINE_SEALED",
    "CONTAINER_PROBED",
    "WORKING_COPY_CREATED",
    "WORKING_COPY_HASHED",
)


def _parse_fraction(text: str | None) -> float:
    if not text or "/" not in str(text):
        try:
            return float(text) if text else 0.0
        except (TypeError, ValueError):
            return 0.0
    num, _, den = str(text).partition("/")
    try:
        denominator = float(den)
        return float(num) / denominator if denominator else 0.0
    except ValueError:
        return 0.0


def probe_container(path: str | Path) -> dict:
    """Structured container facts from ffprobe. Raises IntakeError if unreadable."""
    target = Path(path)
    if not target.is_file():
        raise IntakeError(f"exhibit not found: {target}")

    argv = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(target),
    ]
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not completed.stdout.strip():
        raise IntakeError(f"ffprobe could not read the exhibit: {target.name}")
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise IntakeError(f"ffprobe returned malformed JSON for {target.name}") from exc

    fmt = raw.get("format") or {}
    streams = raw.get("streams") or []
    if not streams:
        raise IntakeError(f"exhibit contains no decodable streams: {target.name}")

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise IntakeError(f"exhibit contains no video stream: {target.name}")

    return {
        "format_name": fmt.get("format_name", "unknown"),
        "format_long_name": fmt.get("format_long_name", "unknown"),
        "duration_s": q(float(fmt.get("duration", 0.0) or 0.0)),
        "bit_rate": int(fmt.get("bit_rate", 0) or 0),
        "size_bytes": int(fmt.get("size", 0) or 0),
        "stream_count": len(streams),
        "video": {
            "codec": video.get("codec_name", "unknown"),
            "profile": str(video.get("profile", "unknown")),
            "width": int(video.get("width", 0) or 0),
            "height": int(video.get("height", 0) or 0),
            "fps": q(_parse_fraction(video.get("avg_frame_rate"))),
            "nb_frames": int(video.get("nb_frames", 0) or 0),
            "pix_fmt": video.get("pix_fmt", "unknown"),
        },
        "audio": (
            {
                "codec": audio.get("codec_name", "unknown"),
                "sample_rate": int(audio.get("sample_rate", 0) or 0),
                "channels": int(audio.get("channels", 0) or 0),
            }
            if audio
            else None
        ),
        "tags": {str(k): str(v) for k, v in (fmt.get("tags") or {}).items()},
    }


def make_read_only(path: str | Path) -> bool:
    """Mark a file read-only and confirm it. Returns the confirmed state."""
    target = Path(path)
    os.chmod(target, stat.S_IREAD)
    return not os.access(target, os.W_OK)


def new_evidence_id(original_sha256: str, now: datetime | None = None) -> str:
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"EVD_{moment.strftime('%Y%m%dT%H%M%SZ')}_{original_sha256[:8]}"


@dataclass(frozen=True)
class ExhibitRecord:
    evidence_id: str
    evidence_dir: Path
    original_path: Path
    working_path: Path
    original_sha256: str
    working_sha256: str
    original_filename: str
    size_bytes: int
    container: dict
    read_only: bool
    ledger_path: Path

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "original_filename": self.original_filename,
            "size_bytes": self.size_bytes,
            "original_sha256": self.original_sha256,
            "working_sha256": self.working_sha256,
            "working_copy_method": "byte-identical-copy",
            "original_read_only": self.read_only,
            "container": self.container,
        }


def intake(
    src_path: str | Path,
    evidence_root: str | Path = "evidence",
    original_filename: str | None = None,
) -> ExhibitRecord:
    """Perform the six custody acts, recording each one in the ledger."""
    src = Path(src_path)
    if not src.is_file():
        raise IntakeError(f"exhibit not found: {src}")

    filename = original_filename or src.name
    size_bytes = src.stat().st_size
    if size_bytes == 0:
        raise IntakeError(f"exhibit is empty: {filename}")

    original_sha256 = sha256_file(src)
    evidence_id = new_evidence_id(original_sha256)
    evidence_dir = Path(evidence_root) / evidence_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(evidence_dir / "ledger.jsonl")

    ledger.append(
        INTAKE_EVENTS[0],
        evidence_id,
        {"original_filename": filename, "size_bytes": size_bytes, "source": "upload"},
    )
    ledger.append(
        INTAKE_EVENTS[1],
        evidence_id,
        {"algorithm": "SHA-256", "sha256": original_sha256},
    )

    original_path = evidence_dir / "original.ro"
    shutil.copy2(src, original_path)
    read_only = make_read_only(original_path)
    ledger.append(
        INTAKE_EVENTS[2],
        evidence_id,
        {
            "quarantine_filename": original_path.name,
            "read_only": read_only,
            "mode_octal": oct(stat.S_IMODE(original_path.stat().st_mode)),
            "statement": "the original was never opened for writing",
        },
    )

    container = probe_container(original_path)
    ledger.append(
        INTAKE_EVENTS[3],
        evidence_id,
        {
            "tool": "ffprobe",
            "format_name": container["format_name"],
            "duration_s": container["duration_s"],
            "video_codec": container["video"]["codec"],
            "width": container["video"]["width"],
            "height": container["video"]["height"],
            "fps": container["video"]["fps"],
            "tag_count": len(container["tags"]),
        },
    )

    suffix = Path(filename).suffix.lower() or ".mp4"
    working_path = evidence_dir / f"working{suffix}"
    shutil.copy2(original_path, working_path)
    os.chmod(working_path, stat.S_IREAD | stat.S_IWRITE)
    ledger.append(
        INTAKE_EVENTS[4],
        evidence_id,
        {"working_filename": working_path.name, "method": "byte-identical-copy"},
    )

    working_sha256 = sha256_file(working_path)
    ledger.append(
        INTAKE_EVENTS[5],
        evidence_id,
        {
            "algorithm": "SHA-256",
            "sha256": working_sha256,
            "matches_original": working_sha256 == original_sha256,
        },
    )
    if working_sha256 != original_sha256:
        raise IntakeError(
            "working copy hash does not match the original; intake aborted"
        )

    return ExhibitRecord(
        evidence_id=evidence_id,
        evidence_dir=evidence_dir,
        original_path=original_path,
        working_path=working_path,
        original_sha256=original_sha256,
        working_sha256=working_sha256,
        original_filename=filename,
        size_bytes=size_bytes,
        container=container,
        read_only=read_only,
        ledger_path=evidence_dir / "ledger.jsonl",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_intake.py -v`
Expected: PASS, 13 passed.

Known Windows wrinkle: `test_writing_to_the_original_raises` fails if the test process
runs elevated, because Administrators bypass the read-only attribute. Run the suite
from a non-elevated shell. If elevation is unavoidable, additionally deny write via
`icacls <path> /deny "%USERNAME%":(W)` inside `make_read_only` - but prefer the
non-elevated shell; the extra ACL call is one more thing to explain on stage.

- [ ] **Step 5: Commit**

```bash
git add peri/core/intake.py tests/test_intake.py
git commit -m "feat(intake): hash, quarantine, probe, and copy with six ledger events"
```

---

### Task 4: Examination manifest (`peri/core/manifest.py`)

**Files:**
- Create: `peri/core/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `canon`, `intake.ExhibitRecord`.
- Produces:
  - `MANIFEST_SCHEMA: str = "peri.manifest/1"`
  - `artifact_checksums(artifacts_dir="artifacts") -> dict[str, str]` - sorted mapping
    of every `*.pt` and `*.json` filename to its SHA-256, `{}` if the directory is
    absent
  - `build_manifest(record: ExhibitRecord, config: dict, artifacts_dir="artifacts")
     -> dict` - keys: `schema`, `evidence_id`, `exhibit`, `config`, `seed`,
    `artifacts`, `environment`, `manifest_hash`
  - `write_manifest(manifest: dict, evidence_dir: Path) -> Path` - writes
    `manifest.json`, returns the path

- [ ] **Step 1: Write the failing test**

Create `tests/test_manifest.py`:

```python
import json

import pytest

from peri.core.intake import intake
from peri.core.manifest import (
    MANIFEST_SCHEMA,
    artifact_checksums,
    build_manifest,
    write_manifest,
)
from tools.make_demo_clip import make_demo_clip


@pytest.fixture()
def record(tmp_path):
    clip = make_demo_clip(tmp_path / "src" / "exhibit.mp4", seconds=2)
    return intake(clip, evidence_root=tmp_path / "evidence")


def test_manifest_has_the_required_keys(record):
    manifest = build_manifest(record, {"mode": "full"})
    for key in (
        "schema", "evidence_id", "exhibit", "config", "seed", "artifacts",
        "environment", "manifest_hash",
    ):
        assert key in manifest
    assert manifest["schema"] == MANIFEST_SCHEMA


def test_manifest_pins_the_exhibit_hash(record):
    manifest = build_manifest(record, {})
    assert manifest["exhibit"]["original_sha256"] == record.original_sha256


def test_manifest_records_the_global_seed(record):
    from peri.core.canon import PERI_SEED

    assert build_manifest(record, {})["seed"] == PERI_SEED


def test_manifest_hash_is_stable_for_identical_inputs(record):
    a = build_manifest(record, {"mode": "full"})
    b = build_manifest(record, {"mode": "full"})
    assert a["manifest_hash"] == b["manifest_hash"]


def test_manifest_hash_changes_when_config_changes(record):
    a = build_manifest(record, {"mode": "full"})
    b = build_manifest(record, {"mode": "fast"})
    assert a["manifest_hash"] != b["manifest_hash"]


def test_artifact_checksums_lists_pt_and_json_files(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"weights")
    (tmp_path / "calibration.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    sums = artifact_checksums(tmp_path)
    assert set(sums) == {"model.pt", "calibration.json"}
    assert all(len(v) == 64 for v in sums.values())


def test_artifact_checksums_of_a_missing_directory_is_empty(tmp_path):
    assert artifact_checksums(tmp_path / "absent") == {}


def test_manifest_is_written_and_reloadable(record):
    manifest = build_manifest(record, {"mode": "full"})
    path = write_manifest(manifest, record.evidence_dir)
    assert path.name == "manifest.json"
    assert json.loads(path.read_text(encoding="utf-8"))["manifest_hash"] == manifest["manifest_hash"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_manifest.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'peri.core.manifest'`

- [ ] **Step 3: Write the implementation**

Create `peri/core/manifest.py`:

```python
"""The examination manifest: everything a replay would have to reproduce.

If any value in this manifest differs between two runs, the two runs are not the
same examination and their findings hashes are not required to match. That is the
whole contract of L8.
"""

from __future__ import annotations

import json
from pathlib import Path

from peri.core.canon import PERI_SEED, hash_obj, sha256_file
from peri.core.intake import ExhibitRecord

MANIFEST_SCHEMA = "peri.manifest/1"


def artifact_checksums(artifacts_dir: str | Path = "artifacts") -> dict[str, str]:
    directory = Path(artifacts_dir)
    if not directory.is_dir():
        return {}
    out: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix in (".pt", ".json"):
            out[path.name] = sha256_file(path)
    return out


def _environment_record(artifacts_dir: str | Path) -> dict:
    path = Path(artifacts_dir) / "environment.json"
    if not path.is_file():
        return {"available": False}
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"available": False}
    return {
        "available": True,
        "python": record.get("python", {}),
        "packages": record.get("packages", {}),
        "binaries": record.get("binaries", {}),
        "record_hash": record.get("record_hash", ""),
    }


def build_manifest(
    record: ExhibitRecord,
    config: dict,
    artifacts_dir: str | Path = "artifacts",
) -> dict:
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "evidence_id": record.evidence_id,
        "exhibit": {
            "original_filename": record.original_filename,
            "size_bytes": record.size_bytes,
            "original_sha256": record.original_sha256,
            "working_sha256": record.working_sha256,
            "container": record.container,
        },
        "config": dict(config),
        "seed": PERI_SEED,
        "artifacts": artifact_checksums(artifacts_dir),
        "environment": _environment_record(artifacts_dir),
    }
    manifest["manifest_hash"] = hash_obj(manifest)
    return manifest


def write_manifest(manifest: dict, evidence_dir: str | Path) -> Path:
    path = Path(evidence_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_manifest.py -v`
Expected: PASS, 8 passed.

- [ ] **Step 5: Commit**

```bash
git add peri/core/manifest.py tests/test_manifest.py
git commit -m "feat(manifest): examination manifest with artifact checksums"
```

---

### Task 5: Custody CLI and the phase acceptance script

**Files:**
- Create: `tools/custody_demo.py`
- Test: `tests/test_custody_acceptance.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `run_custody_demo(src_path, evidence_root="evidence") -> dict` returning
  `{"evidence_id", "original_sha256", "read_only", "ledger_events",
    "ledger_ok", "manifest_hash"}`, and a `__main__` printing one line per ledger
  event plus the verification verdict.

- [ ] **Step 1: Write the failing test**

Create `tests/test_custody_acceptance.py`:

```python
from tools.custody_demo import run_custody_demo
from tools.make_demo_clip import make_demo_clip


def test_end_to_end_custody_meets_the_phase_two_criteria(tmp_path):
    clip = make_demo_clip(tmp_path / "src" / "exhibit.mp4", seconds=2)
    summary = run_custody_demo(clip, evidence_root=tmp_path / "evidence")

    assert summary["evidence_id"].startswith("EVD_")
    assert len(summary["original_sha256"]) == 64
    assert summary["read_only"] is True
    assert summary["ledger_events"] == 6
    assert summary["ledger_ok"] is True
    assert len(summary["manifest_hash"]) == 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_custody_acceptance.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'tools.custody_demo'`

- [ ] **Step 3: Write the implementation**

Create `tools/custody_demo.py`:

```python
"""Phase 2 acceptance driver: run intake on a file and print the custody record."""

from __future__ import annotations

import sys
from pathlib import Path

from peri.core.intake import intake
from peri.core.ledger import Ledger, verify_ledger
from peri.core.manifest import build_manifest, write_manifest


def run_custody_demo(src_path: str | Path, evidence_root: str | Path = "evidence") -> dict:
    record = intake(src_path, evidence_root=evidence_root)
    manifest = build_manifest(record, {"mode": "custody-demo"})
    write_manifest(manifest, record.evidence_dir)
    report = verify_ledger(record.ledger_path)
    return {
        "evidence_id": record.evidence_id,
        "evidence_dir": str(record.evidence_dir),
        "original_sha256": record.original_sha256,
        "working_sha256": record.working_sha256,
        "read_only": record.read_only,
        "ledger_events": report.count,
        "ledger_ok": report.ok,
        "manifest_hash": manifest["manifest_hash"],
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m tools.custody_demo <path-to-exhibit>")
        return 2
    summary = run_custody_demo(sys.argv[1])
    print(f"evidence_id     {summary['evidence_id']}")
    print(f"original SHA256 {summary['original_sha256']}")
    print(f"working  SHA256 {summary['working_sha256']}")
    print(f"read-only       {summary['read_only']}")
    print(f"manifest hash   {summary['manifest_hash']}")
    print("-- ledger --")
    for event in Ledger(Path(summary["evidence_dir"]) / "ledger.jsonl").events():
        print(f"  {event.seq}  {event.ts_utc}  {event.event:22s} {event.hash[:16]}")
    print(f"ledger verified: {summary['ledger_ok']} ({summary['ledger_events']} events)")
    return 0 if (summary["ledger_ok"] and summary["read_only"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_custody_acceptance.py -v`
Expected: PASS, 1 passed.

- [ ] **Step 5: Run the acceptance driver by hand**

```bash
.venv/Scripts/python.exe -m tools.make_demo_clip
.venv/Scripts/python.exe -m tools.custody_demo evidence/_fixtures/demo.mp4; echo "exit=$?"
```

Expected: an evidence ID, two identical 64-hex hashes, `read-only True`, six ledger
lines in the order `INTAKE_RECEIVED → ORIGINAL_HASHED → QUARANTINE_SEALED →
CONTAINER_PROBED → WORKING_COPY_CREATED → WORKING_COPY_HASHED`, and
`ledger verified: True (6 events)`, exit 0.

- [ ] **Step 6: Commit**

```bash
git add tools/custody_demo.py tests/test_custody_acceptance.py
git commit -m "feat(tools): custody acceptance driver"
```

---

## Phase 2 acceptance test

```bash
.venv/Scripts/python.exe -m pytest tests/test_demo_clip.py tests/test_ledger.py tests/test_intake.py tests/test_manifest.py tests/test_custody_acceptance.py -q
.venv/Scripts/python.exe -m tools.custody_demo evidence/_fixtures/demo.mp4; echo "exit=$?"
```

**Pass criteria, all six:**
1. 37 tests pass, 0 fail.
2. `custody_demo` exits 0 and prints six ledger events in the specified order.
3. The original at `evidence/<EVD_ID>/original.ro` is not writable
   (`python -c "import os;print(os.access(r'evidence/<EVD_ID>/original.ro', os.W_OK))"`
   prints `False`).
4. `verify_ledger` returns `ok=True, count=6`.
5. Hand-editing one byte of any payload in `ledger.jsonl` and re-running
   `verify_ledger` returns `ok=False` with the correct `broken_at`.
6. `manifest.json` exists in the evidence directory with a 64-hex `manifest_hash`.

**Phase 2 is green when all six hold.** Phase 3 (provenance) and Phase 5 (model layer)
may then start.
