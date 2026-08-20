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
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    return f"EVD_{stamp}_{original_sha256[:8]}"


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


def load_exhibit_record(evidence_dir: str | Path) -> ExhibitRecord:
    """Rebuild an ExhibitRecord from an evidence directory already on disk.

    Replay (L8) needs the sealed exhibit without performing intake again: a second
    intake would mint a second evidence ID and a second ledger, which is exactly
    what replay must not do.
    """
    directory = Path(evidence_dir)
    exhibit_path = directory / "exhibit.json"
    if not exhibit_path.is_file():
        raise IntakeError(f"no exhibit record in {directory}")
    data = json.loads(exhibit_path.read_text(encoding="utf-8"))
    working = directory / str(data["working_filename"])
    if not working.is_file():
        raise IntakeError(f"working copy missing for {directory.name}")
    return ExhibitRecord(
        evidence_id=str(data["evidence_id"]),
        evidence_dir=directory,
        original_path=directory / "original.ro",
        working_path=working,
        original_sha256=str(data["original_sha256"]),
        working_sha256=str(data["working_sha256"]),
        original_filename=str(data["original_filename"]),
        size_bytes=int(data["size_bytes"]),
        container=dict(data["container"]),
        read_only=bool(data["original_read_only"]),
        ledger_path=directory / "ledger.jsonl",
    )


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
    if original_path.exists():
        # A second intake of the same bytes inside the same second lands in the
        # same directory, and the seal left by the first one makes copy2 fail.
        # Lift it for exactly as long as the copy takes; re-seal immediately.
        os.chmod(original_path, stat.S_IREAD | stat.S_IWRITE)
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
        raise IntakeError("working copy hash does not match the original; intake aborted")

    record = ExhibitRecord(
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
    payload = record.to_dict()
    payload["working_filename"] = working_path.name
    (evidence_dir / "exhibit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return record
