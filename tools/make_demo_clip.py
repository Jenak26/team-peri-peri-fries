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
        "-metadata",
        "encoder=",
        "-movflags",
        "+faststart",
        str(out),
    ]
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not out.is_file():
        raise IntakeError(
            f"ffmpeg failed to build the demo clip: {completed.stderr.strip()}"
        )
    return out


def main() -> None:
    out = make_demo_clip("evidence/_fixtures/demo.mp4")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
