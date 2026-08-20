"""Write artifacts/SHA256SUMS over every trained checkpoint.

Checkpoints are produced on the training workstation and carried to the
examination workstation by hand. The report's reproducibility page cites a
SHA-256 per model, and the examination manifest records it, so the digest has to
travel with the weights rather than being recomputed on arrival and trusted.

The output format is the one `sha256sum -c` expects, so the receiving machine can
verify without this repository:

    python -m tools.checksum_artifacts          # write artifacts/SHA256SUMS
    python -m tools.checksum_artifacts --check  # verify against it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from peri.core.canon import sha256_file

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
SUMS_PATH = ARTIFACTS_DIR / "SHA256SUMS"
TRACKED_SUFFIXES = (".pt", ".onnx", ".json")


def tracked_files(root: Path) -> list[Path]:
    """Every artifact worth pinning, in a stable order.

    SHA256SUMS itself is excluded: a file cannot contain its own digest.
    """
    return sorted(
        p
        for p in root.glob("*")
        if p.is_file() and p.suffix in TRACKED_SUFFIXES and p.name != SUMS_PATH.name
    )


def write_sums(root: Path = ARTIFACTS_DIR, out: Path = SUMS_PATH) -> int:
    files = tracked_files(root)
    if not files:
        print(f"no artifacts to checksum under {root}")
        return 1
    lines = [f"{sha256_file(path)}  {path.name}" for path in files]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    print(f"\nwrote {out} ({len(files)} files)")
    return 0


def check_sums(root: Path = ARTIFACTS_DIR, sums: Path = SUMS_PATH) -> int:
    if not sums.is_file():
        print(f"{sums} not found. Run `python -m tools.checksum_artifacts` first.")
        return 1

    failed = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, _, name = line.partition("  ")
        path = root / name
        if not path.is_file():
            print(f"MISSING   {name}")
            failed += 1
        elif sha256_file(path) != expected:
            print(f"MISMATCH  {name}")
            failed += 1
        else:
            print(f"OK        {name}")

    if failed:
        print(f"\n{failed} file(s) did not match. These weights are not the ones recorded.")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Checksum trained artifacts.")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument(
        "--check", action="store_true", help="verify instead of writing"
    )
    args = parser.parse_args()

    sums = args.artifacts / SUMS_PATH.name
    if args.check:
        return check_sums(args.artifacts, sums)
    return write_sums(args.artifacts, sums)


if __name__ == "__main__":
    sys.exit(main())
