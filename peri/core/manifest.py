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


def read_manifest(evidence_dir: str | Path) -> dict:
    path = Path(evidence_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"no manifest in {evidence_dir}")
    return json.loads(path.read_text(encoding="utf-8"))
