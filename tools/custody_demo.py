"""Phase 2 acceptance driver: run intake on a file and print the custody record."""

from __future__ import annotations

import sys
from pathlib import Path

from peri.core.intake import intake
from peri.core.ledger import Ledger, verify_ledger
from peri.core.manifest import build_manifest, write_manifest


def run_custody_demo(
    src_path: str | Path, evidence_root: str | Path = "evidence"
) -> dict:
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
