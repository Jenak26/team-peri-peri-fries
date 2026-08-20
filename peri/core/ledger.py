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
    def from_dict(cls, data: dict) -> LedgerEvent:
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
        return LedgerVerification(
            ok=False, count=0, broken_at=None, reason="ledger file not found"
        )

    events = Ledger(target).events()
    expected_prev = GENESIS_HASH
    for index, event in enumerate(events):
        if event.seq != index:
            return LedgerVerification(
                ok=False,
                count=len(events),
                broken_at=index,
                reason=f"sequence number {event.seq} at position {index}",
            )
        if event.prev_hash != expected_prev:
            return LedgerVerification(
                ok=False,
                count=len(events),
                broken_at=index,
                reason=f"prev_hash mismatch at position {index}",
            )
        recomputed = compute_event_hash(
            event.seq,
            event.ts_utc,
            event.event,
            event.evidence_id,
            event.payload,
            event.prev_hash,
        )
        if recomputed != event.hash:
            return LedgerVerification(
                ok=False,
                count=len(events),
                broken_at=index,
                reason=f"hash mismatch at position {index}",
            )
        expected_prev = event.hash

    return LedgerVerification(ok=True, count=len(events), broken_at=None, reason=None)
