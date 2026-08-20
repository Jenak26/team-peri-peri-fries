# Security Policy

## Scope

This repository is forensic decision support software. Two classes of issue matter
here, and the second is unusual enough to be worth stating explicitly.

### 1. Ordinary software vulnerabilities

Anything that lets untrusted input compromise the host: path traversal through an
uploaded exhibit filename, deserialisation of an attacker-supplied checkpoint,
command injection into an `ffmpeg`/`ffprobe` invocation, or resource exhaustion via a
crafted container.

### 2. Forensic integrity defects

A defect that causes the engine to report a conclusion it is not entitled to. These
are treated at the same severity as a remote code execution, because in this system
they are the more damaging failure:

- **Chain-of-custody breaks** — anything that writes to the quarantined original, or
  that lets a ledger entry be inserted, reordered, or removed without breaking the
  SHA-256 chain.
- **Replay divergence** — any nondeterminism that causes two runs over the same
  manifest to produce different findings hashes.
- **Calibration leakage** — anything that lets training or validation data reach the
  `cal` split, which would silently invalidate every likelihood ratio the system
  reports.
- **Abstention bypass** — anything that returns a definite verdict where the
  in-domain gate, the stability check, or the reporting threshold should have forced
  `INCONCLUSIVE`.
- **Fragility-search circularity** — any overlap between the training augmentation
  family and the fragility-search transform family, which would make the robustness
  claim self-referential.

## Reporting a vulnerability

Please open a [private security advisory](https://github.com/Jenak26/team-peri-peri-fries/security/advisories/new)
rather than a public issue. Include the exhibit or seed that reproduces it where you
can — this codebase is built so that every failure has a seed.

Expect an acknowledgement within 72 hours.

## Handling exhibits

If you run this on real case material, note that `evidence/{EVD_ID}/` contains the
working copy of the exhibit and its complete examination record. That directory is
git-ignored by default and must stay that way. Do not commit exhibits, and do not
report a vulnerability with a real exhibit attached.
