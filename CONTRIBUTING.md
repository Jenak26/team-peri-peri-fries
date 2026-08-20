# Contributing

Contributions are welcome. This project has a few non-negotiable rules that come from
what it is rather than from taste, so they are worth reading before you open a PR.

## The gates

Every change must pass all three locally before it will pass CI:

```bash
python -m pytest -q                 # 88 tests, no GPU required
python -m tools.selftest_lr         # 3 likelihood-ratio acceptance assertions
python -m ruff check .
```

## The rules that are not about style

1. **Determinism is a feature, not an optimisation.** The replay guarantee requires a
   byte-identical findings hash across runs. Float quantisation and JSON key ordering
   live in `peri/core/canon.py` and nowhere else. Never use Python's built-in
   `hash()` anywhere that reaches a seed or a digest — it is salted per process.

2. **Never silently degrade the abstention path, the ledger, the fragility index, or
   the report's limitations page.** Those four *are* the product. A change that makes
   the system more confident is suspect by default.

3. **The `cal` split is sacred.** It is never trained on. If your change gives any
   training or validation code a path to `cal` data, it will be rejected regardless of
   how much it improves a metric.

4. **Training augmentations and fragility-search transforms must stay disjoint.**
   Different families, different parameter ranges. `fragility.assert_transform_disjointness()`
   enforces this in code. If they overlap, the robustness claim becomes circular.

5. **Legal language is CI-enforced.** `tests/test_legal_language.py` fails the build on
   a fixed list of overclaiming phrases. The system *assists* forensic examination; it
   does not determine admissibility or weight. Write to that standard in code
   comments, UI strings, and report text alike.

6. **No LLM-written explanations.** Report prose is templated sentences bound to
   numeric findings. A generated sentence that is not derivable from a number in
   `findings.json` cannot be defended under cross-examination.

## Commit style

Present tense, imperative, describing the effect rather than the file touched:

```
Stop Stage A starting a run that cannot fit in VRAM
Restore Python 3.10 compatibility, broken by a lint autofix
```

## Reporting a security or forensic-integrity issue

See [SECURITY.md](SECURITY.md). Forensic integrity defects are handled at the same
severity as remote code execution.
