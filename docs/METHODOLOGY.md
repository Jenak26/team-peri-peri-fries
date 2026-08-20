# 🔬 Methodology and Honest Limitations

**Why the corpus is built the way it is, what the calibration split can and cannot
support at its current size, how determinism is achieved, and every place this build
deviates from its own specification.**

This page exists because the report's Methods section has to state what was
*actually* run, not what was planned. Everything here is written to be read by
someone looking for the weak point.

---

## Corpus: PPF-ICV-1

`PPF-ICV-1` is an **internal validation corpus, not a public benchmark.** It is built
by compositing a donor region into an authentic source clip using four documented
splice methods, each producing an exact ground-truth mask. Authentic and manipulated
samples pass through an identical write path, so encoding history is not a class cue.

It is not FF++, and its numbers are never compared against published FF++ numbers.

| Property | Value |
|---|---|
| Corpus ID | `PPF-ICV-1` |
| Index hash | `b07d65079449416067387e20b9b400f6da3ad3047bf93af8d21fedc4c2a5fa6b` |
| Source clips | 15 |
| Samples | 64 |
| Frames per clip | 24 |
| Splice methods | `alpha_ellipse` · `warp_affine` · `color_matched` · `poisson` |
| Held-out method | `poisson` |
| Build seed | `20260820` |

---

## Splits are by identity AND by generator, never random

- **train / val** — the three non-held-out splice methods only.
- **cal / test** — all four methods, so the held-out generator is represented.

The `cal` split is sacred. It is never trained on, and exists solely to fit the
likelihood-ratio densities and the Mahalanobis in-domain statistics. A model that has
seen its own calibration data reports a likelihood ratio that means nothing.

## How much source video you need

Splits are assigned by identity, so the number of source clips sets everything:

| source clips | train | val | cal | test |
|---:|---:|---:|---:|---:|
| 15 | 9 | 2 | 2 | 2 |
| 30 | 18 | 4 | 4 | 4 |
| 60 | 36 | 9 | 9 | 6 |
| 100 | 60 | 15 | 15 | 10 |

The `cal` split is the binding constraint. It is never trained on and exists solely
to fit the likelihood-ratio densities and the Mahalanobis gate. `forensic_lr` fits
Gaussian KDE densities only when it has at least **15 samples per class** in `cal`,
and falls back to a ridge-penalised logistic fit below that. The number of authentic
cal samples equals the number of cal identities, so:

- **fewer than ~100 source clips**: the logistic fallback is used. It works and is
  honest, but it is a two-parameter model standing in for a density.
- **~100 or more**: KDE densities, which is what the design assumes.

The corpus currently shipped (`index_hash b07d6507...`) is built from 15 clips, so it
uses the logistic fallback. Adding source video is the highest-leverage improvement
available, and it requires rebuilding and re-transferring the corpus.

Stage A has a floor of its own. Its contrastive objective draws negatives from the
other clips in the batch, so it needs at least two authentic clips in `train` and
warns when the clip count is below the batch size, because most in-batch pairs are
then masked out as same-clip and the effective negative count collapses.

## The held-out generator

The corpus is built with four splice methods. The `poisson` method is kept out of
`train` and `val` entirely. Stage B reports AUROC on it separately, under
`AUROC[poisson]`. That number is the generalisation claim: how the decoder performs
on a manipulation family it was never shown.

## Determinism

The replay guarantee requires that a second run produces a byte-identical findings
hash. Concretely:

- All floats are quantised and JSON is sorted at hash time, in `peri/core/canon.py`
  and nowhere else.
- Seeds derive from `canon.stable_seed()`, which digests its inputs. Python's
  built-in `hash()` is salted per process and must not be used for anything that
  reaches a seed or a hash.
- `train.build_corpus` produces byte-identical frames across separate processes for
  the same source video and seed. Verified by rebuilding and comparing digests.
- The corpus contains no absolute paths and can be copied between machines freely.

## Known deviations from the CLAUDE.md specification

Recorded here rather than quietly ignored, because the report's Methods page has to
state what was actually run.

- **Stage A parameter count.** `CLAUDE.md` section 4 asks for ~18M trainable
  parameters and also for "17 layers, 64ch". Those two are inconsistent with each
  other: a 17-layer DnCNN at 64 channels is about 0.55M parameters. The shipped
  config uses 17 layers at 96 channels, which is 1.25M. The architecture matches the
  stated shape; the parameter count does not match the stated total.
- **Interpreter version.** `artifacts/environment.json` records the interpreter that
  actually produced it and derives its own deviation text. It is generated, never
  hand-edited.
- **Corpus.** `PPF-ICV-1` is an internal validation corpus built by compositing donor
  regions into authentic clips. It is not FF++ and is never compared against
  published FF++ numbers.

