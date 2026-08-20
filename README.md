# Team Peri Peri Fries v2

Judicial digital evidence authentication engine. Not a deepfake detector: a forensic
examination protocol engine that reports a likelihood ratio between two stated
propositions, declares the domain it was calibrated on, abstains when it should, and
seals the whole examination in a replayable hash chain.

Full design context lives in `CLAUDE.md`. This file covers how to run and train it.

- **Hp**: the exhibit is an unmanipulated recording of a real event
- **Hd**: the exhibit is synthetically generated or materially manipulated in the facial region

---

## Repository layout

```
peri/core/      forensic layers: canon, errors, forensic_lr, fragility, videoprint
train/          corpus builder, datasets, augmentation policy, Stage A/B/C training
tools/          environment record, LR self-test, artifact checksums
tests/          pytest suite (76 tests, no GPU required)
artifacts/      checkpoints, calibration.json, environment.json, SHA256SUMS
data/           source video and built corpus (NOT in git, see below)
docs/           build plans
```

---

## Quick start (examination workstation, CPU only)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-cpu.txt

python -m pytest -q                # 76 tests
python -m tools.selftest_lr        # 3 likelihood-ratio acceptance assertions
python -m tools.write_environment  # regenerate artifacts/environment.json
```

`tools.selftest_lr` exercises the likelihood-ratio layer with no trained model in
existence. It must print three `PASS` lines before anything else is worth debugging.

---

## Training on a separate GPU machine

The GPU track is independent of the CPU track. Nothing below needs the API, the
frontend, or any trained model to already exist.

### Step 1: get the code onto the training machine

```bash
git clone https://github.com/Jenak26/team-peri-peri-fries.git
cd team-peri-peri-fries
```

If the training box has no network, copy the working directory across instead, but
**exclude `.venv/`** (it contains CPU-only wheels with absolute paths baked in and
will not work on the other machine).

### Step 2: get the source video across

`data/` is deliberately not in git. It is hundreds of megabytes and fully
regenerable, so committing it would make every clone slow and every diff useless.
You must transfer it yourself:

```bash
# copy your authentic source clips into:
data/authentic/*.mp4        # .mov .avi .mkv .webm .m4v .mpg .mpeg also accepted
```

Only `data/authentic/` needs transferring. `data/corpus/` is rebuilt in step 5.

Filename stems become identities and must be unique, because splits are assigned by
identity. `personA.mp4` and `personA.mov` will be rejected.

### Step 3: install PyTorch with CUDA 12.8

50-series (Blackwell, sm_120) needs CUDA 12.8 wheels or CUDA will not initialise.
Install torch **first**, from its own index, then everything else:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-gpu.txt
```

Confirm the GPU is actually visible before starting a six-hour run:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `torch.cuda.is_available()` prints `False`, stop and fix that. Every training
script silently falls back to CPU, which turns hours into days.

### Step 4: verify the box in five minutes

Run the whole pipeline at toy size first. This catches a broken corpus, a missing
codec, or an out-of-memory config before you commit to the real run.

```bash
python -m pytest -q

python -m train.build_corpus --out /tmp/smoke --frames 6 --limit 8
python -m train.stage_a_videoprint --corpus /tmp/smoke --out /tmp/a.pt \
    --epochs 1 --batch-size 4 --pairs 16 --workers 0
python -m train.stage_b_decoder --corpus /tmp/smoke --out /tmp/b.pt --stage-a /tmp/a.pt \
    --arch unet --epochs 1 --batch-size 2 --crop-size 128 --max-steps 3 --workers 0
python -m train.stage_c_temporal --corpus /tmp/smoke --tokens /tmp/tok.pt \
    --stage-a /tmp/a.pt --stage-b /tmp/b.pt --cache
python -m train.stage_c_temporal --tokens /tmp/tok.pt --out /tmp/c.pt --epochs 2 --batch-size 4
```

On Windows use a real path such as `%TEMP%\smoke` in place of `/tmp/smoke`.

### Step 5: build the corpus

**Do this before any training script.** All three stages read
`data/corpus/index.json` and will refuse to start without it.

```bash
python -m train.build_corpus --frames 24
```

This decodes every clip in `data/authentic/`, writes an authentic sample plus one
manipulated sample per splice method, and prints the split table. It is
deterministic: the same source video and seed give byte-identical frames and the
same `index_hash` on any machine.

The `poisson` splice method is held out of `train` and `val` entirely. That hold-out
is the generalisation claim, and Stage B reports AUROC on it separately.

### Step 6: train the three stages, in order

```bash
# Stage A - acquisition fingerprint, authentic video only, 6-8 h
python -m train.stage_a_videoprint --epochs 30 --batch-size 256

# Stage B - tamper mask + reliability map, 3-4 h
python -m train.stage_b_decoder --epochs 24 --batch-size 12

# Stage C - temporal transformer, ~25 min. Caching is a separate pass.
python -m train.stage_c_temporal --cache
python -m train.stage_c_temporal --epochs 60
```

Notes that matter:

- **Stage B does not need Stage A.** With no Stage A checkpoint it fills the
  fingerprint channels with the fixed SRM acquisition residual and prints
  `fingerprint source: srm-residual`. This is the deliberate fallback from
  `CLAUDE.md` section 3, so Stage B can be trained and shipped while Stage A is
  still running. Re-run Stage B afterwards to hot-swap the learned fingerprint in.
- **Stage C needs `--cache` first.** The cache pass runs Stage B over the whole
  corpus once and writes per-frame tokens. Training without it exits with a message
  telling you so.
- **Stage B falls back to U-Net** if `transformers` cannot fetch `nvidia/mit-b2`
  (no network, no cache). It prints a warning and keeps going. Pass `--arch unet`
  to choose that deliberately.
- Every stage takes `--device`, `--seed`, `--batch-size`, and `--out`. Stage B and
  Stage C take `--max-steps` for short runs.

Checkpoints land in `artifacts/` and carry their own config, seed, corpus id,
held-out method, and epoch history inside the `.pt` file.

### Step 7: checksum and hand back

```bash
python -m tools.checksum_artifacts        # writes artifacts/SHA256SUMS
```

Copy `artifacts/*.pt` **and** `artifacts/SHA256SUMS` to the examination
workstation's `artifacts/` directory, then verify there:

```bash
python -m tools.checksum_artifacts --check
```

Every line must read `OK`. A `MISMATCH` means the weights that arrived are not the
weights that were trained, and the report's reproducibility page would be false.

---

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

Stage A has a floor of its own. Its contrastive objective draws negatives from the
other clips in the batch, so it needs at least two authentic clips in `train` and
warns when the clip count is below the batch size, because most in-batch pairs are
then masked out as same-clip and the effective negative count collapses.

---

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

---

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
- **Corpus.** `PPF-ICV-1` is an internal validation corpus built by compositing
  donor regions into authentic clips. It is not FF++ and is never compared against
  published FF++ numbers.

---

## Legal framing

This system assists forensic examination. It does not replace judicial determination
of admissibility or weight. Section 63(4) of the Bharatiya Sakshya Adhiniyam, 2023
requires a certificate signed by a person in charge of the device and an expert; this
repository generates inputs for that human expert and signs nothing.

Where C2PA provenance and forensic findings disagree, both are reported and the
forensic findings take precedence. C2PA verifies that provenance claims have not been
tampered with, not that they are truthful.
