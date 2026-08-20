# Team Peri Peri Fries v2

Judicial digital evidence authentication engine. Not a deepfake detector. It is a
forensic examination protocol engine: it reports a likelihood ratio between two
stated propositions, declares the domain it was calibrated on, abstains when the
evidence is too weak or too fragile, and seals the whole examination in a
replayable hash chain.

- **Hp**: the exhibit is an unmanipulated recording of a real event
- **Hd**: the exhibit is synthetically generated or materially manipulated in the facial region

Full design context is in `CLAUDE.md`. This file tells you how to run it.

---

# TRAINING LAPTOP GUIDE

**Read this whole section before typing anything.** It goes from a fresh machine to
finished model files. You do not need to understand the forensics to follow it.

## What you are doing

Three neural networks need training. They must be trained in order, because each one
uses the output of the one before it.

| Stage | What it learns | How long |
|---|---|---|
| **A** Videoprint | What a genuine camera pipeline looks like | 6 to 8 hours |
| **B** Decoder | Which pixels were tampered with | 3 to 4 hours |
| **C** Temporal | Which moments in the video were tampered with | about 30 minutes |

Total: roughly half a day. You can leave it running overnight.

## Before you start, you need

1. A laptop with an **NVIDIA GPU** (RTX 5060 or better, 8 GB VRAM minimum)
2. **Python 3.10 or newer** installed. Check by opening a terminal and typing
   `python --version`. If that fails, try `python3 --version`
3. **Git** installed. Check with `git --version`
4. The **corpus folder** copied from the main laptop. This is the training data.
   It is about 294 MB and is named `corpus`
5. Internet, for downloading PyTorch (about 3 GB)

> **About the corpus:** the training data has already been built on the main laptop.
> You do **not** need any video files, and you do **not** need to build anything.
> Just copy the folder across. If you were not given a corpus folder, jump to
> [If you have no corpus folder](#if-you-have-no-corpus-folder) at the end.

---

## Step 1: Download the code

Open a terminal and run:

```bash
git clone https://github.com/Jenak26/team-peri-peri-fries.git
cd team-peri-peri-fries
```

**Stay in this folder for every remaining step.** If you close the terminal, `cd`
back into it before continuing.

## Step 2: Put the corpus in place

Copy the `corpus` folder you were given into the `data` folder inside the project.

When you are done, this exact file must exist:

```
team-peri-peri-fries/data/corpus/index.json
```

Check it with:

```bash
python -c "import json; print(json.load(open('data/corpus/index.json'))['index_hash'])"
```

You should see:

```
b07d65079449416067387e20b9b400f6da3ad3047bf93af8d21fedc4c2a5fa6b
```

If you see that line, the data arrived intact. If you get a "file not found" error,
the folder is in the wrong place. If the number is different, the copy was corrupted
or the corpus was rebuilt, so ask before continuing.

## Step 3: Create a Python environment

This keeps the project's packages separate from the rest of your machine.

**On Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**On Linux or Mac:**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Your terminal prompt should now start with `(.venv)`. If it does not, the environment
is not active and everything after this will install to the wrong place.

> **Every time you open a new terminal, run the activate line again.** This is the
> single most common thing to forget.

## Step 4: Install PyTorch (the GPU part)

Order matters here. PyTorch must be installed **first**, from its own download
address, or you will get a version that ignores your GPU.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

This downloads about 3 GB. It will take a while.

Then install everything else:

```bash
pip install -r requirements-gpu.txt
```

## Step 5: Check the GPU actually works

**Do not skip this.** If the GPU is not detected, training silently runs on the
processor instead, which turns 8 hours into several days.

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Good result** looks like:

```
True NVIDIA GeForce RTX 5060
```

**Bad result** is `False`, or an error. If you get that, stop and fix it:

- Make sure you ran the `--index-url` command in Step 4 and not a plain `pip install torch`
- Update your NVIDIA graphics driver, then restart the laptop
- Run `nvidia-smi` in a terminal. If that command is not found, the driver is missing

## Step 6: Check the code is healthy

```bash
python -m pytest -q
```

You should see `76 passed`. If anything fails, stop and report it. Do not start
training on a broken checkout.

## Step 7: Do a quick practice run

This runs Stage A and Stage B at tiny size, about a minute each. It catches problems
before you commit to an overnight run. Nothing it produces is kept.

Stage C is not included here, because it is short enough to simply run for real in
Step 10 and its preparation pass reads the whole corpus.

**On Windows (PowerShell):**

```powershell
python -m train.stage_a_videoprint --out "$env:TEMP\a.pt" --epochs 1 --batch-size 4 --pairs 16 --workers 0
python -m train.stage_b_decoder --out "$env:TEMP\b.pt" --stage-a "$env:TEMP\a.pt" --arch unet --epochs 1 --batch-size 2 --crop-size 128 --max-steps 3 --workers 0
```

**On Linux or Mac:**

```bash
python -m train.stage_a_videoprint --out /tmp/a.pt --epochs 1 --batch-size 4 --pairs 16 --workers 0
python -m train.stage_b_decoder --out /tmp/b.pt --stage-a /tmp/a.pt --arch unet --epochs 1 --batch-size 2 --crop-size 128 --max-steps 3 --workers 0
```

Both should finish without a traceback and print a `saved` line. Some numbers will
look poor, and Stage B will print `AUROC nan` and `AUROC[poisson] n/a`. That is
expected at this size and does not mean anything is wrong: three steps of data is not
enough for those numbers to exist.

If both ran, you are ready for the real thing.

## Step 8: Train Stage A

**Pick the command that matches your GPU.** The batch size is set by how much VRAM
you have, and getting it wrong is the most common way this fails.

| Your VRAM | Command |
|---|---|
| **8 GB** | `python -m train.stage_a_videoprint --epochs 30 --batch-size 64` |
| **12 GB** | `python -m train.stage_a_videoprint --epochs 30 --batch-size 128` |
| **16 GB or more** | `python -m train.stage_a_videoprint --epochs 30 --batch-size 256` |

Not sure? The command checks for you. It prints a line like:

```
VRAM: 8.6 GB total, assuming 6.9 GB usable; activations need ~3.2 GB at batch 64
```

and refuses to start if the numbers do not work, telling you which batch size to use
instead. It will not let you begin a run that cannot finish.

It then prints one line per epoch, 30 in total, and saves to `artifacts/` whenever
the loss improves. **Leave it alone for 6 to 8 hours.**

The `train` number should generally drift downwards. It will not fall smoothly, and
that is normal for this kind of training.

> **Does a smaller batch make the model worse?** Barely, with this corpus. Stage A
> learns by comparing patches against patches from *other source clips* in the same
> batch. This corpus has 9 source clips in the training split, so once the batch is
> comfortably larger than that, adding more mostly adds patches from clips already
> represented, which get excluded from the comparison anyway. Batch 256 is specified
> for a corpus of roughly 2000 clips.

## Step 9: Train Stage B

**Always pass `--crop-size 256` with this corpus.** The frames in it are 256x256.
The default of 512 upscales every frame, which adds no detail at all but costs four
times the memory and four times the time. The command warns you if you forget.

| Your VRAM | Command |
|---|---|
| **8 GB** | `python -m train.stage_b_decoder --epochs 24 --batch-size 6 --crop-size 256` |
| **12 GB or more** | `python -m train.stage_b_decoder --epochs 24 --batch-size 12 --crop-size 256` |

Takes 3 to 4 hours. Each line shows `IoU` (how well it finds the tampered pixels,
higher is better) and `ECE` (how honest its confidence is, **lower** is better).

**If you still get an "out of memory" error**, halve the batch again:

```bash
python -m train.stage_b_decoder --epochs 24 --batch-size 3 --crop-size 256
```

**If it prints `[warn] SegFormer unavailable`**, it could not download the pretrained
model, usually because of no internet. It automatically switches to a simpler network
and keeps going. Training is still valid, just weaker. Reconnect and rerun if you can.

> **Tip if you want to save time:** Stage B does not actually need Stage A. If you
> have a second machine or want to start it early, you can run Stage B at the same
> time as Stage A. It will print `fingerprint source: srm-residual`, which means it
> is using a built in stand in. Once Stage A finishes, just run Stage B again to
> upgrade it.

## Step 10: Train Stage C

Two commands. The first prepares data, the second trains. **You must run them in this
order.**

```bash
python -m train.stage_c_temporal --cache
python -m train.stage_c_temporal --epochs 60
```

The first one takes a few minutes and prints `wrote ...tokens_stage_c.pt`. The second
takes about 25 minutes.

If you skip the `--cache` command, the second one exits and tells you so.

## Step 11: Lock in the results

```bash
python -m tools.checksum_artifacts
```

This creates `artifacts/SHA256SUMS`, a fingerprint of every model file. The court
report cites these numbers, so they have to travel with the models.

## Step 12: Send the results back

Copy these files from the `artifacts` folder back to the main laptop's `artifacts`
folder:

- `stage_a_videoprint.pt`
- `stage_b_decoder.pt`
- `stage_c_temporal.pt`
- `SHA256SUMS`

On the main laptop, verify they survived the trip:

```bash
python -m tools.checksum_artifacts --check
```

Every line must say `OK`. If any line says `MISMATCH`, the file was damaged in
transfer. Copy it again.

**You are done.**

---

## If something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| `corpus index not found` | The corpus is missing or in the wrong folder | Redo Step 2 |
| `torch.cuda.is_available()` is `False` | Training would run on the processor | Redo Step 4, update your NVIDIA driver |
| `CUDA out of memory` | Batch too large for your GPU | Lower `--batch-size`, see Steps 8 and 9 |
| `This run needs about N GB ... but only about M GB is usable` | The VRAM check stopped a run that could not finish | Use the batch size it suggests |
| Whole laptop freezes, or Task Manager shows memory exhausted | Too many dataloader workers, each a full process | Add `--workers 2` |
| `[warn] corpus frames are 256x256 but crop_size is 512` | Stage B is upscaling for nothing | Add `--crop-size 256` |
| `tokens not found ... run with --cache first` | Stage C run in the wrong order | Run the `--cache` command first |
| `ModuleNotFoundError` | Environment not active | Run the activate line from Step 3 |
| `cannot import name 'UTC' from 'datetime'` | Code using a Python 3.11 feature on an older interpreter | Fixed. Run `git pull` and try again |
| `SKIPPED ... ffprobe not on PATH` | ffprobe is missing | Harmless on this machine. Training does not use it |
| `[warn] SegFormer unavailable` | No internet for the pretrained model | Harmless, it falls back. Reconnect and rerun if you can |
| `[warn] ... authentic clips vs batch size` | Not much source video | Harmless, training continues |
| Training is extremely slow | Almost certainly running on the processor | Stop it, redo Step 5 |

**Safe to stop and restart?** Yes. Each stage saves its best result as it goes, so
pressing Ctrl+C loses at most the current epoch. Rerunning a stage starts it over
from scratch, it does not resume.

---

## If you have no corpus folder

Only do this if you were **not** given a corpus. It builds the training data from
scratch and needs the original video files.

1. Put the source video into `data/authentic/`. Accepted formats: `.mp4`, `.mov`,
   `.avi`, `.mkv`, `.webm`, `.m4v`, `.mpg`, `.mpeg`
2. Every filename must be different, because filenames are used as identities
3. Run:

```bash
python -m train.build_corpus --frames 24
```

It prints a table of how the data was split. Then continue from Step 3 above.

Note that a corpus built here will have a different fingerprint from one built
elsewhere if the OpenCV versions differ, so prefer copying the corpus when you can.

---

# Reference

## Examination laptop setup (no GPU needed)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-cpu.txt

python -m pytest -q                # 76 tests
python -m tools.selftest_lr        # 3 likelihood-ratio acceptance assertions
python -m tools.write_environment  # regenerate artifacts/environment.json
```

`tools.selftest_lr` exercises the likelihood-ratio layer with no trained model in
existence. It must print three `PASS` lines before anything else is worth debugging.

## Repository layout

```
peri/core/      forensic layers: canon, errors, forensic_lr, fragility, videoprint
train/          corpus builder, datasets, augmentation policy, Stage A/B/C training
tools/          environment record, LR self-test, artifact checksums
tests/          pytest suite (76 tests, no GPU required)
artifacts/      checkpoints, calibration.json, environment.json, SHA256SUMS
data/           source video and built corpus (NOT in git)
docs/           build plans
```

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

## Legal framing

This system assists forensic examination. It does not replace judicial determination
of admissibility or weight. Section 63(4) of the Bharatiya Sakshya Adhiniyam, 2023
requires a certificate signed by a person in charge of the device and an expert; this
repository generates inputs for that human expert and signs nothing.

Where C2PA provenance and forensic findings disagree, both are reported and the
forensic findings take precedence. C2PA verifies that provenance claims have not been
tampered with, not that they are truthful.
