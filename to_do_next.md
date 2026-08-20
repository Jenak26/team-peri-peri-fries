# TO DO NEXT - from trained weights to a demo a judge can read

**Written 2026-08-20. Authoritative build order for everything after training.**
Spec is `CLAUDE.md`; phase schedule is `docs/superpowers/plans/2026-08-20-ppf-00-ROADMAP.md`.
This file does not restate them, it tells you exactly what to type and in what order,
with the real function names and payload keys already present in the repository.

---

## 0. Where we actually are

Verified against the working tree, not against the plan.

### Done and green

| Module | State |
|---|---|
| `peri/core/canon.py` | Determinism spine: `q`, `qdeep`, `canonical_json`, `hash_obj`, `sha256_file`, `stable_seed`, `seed_everything`, `PERI_SEED = 20260820` |
| `peri/core/errors.py` | `PeriError` and its four subclasses |
| `peri/core/forensic_lr.py` | Complete. `fit_stream_calibration`, `log10_lr`, `mahalanobis_distance`, `evaluate_stream`, `fuse_and_decide`, `enfsi_sentence`, all constants, all six reason codes |
| `peri/core/videoprint.py` | `DnCNN`, `srm_residual`, `VideoprintExtractor(checkpoint, device)` with `.extract()`, `.describe()`, `.mode` in `{learned-videoprint, srm-residual}` |
| `peri/core/ledger.py` | `Ledger`, `LedgerEvent`, `verify_ledger`, `GENESIS_HASH` |
| `peri/core/intake.py` | Six custody acts, `ExhibitRecord`, `probe_container`, `load_exhibit_record` |
| `peri/core/manifest.py` | `build_manifest`, `write_manifest`, `read_manifest`, `artifact_checksums` |
| `tools/make_demo_clip.py`, `tools/custody_demo.py` | Deterministic fixture + acceptance driver |
| `train/*` | Corpus builder, datasets, Stage A / B / C training scripts |

Custody spine acceptance, run 2026-08-20, exit 0:

```
evidence_id     EVD_20260820T161121Z_841ffbf3
original SHA256 841ffbf369d18194ab3ce1b6e47c5d23628251b2266cf4b524e49c4a9a32ac58
working  SHA256 841ffbf369d18194ab3ce1b6e47c5d23628251b2266cf4b524e49c4a9a32ac58
read-only       True
6 chained events, INTAKE_RECEIVED -> WORKING_COPY_HASHED, ledger verified: True
```

### Not built - this is the whole remaining product

| Module | State |
|---|---|
| `peri/core/fragility.py` | **Stub.** Constants, `axis_label`, `describe_transform_sets`, `assert_transform_disjointness` exist. The search does not. |
| `peri/core/provenance.py` | Missing |
| `peri/core/decoder.py` | Missing |
| `peri/core/temporal.py` | Missing |
| `peri/core/pipeline.py` | Missing - not even in the original file tree, added below because it is the missing hinge |
| `peri/core/localize.py` | Missing |
| `peri/core/report.py` | Missing |
| `train/stage_d_calibrate.py` | Missing |
| `api/main.py` | Empty file |
| `web/index.html` | Missing |
| `/replay` | Missing |

### Blocking fact

`artifacts/` on this machine contains only `environment.json`. **No `.pt` files.**
Training finishing on the other laptop does not put weights here. Task 0 below is the
import, and nothing that reports a learned result can be believed until it is done.

---

## 1. Build order

Dependencies are hard. Do not start a task until its dependencies are green.

| # | Task | File | Dep | Budget |
|---|---|---|---|---|
| 0 | Import checkpoints | `artifacts/*.pt` | - | 0.3 h |
| 1 | Provenance stream | `peri/core/provenance.py` | - | 1.5 h |
| 2 | Stage B inference | `peri/core/decoder.py` | 0 | 1.0 h |
| 3 | Stage C inference | `peri/core/temporal.py` | 2 | 0.8 h |
| 4 | **Scoring hinge** | `peri/core/pipeline.py` | 1,2,3 | 1.5 h |
| 5 | Fragility search | `peri/core/fragility.py` | 4 | 1.5 h |
| 6 | Calibration | `train/stage_d_calibrate.py` | 4 | 1.0 h |
| 7 | Examination orchestrator | `peri/core/pipeline.py` (part 2) | 5,6 | 1.5 h |
| 8 | Localisation | `peri/core/localize.py` | 7 | 1.0 h |
| 9 | API | `api/main.py` | 7,8 | 2.0 h |
| 10 | Dashboard | `web/index.html` | 9 | 3.0 h |
| 11 | Report | `peri/core/report.py` | 7 | 3.0 h |
| 12 | Replay + language gate | `api/main.py`, CI | 10,11 | 2.0 h |

Total 20 hours. Tasks 4 and 6 unblock everything visual; if you are short on people,
put your best person on 4 and 6 and let the frontend wait.

Parallel split for two people:

* **Person A:** 1 -> 5 -> 11 (provenance, fragility, report)
* **Person B:** 2 -> 3 -> 4 -> 6 -> 7 -> 8 (the model path and the hinge)
* Whoever finishes first takes 9, then both converge on 10 and 12.

---

## Task 0 - Import the trained checkpoints

Nothing below this line means anything until these files exist and their hashes are
recorded.

```bash
# on the training box
sha256sum artifacts/stage_a_videoprint.pt artifacts/stage_b_decoder.pt artifacts/stage_c_temporal.pt

# copy the three .pt files into artifacts/ on the dev box, then here:
.venv/Scripts/python.exe -m tools.checksum_artifacts
.venv/Scripts/python.exe -c "import torch,json; p=torch.load('artifacts/stage_b_decoder.pt',map_location='cpu',weights_only=False); print(json.dumps(p['meta'],indent=2)); print(json.dumps(p['config'],indent=2))"
```

**Check, and write the answers down - they go on the report's Methods page:**

1. Do the three SHA-256 values match what the training box printed? If not, the copy
   is corrupt, not "probably fine".
2. `meta.corpus_id` is `PPF-ICV-1` and `meta.held_out_method` is `poisson`.
3. `meta.extra.fingerprint_source` - if it says `srm-residual`, Stage B was trained on
   the fallback fingerprint and **Stage A is decorative**. That is survivable and is
   exactly the hour-14 kill switch, but you must say "acquisition residual", never
   "learned fingerprint", on stage and in the report.
4. `meta.extra.val_auroc`, `val_ece`, `val_auroc_held_out_method`. The held-out number
   is the one you quote. A well-calibrated 0.85 beats an overconfident 0.97 and you
   say so out loud.

**If any checkpoint is missing, keep going anyway.** Every wrapper below falls back to
a deterministic classical operator. The fallback path is the one you rehearse.

---

## Task 1 - `peri/core/provenance.py` (S4)

**Why first:** it is the only stream with zero dependence on training. If every model
is missing or wrong, this still produces a scored stream, a findings file, and a PDF.
It is the insurance policy on the whole demo.

Full plan already written: `docs/superpowers/plans/2026-08-20-ppf-04-provenance.md`.
Follow it. Contract summary:

```python
C2PA_STATUSES = ("present", "absent", "invalid", "unavailable")
PROVENANCE_FEATURE_DIM = 6

def read_c2pa(path) -> dict          # never raises; status + validation errors
def collect_facts(path, container=None, now_utc=None) -> dict
def evaluate_rules(facts) -> list[dict]     # 10 rules, PRV01..PRV10
def provenance_score(rule_results) -> float # weighted fraction in [0,1], higher = Hd
def provenance_feature(facts, rule_results) -> tuple[float, ...]  # 6 dims
def provenance_stress_scores(facts) -> tuple[float, ...]          # leave-one-tag-out
def analyse(path, container=None, now_utc=None) -> dict
def to_observation(analysis, weight=1.0) -> StreamObservation
```

**Non-negotiables:**

* Rule-based. No ML. No LLM-written prose - templated sentences bound to the numeric
  facts only.
* A rule that cannot be evaluated is `evaluated: False` and is excluded from the
  denominator of the score. Missing information is not evidence of manipulation.
* C2PA precedence, verbatim in the rule statements and in the report: C2PA verifies
  that provenance claims have not been tampered with, **not** that they are truthful.
  Where forensic findings contradict a manifest, forensic findings take precedence and
  both are reported.
* Rule statements must survive `tests/test_legal_language.py`. Nothing from the
  forbidden list in CLAUDE.md section 8 may appear in a statement string.

**Acceptance:**

```bash
.venv/Scripts/python.exe -m pytest tests/test_provenance_c2pa.py tests/test_provenance_rules.py tests/test_provenance_stream.py -q
.venv/Scripts/python.exe -c "from peri.core.provenance import analyse; import json; print(json.dumps(analyse('evidence/_fixtures/demo.mp4'), indent=2)[:1200])"
```

A plain ffmpeg clip must trigger the C2PA-absence rule and, because `testsrc2` writes
no device tags, the missing-device-tag rule. It must not trigger the future-timestamp
rule. Score is a fraction in `[0, 1]`, not a percentage, and never rendered as one.

---

## Task 2 - `peri/core/decoder.py` (Stage B inference)

**Purpose:** RGB frame + Videoprint field -> tamper mask + reliability map.

```python
MODE_LEARNED = "learned-decoder"
MODE_THRESHOLD = "residual-threshold"

class TamperDecoder:
    def __init__(self, checkpoint: str | Path | None = None, device: str = "cpu") -> None
    @torch.no_grad()
    def infer(self, rgb: torch.Tensor, fingerprint: torch.Tensor) -> dict
        # rgb, fingerprint: (B,3,H,W). Returns
        # {"mask_prob": (B,1,H,W), "reliability": (B,1,H,W), "frame_score": (B,)}
    def describe(self) -> dict
```

**Loading the checkpoint - use the real payload keys:**

```python
payload = torch.load(path, map_location=device, weights_only=False)
config  = payload["config"]     # arch, requested_arch, backbone, in_channels=6, out_channels=2
model.load_state_dict(payload["model"])
self.meta = payload["meta"]     # stage, corpus_id, held_out_method, seed, extra{...}
```

`config["arch"]` is the class name: `"UNetDecoder"` or `"SegformerDecoder"`. Import
both from `train.stage_b_decoder` and select on that string. Do not guess from the
backbone name - a Segformer run that fell back to U-Net at train time records
`requested_arch: "segformer"` but `arch: "UNetDecoder"`, and loading the wrong class
throws a state-dict mismatch that costs you twenty minutes at the worst moment.

**Output channel convention, fixed by training:** `OUT_CHANNELS = 2`, channel 0 is the
mask logit, channel 1 is the TCP-style confidence logit. Both go through sigmoid.

**Frame score:** reuse `train.stage_b_decoder.frame_score` unchanged - the mean of the
top 2% most suspicious pixels. Do not re-implement it. Calibration and inference must
compute the identical statistic or the LR is fitted to a different quantity than the
one it scores.

**Fallback when no checkpoint (`MODE_THRESHOLD`):** no learned decoder. Take the
fingerprint field, compute a local energy map (box filter over the squared residual),
robust-normalise it with the median and the MAD of the frame, and take the sigmoid of
the z-score as `mask_prob`. Set `reliability` to a constant derived from the frame's
residual dispersion - low dispersion means the map means little, and the UI greys it
out. This is deterministic and it visibly separates a splice in most clips. Say
"acquisition residual", not "fingerprint".

**Acceptance:**

```bash
.venv/Scripts/python.exe -c "
import torch
from peri.core.decoder import TamperDecoder
from peri.core.videoprint import VideoprintExtractor
vp = VideoprintExtractor('artifacts/stage_a_videoprint.pt')
td = TamperDecoder('artifacts/stage_b_decoder.pt')
x = torch.rand(2,3,256,256)
out = td.infer(x, vp.extract(x))
print(td.describe()['mode'], vp.mode, out['mask_prob'].shape, out['frame_score'].tolist())
"
```

Then delete the checkpoint path and run it again with `None` for both. It must still
print shapes and finite scores. **Run that second form once per session** - it is the
kill-switch rehearsal.

---

## Task 3 - `peri/core/temporal.py` (Stage C inference)

**Purpose:** per-frame tokens -> video-level anomaly score + per-frame timeline.

```python
TOKEN_DIM = 8
MODE_LEARNED = "learned-temporal"
MODE_WINDOW = "moving-window"

def build_token(mask_logits, mask_prob, conf, field) -> list[float]   # 8 dims
class TemporalAggregator:
    def __init__(self, checkpoint=None, device="cpu") -> None
    @torch.no_grad()
    def infer(self, tokens: np.ndarray) -> dict
        # (T, 8) -> {"video_score": float, "frame_scores": (T,), "mode": str}
    def describe(self) -> dict
```

**The token layout is already fixed by `train/stage_c_temporal.cache_tokens`.** Copy it
exactly, in this order, or the transformer is being fed a permuted vector and will
output confident nonsense:

```
0  frame_score(mask_logits)          top-2% mean of the mask probability
1  probability.mean()
2  probability.max()
3  (probability > 0.5).float().mean()   suspicious area fraction
4  conf.mean()
5  conf.min()
6  field.abs().mean()                fingerprint energy
7  field.std()                       fingerprint dispersion
```

**Loading:** `payload["config"]` holds `d_model`, `n_heads`, `n_layers`, `max_frames`.
Construct `TemporalTransformer` from `train.stage_c_temporal`, load
`payload["model"]`, call `model(tokens, valid)` which returns
`(frame_logits, video_logit)`. Pad or truncate to `max_frames`; `valid` is 1.0 for real
frames and 0.0 for padding.

**Fallback (`MODE_WINDOW`):** a moving-window statistic over token dimension 0. Window
of 5 frames, take the median inside the window to reject single-frame decode noise,
then the video score is the 90th percentile of the windowed series. A splice that lasts
a second survives that; a single flickering frame does not, which is the correct
behaviour.

---

## Task 4 - `peri/core/pipeline.py` part 1: the scoring hinge

**This is the piece whose absence is the actual problem.** It is not in the CLAUDE.md
file tree and it must exist anyway, for one reason:

> Calibration and examination must compute their scores with the *same code path*.
> If `stage_d_calibrate.py` scores corpus PNG directories with one code path and
> `/examine` scores decoded video frames with another, the LR densities are fitted to
> a different quantity than the one they are asked to interpret. That is not a style
> problem, it is a silent correctness bug that produces a confidently wrong likelihood
> ratio, which is the single worst failure this system can have.

So the shared primitive takes **frames**, not a path:

```python
STREAM_NAMES = ("provenance", "acquisition", "temporal")

@dataclass(frozen=True)
class StreamScores:
    scores: dict[str, float]                    # higher = Hd, always
    features: dict[str, tuple[float, ...]]      # for the Mahalanobis gate
    frame_scores: tuple[float, ...]             # the timeline
    tokens: tuple[tuple[float, ...], ...]
    masks: list[np.ndarray] | None              # kept only when want_masks
    modes: dict[str, str]                       # which implementation ran

def sample_frames(video_path, max_frames=64, stride=None) -> list[np.ndarray]
def score_frames(frames, *, extractor, decoder, aggregator, want_masks=False) -> StreamScores
```

**Frame sampling must be deterministic.** Fixed `max_frames`, uniform indices computed
from the reported frame count, `cv2.CAP_PROP_POS_FRAMES` seeks, BGR to RGB, float32 in
`[0,1]`. No random sampling, no time-based seeking. Two runs of `/examine` on the same
file must select the identical frame indices or the replay hash will not match and the
whole L8 claim dies.

**Score orientation - the invariant that catches people out:** every stream score is
oriented so **higher means Hd**. Provenance already is. Acquisition and temporal
already are. If you ever add a stream that points the other way, negate it *at its
wrapper boundary*, never inside `forensic_lr.py`.

**Feature vectors for the validated-domain gate.** These are what `mahalanobis_distance`
consumes, and their dimension must be byte-stable between calibration and inference:

| Stream | Dim | Contents |
|---|---|---|
| `provenance` | 6 | `provenance_feature(facts, rules)` |
| `acquisition` | 8 | Mean of the 8-dim token across sampled frames |
| `temporal` | 6 | `[mean, std, max, p90, frac>0.5, longest_run_fraction]` of the frame-score series |

Write these dimensions into `artifacts/calibration.json` and assert them on load.
A dimension mismatch must raise `CalibrationError`, not broadcast silently.

**Stress replicas for the LR stability check.** `evaluate_stream` needs
`stress_scores` - the same stream re-scored under mild degradation - to detect a
conclusion resting on a knife edge. Use the **three mildest rungs** of each fragility
ladder: CRF 20, 90% rescale, JPEG q90. These are from the fragility transform families,
which are already asserted disjoint from the training augmentations, so the stability
check is not circular either.

---

## Task 5 - Finish `peri/core/fragility.py`

The constants, `axis_label`, `describe_transform_sets` and
`assert_transform_disjointness` are already there. Append:

```python
FRAGILITY_BANDS = ("LOW", "MODERATE", "HIGH")

def apply_axis_transform(src: Path, axis: str, level: float, out: Path) -> Path
def search_axis(axis, scorer, baseline_outcome, work_dir) -> dict
    # -> {"axis", "survives_to", "flips_at", "label_survives", "label_flips",
    #     "evaluated_levels", "n_evaluations"}
def assess_fragility(video_path, scorer, work_dir) -> dict
    # -> {"axes": {...}, "band": "LOW"|"MODERATE"|"HIGH", "statement": str}
```

**The three transforms, via ffmpeg on the working copy (never the original):**

* `reencode_crf` - `-c:v libx264 -crf {level} -preset veryfast -pix_fmt yuv420p`
* `rescale` - `-vf scale=iw*{level}:ih*{level}` then scale back up to the original
  dimensions, so the scorer always sees the same input size. The information is gone;
  the pixel count is restored. That is what social-media resizing actually does.
* `jpeg_quality` - decode to JPEG at `-q:v` mapped from the quality rung, re-encode.

**The search:** the ladders are ordered mild to harsh. Binary-search for the first rung
whose outcome differs from the baseline outcome. Roughly 4 evaluations per axis instead
of 15. Each evaluation is a full re-score, so budget 12 evaluations total and make the
progress observable - this is the panel the judges watch.

**Bands:**

* `LOW` - survives past CRF 32, past 50% rescale, past JPEG q50
* `HIGH` - flips at or before CRF 28, or 70% rescale, or JPEG q70. Ordinary social-media
  recompression breaks it.
* `MODERATE` - anything between.

**HARD RULE, and it is a rule you will be asked about:** `HIGH` forces the outcome to
`INCONCLUSIVE`. A conclusion that does not survive a WhatsApp forward is not a
conclusion you put in front of a court. Implement that as a decision-gate override in
Task 7, record the reason code, and say it on stage before anyone asks.

**Determinism note:** the ladder rungs are fixed integers and fixed fractions, so the
reported critical levels are stable across runs even though the intermediate encoded
bytes are not. Never hash the transcoded files.

**Court-legible output string, this exact shape:**

```
Conclusion survives to CRF 34 / 41% rescale / JPEG q38. Flips at CRF 36. FRAGILITY: LOW.
```

---

## Task 6 - `train/stage_d_calibrate.py`

**This is the bridge from "training is done" to "I can examine a video".** Weights give
a raw score. A raw score is not a finding. This step is what makes the number mean
something.

```bash
.venv/Scripts/python.exe -m train.stage_d_calibrate
# -> artifacts/calibration.json  (~10 min, CPU)
```

**What it does:**

1. `load_index(CORPUS_DIR)`, keep `sample["split"] == "cal"` **only**. The cal split is
   sacred: never trained on, exists solely for this. If you calibrate on `train` you
   will get beautiful separation and a system that lies.
2. For each cal sample, read its frames with `frame_paths(corpus_dir, sample)` and call
   the **same** `score_frames` from Task 4.
3. Partition by `sample["label"]`: 0 -> Hp scores, 1 -> Hd scores.
4. Per stream, `fit_stream_calibration(name, hp_scores, hd_scores, features)`. It picks
   KDE at 15+ samples per class and ridge logistic below that automatically.
5. Write `artifacts/calibration.json`:

```json
{
  "schema": "peri.calibration/1",
  "corpus_id": "PPF-ICV-1",
  "corpus_description": "Internal validation corpus, not a public benchmark. ...",
  "held_out_method": "poisson",
  "split_counts": {"cal": {"authentic": 12, "manipulated": 48}},
  "validated_domain": {
    "resolutions": ["...*"], "codecs": ["h264"], "duration_range_s": [2.0, 30.0],
    "statement": "Findings are conditional on this declared domain."
  },
  "streams": { "provenance": {...}, "acquisition": {...}, "temporal": {...} },
  "metrics": {"auroc_held_out_method": 0.0, "ece": 0.0},
  "calibration_hash": "<64 hex>"
}
```

6. Also compute and record AUROC on the **held-out generator** (`poisson`) and ECE.
   These two numbers go on the Methods page and you quote them on stage.

**Re-run this after every checkpoint swap.** Stale densities fitted to old weights are
a correctness bug, not a cosmetic one. Put it in the checkpoint import checklist.

**Acceptance:**

```bash
.venv/Scripts/python.exe -m pytest tests/test_calibration_artifact.py -q
.venv/Scripts/python.exe -c "
import json; c=json.load(open('artifacts/calibration.json'))
print(c['corpus_id'], list(c['streams']), c['metrics'])
for n,s in c['streams'].items(): print(n, s['method'], s['feature_dim'], len(s['hp_scores']), len(s['hd_scores']))
"
```

Every stream must report at least one Hp and one Hd score or `fit_stream_calibration`
raises `CalibrationError` - which is the correct behaviour, not something to work
around.

---

## Task 7 - `peri/core/pipeline.py` part 2: the examination orchestrator

The answer to "how do I test my video".

```python
def examine(src_path, evidence_root="evidence", examiner="unattributed",
            progress=None) -> dict
def replay(evidence_dir) -> dict
def canonical_findings(findings: dict) -> dict
```

**Order, with a ledger append after each numbered step:**

1. `intake()` - 6 events, original sealed read-only
2. `build_manifest()` + `write_manifest()` -> `MANIFEST_BUILT`
3. `provenance.analyse(working_path, container)` -> `STREAM_SCORED`
4. `sample_frames()` + `score_frames(want_masks=True)` -> `STREAM_SCORED` x2
5. `assess_fragility()` on the working copy -> `FRAGILITY_ASSESSED`
6. Build three `StreamObservation`s, `evaluate_stream` each, `fuse_and_decide`
7. Apply the fragility override: band `HIGH` -> force `INCONCLUSIVE`, append reason
   code `unstable-under-degradation` -> `DECISION_MADE`
8. `localize()` -> timeline, top-k suspect frames, overlays -> `LOCALISED`
9. Write `findings.json`, hash it, append `FINDINGS_SEALED` with the hash

**`canonical_findings` strips the volatile fields before hashing:** timestamps, wall
clock durations, absolute paths, hostname, examiner name. They still appear in the
findings file, the ledger and the report - just not in the hash. Everything else is in.
This is the contract that makes Task 12 possible; get it wrong and replay never matches.

**`findings.json` top-level keys, fixed:**

```
schema  evidence_id  exhibit  propositions  streams  decision  fragility
localisation  provenance  models  calibration  manifest_hash  findings_hash
generated_utc  generated_ist  examiner
```

`propositions` carries `Hp` and `Hd` verbatim from `forensic_lr.HP_TEXT` / `HD_TEXT`.
Not paraphrased. The same words in the code, the JSON, the UI and the PDF.

**Acceptance:**

```bash
.venv/Scripts/python.exe -c "
from peri.core.pipeline import examine; import json
f = examine('evidence/_fixtures/demo.mp4')
print(json.dumps({k: f[k] for k in ('evidence_id','findings_hash')}, indent=2))
print(f['decision']['outcome'], f['decision']['log10lr_total'], f['decision']['reason_codes'])
print(f['fragility']['band'], f['fragility']['statement'])
"
```

Then run it with `artifacts/` renamed away. It must still produce a complete findings
JSON on the fallback path. That is the kill switch, exercised rather than assumed.

---

## Task 8 - `peri/core/localize.py`

```python
def build_timeline(frame_scores, reliability, fps) -> list[dict]
    # per frame: {"index", "timestamp_s", "score", "reliability", "confident"}
def top_suspect_frames(timeline, k=5) -> list[dict]
def write_overlays(frames, masks, indices, out_dir) -> list[Path]
def localize(frames, masks, frame_scores, reliability, fps, out_dir) -> dict
```

**Rules:**

* Timestamps in seconds from the ffprobe fps, not from an assumed 25.
* `confident` is False where reliability is below its threshold. The UI greys those
  regions. Showing where the model does not know is worth more than one more
  percentage point of accuracy, and it is the thing an expert witness gets asked about.
* Overlays: red channel scaled by mask probability, alpha scaled by reliability, over
  the RGB frame. A low-reliability hot region renders faint - visibly hedged.
* Write PNGs to `evidence/<EVD_ID>/frames/`. Filenames `frame_{index:04d}.png` and
  `overlay_{index:04d}.png` so the API can serve them by name without a lookup table.

---

## Task 9 - `api/main.py`

FastAPI, six routes, no auth, no accounts, no database.

| Route | Method | Returns |
|---|---|---|
| `/examine` | POST multipart | `{"evidence_id": ...}` immediately; work runs in a background task |
| `/status/{evidence_id}` | GET, SSE | Progress events as each ledger event lands |
| `/findings/{evidence_id}` | GET | `findings.json` |
| `/report/{evidence_id}` | GET | `report.pdf` as a file response |
| `/ledger/{evidence_id}` | GET | Ledger events + `verify_ledger` verdict |
| `/replay/{evidence_id}` | POST | `{"original_hash", "replay_hash", "match": bool}` |
| `/frames/{evidence_id}/{name}` | GET | PNG from the evidence directory |

**Details that matter:**

* Mount `web/` as static and serve `index.html` at `/`. One process, one port, no CORS.
* SSE: the ledger is the progress log. Poll the JSONL for new lines and emit each as an
  event. Do not invent a second progress channel that can disagree with the record.
* Every `PeriError` becomes a JSON error response with the reason. A raw traceback must
  never reach a findings file or the UI.
* Path traversal: `/frames/` resolves the requested name against the evidence directory
  and rejects anything that escapes it.
* No network calls anywhere. Demo runs with the wifi off.

```bash
.venv/Scripts/python.exe -m uvicorn api.main:app --port 8000
curl -F file=@evidence/_fixtures/demo.mp4 http://localhost:8000/examine
```

---

## Task 10 - `web/index.html`

Single file. Tailwind is **vendored** at `web/vendor/tailwind.js` and referenced with a
relative path - CDN is a demo-day failure waiting to happen. Legible to a judge with
zero ML background. Vertical order, and the order is the argument:

1. **Drop zone** -> evidence ID badge, SHA-256 typing out character by character
2. **RGB / Videoprint toggle** - the manipulated region is visibly a different texture.
   This is the demo shot. Build it first and build it well.
3. **Tamper timeline** - per-frame anomaly, greyed where reliability is low, click a
   frame to load its mask overlay
4. **Fragility panel** - live self-attack progress, then the three critical thresholds
   and the band
5. **log10LR dial + ENFSI verbal + outcome** - `INCONCLUSIVE` in amber with the reason
   in plain English. Both propositions on screen, verbatim.
6. **Ledger table**, live-appending
7. **Generate Report** and **Verify Replay** - replay success shows two identical
   hashes side by side in green

**Never render a percentage where an LR belongs.** No `%`, no "confidence", no
"97% fake". Reviewed by eye before the demo and grepped in CI.

---

## Task 11 - `peri/core/report.py`

ReportLab. Nine pages, in this order, every field populated from `findings.json` - no
free text, no LLM-written explanation, templated sentences bound to numbers only.

1. Examination summary - IDs, examiner, IST and UTC timestamps, software and model
   versions with SHA-256, calibration corpus ID
2. Exhibit - filename, type, size, **SHA-256 of the original**, working-copy hash,
   stated acquisition source, container, codec, duration, fps, resolution, EXIF,
   C2PA status
3. Methods - stages, frozen vs trained parameter counts, calibration corpus
   composition, **declared validated domain**, both transform sets from
   `describe_transform_sets()`, library versions, and the prior-art credits:
   Noiseprint (2019), TruFor (CVPR 2023), DiCoME (ICML 2026), DTRA (ICMR 2026),
   GenD (WACV 2026), NTIRE 2026 Robust Deepfake Detection Challenge, C2PA
4. Findings - both propositions verbatim, per-stream score / log10LR / fragility /
   in-domain, fused log10LR, verbal equivalent, three-way outcome, reason codes
5. Localised findings - timestamp, frame index, region bbox, anomaly type, contributing
   streams, frame log10LR, thumbnail, reliability
6. Integrity and chain of custody - the full ledger table and the statement that the
   original was never written to
7. Reproducibility - manifest hash, model checksums, config, seeds, hardware, and the
   literal replay command
8. **Limitations - the paragraph from CLAUDE.md section 7, verbatim, non-negotiable,
   copy-pasted, not retyped**
9. Section 63(4) Part-B draft input sheet, watermarked
   `DRAFT - REQUIRES EXPERT REVIEW AND SIGNATURE`

After writing the PDF, hash it and append `REPORT_GENERATED` to the ledger with the
hash. The report is an examination event like any other.

**We sign nothing.** Section 63(4) BSA requires a certificate signed by a person in
charge of the device and an expert. We generate inputs for that human expert.

---

## Task 12 - Replay and the CI language gate

**Replay:**

```python
def replay(evidence_dir) -> dict:
    # load_exhibit_record -> re-run steps 3..8 -> canonical_findings -> hash
    # compare to the FINDINGS_SEALED hash already in the ledger
    # append REPLAY_VERIFIED with both hashes and the match verdict
```

It must **not** re-run intake. A second intake mints a second evidence ID and a second
ledger, which is exactly what replay must not do. That is why `load_exhibit_record` and
`exhibit.json` exist.

```bash
.venv/Scripts/python.exe -m pytest tests/test_replay.py -q
curl -X POST http://localhost:8000/replay/EVD_...
```

If the hashes differ, do not adjust `canonical_findings` until you have found *which*
field moved. Diff the two canonical JSON strings. The usual culprits, in order of
likelihood: non-deterministic frame sampling, an unseeded RNG in a fallback path, a
float that skipped `q()`, a dict that got serialised without `sort_keys`.

**CI language gate** - extend `.github/workflows/ci.yml` to grep `peri/ api/ train/
tools/ web/` for the forbidden strings with word boundaries:

The list is the `FORBIDDEN` tuple in `tests/test_legal_language.py`, which mirrors
CLAUDE.md section 8. Do not retype it here or anywhere else - this document is
scanned by the same gate, and the only files allowed to name the phrases are the
spec, the build plans, and the gate itself.

Word boundaries matter: legitimate words that contain a forbidden token, such as
`improves` and `approves`, must not trip the gate. There are already tests for
both directions - one that every phrase is caught, one that benign substrings are
not.
`tests/test_legal_language.py` already exists - wire it into the workflow and make the
job fail on a hit.

---

## 2. Demo script - four minutes, one tab, wifi off

Rehearse this end to end at least twice. The demo is a deliverable, not an afterthought.

| Beat | Action | What you say |
|---|---|---|
| 0:00 | Drag the manipulated exhibit in | "The original is now read-only. Every act from here is in a hash chain." |
| 0:30 | SHA-256 types out, ledger starts filling | "That is the exhibit's fingerprint. Nothing we do touches the original." |
| 1:00 | **Flip the RGB / Videoprint toggle** | Say nothing for three seconds. Let them see the texture change. |
| 1:30 | Tamper timeline, point at the grey regions | "Grey is where the model is not confident. We are telling you where we do not know." |
| 2:00 | Fragility panel runs live | "It is now attacking its own conclusion. Survives to CRF 34, flips at 36. Fragility: LOW." |
| 2:45 | The dial | "log10 LR 2.4 - moderately strong support for Hd. Here are both propositions, verbatim." |
| 3:15 | **Run a clean video, get INCONCLUSIVE** | "It refuses to answer. That refusal is the feature." |
| 3:40 | Generate Report, open page 8 | "Limitations. This does not determine admissibility - that is the Court's." |
| 3:55 | Verify Replay, two green hashes | "Byte-identical. Anyone can reproduce this examination." |

**The line to land:** every other tool says 97% fake. We say how much more likely these
findings are under manipulation than under authenticity, on which validated domain, and
at what laundering strength that conclusion dies.

**Prepare three exhibits and know each one's outcome before you walk in:** one clearly
manipulated, one clean, one heavily recompressed that comes back `HIGH` fragility and
therefore `INCONCLUSIVE`. Never demo a file you have not run.

---

## 3. Questions you will be asked, and the answer

| Question | Answer |
|---|---|
| "What is your accuracy?" | AUROC on the held-out generator, plus ECE. Then: a well-calibrated 0.85 is worth more in court than an overconfident 0.97. |
| "Can this go straight into a courtroom?" | We make no claim about admissibility, and the phrases that would claim it are banned in our codebase and grepped in CI. Admissibility and weight are the Court's. We produce decision support for a human expert, and we sign nothing. |
| "Did you train on the test data?" | Splits are by identity **and** by generator. One generator is held out entirely. The cal split is never trained on and exists only to fit the LR densities. |
| "Your robustness test is circular." | Training augmentation families and fragility transform families are disjoint, asserted at import time, and both sets are printed on the Methods page. |
| "What if C2PA says it is authentic?" | C2PA verifies that provenance claims have not been tampered with, not that they are truthful. Forensic findings take precedence and both are reported. |
| "Why not blockchain?" | The hash chain gives tamper-evidence. The trust anchor is the laboratory, not a ledger. Adding a chain moves no trust and costs demo reliability. |
| "It said INCONCLUSIVE - is that a failure?" | It is the product. A tool that always answers is a tool that is sometimes confidently wrong in front of a judge. |

---

## 4. What may be cut, in order

Cut from the top. Everything below the line is the product.

1. S5 reference-based identity stream (already a stretch, abstains by design)
2. Stage C learned transformer -> moving-window fallback
3. Stage A learned Videoprint -> SRM residual (the kill switch)
4. Frontend polish beyond the seven panels
5. C2PA manifest reading -> metadata rules alone, reported as `c2pa: absent`

- - - never cut below this line - - -

* The abstention path and its reason codes
* The ledger and the read-only original
* The Evidence Fragility Index
* The report's limitations page, verbatim
* Replay hash equality

---

## 5. Standing invariants

Every task above touches at least one of these. Breaking one silently is worse than
missing a feature.

* **Determinism.** Fixed seeds, `sort_keys`, tight separators, every float through
  `q()`. Replay hash equality is a test, not an aspiration.
* **The findings hash excludes volatile fields** and nothing else.
* **Higher score means Hd**, always, re-oriented at the wrapper boundary only.
* **Never a percentage where an LR belongs.**
* **Fragility transforms stay disjoint from training augmentations.**
* **We sign nothing.** The Part-B sheet is a draft input for a human expert.
* **No network at demo time.**
