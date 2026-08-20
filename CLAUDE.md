# Team Peri Peri Fries v2 - Judicial Digital Evidence Authentication Engine

**Authoritative context for this repository. Read fully before writing code. Do not redesign. Do not add components not listed here.**

---

## 0. WHAT THIS IS

Not a deepfake detector. A **forensic examination protocol engine**.

Every existing tool outputs `P(fake)=0.97`. A court cannot weigh that: no stated propositions, no declared population, no validated domain, no uncertainty, no reproducibility.

Team Peri Peri Fries produces, for one exhibit:

1. A **log₁₀ likelihood ratio** between two explicitly stated propositions
2. A **pixel-level tamper mask + reliability map + tamper timeline** from a trained acquisition-consistency network
3. An **Evidence Fragility Index** - the adversarially-found minimum laundering strength at which our own verdict flips
4. A **validated-domain check** (Mahalanobis) - is this exhibit inside what we calibrated on?
5. A **mandatory abstention path** - `INCONCLUSIVE` with machine-generated reason codes
6. A **tamper-evident, replayable examination record** - a second run yields a byte-identical findings hash
7. A **court-oriented PDF** structured against India's Section 63(4) BSA Part-B expert declaration

Propositions, verbatim in code and report:

- **Hp:** the exhibit is an unmanipulated recording of a real event
- **Hd:** the exhibit is synthetically generated or materially manipulated in the facial region

**Novelty claim - never overstate it in code, UI or report:**
> We extend learned acquisition-fingerprint forgery localization (Noiseprint/TruFor, CVPR 2023) from images to video via codec-trace-conditioned self-supervised fingerprint learning and temporal aggregation, and embed it in a judicial examination protocol reporting an ENFSI likelihood ratio, a per-exhibit Evidence Fragility Index, a mandatory abstention path, and a replayable hash-sealed record structured to Section 63(4) BSA. The fingerprint paradigm is not ours; its video formulation, its adversarial fragility reporting, and its statutory packaging are.

Prior art we consume and MUST credit in the report's Methods page: Noiseprint (2019), TruFor (CVPR 2023), DiCoME (ICML 2026), DTRA (ICMR 2026), GenD (WACV 2026), NTIRE 2026 Robust Deepfake Detection Challenge, C2PA/`c2pa-python`.

---

## 1. HARD CONSTRAINTS

- **25 hours total.** Optimise for finished-and-demoable.
- **GPU:** RTX 5060 (Blackwell sm_120, 8GB) minimum; more compute available and should be used. Stages A and B run in parallel if two GPUs exist.
- Python 3.11 backend, single-file HTML/JS frontend. No bundler, no npm.
- No network at demo time.
- Prefer deterministic Python over ML wherever a rule will do.

### Environment - do this first

50-series needs CUDA 12.8 wheels or CUDA will not initialise.

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_available())"
pip install open_clip_torch transformers timm opencv-python-headless numpy scipy \
            scikit-learn pandas fastapi uvicorn python-multipart pillow reportlab c2pa-python
sudo apt-get install -y ffmpeg
```

Pin all versions into `artifacts/environment.json` - it goes in the report.

---

## 2. ARCHITECTURE

```
L0 INTAKE     SHA-256 of original → read-only quarantine → ffprobe/EXIF → working copy + hash
L1 MODEL      Stage A Videoprint  : self-supervised acquisition fingerprint (18M, trained on AUTHENTIC ONLY)
              Stage B Decoder     : SegFormer-B2 on RGB ⊕ Videoprint → tamper mask + reliability map (28M)
              Stage C Temporal    : transformer over per-frame tokens → video verdict + tamper timeline (3M)
              S4 Provenance       : c2pa manifest + metadata contradiction rules (RULE-BASED, no ML)
              S5 (stretch)        : reference-based identity consistency, abstains without reference material
L2 GATE       Mahalanobis distance to the calibration feature population
L3 FRAGILITY  Adaptive adversarial binary search for the verdict's breaking point
L4 LR+FUSION  Per-stream log₁₀LR from densities fitted on the held-out cal split → shrunk fusion → 3-way gate
L5 LOCALISE   Tamper timeline, top-k suspect frames, pixel masks with reliability overlay
L6 LEDGER     Append-only SHA-256 hash chain of every examination event
L7 REPORT     ReportLab PDF + Section 63(4) Part-B DRAFT input sheet; report hash → ledger
L8 REPLAY     Re-run from the examination manifest → identical findings hash
```

### File tree

```
team-peri-peri-fries/
├─ core/
│  ├─ intake.py        # hash, quarantine, ffprobe, EXIF, working copy
│  ├─ provenance.py    # S4: c2pa read + metadata contradiction rules
│  ├─ videoprint.py    # Stage A inference wrapper
│  ├─ decoder.py       # Stage B inference: tamper mask + reliability map
│  ├─ temporal.py      # Stage C inference: verdict + timeline
│  ├─ forensic_lr.py   # LR calibration, in-domain gate, fusion, decision   ← BUILD FIRST
│  ├─ fragility.py     # adversarial breaking-point search
│  ├─ localize.py      # timeline assembly, suspect frames, overlays
│  ├─ ledger.py        # append-only SHA-256 chain
│  ├─ manifest.py      # examination manifest: model hashes, config, seeds, versions
│  └─ report.py        # ReportLab PDF + Part-B draft sheet
├─ train/
│  ├─ stage_a_videoprint.py
│  ├─ stage_b_decoder.py
│  ├─ stage_c_temporal.py
│  └─ stage_d_calibrate.py
├─ api/main.py         # FastAPI: /examine /status /findings /report /ledger /replay
├─ web/index.html      # single file, Tailwind CDN, fetch + SSE
├─ artifacts/          # *.pt, calibration.json, environment.json, SHA256SUMS
└─ evidence/{EVD_ID}/  # original.ro, working.mp4, findings.json, ledger.jsonl, report.pdf
```

---

## 3. BUILD ORDER

Two tracks run in parallel. GPU track starts at hour 2 and runs unattended; CPU track is built while it trains.

**GPU track:** Stage A (h2, 6–8h) → Stage C (h12) → hot-swap into Stage B (h14 checkpoint).
**CPU track (start immediately):**

| # | Step | Acceptance test |
|---|---|---|
| 1 | `core/forensic_lr.py` + synthetic self-test | Three assertions pass: clear-manipulation, unstable-abstains, out-of-domain-abstains |
| 2 | `core/intake.py` + `core/ledger.py` | Upload → evidence ID, original chmod 444, 6 chained ledger events, `ledger.jsonl` verifies |
| 3 | `core/provenance.py` (S4) | Zero-ML structured facts + rule score. **Must work even if all training fails** |
| 4 | `train/stage_b_decoder.py` on **SRM placeholder** | Trains and produces a mask before Stage A exists |
| 5 | `core/fragility.py` | Binary search returns critical CRF / rescale / JPEG-q on a stub scorer |
| 6 | `train/stage_d_calibrate.py` | Writes `artifacts/calibration.json` |
| 7 | `api/main.py` | `curl -F file=@demo.mp4 localhost:8000/examine` → complete findings JSON |
| 8 | `web/index.html` | Legible to a judge with zero ML background |
| 9 | `core/report.py` | PDF opens, every field populated, report hash appended to ledger |
| 10 | `/replay` | Two runs → identical findings hash, green in the UI |

**HOUR 14 KILL SWITCH.** If Stage A has not produced a fingerprint that visibly separates a spliced region, ship Stage B on SRM residuals, never mention it, and spend the recovered time on the report and rehearsal. Stage A is a hot-swap upgrade, never a dependency.

**Minimum viable system:** steps 1, 2, 3, 5, 7, 9, 10 plus Stage B on SRM. That still wins.

---

## 4. TRAINING SPEC

| Stage | Architecture | Data | Trainable | VRAM | Time |
|---|---|---|---|---|---|
| **A Videoprint** | DnCNN/U-Net residual extractor, 17 layers, 64ch. Contrastive: patches from same video+codec+GOP position pull together; different source pushes apart | **Unlabeled authentic video only.** ~2000 clips, ~2M 64×64 patch pairs | ~18M | 12–16GB @ batch 256 | 6–8h |
| **B Decoder** | SegFormer-B2 encoder on RGB ⊕ Videoprint; dual decoder - tamper mask + TCP-style confidence map | FF++ c23 **with ground-truth masks**, 512² crops, batch 12 | ~28M | 14GB | 3–4h |
| **C Temporal** | 4-layer transformer, d=256, 8 heads, over per-frame (anomaly, confidence, fingerprint-stat) tokens | Cached Stage-B outputs | ~3M | 3GB | 25m |
| **D Calibrate** | KDE/logistic LR densities + Mahalanobis stats | held-out `cal` split | 0 | CPU | 10m |

Common: AdamW, cosine schedule. Stage A lr 1e-4; Stage B lr 6e-5 encoder / 6e-4 decoder. `torch.autocast('cuda', torch.bfloat16)`, no GradScaler. Fixed seeds everywhere.

**Splits: by identity AND by generator, never random.** Hold one generator out entirely - that is the generalisation claim. The `cal` split is sacred: never trained on; it exists solely to fit LR densities and Mahalanobis stats.

**Report AUROC on the unseen generator AND ECE.** A well-calibrated 0.85 beats an overconfident 0.97; say so on stage.

Dataset: FF++ c23 with masks for Stage B (worth the EULA wait - Stage A does not need it). Fallback for B: any masked manipulation corpus, or self-built splices with known masks, labelled honestly as "internal validation corpus, not a public benchmark." Hard stop at 45 minutes of acquisition.

---

## 5. LIKELIHOOD-RATIO LAYER (`core/forensic_lr.py`)

Highest-risk component. Unit-test against synthetic scores **before any model exists**.

Fit `f(score|Hp)` and `f(score|Hd)` per stream on the held-out `cal` split - Gaussian KDE with shared Silverman bandwidth; logistic fallback under 15 samples per class (subtract the fitted prior log-odds to recover an LR from a posterior). Fit Mahalanobis stats on the stream's **feature vectors**, not its scores.

```
LOG10LR_DECISION_THRESHOLD = 1.0
STABILITY_IQR_MAX          = 1.0
MAHALANOBIS_QUANTILE       = 0.99
DEPENDENCE_SHRINKAGE       = 0.5
LR_CLIP                    = 6.0
```

Fusion: `log10LR_total = clip(λ · Σ wₛ · median(stressₛ))`. Streams are correlated - never claim independence; shrinkage is stated conservatism and must be justified in the report.

Stream exclusion (reason code recorded): `out-of-validated-domain` · `sign-unstable-under-degradation` · `unstable-under-degradation`.

Decision gate, three outcomes only: no usable stream → `INCONCLUSIVE` · two usable streams |LR|>1 opposing → `INCONCLUSIVE` (`cross-stream-contradiction`) · `|total| < 1.0` → `INCONCLUSIVE` (`evidence-strength-below-reporting-threshold`) · `total > 0` → `MANIPULATION INDICATED` · `total < 0` → `AUTHENTICITY SUPPORTED`.

ENFSI verbal scale on `|log₁₀LR|`: <1 none · 1–2 moderate · 2–3 moderately strong · 3–4 strong · 4–5 very strong · >5 extremely strong. Always name the supported proposition.

---

## 6. EVIDENCE FRAGILITY INDEX (`core/fragility.py`)

Binary-search the minimum laundering strength at which the verdict flips, on three independent axes: re-encode CRF, rescale factor, JPEG quality. Report in court-legible units:

> Conclusion survives to CRF 34 / 41% rescale / JPEG q38. Flips at CRF 36. **FRAGILITY: LOW.**

Bands: LOW (survives heavy laundering) · MODERATE · HIGH (flips under ordinary social-media recompression → force `INCONCLUSIVE`).

**HARD RULE:** training augmentations and fragility-search transforms must be **disjoint** - different families, different parameter ranges. Assert it in code and list both sets in the report. If they overlap, the robustness claim is circular and gets destroyed in Q&A.

---

## 7. REPORT (`core/report.py`)

Nine pages: (1) examination summary - IDs, examiner, IST+UTC timestamp, software and model versions with SHA-256, calibration corpus ID; (2) exhibit - filename, type, size, **SHA-256 of original**, working-copy hash, stated acquisition source, container/codec/duration/fps/resolution, EXIF, C2PA status; (3) methods - stages, frozen vs trained params, calibration corpus composition, **declared validated domain**, fragility axes with parameters, library versions, prior-art credits; (4) findings - both propositions verbatim, per-stream score/log₁₀LR/fragility/in-domain, fused log₁₀LR, verbal equivalent, three-way outcome, reason codes; (5) localised findings - timestamp, frame index, region bbox, anomaly type, contributing streams, frame log₁₀LR, thumbnail, reliability; (6) integrity and chain of custody - full ledger table, statement that the original was never written to; (7) reproducibility - manifest hash, model checksums, config, seeds, hardware, replay command; (8) limitations; (9) Section 63(4) Part-B draft input sheet, watermarked **DRAFT - REQUIRES EXPERT REVIEW AND SIGNATURE**.

Page 8, verbatim, non-negotiable:

> Automated detection is probabilistic. Absence of detected manipulation does not establish authenticity. Findings are conditional on the declared validated domain; exhibits outside that domain are reported as inconclusive. This report is forensic decision support prepared to assist an examiner. It does not itself constitute the certificate under Section 63(4) of the Bharatiya Sakshya Adhiniyam, 2023, and does not determine admissibility or evidentiary weight, which are matters for the Court.

---

## 8. LEGAL LANGUAGE RULES - ENFORCED BY CI GREP

**Forbidden strings anywhere in this repo:** `court-admissible`, `legally valid`, `legally admissible`, `certified evidence`, `meets Section 63`, `proves`, `guaranteed authentic`.

Required framing: the system assists forensic examination. It does not replace judicial determination of admissibility or weight. Section 63(4) BSA requires a certificate signed by a person in charge of the device **and an expert** - we generate inputs for that human expert; we sign nothing.

C2PA precedence: C2PA verifies that provenance claims have not been tampered with, **not** that they are truthful. Where forensic findings contradict a manifest, forensic findings take precedence and both are reported.

---

## 9. FRONTEND (`web/index.html`)

Single file, Tailwind CDN, legible to a judge with zero ML background. Vertical order:

1. Drop zone → evidence ID badge + SHA-256 typing out
2. **RGB ⇄ Videoprint toggle** - the manipulated region is visibly a different texture. *This is the demo shot.*
3. Tamper timeline - per-frame anomaly with reliability shading; greyed where the model is not confident; click → pixel mask overlay
4. **Fragility panel** - live self-attack progress, then the critical thresholds and the band
5. log₁₀LR dial + ENFSI verbal + outcome; `INCONCLUSIVE` in amber with the reason in plain English
6. Ledger table, live-appending
7. `Generate Report` and `Verify Replay`; replay success shows two identical hashes side by side in green

Never render a percentage where an LR belongs.

---

## 10. DO NOT BUILD

Blockchain/IPFS (hash chain gives tamper-evidence; the lab is the trust anchor - reject on the merits and say so). Beauty/attractiveness scoring (encodes rater-population bias in skin tone, gender, age; in a judicial instrument that is the headline, not a caveat - the real signal is already captured by Videoprint). User accounts. Audio or lip-sync stream. Training a generator. Multi-file case management. Docker. Cloud deploy. **LLM-written explanations** - templated sentences bound to numeric findings only. SOTA benchmark chasing. Mobile responsiveness.

---

## 11. WORKING AGREEMENT

- Make judgment calls; do not stop to ask which option I prefer. State assumptions inline and continue.
- Complete working files, not fragments. Copy-paste-ready terminal commands.
- Run the acceptance test after every step and report actual output, not intent.
- If 30+ minutes behind, cut from §10's neighbours and say what you cut. Never silently degrade the abstention path, the ledger, the fragility index, or the report's limitations page - those are the product.
- Determinism is a feature: fixed seeds, sorted JSON keys, stable float formatting. The replay hash must match byte-for-byte.
