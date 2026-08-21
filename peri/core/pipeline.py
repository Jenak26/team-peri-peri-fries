"""Shared scoring hinge and examination orchestration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from peri.core.canon import hash_obj, ist_now_iso, q, qdeep, utc_now_iso
from peri.core.decoder import TamperDecoder
from peri.core.forensic_lr import HD_TEXT, HP_TEXT, OUTCOME_INCONCLUSIVE
from peri.core.intake import intake
from peri.core.ledger import Ledger
from peri.core.localize import localize
from peri.core.manifest import build_manifest, write_manifest
from peri.core.provenance import analyse as analyse_provenance
from peri.core.temporal import TemporalAggregator, build_token
from peri.core.videoprint import VideoprintExtractor

STREAM_NAMES = ("provenance", "acquisition", "temporal")

# Frames drawn for the reported examination.
EXAMINATION_FRAMES = 64
# Frames drawn for each fragility probe. The fragility search asks only whether the
# outcome flips, and it runs one full scoring pass per rung of three ladders, so it
# reads a coarser sample. Baseline and probes use the same budget, which keeps the
# comparison like-for-like; the number is recorded in the findings and the report.
FRAGILITY_PROBE_FRAMES = 12


@dataclass(frozen=True)
class StreamScores:
    scores: dict[str, float]
    features: dict[str, tuple[float, ...]]
    frame_scores: tuple[float, ...]
    tokens: tuple[tuple[float, ...], ...]
    masks: list[np.ndarray] | None
    fields: list[np.ndarray] | None
    reliability: tuple[float, ...]
    modes: dict[str, str]


def select_device(preferred: str | None = None) -> str:
    """Prefer CUDA when the installed torch build exposes it."""
    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def sample_frames(video_path: str | Path, max_frames: int = 64, stride=None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count <= 0:
        indices = list(range(max_frames))
    elif stride:
        indices = list(range(0, count, int(stride)))[:max_frames]
    else:
        indices = np.linspace(0, max(count - 1, 0), num=min(max_frames, count), dtype=int).tolist()
    frames: list[np.ndarray] = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = cap.read()
        if ok and bgr is not None:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    cap.release()
    return frames


def _temporal_feature(frame_scores: np.ndarray) -> tuple[float, ...]:
    if frame_scores.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    hot = frame_scores > 0.5
    longest = run = 0
    for value in hot:
        run = run + 1 if value else 0
        longest = max(longest, run)
    return (
        q(float(frame_scores.mean())),
        q(float(frame_scores.std())),
        q(float(frame_scores.max())),
        q(float(np.percentile(frame_scores, 90))),
        q(float(hot.mean())),
        q(float(longest / frame_scores.size)),
    )


def score_frames(
    frames,
    *,
    extractor: VideoprintExtractor,
    decoder: TamperDecoder,
    aggregator: TemporalAggregator,
    want_masks: bool = False,
) -> StreamScores:
    tokens: list[tuple[float, ...]] = []
    masks: list[np.ndarray] = []
    fields: list[np.ndarray] = []
    reliability: list[float] = []
    for frame in frames:
        tensor = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)))[None]
        field = extractor.extract(tensor)
        out = decoder.infer(tensor, field)
        token = build_token(
            out["mask_logits"],
            out["mask_prob"],
            out["reliability"],
            field,
        )
        tokens.append(tuple(q(v) for v in token))
        reliability.append(q(float(out["reliability"].mean().item())))
        if want_masks:
            masks.append(out["mask_prob"][0, 0].detach().cpu().numpy())
            # The acquisition field is what the RGB/Videoprint toggle renders. It is
            # only retained when masks are, so a fragility probe does not pay for it.
            fields.append(field[0].detach().cpu().numpy())

    token_matrix = np.asarray(tokens, dtype=np.float32) if tokens else np.zeros((0, 8), dtype=np.float32)
    temporal = aggregator.infer(token_matrix)
    frame_scores = np.asarray(temporal["frame_scores"], dtype=float)
    acquisition_score = float(token_matrix[:, 0].mean()) if len(token_matrix) else 0.0
    temporal_score = float(temporal["video_score"])
    features = {
        "acquisition": tuple(q(v) for v in token_matrix.mean(axis=0)) if len(token_matrix) else (0.0,) * 8,
        "temporal": _temporal_feature(frame_scores),
    }
    return StreamScores(
        scores={"acquisition": q(acquisition_score), "temporal": q(temporal_score)},
        features=features,
        frame_scores=tuple(q(v) for v in frame_scores),
        tokens=tuple(tokens),
        masks=masks if want_masks else None,
        fields=fields if want_masks else None,
        reliability=tuple(reliability),
        modes={
            "acquisition": extractor.mode,
            "decoder": decoder.mode,
            "temporal": temporal["mode"],
        },
    )


def finite_or_none(obj):
    """Replace non-finite floats with None, recursively.

    Checkpoint metadata carries training metrics, and some of them are legitimately
    undefined: `val_auroc_held_out_method` is NaN on the val split by construction,
    because the held-out manipulation method never appears there. JSON has no NaN
    literal and the determinism spine refuses to canonicalise one, so an undefined
    metric is recorded as null -- "not computed on this split" -- rather than being
    silently coerced to a number that would then be quotable.
    """
    if isinstance(obj, dict):
        return {k: finite_or_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_or_none(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def canonical_findings(findings: dict) -> dict:
    volatile = {"generated_utc", "generated_ist", "examiner"}
    return {k: qdeep(v) for k, v in findings.items() if k not in volatile}


def _heuristic_decision(scores: StreamScores) -> dict:
    total = q((scores.scores["acquisition"] + scores.scores["temporal"]) / 2.0)
    return {
        "outcome": OUTCOME_INCONCLUSIVE,
        "log10lr_total": 0.0,
        "raw_fallback_score": total,
        "verbal": "no support",
        "sentence": "Calibration has not been run on this machine, so the fallback score is not reported as a likelihood ratio.",
        "reason_codes": ["calibration-missing"],
        "propositions": {"Hp": HP_TEXT, "Hd": HD_TEXT},
    }


def build_findings(
    *,
    record,
    provenance: dict,
    scored: StreamScores,
    decision: dict,
    fragility: dict,
    localisation: dict,
    extractor: VideoprintExtractor,
    decoder: TamperDecoder,
    aggregator: TemporalAggregator,
    calibration: dict,
    manifest_hash: str,
    generated_utc: str,
    generated_ist: str,
    examiner: str,
) -> dict:
    """Assemble the findings document.

    Both `examine` and `replay` build this document, and the replay guarantee is that
    they build the *same* one. They therefore share this single constructor: when the
    two were assembled separately, a field added to one and not the other silently
    broke the byte-identical findings hash, which is the guarantee itself.

    `findings_hash` is left empty here on purpose. The hash is taken over the document
    in exactly this state, then written into it, so that hashing is reproducible from
    the stored document.
    """
    return {
        "schema": "peri.findings/1",
        "evidence_id": record.evidence_id,
        "exhibit": record.to_dict(),
        "propositions": {"Hp": HP_TEXT, "Hd": HD_TEXT},
        "streams": {
            "provenance": provenance,
            "acquisition": {
                "score": scored.scores["acquisition"],
                "feature": scored.features["acquisition"],
            },
            "temporal": {
                "score": scored.scores["temporal"],
                "feature": scored.features["temporal"],
            },
        },
        "decision": decision,
        "fragility": fragility,
        "localisation": localisation,
        "provenance": provenance,
        "models": finite_or_none(
            {
                "videoprint": extractor.describe(),
                "decoder": decoder.describe(),
                "temporal": aggregator.describe(),
                "modes": scored.modes,
            }
        ),
        "calibration": calibration,
        "sampling": {
            "examination_frames": EXAMINATION_FRAMES,
            "fragility_probe_frames": FRAGILITY_PROBE_FRAMES,
        },
        "manifest_hash": manifest_hash,
        "findings_hash": "",
        "generated_utc": generated_utc,
        "generated_ist": generated_ist,
        "examiner": examiner,
    }


def examine(
    src_path: str | Path,
    evidence_root: str | Path = "evidence",
    examiner: str = "unattributed",
    progress=None,
    original_filename: str | None = None,
) -> dict:
    from peri.core.forensic_lr import (
        StreamCalibration,
        StreamObservation,
        evaluate_stream,
        fuse_and_decide,
    )

    # The submitted filename is a custody fact. An upload arrives in a temporary
    # file, so the caller passes the name the exhibit was actually submitted under.
    record = intake(
        src_path, evidence_root=evidence_root, original_filename=original_filename
    )
    if progress:
        progress({"evidence_id": record.evidence_id})
    ledger = Ledger(record.ledger_path)
    device = select_device()
    manifest = build_manifest(
        record,
        {
            "max_frames": EXAMINATION_FRAMES,
            "fragility_probe_frames": FRAGILITY_PROBE_FRAMES,
            "device": device,
        },
    )
    write_manifest(manifest, record.evidence_dir)
    ledger.append("MANIFEST_BUILT", record.evidence_id, {"manifest_hash": manifest["manifest_hash"]})

    provenance = analyse_provenance(record.working_path, record.container)
    ledger.append("STREAM_SCORED", record.evidence_id, {"stream": "provenance", "score": provenance["score"]})

    extractor = VideoprintExtractor("artifacts/stage_a_videoprint.pt", device=device)
    decoder = TamperDecoder("artifacts/stage_b_decoder.pt", device=device)
    aggregator = TemporalAggregator("artifacts/stage_c_temporal.pt", device=device)

    cal_path = Path("artifacts") / "calibration.json"
    if cal_path.exists():
        cal_data = json.loads(cal_path.read_text(encoding="utf-8"))
        cal_acq = StreamCalibration.from_dict(cal_data["streams"]["acquisition"])
        cal_tem = StreamCalibration.from_dict(cal_data["streams"]["temporal"])
    else:
        cal_acq = cal_tem = None

    def run_scorer(video_path: Path) -> dict:
        frames_subset = sample_frames(video_path, max_frames=FRAGILITY_PROBE_FRAMES)
        scored_subset = score_frames(
            frames_subset,
            extractor=extractor,
            decoder=decoder,
            aggregator=aggregator,
            want_masks=False,
        )
        if cal_acq and cal_tem:
            obs_acq = StreamObservation(
                name="acquisition",
                score=scored_subset.scores["acquisition"],
                feature=scored_subset.features["acquisition"],
            )
            obs_tem = StreamObservation(
                name="temporal",
                score=scored_subset.scores["temporal"],
                feature=scored_subset.features["temporal"],
            )
            dec = fuse_and_decide([
                evaluate_stream(cal_acq, obs_acq),
                evaluate_stream(cal_tem, obs_tem)
            ])
            return dec.outcome
        else:
            return _heuristic_decision(scored_subset)["outcome"]

    frames = sample_frames(record.working_path, max_frames=EXAMINATION_FRAMES)
    scored = score_frames(
        frames,
        extractor=extractor,
        decoder=decoder,
        aggregator=aggregator,
        want_masks=True,
    )
    ledger.append("STREAM_SCORED", record.evidence_id, {"stream": "acquisition", "score": scored.scores["acquisition"]})
    ledger.append("STREAM_SCORED", record.evidence_id, {"stream": "temporal", "score": scored.scores["temporal"]})

    if cal_acq and cal_tem:
        obs_acq = StreamObservation(
            name="acquisition",
            score=scored.scores["acquisition"],
            feature=scored.features["acquisition"],
        )
        obs_tem = StreamObservation(
            name="temporal",
            score=scored.scores["temporal"],
            feature=scored.features["temporal"],
        )
        decision_obj = fuse_and_decide([
            evaluate_stream(cal_acq, obs_acq),
            evaluate_stream(cal_tem, obs_tem)
        ])
        decision = decision_obj.to_dict()
    else:
        decision = _heuristic_decision(scored)

    from peri.core.fragility import assess_fragility
    fragility = assess_fragility(record.working_path, run_scorer, record.evidence_dir)

    if fragility["band"] == "HIGH" and decision["outcome"] != OUTCOME_INCONCLUSIVE:
        decision["outcome"] = OUTCOME_INCONCLUSIVE
        decision["reason_codes"].append("fragility-high")
        decision["verbal"] = "no support"
        decision["sentence"] = "Evidence fragility was HIGH, overriding outcome to INCONCLUSIVE."
        decision["log10lr_total"] = 0.0

    ledger.append("DECISION_MADE", record.evidence_id, {"outcome": decision["outcome"], "reason_codes": decision["reason_codes"]})

    fps = float(record.container.get("video", {}).get("fps", 25.0) or 25.0)
    localisation = localize(
        frames,
        scored.masks,
        scored.frame_scores,
        scored.reliability,
        fps,
        record.evidence_dir / "frames",
        fields=scored.fields,
    )
    ledger.append("LOCALISED", record.evidence_id, {"top_frames": [r["index"] for r in localisation["top_suspect_frames"]]})

    findings = build_findings(
        record=record,
        provenance=provenance,
        scored=scored,
        decision=decision,
        fragility=fragility,
        localisation=localisation,
        extractor=extractor,
        decoder=decoder,
        aggregator=aggregator,
        calibration={"available": bool(cal_acq and cal_tem)},
        manifest_hash=manifest["manifest_hash"],
        generated_utc=utc_now_iso(),
        generated_ist=ist_now_iso(),
        examiner=examiner,
    )
    findings["findings_hash"] = hash_obj(canonical_findings(findings))
    (record.evidence_dir / "findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8"
    )
    ledger.append("FINDINGS_SEALED", record.evidence_id, {"findings_hash": findings["findings_hash"]})
    return findings


def replay(evidence_dir: str | Path) -> dict:
    from peri.core.forensic_lr import (
        StreamCalibration,
        StreamObservation,
        evaluate_stream,
        fuse_and_decide,
    )
    from peri.core.fragility import assess_fragility
    from peri.core.intake import load_exhibit_record

    evidence_dir = Path(evidence_dir)
    findings_path = evidence_dir / "findings.json"
    findings = json.loads(findings_path.read_text(encoding="utf-8"))

    # Rebuild the sealed exhibit from disk rather than performing intake again: a
    # second intake would mint a second evidence ID and a second ledger, which is
    # precisely what replay must not do.
    record = load_exhibit_record(evidence_dir)

    # The provenance rules take the examination time as an input -- one of them asks
    # whether a recorded timestamp lies in the future. Replaying against the wall
    # clock would re-ask that question at a different instant, so the replay uses the
    # instant the original examination recorded. This is what makes the run a replay
    # of a record rather than a fresh examination that happens to reuse the exhibit.
    recorded_now = (
        findings.get("streams", {}).get("provenance", {}).get("facts", {}).get("now_utc")
    )
    now_utc = datetime.fromisoformat(recorded_now) if recorded_now else None
    provenance = analyse_provenance(record.working_path, record.container, now_utc=now_utc)

    device = select_device()
    extractor = VideoprintExtractor("artifacts/stage_a_videoprint.pt", device=device)
    decoder = TamperDecoder("artifacts/stage_b_decoder.pt", device=device)
    aggregator = TemporalAggregator("artifacts/stage_c_temporal.pt", device=device)

    cal_path = Path("artifacts") / "calibration.json"
    if cal_path.exists():
        cal_data = json.loads(cal_path.read_text(encoding="utf-8"))
        cal_acq = StreamCalibration.from_dict(cal_data["streams"]["acquisition"])
        cal_tem = StreamCalibration.from_dict(cal_data["streams"]["temporal"])
    else:
        cal_acq = cal_tem = None

    def run_scorer(video_path: Path) -> dict:
        frames_subset = sample_frames(video_path, max_frames=FRAGILITY_PROBE_FRAMES)
        scored_subset = score_frames(
            frames_subset,
            extractor=extractor,
            decoder=decoder,
            aggregator=aggregator,
            want_masks=False,
        )
        if cal_acq and cal_tem:
            obs_acq = StreamObservation(
                name="acquisition",
                score=scored_subset.scores["acquisition"],
                feature=scored_subset.features["acquisition"],
            )
            obs_tem = StreamObservation(
                name="temporal",
                score=scored_subset.scores["temporal"],
                feature=scored_subset.features["temporal"],
            )
            dec = fuse_and_decide([
                evaluate_stream(cal_acq, obs_acq),
                evaluate_stream(cal_tem, obs_tem)
            ])
            return dec.outcome
        else:
            return _heuristic_decision(scored_subset)["outcome"]

    frames = sample_frames(record.working_path, max_frames=EXAMINATION_FRAMES)
    scored = score_frames(
        frames,
        extractor=extractor,
        decoder=decoder,
        aggregator=aggregator,
        want_masks=True,
    )

    if cal_acq and cal_tem:
        obs_acq = StreamObservation(
            name="acquisition",
            score=scored.scores["acquisition"],
            feature=scored.features["acquisition"],
        )
        obs_tem = StreamObservation(
            name="temporal",
            score=scored.scores["temporal"],
            feature=scored.features["temporal"],
        )
        decision_obj = fuse_and_decide([
            evaluate_stream(cal_acq, obs_acq),
            evaluate_stream(cal_tem, obs_tem)
        ])
        decision = decision_obj.to_dict()
    else:
        decision = _heuristic_decision(scored)

    fragility = assess_fragility(record.working_path, run_scorer, record.evidence_dir)

    if fragility["band"] == "HIGH" and decision["outcome"] != OUTCOME_INCONCLUSIVE:
        decision["outcome"] = OUTCOME_INCONCLUSIVE
        decision["reason_codes"].append("fragility-high")
        decision["verbal"] = "no support"
        decision["sentence"] = "Evidence fragility was HIGH, overriding outcome to INCONCLUSIVE."
        decision["log10lr_total"] = 0.0

    fps = float(record.container.get("video", {}).get("fps", 25.0) or 25.0)
    localisation = localize(
        frames,
        scored.masks,
        scored.frame_scores,
        scored.reliability,
        fps,
        record.evidence_dir / "frames",
        fields=scored.fields,
    )

    new_findings = build_findings(
        record=record,
        provenance=provenance,
        scored=scored,
        decision=decision,
        fragility=fragility,
        localisation=localisation,
        extractor=extractor,
        decoder=decoder,
        aggregator=aggregator,
        calibration=findings.get("calibration", {"available": False}),
        manifest_hash=findings["manifest_hash"],
        generated_utc=findings["generated_utc"],
        generated_ist=findings["generated_ist"],
        examiner=findings["examiner"],
    )

    new_findings["findings_hash"] = hash_obj(canonical_findings(new_findings))

    replay_hash = new_findings["findings_hash"]
    original_hash = findings.get("findings_hash", "")
    match = replay_hash == original_hash

    # The replay is itself an examination event, so it is recorded on the same chain.
    # A replay that did not match is recorded exactly as faithfully as one that did.
    Ledger(record.ledger_path).append(
        "REPLAY_VERIFIED",
        record.evidence_id,
        {"original_hash": original_hash, "replay_hash": replay_hash, "match": match},
    )

    return {
        "evidence_id": record.evidence_id,
        "original_hash": original_hash,
        "replay_hash": replay_hash,
        "match": match,
        "findings": new_findings,
    }
