"""Stage D calibration artifact builder."""

from __future__ import annotations

import json

import numpy as np

from peri.core.canon import hash_obj, q
from peri.core.decoder import TamperDecoder
from peri.core.forensic_lr import fit_stream_calibration
from peri.core.pipeline import score_frames
from peri.core.temporal import TemporalAggregator
from peri.core.videoprint import VideoprintExtractor
from train.config import ARTIFACTS_DIR, CORPUS_DESCRIPTION, CORPUS_DIR, CORPUS_ID, HELD_OUT_METHOD
from train.dataset import frame_paths, load_index, read_rgb, split_counts
from train.stage_b_decoder import auroc, expected_calibration_error


def main() -> int:
    index = load_index(CORPUS_DIR)
    extractor = VideoprintExtractor(ARTIFACTS_DIR / "stage_a_videoprint.pt")
    decoder = TamperDecoder(ARTIFACTS_DIR / "stage_b_decoder.pt")
    aggregator = TemporalAggregator(ARTIFACTS_DIR / "stage_c_temporal.pt")
    per_stream = {
        "acquisition": {"hp": [], "hd": [], "features": []},
        "temporal": {"hp": [], "hd": [], "features": []},
    }
    all_scores: list[float] = []
    labels: list[int] = []
    held_scores: list[float] = []
    held_labels: list[int] = []
    for sample in index["samples"]:
        if sample["split"] not in {"cal", "test"}:
            continue
        frames = [read_rgb(p) for p in frame_paths(CORPUS_DIR, sample)]
        scored = score_frames(
            frames,
            extractor=extractor,
            decoder=decoder,
            aggregator=aggregator,
            want_masks=False,
        )
        label = int(sample["label"])
        for name in ("acquisition", "temporal"):
            bucket = "hd" if label else "hp"
            per_stream[name][bucket].append(float(scored.scores[name]))
            per_stream[name]["features"].append(tuple(scored.features[name]))
        mean_score = float(np.mean([scored.scores["acquisition"], scored.scores["temporal"]]))
        all_scores.append(mean_score)
        labels.append(label)
        if sample.get("method") == HELD_OUT_METHOD:
            held_scores.append(mean_score)
            held_labels.append(label)

    streams = {}
    for name, payload in per_stream.items():
        cal = fit_stream_calibration(
            name,
            payload["hp"],
            payload["hd"],
            payload["features"],
        )
        streams[name] = cal.to_dict()
    artifact = {
        "schema": "peri.calibration/1",
        "corpus_id": CORPUS_ID,
        "corpus_description": CORPUS_DESCRIPTION,
        "held_out_method": HELD_OUT_METHOD,
        "split_counts": split_counts(CORPUS_DIR),
        "validated_domain": {
            "resolutions": ["corpus-native"],
            "codecs": ["h264"],
            "duration_range_s": [0.0, 60.0],
            "statement": "Findings are conditional on this declared domain.",
        },
        "streams": streams,
        "metrics": {
            "auroc_held_out_method": q(auroc(held_scores, held_labels)) if len(set(held_labels)) > 1 else None,
            "ece": q(expected_calibration_error(all_scores, labels)) if len(set(labels)) > 1 else None,
        },
    }
    artifact["calibration_hash"] = hash_obj(artifact)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / "calibration.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
