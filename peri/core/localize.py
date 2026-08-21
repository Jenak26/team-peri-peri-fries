"""Timeline and frame overlay helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from peri.core.canon import q


def build_timeline(frame_scores, reliability, fps: float) -> list[dict]:
    fps = float(fps) if fps else 25.0
    rel = np.asarray(reliability, dtype=float)
    scores = np.asarray(frame_scores, dtype=float)
    if rel.ndim > 1:
        rel = rel.reshape(rel.shape[0], -1).mean(axis=1)
    return [
        {
            "index": int(i),
            "timestamp_s": q(i / fps),
            "score": q(score),
            "reliability": q(rel[i] if i < len(rel) else 0.0),
            "confident": bool((rel[i] if i < len(rel) else 0.0) >= 0.35),
        }
        for i, score in enumerate(scores)
    ]


def top_suspect_frames(timeline: list[dict], k: int = 5) -> list[dict]:
    return sorted(timeline, key=lambda row: row["score"], reverse=True)[:k]


def render_videoprint(field) -> np.ndarray:
    """Render an acquisition fingerprint field as a viewable greyscale image.

    The field is a signed residual whose absolute scale carries no meaning to a reader;
    what carries meaning is that a manipulated region has a *different texture* from its
    surroundings. So the magnitude is stretched between robust percentiles of this
    frame, which keeps the contrast comparable between frames without letting a single
    outlier pixel flatten everything else.
    """
    array = np.asarray(field, dtype=np.float32)
    if array.ndim == 3:
        array = np.sqrt(np.square(array).sum(axis=0))
    array = np.abs(array)
    low, high = np.percentile(array, 2.0), np.percentile(array, 98.0)
    if high - low < 1e-8:
        high = low + 1e-8
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def write_overlays(frames, masks, indices, out_dir: str | Path, fields=None) -> list[Path]:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index in indices:
        if index >= len(frames) or masks is None or index >= len(masks):
            continue
        rgb = np.asarray(frames[index], dtype=np.float32).clip(0.0, 1.0)
        mask = np.asarray(masks[index], dtype=np.float32).squeeze().clip(0.0, 1.0)
        if mask.shape[:2] != rgb.shape[:2]:
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]))
        heat = rgb.copy()
        heat[..., 0] = np.maximum(heat[..., 0], mask)
        overlay = (0.65 * rgb + 0.35 * heat).clip(0.0, 1.0)
        frame_path = directory / f"frame_{index:04d}.png"
        overlay_path = directory / f"overlay_{index:04d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(
            str(overlay_path),
            cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        written.extend([frame_path, overlay_path])

        if fields is not None and index < len(fields):
            print_image = render_videoprint(fields[index])
            if print_image.shape[:2] != rgb.shape[:2]:
                print_image = cv2.resize(print_image, (rgb.shape[1], rgb.shape[0]))
            videoprint_path = directory / f"videoprint_{index:04d}.png"
            cv2.imwrite(str(videoprint_path), (print_image * 255).astype(np.uint8))
            written.append(videoprint_path)
    return written


def localize(
    frames, masks, frame_scores, reliability, fps: float, out_dir: str | Path, fields=None
) -> dict:
    timeline = build_timeline(frame_scores, reliability, fps)
    suspects = top_suspect_frames(timeline)
    paths = write_overlays(
        frames, masks, [row["index"] for row in suspects], out_dir, fields=fields
    )
    return {
        "timeline": timeline,
        "top_suspect_frames": suspects,
        "overlay_files": [p.name for p in paths],
        "videoprint_files": [p.name for p in paths if p.name.startswith("videoprint_")],
    }
