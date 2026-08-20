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


def write_overlays(frames, masks, indices, out_dir: str | Path) -> list[Path]:
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
    return written


def localize(frames, masks, frame_scores, reliability, fps: float, out_dir: str | Path) -> dict:
    timeline = build_timeline(frame_scores, reliability, fps)
    suspects = top_suspect_frames(timeline)
    paths = write_overlays(frames, masks, [row["index"] for row in suspects], out_dir)
    return {
        "timeline": timeline,
        "top_suspect_frames": suspects,
        "overlay_files": [p.name for p in paths],
    }
