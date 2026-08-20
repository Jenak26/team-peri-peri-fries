"""Stage C inference over per-frame tamper tokens."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from train.config import STAGE_C
from train.stage_b_decoder import frame_score
from train.stage_c_temporal import TOKEN_DIM, TemporalTransformer

MODE_LEARNED = "learned-temporal"
MODE_WINDOW = "moving-window"


def build_token(
    mask_logits: torch.Tensor,
    mask_prob: torch.Tensor,
    conf: torch.Tensor,
    field: torch.Tensor,
) -> list[float]:
    """Build the exact 8-dimensional token layout used by Stage C training."""
    return [
        float(frame_score(mask_logits).item()),
        float(mask_prob.mean().item()),
        float(mask_prob.max().item()),
        float((mask_prob > 0.5).float().mean().item()),
        float(conf.mean().item()),
        float(conf.min().item()),
        float(field.abs().mean().item()),
        float(field.std().item()),
    ]


class TemporalAggregator:
    """Load Stage C when available; otherwise use a deterministic window statistic."""

    def __init__(self, checkpoint: str | Path | None = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.mode = MODE_WINDOW
        self.model: TemporalTransformer | None = None
        self.meta: dict = {}
        self.config: dict = {"max_frames": STAGE_C.max_frames}

        path = Path(checkpoint) if checkpoint else None
        if path is not None and path.is_file():
            payload = torch.load(path, map_location=self.device, weights_only=False)
            self.config = dict(payload["config"])
            cfg = replace(
                STAGE_C,
                d_model=int(self.config.get("d_model", STAGE_C.d_model)),
                n_heads=int(self.config.get("n_heads", STAGE_C.n_heads)),
                n_layers=int(self.config.get("n_layers", STAGE_C.n_layers)),
                max_frames=int(self.config.get("max_frames", STAGE_C.max_frames)),
            )
            model = TemporalTransformer(cfg)
            model.load_state_dict(payload["model"])
            model.eval().to(self.device)
            self.model = model
            self.meta = dict(payload.get("meta", {}))
            self.mode = MODE_LEARNED

    @torch.no_grad()
    def infer(self, tokens: np.ndarray) -> dict:
        matrix = np.asarray(tokens, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != TOKEN_DIM:
            raise ValueError(f"expected tokens shaped (T, {TOKEN_DIM}), got {matrix.shape}")
        if matrix.shape[0] == 0:
            return {"video_score": 0.0, "frame_scores": np.zeros(0), "mode": self.mode}

        if self.model is not None:
            max_frames = int(self.config.get("max_frames", STAGE_C.max_frames))
            length = min(matrix.shape[0], max_frames)
            padded = np.zeros((1, max_frames, TOKEN_DIM), dtype=np.float32)
            valid = np.zeros((1, max_frames), dtype=np.float32)
            padded[0, :length] = matrix[:length]
            valid[0, :length] = 1.0
            frame_logits, video_logit = self.model(
                torch.from_numpy(padded).to(self.device),
                torch.from_numpy(valid).to(self.device),
            )
            frame_scores = torch.sigmoid(frame_logits[0, :length]).cpu().numpy()
            return {
                "video_score": float(torch.sigmoid(video_logit)[0].item()),
                "frame_scores": frame_scores,
                "mode": self.mode,
            }

        base = matrix[:, 0].astype(np.float32)
        out = np.zeros_like(base)
        for i in range(base.size):
            lo = max(0, i - 2)
            hi = min(base.size, i + 3)
            out[i] = float(np.median(base[lo:hi]))
        return {
            "video_score": float(np.percentile(out, 90)),
            "frame_scores": out,
            "mode": self.mode,
        }

    def describe(self) -> dict:
        return {
            "mode": self.mode,
            "checkpoint_loaded": self.model is not None,
            "meta": dict(self.meta),
            "config": dict(self.config),
        }
