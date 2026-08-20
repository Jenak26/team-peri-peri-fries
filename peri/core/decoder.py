"""Stage B inference: RGB + acquisition field to tamper mask and reliability."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from train.stage_b_decoder import SegformerDecoder, UNetDecoder, frame_score

MODE_LEARNED = "learned-decoder"
MODE_THRESHOLD = "residual-threshold"

# SegFormer's stage-1 spatial-reduction convolution has an 8x8 kernel over a
# feature map already downsampled 4x, so any exhibit smaller than 32px on a side
# would make the kernel larger than its input. Small exhibits are real -- the
# rescale fragility axis manufactures them on purpose -- so we reflect-pad up to
# the floor and crop the outputs back rather than failing the examination.
MIN_INPUT_SIDE = 32


def _pad_to_floor(stacked: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
    """Reflect-pad `stacked` up to MIN_INPUT_SIDE, returning the crop back to size."""
    height, width = stacked.shape[-2:]
    pad_h = max(0, MIN_INPUT_SIDE - height)
    pad_w = max(0, MIN_INPUT_SIDE - width)
    if not pad_h and not pad_w:
        return stacked, None
    # Reflection needs the source to be larger than the padding on each side, so fall
    # back to edge replication for the degenerate cases.
    mode = "reflect" if pad_h < height and pad_w < width else "replicate"
    padded = F.pad(stacked, (0, pad_w, 0, pad_h), mode=mode)
    return padded, (height, width)


class TamperDecoder:
    """Load the learned Stage B decoder when present; otherwise use a fixed map."""

    def __init__(self, checkpoint: str | Path | None = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.mode = MODE_THRESHOLD
        self.model: torch.nn.Module | None = None
        self.meta: dict = {}
        self.config: dict = {}

        path = Path(checkpoint) if checkpoint else None
        if path is not None and path.is_file():
            payload = torch.load(path, map_location=self.device, weights_only=False)
            self.config = dict(payload["config"])
            arch = self.config.get("arch")
            if arch == "UNetDecoder":
                model = UNetDecoder(in_channels=int(self.config.get("in_channels", 6)))
            elif arch == "SegformerDecoder":
                model = SegformerDecoder(
                    backbone=str(self.config.get("backbone", "nvidia/mit-b2"))
                )
            else:
                raise ValueError(f"unknown Stage B architecture in checkpoint: {arch!r}")
            model.load_state_dict(payload["model"])
            model.eval().to(self.device)
            self.model = model
            self.meta = dict(payload.get("meta", {}))
            self.mode = MODE_LEARNED

    @torch.no_grad()
    def infer(self, rgb: torch.Tensor, fingerprint: torch.Tensor) -> dict:
        """Return mask probability, reliability, and frame score.

        `rgb` and `fingerprint` are `(B, 3, H, W)` tensors in floating point.
        """
        rgb = rgb.to(self.device).float()
        fingerprint = fingerprint.to(self.device).float()
        if rgb.shape != fingerprint.shape or rgb.dim() != 4 or rgb.shape[1] != 3:
            raise ValueError(
                "expected rgb and fingerprint tensors shaped (B, 3, H, W), got "
                f"{tuple(rgb.shape)} and {tuple(fingerprint.shape)}"
            )

        if self.model is not None:
            stacked, crop = _pad_to_floor(torch.cat([rgb, fingerprint], dim=1))
            logits = self.model(stacked)
            if crop is not None:
                height, width = crop
                logits = logits[..., :height, :width]
            mask_logits = logits[:, 0:1].float()
            conf_logits = logits[:, 1:2].float()
            mask_prob = torch.sigmoid(mask_logits)
            reliability = torch.sigmoid(conf_logits)
            scores = frame_score(mask_logits)
            return {
                "mask_prob": mask_prob,
                "reliability": reliability,
                "frame_score": scores,
                "mask_logits": mask_logits,
            }

        energy = fingerprint.square().mean(dim=1, keepdim=True)
        energy = F.avg_pool2d(energy, kernel_size=15, stride=1, padding=7)
        flat = energy.flatten(1)
        median = flat.median(dim=1).values.view(-1, 1, 1, 1)
        mad = (flat - flat.median(dim=1).values[:, None]).abs().median(dim=1).values
        mad = mad.view(-1, 1, 1, 1).clamp(min=1e-6)
        z = (energy - median) / (1.4826 * mad)
        mask_prob = torch.sigmoid(z - 2.0).clamp(0.0, 1.0)
        dispersion = flat.std(dim=1, unbiased=False).view(-1, 1, 1, 1)
        reliability_value = torch.sigmoid((dispersion - 0.01) * 40.0).clamp(0.15, 0.85)
        reliability = reliability_value.expand_as(mask_prob).contiguous()
        return {
            "mask_prob": mask_prob,
            "reliability": reliability,
            "frame_score": frame_score(torch.logit(mask_prob.clamp(1e-5, 1.0 - 1e-5))),
            "mask_logits": torch.logit(mask_prob.clamp(1e-5, 1.0 - 1e-5)),
        }

    def describe(self) -> dict:
        return {
            "mode": self.mode,
            "checkpoint_loaded": self.model is not None,
            "meta": dict(self.meta),
            "config": dict(self.config),
        }
