"""Stage A: the acquisition fingerprint extractor, and its classical fallback.

Two implementations behind one interface:

* `DnCNN` - the learned residual extractor trained self-supervised on authentic
  video only (train/stage_a_videoprint.py). This is the Videoprint.
* `srm_residual` - three fixed SRM high-pass kernels. No training, no checkpoint,
  deterministic. This is the fallback and the hour-14 kill switch: if Stage A has
  not produced a usable fingerprint, Stage B runs on SRM residuals instead and the
  system still works end to end.

`VideoprintExtractor` picks whichever is available and reports which one it used, so
the report can state the truth about what produced the findings.

Prior art this extends, credited on the report's Methods page: Noiseprint (2019) and
TruFor (CVPR 2023). The fingerprint paradigm is theirs; the video formulation is
what we add.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

MODE_LEARNED = "learned-videoprint"
MODE_SRM = "srm-residual"

# Three classical SRM high-pass kernels (first-order, second-order, and the 3x3
# SQUARE variant), normalised. Fixed, not learned.
_SRM_KERNELS = np.array(
    [
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
    ],
    dtype=np.float32,
)
_SRM_NORMALISERS = np.array([2.0, 12.0, 2.0], dtype=np.float32)


def srm_filter_bank() -> torch.Tensor:
    """(3, 1, 5, 5) normalised SRM kernels."""
    kernels = _SRM_KERNELS / _SRM_NORMALISERS[:, None, None]
    return torch.from_numpy(kernels).unsqueeze(1).contiguous()


def srm_residual(images: torch.Tensor) -> torch.Tensor:
    """Deterministic 3-channel acquisition residual.

    `images` is (B, 3, H, W) float in [0, 1]. Returns (B, 3, H, W): the input is
    converted to luminance and passed through the three fixed high-pass kernels.
    """
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(f"expected (B, 3, H, W), got {tuple(images.shape)}")
    weights = torch.tensor([0.299, 0.587, 0.114], device=images.device, dtype=images.dtype)
    luma = (images * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    bank = srm_filter_bank().to(device=images.device, dtype=images.dtype)
    return torch.nn.functional.conv2d(luma, bank, padding=2)


class DnCNN(nn.Module):
    """17-layer residual extractor in the DnCNN family.

    Conv-BN-ReLU stack with no downsampling, so the output fingerprint is at full
    input resolution and a spliced region shows up as a texture change in place.
    """

    def __init__(self, depth: int = 17, width: int = 96, in_channels: int = 3,
                 out_channels: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, width, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(width, width, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(width))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(width, out_channels, 3, padding=1, bias=True))
        self.body = nn.Sequential(*layers)
        self.depth = depth
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ProjectionHead(nn.Module):
    """Patch embedding used only by the Stage A contrastive objective.

    It is discarded at inference: what we keep is the fingerprint field, not the
    embedding.
    """

    def __init__(self, in_channels: int = 3, hidden: int = 256, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, fingerprint: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.net(fingerprint), dim=1)


class VideoprintExtractor:
    """Load the learned fingerprint if it exists; fall back to SRM if it does not."""

    def __init__(self, checkpoint: str | Path | None = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.mode = MODE_SRM
        self.model: DnCNN | None = None
        self.meta: dict = {}

        path = Path(checkpoint) if checkpoint else None
        if path is not None and path.is_file():
            payload = torch.load(path, map_location=self.device, weights_only=False)
            config = payload.get("config", {})
            model = DnCNN(
                depth=int(config.get("depth", 17)),
                width=int(config.get("width", 96)),
            )
            model.load_state_dict(payload["model"])
            model.eval().to(self.device)
            self.model = model
            self.mode = MODE_LEARNED
            self.meta = payload.get("meta", {})

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) in [0, 1] -> (B, 3, H, W) fingerprint field."""
        images = images.to(self.device)
        if self.model is None:
            return srm_residual(images)
        return self.model(images)

    def describe(self) -> dict:
        return {
            "mode": self.mode,
            "checkpoint_loaded": self.model is not None,
            "meta": dict(self.meta),
            "statement": (
                "Fingerprint produced by the learned acquisition model."
                if self.model is not None
                else "Fingerprint produced by fixed high-pass acquisition residual "
                "filters. No learned model was used for this field."
            ),
        }
