"""Shared training configuration: paths, splits, and hyperparameters.

Every training script imports from here so that a change to a split rule or a seed
happens in exactly one place. Anything in this file that affects a checkpoint is
written into that checkpoint's metadata, and from there into the examination
manifest and the report's Methods page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from peri.core.canon import PERI_SEED

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
AUTHENTIC_DIR = DATA_DIR / "authentic"
CORPUS_DIR = DATA_DIR / "corpus"
CORPUS_INDEX = CORPUS_DIR / "index.json"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

STAGE_A_CKPT = ARTIFACTS_DIR / "stage_a_videoprint.pt"
STAGE_B_CKPT = ARTIFACTS_DIR / "stage_b_decoder.pt"
STAGE_C_CKPT = ARTIFACTS_DIR / "stage_c_temporal.pt"

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

CORPUS_ID = "PPF-ICV-1"
CORPUS_DESCRIPTION = (
    "Internal validation corpus, not a public benchmark. Manipulated samples are "
    "synthesised by compositing a donor region into an authentic source clip using "
    "four documented splice methods, each producing an exact ground-truth mask. "
    "Authentic and manipulated samples pass through an identical write path, so "
    "encoding history is not a class cue."
)

# The four splice methods play the role of "generators". One is held out of
# training entirely; that hold-out is the generalisation claim, and the number we
# report AUROC on.
SPLICE_METHODS = ("alpha_ellipse", "warp_affine", "color_matched", "poisson")
HELD_OUT_METHOD = "poisson"

# Splits are by source identity AND by generator, never random.
#   train / val : the three non-held-out methods only
#   cal / test  : all four methods, so the held-out generator is represented
# The cal split is sacred: it is never trained on and exists solely to fit the
# likelihood-ratio densities and the Mahalanobis statistics.
SPLIT_FRACTIONS = {"train": 0.60, "val": 0.15, "cal": 0.15, "test": 0.10}
SPLIT_NAMES = ("train", "val", "cal", "test")

FRAMES_PER_CLIP = 24
MASK_MIN_AREA_FRACTION = 0.01
MASK_MAX_AREA_FRACTION = 0.25

# ---------------------------------------------------------------------------
# Stage hyperparameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageAConfig:
    """Self-supervised acquisition fingerprint. Trained on AUTHENTIC video only."""

    patch_size: int = 64
    width: int = 96
    depth: int = 17
    embed_dim: int = 128
    batch_size: int = 256
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0.01
    temperature: float = 0.10
    gop_buckets: int = 4
    pairs_per_epoch: int = 60_000
    seed: int = PERI_SEED
    amp_dtype: str = "bfloat16"


@dataclass(frozen=True)
class StageBConfig:
    """Tamper mask + reliability map on RGB concatenated with the Videoprint."""

    arch: str = "segformer"  # "segformer" | "unet"
    backbone: str = "nvidia/mit-b2"
    crop_size: int = 512
    batch_size: int = 12
    epochs: int = 24
    lr_encoder: float = 6e-5
    lr_decoder: float = 6e-4
    weight_decay: float = 0.01
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    confidence_weight: float = 0.5
    seed: int = PERI_SEED
    amp_dtype: str = "bfloat16"


@dataclass(frozen=True)
class StageCConfig:
    """Transformer over per-frame tokens cached from Stage B."""

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    max_frames: int = 256
    token_dim: int = 8
    batch_size: int = 32
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 0.01
    seed: int = PERI_SEED
    amp_dtype: str = "bfloat16"


STAGE_A = StageAConfig()
STAGE_B = StageBConfig()
STAGE_C = StageCConfig()


@dataclass(frozen=True)
class RunMeta:
    """What gets embedded in every checkpoint alongside the weights."""

    stage: str
    corpus_id: str = CORPUS_ID
    held_out_method: str = HELD_OUT_METHOD
    seed: int = PERI_SEED
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "corpus_id": self.corpus_id,
            "held_out_method": self.held_out_method,
            "seed": self.seed,
            "extra": dict(self.extra),
        }


def ensure_dirs() -> None:
    for path in (DATA_DIR, AUTHENTIC_DIR, CORPUS_DIR, ARTIFACTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
