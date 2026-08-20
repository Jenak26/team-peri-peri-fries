"""Datasets for Stage A (contrastive fingerprint) and Stage B (mask decoder).

Both read the corpus index written by train/build_corpus.py. Stage A uses ONLY the
authentic samples - the fingerprint is learned from unmanipulated video and never
sees a manipulated frame, which is what lets it act as an anomaly detector rather
than a classifier of the manipulations we happened to synthesise.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from peri.core.canon import PERI_SEED
from peri.core.errors import CorpusError
from train.augment import augment
from train.config import CORPUS_DIR


def load_index(corpus_dir: str | Path = CORPUS_DIR) -> dict:
    path = Path(corpus_dir) / "index.json"
    if not path.is_file():
        raise CorpusError(
            f"corpus index not found at {path}. Run: python -m train.build_corpus"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def frame_paths(corpus_dir: Path, sample: dict) -> list[Path]:
    """Sorted frame files for one corpus sample. Public: Stage C caching uses it."""
    return sorted((corpus_dir / sample["frames_dir"]).glob("*.png"))


def mask_paths(corpus_dir: Path, sample: dict) -> list[Path]:
    if not sample.get("masks_dir"):
        return []
    return sorted((corpus_dir / sample["masks_dir"]).glob("*.png"))


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise CorpusError(f"could not read frame: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise CorpusError(f"could not read mask: {path}")
    return (mask.astype(np.float32) / 255.0 > 0.5).astype(np.float32)


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------


class ContrastivePatchDataset(Dataset):
    """Positive pairs: two patches from the same clip at the same GOP-position bucket.

    Negatives are every other clip in the batch. The objective therefore asks the
    network to encode *where the pixels came from*, not what they depict.
    """

    def __init__(
        self,
        corpus_dir: str | Path = CORPUS_DIR,
        split: str = "train",
        patch_size: int = 64,
        gop_buckets: int = 4,
        length: int = 60_000,
        seed: int = PERI_SEED,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.patch_size = patch_size
        self.gop_buckets = gop_buckets
        self.length = length
        self.seed = seed

        index = load_index(self.corpus_dir)
        self.samples = [
            s for s in index["samples"] if s["label"] == 0 and s["split"] == split
        ]
        if not self.samples:
            raise CorpusError(
                f"no authentic samples in split {split!r}; Stage A trains on "
                f"authentic video only"
            )
        self.frames: list[list[Path]] = [
            frame_paths(self.corpus_dir, s) for s in self.samples
        ]
        self.frames = [f for f in self.frames if len(f) >= 2]
        if not self.frames:
            raise CorpusError("authentic samples contain fewer than two frames each")

    @property
    def n_clips(self) -> int:
        """Distinct source clips backing this stream.

        NT-Xent draws its negatives from the other clips in the batch, so this
        number - not `len(self)` - is what determines whether the objective has
        anything to push apart. Stage A checks it before training.
        """
        return len(self.frames)

    def __len__(self) -> int:
        return self.length

    def _crop(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        height, width = image.shape[:2]
        size = min(self.patch_size, height, width)
        top = int(rng.integers(0, height - size + 1))
        left = int(rng.integers(0, width - size + 1))
        patch = image[top : top + size, left : left + size]
        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
            patch = cv2.resize(
                patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST
            )
        return patch

    def __getitem__(self, item: int):
        rng = np.random.default_rng(self.seed + item)
        clip_index = int(rng.integers(0, len(self.frames)))
        paths = self.frames[clip_index]

        bucket = int(rng.integers(0, self.gop_buckets))
        candidates = [p for i, p in enumerate(paths) if i % self.gop_buckets == bucket]
        if len(candidates) < 2:
            candidates = paths

        first, second = rng.choice(len(candidates), size=2, replace=len(candidates) < 2)
        image_a = read_rgb(candidates[int(first)])
        image_b = read_rgb(candidates[int(second)])

        patch_a, _, _ = augment(self._crop(image_a, rng), None, rng, enable=("flip", "crop"))
        patch_b, _, _ = augment(self._crop(image_b, rng), None, rng, enable=("flip", "crop"))

        return (
            torch.from_numpy(np.ascontiguousarray(patch_a.transpose(2, 0, 1))),
            torch.from_numpy(np.ascontiguousarray(patch_b.transpose(2, 0, 1))),
            torch.tensor(clip_index, dtype=torch.long),
            torch.tensor(bucket, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Stage B
# ---------------------------------------------------------------------------


class MaskDataset(Dataset):
    """Frames with exact ground-truth masks. Authentic frames carry an all-zero mask."""

    def __init__(
        self,
        corpus_dir: str | Path = CORPUS_DIR,
        split: str = "train",
        crop_size: int = 512,
        seed: int = PERI_SEED,
        augment_enabled: bool = True,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.crop_size = crop_size
        self.seed = seed
        self.augment_enabled = augment_enabled

        index = load_index(self.corpus_dir)
        self.items: list[tuple[Path, Path | None, int, str]] = []
        for sample in index["samples"]:
            if sample["split"] != split:
                continue
            frames = frame_paths(self.corpus_dir, sample)
            masks = mask_paths(self.corpus_dir, sample)
            for position, frame in enumerate(frames):
                mask = masks[position] if position < len(masks) else None
                self.items.append((frame, mask, int(sample["label"]), sample["method"]))
        if not self.items:
            raise CorpusError(f"split {split!r} contains no frames")

    def __len__(self) -> int:
        return len(self.items)

    def _crop_pair(self, image: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
        height, width = image.shape[:2]
        size = min(self.crop_size, height, width)
        top = int(rng.integers(0, height - size + 1))
        left = int(rng.integers(0, width - size + 1))
        image = image[top : top + size, left : left + size]
        mask = mask[top : top + size, left : left + size]
        if size != self.crop_size:
            image = cv2.resize(
                image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST
            )
            mask = cv2.resize(
                mask, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST
            )
        return image, mask

    def __getitem__(self, item: int):
        frame_path, mask_path, label, method = self.items[item]
        rng = np.random.default_rng(self.seed + item)

        image = read_rgb(frame_path)
        mask = (
            read_mask(mask_path)
            if mask_path is not None
            else np.zeros(image.shape[:2], dtype=np.float32)
        )
        image, mask = self._crop_pair(image, mask, rng)
        if self.augment_enabled:
            image, mask, _ = augment(image, mask, rng)

        return (
            torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            torch.from_numpy(np.ascontiguousarray(mask))[None, ...],
            torch.tensor(label, dtype=torch.long),
            method,
        )


def split_counts(corpus_dir: str | Path = CORPUS_DIR) -> dict[str, dict[str, int]]:
    index = load_index(corpus_dir)
    out: dict[str, dict[str, int]] = {}
    for sample in index["samples"]:
        row = out.setdefault(sample["split"], {"authentic": 0, "manipulated": 0})
        row["authentic" if sample["label"] == 0 else "manipulated"] += 1
    return out
