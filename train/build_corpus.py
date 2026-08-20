"""Build the PPF-ICV-1 internal validation corpus.

Honest labelling, stated here and repeated in the report: this is an internal
validation corpus, not a public benchmark. Manipulated samples are synthesised by
compositing a donor region into an authentic source clip. Nothing here is compared
against published FF++ numbers, and the corpus is never described as FF++.

Design decisions that matter, and why:

1. Authentic and manipulated samples pass through an IDENTICAL write path (decode
   to RGB, composite or not, write PNG). No re-encode happens for either class.
   If manipulated clips were re-encoded and authentic ones were not, the model
   would learn "re-encoded means fake" and every number afterwards would be a lie.

2. Splits are by source identity AND by generator, never random. A source clip
   appears in exactly one split. The `poisson` splice method is held out of
   training entirely: the generalisation claim is the AUROC on that unseen method.

3. The `cal` split is sacred. It is never trained on. It exists to fit the
   likelihood-ratio densities and the Mahalanobis statistics, and nothing else.

Usage:
    python -m train.build_corpus --frames 24
    python -m train.build_corpus --frames 24 --limit 200 --out data/corpus
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from peri.core.canon import PERI_SEED, hash_obj, stable_seed
from peri.core.errors import CorpusError
from train.config import (
    AUTHENTIC_DIR,
    CORPUS_DESCRIPTION,
    CORPUS_DIR,
    CORPUS_ID,
    FRAMES_PER_CLIP,
    HELD_OUT_METHOD,
    MASK_MAX_AREA_FRACTION,
    MASK_MIN_AREA_FRACTION,
    SPLICE_METHODS,
    SPLIT_FRACTIONS,
    SPLIT_NAMES,
)

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")

# Methods available to each split. train/val see three methods; cal/test see all
# four, so the held-out generator is represented where it must be.
SPLIT_METHODS = {
    "train": tuple(m for m in SPLICE_METHODS if m != HELD_OUT_METHOD),
    "val": tuple(m for m in SPLICE_METHODS if m != HELD_OUT_METHOD),
    "cal": SPLICE_METHODS,
    "test": SPLICE_METHODS,
}


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def list_source_clips(root: Path) -> list[Path]:
    clips = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )
    if not clips:
        raise CorpusError(
            f"no source clips found under {root}. Drop authentic video files there "
            f"(any of {', '.join(VIDEO_SUFFIXES)}) and run this again."
        )
    return clips


def extract_frames(path: Path, count: int) -> list[np.ndarray]:
    """Sample `count` frames evenly across the clip, as uint8 BGR."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    try:
        if total >= count > 0:
            indices = np.linspace(0, total - 1, count).round().astype(int)
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                if ok:
                    frames.append(frame)
        else:
            while len(frames) < count:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
    finally:
        cap.release()
    return frames


# ---------------------------------------------------------------------------
# Splice methods. Each returns (composited BGR uint8, mask float32 in {0,1}).
# ---------------------------------------------------------------------------


def _elliptical_mask(
    shape: tuple[int, int], rng: np.random.Generator, feather: int = 9
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Return (soft alpha HxWx1, hard binary HxW, bbox)."""
    height, width = shape
    area_fraction = float(rng.uniform(MASK_MIN_AREA_FRACTION, MASK_MAX_AREA_FRACTION))
    target_area = area_fraction * height * width
    aspect = float(rng.uniform(0.7, 1.4))
    axis_a = int(np.sqrt(target_area * aspect / np.pi))
    axis_b = int(np.sqrt(target_area / (aspect * np.pi)))
    axis_a = max(8, min(axis_a, width // 2 - 2))
    axis_b = max(8, min(axis_b, height // 2 - 2))
    cx = int(rng.integers(axis_a + 1, max(axis_a + 2, width - axis_a - 1)))
    cy = int(rng.integers(axis_b + 1, max(axis_b + 2, height - axis_b - 1)))
    angle = float(rng.uniform(0, 180))

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (axis_a, axis_b), angle, 0, 360, 255, -1)
    soft = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), 0)
    alpha = (soft.astype(np.float32) / 255.0)[..., None]
    binary = (mask > 127).astype(np.float32)
    bbox = (cx - axis_a, cy - axis_b, 2 * axis_a, 2 * axis_b)
    return alpha, binary, bbox


def _match_color(donor: np.ndarray, target: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Match the donor's per-channel mean and standard deviation to the target's,
    inside the composited region only."""
    weight = alpha[..., 0]
    total = weight.sum() + 1e-6
    out = donor.astype(np.float32).copy()
    for c in range(3):
        d_mean = (donor[:, :, c] * weight).sum() / total
        t_mean = (target[:, :, c] * weight).sum() / total
        d_std = np.sqrt(((donor[:, :, c] - d_mean) ** 2 * weight).sum() / total) + 1e-6
        t_std = np.sqrt(((target[:, :, c] - t_mean) ** 2 * weight).sum() / total) + 1e-6
        out[:, :, c] = (donor[:, :, c] - d_mean) * (t_std / d_std) + t_mean
    return np.clip(out, 0, 255)


def splice(
    target: np.ndarray,
    donor: np.ndarray,
    method: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Composite a donor region into `target`. Returns (image, mask, bbox)."""
    height, width = target.shape[:2]
    donor = cv2.resize(donor, (width, height), interpolation=cv2.INTER_AREA)
    alpha, binary, bbox = _elliptical_mask((height, width), rng)

    if method == "alpha_ellipse":
        source = donor.astype(np.float32)

    elif method == "warp_affine":
        angle = float(rng.uniform(-12, 12))
        scale = float(rng.uniform(0.9, 1.1))
        shift_x = float(rng.uniform(-0.03, 0.03) * width)
        shift_y = float(rng.uniform(-0.03, 0.03) * height)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
        matrix[0, 2] += shift_x
        matrix[1, 2] += shift_y
        source = cv2.warpAffine(
            donor, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        ).astype(np.float32)

    elif method == "color_matched":
        source = _match_color(donor.astype(np.float32), target.astype(np.float32), alpha)

    elif method == "poisson":
        centre = (int(bbox[0] + bbox[2] // 2), int(bbox[1] + bbox[3] // 2))
        centre = (
            int(np.clip(centre[0], bbox[2] // 2 + 2, width - bbox[2] // 2 - 2)),
            int(np.clip(centre[1], bbox[3] // 2 + 2, height - bbox[3] // 2 - 2)),
        )
        hard = (binary * 255).astype(np.uint8)
        try:
            blended = cv2.seamlessClone(donor, target, hard, centre, cv2.NORMAL_CLONE)
            return blended, binary, bbox
        except cv2.error:
            source = donor.astype(np.float32)  # fall back to a plain blend

    else:
        raise CorpusError(f"unknown splice method: {method!r}")

    composited = alpha * source + (1.0 - alpha) * target.astype(np.float32)
    return np.clip(composited, 0, 255).astype(np.uint8), binary, bbox


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------


def assign_splits(identities: list[str], seed: int = PERI_SEED) -> dict[str, str]:
    """Deterministically assign each source identity to exactly one split."""
    rng = np.random.default_rng(seed)
    ordered = sorted(identities)
    permutation = rng.permutation(len(ordered))
    shuffled = [ordered[i] for i in permutation]

    assignment: dict[str, str] = {}
    cursor = 0
    for index, split in enumerate(SPLIT_NAMES):
        if index == len(SPLIT_NAMES) - 1:
            take = len(shuffled) - cursor
        else:
            take = max(1, int(round(SPLIT_FRACTIONS[split] * len(shuffled))))
            take = min(take, len(shuffled) - cursor - (len(SPLIT_NAMES) - index - 1))
        for identity in shuffled[cursor : cursor + take]:
            assignment[identity] = split
        cursor += take
    for identity in shuffled[cursor:]:
        assignment[identity] = SPLIT_NAMES[-1]
    return assignment


def _write_sample(
    out_root: Path,
    sample_id: str,
    frames: list[np.ndarray],
    masks: list[np.ndarray] | None,
) -> tuple[str, str | None]:
    frame_dir = out_root / "frames" / sample_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        cv2.imwrite(str(frame_dir / f"{index:04d}.png"), frame)

    mask_rel: str | None = None
    if masks is not None:
        mask_dir = out_root / "masks" / sample_id
        mask_dir.mkdir(parents=True, exist_ok=True)
        for index, mask in enumerate(masks):
            cv2.imwrite(str(mask_dir / f"{index:04d}.png"), (mask * 255).astype(np.uint8))
        mask_rel = str(Path("masks") / sample_id)
    return str(Path("frames") / sample_id), mask_rel


def build(
    source_dir: Path,
    out_dir: Path,
    frames_per_clip: int,
    limit: int | None,
    seed: int,
) -> dict:
    clips = list_source_clips(source_dir)
    if limit:
        clips = clips[:limit]

    identities = [clip.stem for clip in clips]
    if len(set(identities)) != len(identities):
        raise CorpusError(
            "two source clips share a filename stem; identities must be unique "
            "because splits are assigned by identity"
        )
    splits = assign_splits(identities, seed=seed)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # data/corpus is tracked but its contents are not; restore the marker the
    # rmtree above just removed, or the next `git status` reports a deletion.
    (out_dir / ".gitkeep").touch()

    samples: list[dict] = []
    skipped: list[str] = []
    donor_cache: dict[str, list[np.ndarray]] = {}

    for position, clip in enumerate(clips):
        identity = clip.stem
        split = splits[identity]
        frames = extract_frames(clip, frames_per_clip)
        if len(frames) < 2:
            skipped.append(clip.name)
            continue

        authentic_id = f"{identity}__authentic"
        frame_rel, _ = _write_sample(out_dir, authentic_id, frames, None)
        samples.append(
            {
                "sample_id": authentic_id,
                "split": split,
                "label": 0,
                "method": "none",
                "source_identity": identity,
                "donor_identity": None,
                "n_frames": len(frames),
                "frames_dir": frame_rel,
                "masks_dir": None,
                "source_file": clip.name,
            }
        )

        # Donor comes from a different identity in the SAME split, so no split
        # boundary is crossed by donor pixels.
        same_split = [c for c in clips if splits[c.stem] == split and c.stem != identity]
        donor_clip = same_split[position % len(same_split)] if same_split else clip

        # Decode each donor clip once. It was previously re-decoded for every
        # splice method, which is four full decodes of the same file per source.
        if donor_clip.stem not in donor_cache:
            decoded = extract_frames(donor_clip, frames_per_clip)
            donor_cache[donor_clip.stem] = decoded or list(reversed(frames))
        donor_frames = donor_cache[donor_clip.stem]

        for method in SPLIT_METHODS[split]:
            spliced: list[np.ndarray] = []
            masks: list[np.ndarray] = []
            # stable_seed, not hash(): see peri.core.canon.stable_seed. A salted
            # hash here gave a different corpus on every run.
            method_rng = np.random.default_rng(
                stable_seed(identity, method, base=seed)
            )
            for index, frame in enumerate(frames):
                donor = donor_frames[index % len(donor_frames)]
                image, mask, _ = splice(frame, donor, method, method_rng)
                spliced.append(image)
                masks.append(mask)

            sample_id = f"{identity}__{method}"
            frame_rel, mask_rel = _write_sample(out_dir, sample_id, spliced, masks)
            samples.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "label": 1,
                    "method": method,
                    "source_identity": identity,
                    "donor_identity": donor_clip.stem,
                    "n_frames": len(spliced),
                    "frames_dir": frame_rel,
                    "masks_dir": mask_rel,
                    "source_file": clip.name,
                }
            )

        print(
            f"[{position + 1}/{len(clips)}] {identity} -> {split} "
            f"({1 + len(SPLIT_METHODS[split])} samples)"
        )

    index = {
        "corpus_id": CORPUS_ID,
        "description": CORPUS_DESCRIPTION,
        "seed": seed,
        "frames_per_clip": frames_per_clip,
        "splice_methods": list(SPLICE_METHODS),
        "held_out_method": HELD_OUT_METHOD,
        "split_methods": {k: list(v) for k, v in SPLIT_METHODS.items()},
        "n_source_clips": len(clips),
        "n_samples": len(samples),
        "skipped_unreadable": skipped,
        "samples": sorted(samples, key=lambda s: s["sample_id"]),
    }
    # Hash the whole index, samples included. Hashing only the header would let
    # the corpus contents change without the recorded hash moving, and that hash
    # is what the report cites as the calibration corpus identity.
    index["index_hash"] = hash_obj(index)
    (out_dir / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True), encoding="utf-8"
    )
    return index


def summarise(index: dict) -> str:
    lines = [
        f"corpus        {index['corpus_id']}",
        f"source clips  {index['n_source_clips']}",
        f"samples       {index['n_samples']}",
        f"held out      {index['held_out_method']} (absent from train and val)",
        "",
        f"{'split':<8}{'authentic':<12}{'manipulated':<14}methods",
    ]
    for split in SPLIT_NAMES:
        rows = [s for s in index["samples"] if s["split"] == split]
        authentic = sum(1 for s in rows if s["label"] == 0)
        manipulated = sum(1 for s in rows if s["label"] == 1)
        methods = sorted({s["method"] for s in rows if s["label"] == 1})
        lines.append(
            f"{split:<8}{authentic:<12}{manipulated:<14}{', '.join(methods) or '-'}"
        )
    if index["skipped_unreadable"]:
        lines.append("")
        lines.append(f"skipped (unreadable): {', '.join(index['skipped_unreadable'])}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the PPF-ICV-1 corpus.")
    parser.add_argument("--source", type=Path, default=AUTHENTIC_DIR)
    parser.add_argument("--out", type=Path, default=CORPUS_DIR)
    parser.add_argument("--frames", type=int, default=FRAMES_PER_CLIP)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=PERI_SEED)
    args = parser.parse_args()

    index = build(args.source, args.out, args.frames, args.limit, args.seed)
    print()
    print(summarise(index))
    print()
    print(f"wrote {args.out / 'index.json'}  index_hash={index['index_hash'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
