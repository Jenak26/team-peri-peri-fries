"""L3: the Evidence Fragility Index.

We attack our own conclusion and report where it breaks. Three laundering axes,
each an ordered ladder from mildest to harshest, searched for the minimum strength
at which our verdict changes.

Hard rule, asserted below: the transforms used here and the augmentations used to
train the models are drawn from disjoint families with non-overlapping parameter
ranges. If a model were trained on the same degradation we then use to test it, the
robustness claim would be circular.

NOTE FOR THE BUILD: this file currently carries the constants and the disjointness
assertion only, because the training scripts import them. The search itself
(search_axis, assess_fragility, apply_axis_transform) is appended during Phase 4,
per docs/superpowers/plans/2026-08-20-ppf-05-fragility.md.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

# The only augmentations any training script may use. train/augment.py imports this
# constant and must not add to it locally.
TRAINING_AUGMENTATIONS: dict[str, dict] = {
    "blur": {"family": "gaussian_blur", "params": {"sigma_min": 0.5, "sigma_max": 1.5}},
    "noise": {
        "family": "additive_gaussian_noise",
        "params": {"sigma_min_255": 1.0, "sigma_max_255": 5.0},
    },
    "flip": {"family": "horizontal_flip", "params": {"probability": 0.5}},
    "crop": {
        "family": "random_crop",
        "params": {"min_fraction": 0.85, "max_fraction": 1.0},
    },
}

AXIS_NAMES = ("reencode_crf", "rescale", "jpeg_quality")

FRAGILITY_AXES: dict[str, dict] = {
    "reencode_crf": {
        "family": "codec_reencode",
        "unit": "CRF",
        "ladder": (18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 44, 48, 51),
    },
    "rescale": {
        "family": "spatial_rescale",
        "unit": "scale factor",
        "ladder": (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.41, 0.35, 0.25, 0.15, 0.10),
    },
    "jpeg_quality": {
        "family": "jpeg_recompression",
        "unit": "JPEG quality",
        "ladder": (95, 90, 85, 80, 70, 60, 50, 40, 38, 30, 20, 10, 5),
    },
}

_FRAGILITY_OPERATION_TOKENS = ("crf", "jpeg", "quality", "rescale", "resize", "bitrate")


def assert_transform_disjointness() -> None:
    """Enforce the hard rule from CLAUDE.md section 6.

    Called at import time, and again by every training script, so an overlap stops
    the build instead of quietly invalidating the robustness claim.
    """
    training_families = {spec["family"] for spec in TRAINING_AUGMENTATIONS.values()}
    search_families = {spec["family"] for spec in FRAGILITY_AXES.values()}
    overlap = training_families & search_families
    assert not overlap, (
        "training augmentations and fragility transforms share the families "
        f"{sorted(overlap)}; the robustness claim would be circular"
    )
    for name, spec in TRAINING_AUGMENTATIONS.items():
        blob = f"{name} {spec['family']} {' '.join(spec['params'])}".lower()
        for token in _FRAGILITY_OPERATION_TOKENS:
            assert token not in blob, (
                f"training augmentation {name!r} performs a fragility-axis operation "
                f"({token!r}); the two sets must stay disjoint"
            )


assert_transform_disjointness()


def axis_label(axis: str, level: float) -> str:
    """Court-legible rendering of one ladder rung."""
    if axis == "reencode_crf":
        return f"CRF {int(level)}"
    if axis == "rescale":
        return f"{int(round(level * 100))}% rescale"
    if axis == "jpeg_quality":
        return f"JPEG q{int(level)}"
    raise ValueError(f"unknown fragility axis: {axis!r}")


def describe_transform_sets() -> dict:
    """Both transform sets, for the report's Methods page."""
    return {
        "training_augmentations": {
            name: {"family": spec["family"], "params": dict(spec["params"])}
            for name, spec in sorted(TRAINING_AUGMENTATIONS.items())
        },
        "fragility_axes": {
            name: {
                "family": spec["family"],
                "unit": spec["unit"],
                "ladder": list(spec["ladder"]),
            }
            for name, spec in sorted(FRAGILITY_AXES.items())
        },
        "disjointness": (
            "The augmentation families used during training and the transform "
            "families used by the fragility search share no member and no parameter "
            "range. The assertion that enforces this runs at import time."
        ),
    }


FRAGILITY_BANDS = ("LOW", "MODERATE", "HIGH")


def _run_ffmpeg(argv: list[str]) -> None:
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "ffmpeg failed")[-500:])


def source_dimensions(src: Path) -> tuple[int, int]:
    """Return the (width, height) of the first video stream in `src`."""
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-print_format",
            "json",
            str(src),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or "ffprobe failed")[-500:])
    streams = json.loads(completed.stdout or "{}").get("streams") or []
    if not streams:
        raise RuntimeError(f"no video stream in {src}")
    return int(streams[0]["width"]), int(streams[0]["height"])


def apply_axis_transform(src: Path, axis: str, level: float, out: Path) -> Path:
    """Apply one fixed fragility transform to a working copy."""
    src = Path(src)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if axis == "reencode_crf":
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-crf",
                str(int(level)),
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(out),
            ]
        )
        return out
    if axis == "rescale":
        scale = float(level)
        # Rescale laundering is downscale-then-restore: the resampling destroys fine
        # detail while the exhibit keeps its original geometry. The restoring leg has
        # to name the source dimensions explicitly, because a second `scale=iw:ih`
        # would read iw/ih from its own already-downscaled input and do nothing.
        width, height = source_dimensions(src)
        down_w = max(2, (int(width * scale) // 2) * 2)
        down_h = max(2, (int(height * scale) // 2) * 2)
        vf = f"scale={down_w}:{down_h},scale={width}:{height}"
        _run_ffmpeg(["ffmpeg", "-y", "-i", str(src), "-vf", vf, "-an", str(out)])
        return out
    if axis == "jpeg_quality":
        # ffmpeg q:v is inverse quality. Map 95..5 to roughly 2..31.
        qv = max(2, min(31, int(round(32 - (float(level) / 100.0) * 30))))
        vf = "fps=25,format=yuvj420p"
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-vf",
                vf,
                "-q:v",
                str(qv),
                "-c:v",
                "mjpeg",
                "-an",
                str(out),
            ]
        )
        return out
    raise ValueError(f"unknown fragility axis: {axis!r}")


def search_axis(axis, scorer, baseline_outcome, work_dir) -> dict:
    """Find the first rung where the scorer's outcome changes."""
    ladder = tuple(FRAGILITY_AXES[axis]["ladder"])
    work = Path(work_dir)
    evaluations = []
    flips_at = None
    survives_to = ladder[0]
    lo, hi = 0, len(ladder) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        level = ladder[mid]
        out = work / f"fragility_{axis}_{str(level).replace('.', '_')}.mp4"
        transformed = apply_axis_transform(Path(scorer.source_path), axis, level, out)
        outcome = scorer(transformed)
        evaluations.append({"level": level, "outcome": outcome})
        if outcome != baseline_outcome:
            flips_at = level
            hi = mid - 1
        else:
            survives_to = level
            lo = mid + 1
    return {
        "axis": axis,
        "survives_to": survives_to,
        "flips_at": flips_at,
        "label_survives": axis_label(axis, survives_to),
        "label_flips": axis_label(axis, flips_at) if flips_at is not None else None,
        "evaluated_levels": evaluations,
        "n_evaluations": len(evaluations),
    }


def _band(axes: dict) -> str:
    crf = axes["reencode_crf"].get("flips_at")
    scale = axes["rescale"].get("flips_at")
    jpeg = axes["jpeg_quality"].get("flips_at")
    if (
        (crf is not None and crf <= 28)
        or (scale is not None and scale >= 0.70)
        or (jpeg is not None and jpeg >= 70)
    ):
        return "HIGH"
    if (
        (crf is None or crf > 32)
        and (scale is None or scale < 0.50)
        and (jpeg is None or jpeg < 50)
    ):
        return "LOW"
    return "MODERATE"


def assess_fragility(video_path, scorer, work_dir) -> dict:
    baseline = scorer(Path(video_path))
    scorer.source_path = Path(video_path)
    axes = {
        axis: search_axis(axis, scorer, baseline, work_dir)
        for axis in AXIS_NAMES
    }
    band = _band(axes)
    survive = " / ".join(axes[a]["label_survives"] for a in AXIS_NAMES)
    first_flip = next(
        (axes[a]["label_flips"] for a in AXIS_NAMES if axes[a]["label_flips"]),
        "no tested rung",
    )
    return {
        "axes": axes,
        "band": band,
        "statement": (
            f"Conclusion survives to {survive}. Flips at {first_flip}. "
            f"FRAGILITY: {band}."
        ),
    }
