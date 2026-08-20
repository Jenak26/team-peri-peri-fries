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
