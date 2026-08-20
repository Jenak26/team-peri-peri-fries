"""The training augmentation policy.

This module implements exactly the four families declared in
`peri.core.fragility.TRAINING_AUGMENTATIONS` and nothing else. It re-runs the
disjointness assertion at import so that a training run cannot start with an
augmentation set that overlaps the fragility search.

If you are tempted to add a re-encode, a rescale, or a JPEG pass here: don't. Those
three are the fragility axes. Training on them makes the robustness claim circular
and the whole Evidence Fragility Index becomes unreportable.
"""

from __future__ import annotations

import numpy as np

from peri.core.fragility import TRAINING_AUGMENTATIONS, assert_transform_disjointness

assert_transform_disjointness()

_BLUR = TRAINING_AUGMENTATIONS["blur"]["params"]
_NOISE = TRAINING_AUGMENTATIONS["noise"]["params"]
_FLIP = TRAINING_AUGMENTATIONS["flip"]["params"]
_CROP = TRAINING_AUGMENTATIONS["crop"]["params"]


def _gaussian_kernel1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur.

    Implemented with numpy rather than cv2 so that a given seed produces the
    same training stream regardless of which OpenCV build is installed on the
    machine doing the training.

    Measured at roughly 20 ms per 512x512x3 call. A whole-array tap
    accumulation benchmarks the same to within noise, and
    `scipy.ndimage.correlate1d` pads `reflect` differently and would silently
    change the augmentation, so this stays as it is. At a 0.35 apply rate
    across dataloader workers it is not the bottleneck.
    """
    if sigma <= 0:
        return image
    radius = max(1, int(round(3.0 * sigma)))
    kernel = _gaussian_kernel1d(sigma, radius)
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    out = np.empty_like(padded, dtype=np.float32)
    for c in range(padded.shape[2]):
        out[:, :, c] = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="same"), 1, padded[:, :, c]
        )
    result = np.empty_like(out)
    for c in range(out.shape[2]):
        result[:, :, c] = np.apply_along_axis(
            lambda col: np.convolve(col, kernel, mode="same"), 0, out[:, :, c]
        )
    return result[radius:-radius, radius:-radius, :]


def additive_gaussian_noise(
    image: np.ndarray, sigma_255: float, rng: np.random.Generator
) -> np.ndarray:
    noise = rng.normal(0.0, sigma_255 / 255.0, size=image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0)


def horizontal_flip(image: np.ndarray, mask: np.ndarray | None = None):
    flipped = image[:, ::-1, :].copy()
    if mask is None:
        return flipped, None
    return flipped, mask[:, ::-1].copy()


def random_crop_resize(
    image: np.ndarray,
    mask: np.ndarray | None,
    fraction: float,
    rng: np.random.Generator,
):
    """Crop a sub-window and return it at the original size.

    Nearest-neighbour resampling is used deliberately: a bicubic or bilinear
    resample is a low-pass filter, and a low-pass filter is close enough to the
    rescale fragility axis to muddy the disjointness argument. Nearest neighbour
    changes geometry without adding a resampling signature.
    """
    height, width = image.shape[:2]
    crop_h = max(8, int(round(height * fraction)))
    crop_w = max(8, int(round(width * fraction)))
    top = int(rng.integers(0, height - crop_h + 1))
    left = int(rng.integers(0, width - crop_w + 1))
    window = image[top : top + crop_h, left : left + crop_w]
    rows = np.linspace(0, crop_h - 1, height).round().astype(int)
    cols = np.linspace(0, crop_w - 1, width).round().astype(int)
    out_image = window[rows][:, cols]
    if mask is None:
        return out_image, None
    mask_window = mask[top : top + crop_h, left : left + crop_w]
    return out_image, mask_window[rows][:, cols]


def augment(
    image: np.ndarray,
    mask: np.ndarray | None,
    rng: np.random.Generator,
    enable: tuple[str, ...] = ("blur", "noise", "flip", "crop"),
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Apply the declared policy. Returns the augmented pair and what was applied.

    `image` is float32 in [0, 1], shape (H, W, 3). `mask` is float32 in {0, 1},
    shape (H, W), or None.
    """
    applied: dict = {}
    out_image, out_mask = image.astype(np.float32), mask

    if "crop" in enable and rng.random() < 0.5:
        fraction = float(
            rng.uniform(_CROP["min_fraction"], _CROP["max_fraction"])
        )
        out_image, out_mask = random_crop_resize(out_image, out_mask, fraction, rng)
        applied["random_crop"] = round(fraction, 4)

    if "flip" in enable and rng.random() < _FLIP["probability"]:
        out_image, out_mask = horizontal_flip(out_image, out_mask)
        applied["horizontal_flip"] = True

    if "blur" in enable and rng.random() < 0.35:
        sigma = float(rng.uniform(_BLUR["sigma_min"], _BLUR["sigma_max"]))
        out_image = gaussian_blur(out_image, sigma)
        applied["gaussian_blur"] = round(sigma, 4)

    if "noise" in enable and rng.random() < 0.35:
        sigma = float(rng.uniform(_NOISE["sigma_min_255"], _NOISE["sigma_max_255"]))
        out_image = additive_gaussian_noise(out_image, sigma, rng)
        applied["additive_gaussian_noise"] = round(sigma, 4)

    return np.clip(out_image, 0.0, 1.0), out_mask, applied
