"""Regressions for four defects that survived into the finished tree.

Each of these was found by running the pipeline end to end rather than by reading it,
and each one broke something the specification says must never degrade quietly: the
fragility index, the replay guarantee, and the findings document itself. A full
examination takes minutes, so these tests pin the specific mechanisms instead.
"""

from __future__ import annotations

import math
import shutil
import subprocess

import pytest
import torch

from peri.core import pipeline
from peri.core.decoder import MIN_INPUT_SIDE, TamperDecoder
from peri.core.fragility import apply_axis_transform, source_dimensions
from tools.make_demo_clip import make_demo_clip

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required to exercise the fragility transforms",
)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    return make_demo_clip(tmp_path_factory.mktemp("fixture") / "demo.mp4", seconds=1)


def _dimensions(path):
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    width, height = completed.stdout.strip().split("x")
    return int(width), int(height)


def test_rescale_restores_the_source_geometry(clip, tmp_path):
    """Rescale laundering is downscale-then-restore, not downscale-and-leave.

    The first implementation chained `scale=iw:ih` to restore the frame, but iw/ih in
    the second filter read from its own already-downscaled input, so the restoring leg
    did nothing. At the bottom of the ladder that left a 320x240 exhibit at 32x24 --
    smaller than the decoder's own convolution kernel.
    """
    source = _dimensions(clip)
    assert source == (320, 240)

    for level in (0.9, 0.5, 0.10):
        out = apply_axis_transform(clip, "rescale", level, tmp_path / f"r_{level}.mp4")
        assert _dimensions(out) == source, f"rescale({level}) changed the exhibit geometry"


def test_source_dimensions_reads_the_video_stream(clip):
    assert source_dimensions(clip) == (320, 240)


@pytest.mark.parametrize("size", [(8, 8), (16, 24), (MIN_INPUT_SIDE, MIN_INPUT_SIDE)])
def test_decoder_accepts_exhibits_smaller_than_its_kernel(size):
    """A small exhibit must produce findings, not an exception.

    The learned decoder reduces spatially by 4 and then convolves with an 8x8 kernel,
    so anything under 32px on a side made the kernel larger than its input. Small
    exhibits are real, and the fragility search manufactures them deliberately.
    """
    height, width = size
    decoder = TamperDecoder(None)  # threshold mode; no checkpoint needed
    rgb = torch.rand(1, 3, height, width)
    out = decoder.infer(rgb, torch.rand(1, 3, height, width))
    assert out["mask_prob"].shape == (1, 1, height, width)
    assert torch.isfinite(out["mask_prob"]).all()


def test_non_finite_metrics_become_null_not_a_number():
    """An undefined training metric is recorded as null, never coerced.

    `val_auroc_held_out_method` is NaN on the val split by construction, because the
    held-out manipulation method never appears there. JSON has no NaN literal and the
    determinism spine refuses to canonicalise one, so it is recorded as "not computed"
    rather than silently becoming a number somebody could then quote.
    """
    cleaned = pipeline.finite_or_none(
        {
            "meta": {"extra": {"val_auroc": 1.0, "val_auroc_held_out_method": float("nan")}},
            "list": [1.0, float("inf"), -float("inf")],
            "text": "unchanged",
        }
    )
    assert cleaned["meta"]["extra"]["val_auroc"] == 1.0
    assert cleaned["meta"]["extra"]["val_auroc_held_out_method"] is None
    assert cleaned["list"] == [1.0, None, None]
    assert cleaned["text"] == "unchanged"

    for value in cleaned["list"]:
        assert value is None or math.isfinite(value)


def test_examine_and_replay_share_one_findings_constructor():
    """The replay guarantee is that both paths build the same document.

    They previously built it separately, and a field added to one and not the other
    broke the byte-identical findings hash without breaking anything visible. Both now
    call `build_findings`, and nothing else may assemble a findings document.
    """
    source = pipeline.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()

    assert text.count('"schema": "peri.findings/1"') == 1, (
        "the findings document is assembled in more than one place"
    )
    assert text.count("build_findings(") == 3, (
        "expected one definition of build_findings and one call from each of "
        "examine and replay"
    )


def test_findings_document_carries_every_reported_section():
    """The report and the dashboard both read these keys by name."""
    import inspect

    body = inspect.getsource(pipeline.build_findings)
    for key in (
        "schema", "evidence_id", "exhibit", "propositions", "streams", "decision",
        "fragility", "localisation", "provenance", "models", "calibration",
        "sampling", "manifest_hash", "findings_hash", "generated_utc",
        "generated_ist", "examiner",
    ):
        assert f'"{key}"' in body, f"findings document lost the {key!r} section"
