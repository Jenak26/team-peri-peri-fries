"""L4 self-test: exercise the likelihood-ratio layer with no model in existence.

Three synthetic exhibits, three required behaviours:

  1. clear manipulation  -> MANIPULATION INDICATED
  2. unstable stream     -> INCONCLUSIVE, stream excluded for instability
  3. out-of-domain       -> INCONCLUSIVE, stream excluded by the domain gate

If any of these regress, the abstention path is broken, and the abstention path
is the product.
"""

from __future__ import annotations

import numpy as np

from peri.core import forensic_lr as flr

_CAL_SEED = 20260820
_N_PER_CLASS = 200
_FEATURE_DIM = 4


def _build_calibrations() -> dict[str, flr.StreamCalibration]:
    rng = np.random.default_rng(_CAL_SEED)
    cals: dict[str, flr.StreamCalibration] = {}
    for name, separation in (("videoprint", 3.0), ("temporal", 2.5), ("provenance", 2.0)):
        hp = rng.normal(0.0, 1.0, _N_PER_CLASS).tolist()
        hd = rng.normal(separation, 1.0, _N_PER_CLASS).tolist()
        features = rng.normal(0.0, 1.0, (2 * _N_PER_CLASS, _FEATURE_DIM)).tolist()
        cals[name] = flr.fit_stream_calibration(name, hp, hd, features)
    return cals


def run_selftest() -> dict[str, flr.Decision]:
    cals = _build_calibrations()
    inside = (0.1, -0.2, 0.05, 0.15)
    outside = (25.0, -30.0, 28.0, -26.0)

    clear = flr.fuse_and_decide(
        [
            flr.evaluate_stream(
                cals["videoprint"],
                flr.StreamObservation("videoprint", 4.2, inside, (4.1, 4.2, 4.3, 4.15)),
            ),
            flr.evaluate_stream(
                cals["temporal"],
                flr.StreamObservation("temporal", 3.6, inside, (3.5, 3.6, 3.7, 3.55)),
            ),
        ]
    )

    unstable = flr.fuse_and_decide(
        [
            flr.evaluate_stream(
                cals["videoprint"],
                flr.StreamObservation("videoprint", 3.5, inside, (4.5, -1.0, 3.9, -0.8)),
            ),
            flr.evaluate_stream(
                cals["temporal"],
                flr.StreamObservation("temporal", 3.0, inside, (5.0, 0.5, 4.0, 1.0)),
            ),
        ]
    )

    out_of_domain = flr.fuse_and_decide(
        [
            flr.evaluate_stream(
                cals["videoprint"],
                flr.StreamObservation("videoprint", 4.2, outside, (4.1, 4.2, 4.3, 4.15)),
            ),
            flr.evaluate_stream(
                cals["temporal"],
                flr.StreamObservation("temporal", 3.6, outside, (3.5, 3.6, 3.7, 3.55)),
            ),
        ]
    )

    return {
        "clear_manipulation": clear,
        "unstable": unstable,
        "out_of_domain": out_of_domain,
    }

def main() -> int:
    expectations = {
        "clear_manipulation": flr.OUTCOME_MANIPULATION,
        "unstable": flr.OUTCOME_INCONCLUSIVE,
        "out_of_domain": flr.OUTCOME_INCONCLUSIVE,
    }
    
    results = run_selftest()
    failed = False
    
    for name, expected in expectations.items():
        actual = results[name].outcome
        if actual == expected:
            print(f"PASS: {name} -> {actual}")
        else:
            print(f"FAIL: {name} -> expected {expected}, got {actual}")
            failed = True
            
    return 1 if failed else 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
