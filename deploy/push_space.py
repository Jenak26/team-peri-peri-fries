"""Create the Hugging Face Space and push the examination engine to it.

Run this after `huggingface-cli login`. It creates a Docker Space if one does not
exist, uploads the source, uploads the three checkpoints, and prints the engine URL
to give to Vercel.

    python -m deploy.push_space --space <your-hf-username>/peri-peri-fries

The checkpoints are uploaded separately from the source because they are large and
because they are deliberately absent from the GitHub repository - `stage_b_decoder.pt`
alone is past GitHub's file size limit. The Hub stores them over LFS automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Source the Space needs to run an examination. Everything else -- the corpus, the
# training scripts' outputs, prior evidence, the site -- is not part of the engine.
INCLUDE = ("peri", "api", "train", "tools", "web")
INCLUDE_FILES = ("requirements-cpu.txt", "LICENSE", "NOTICE.md")
CHECKPOINTS = ("stage_a_videoprint.pt", "stage_b_decoder.pt", "stage_c_temporal.pt")

IGNORE = ["__pycache__/*", "*.pyc", ".venv/*", "evidence/*", "data/*"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--space",
        required=True,
        help="target Space id, for example jenak26/peri-peri-fries",
    )
    parser.add_argument(
        "--private", action="store_true", help="create the Space private"
    )
    parser.add_argument(
        "--skip-checkpoints",
        action="store_true",
        help="push source only; the engine then runs its documented fallback operators",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import LocalTokenNotFoundError
    except ImportError:
        print("huggingface_hub is not installed. Run: pip install huggingface_hub")
        return 1

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except (LocalTokenNotFoundError, OSError, ValueError):
        print(
            "Not logged in to Hugging Face.\n"
            "Run this first, and paste a WRITE token from "
            "https://huggingface.co/settings/tokens:\n\n"
            "    huggingface-cli login\n"
        )
        return 1
    print(f"authenticated as {user}")

    api.create_repo(
        repo_id=args.space,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )
    print(f"space ready: {args.space}")

    # The Space's own Dockerfile and card live at its root, not under deploy/.
    for name in ("Dockerfile", "README.md"):
        api.upload_file(
            path_or_fileobj=str(ROOT / "deploy" / "space" / name),
            path_in_repo=name,
            repo_id=args.space,
            repo_type="space",
        )
    print("uploaded Dockerfile and Space card")

    for folder in INCLUDE:
        api.upload_folder(
            folder_path=str(ROOT / folder),
            path_in_repo=folder,
            repo_id=args.space,
            repo_type="space",
            ignore_patterns=IGNORE,
        )
        print(f"uploaded {folder}/")

    for name in INCLUDE_FILES:
        path = ROOT / name
        if path.is_file():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=name,
                repo_id=args.space,
                repo_type="space",
            )
    print("uploaded root files")

    if args.skip_checkpoints:
        print(
            "\nSkipped checkpoints. The engine will report srm-residual and "
            "residual-threshold modes, which is the documented fallback path."
        )
    else:
        missing = [n for n in CHECKPOINTS if not (ROOT / "artifacts" / n).is_file()]
        if missing:
            print(f"\nmissing checkpoints in artifacts/: {', '.join(missing)}")
            print("Add them, or re-run with --skip-checkpoints to deploy the fallback.")
            return 1
        for name in CHECKPOINTS:
            source = ROOT / "artifacts" / name
            size_mb = source.stat().st_size / 1048576
            print(f"uploading {name} ({size_mb:.0f} MB) - this is the slow part")
            api.upload_file(
                path_or_fileobj=str(source),
                path_in_repo=f"artifacts/{name}",
                repo_id=args.space,
                repo_type="space",
            )
        # calibration.json is small but the decision layer is meaningless without it.
        api.upload_file(
            path_or_fileobj=str(ROOT / "artifacts" / "calibration.json"),
            path_in_repo="artifacts/calibration.json",
            repo_id=args.space,
            repo_type="space",
        )
        print("uploaded checkpoints and calibration")

    owner, _, name = args.space.partition("/")
    engine = f"https://{owner}-{name}.hf.space".lower()
    print(
        "\ndone.\n"
        f"  Space    https://huggingface.co/spaces/{args.space}\n"
        f"  Engine   {engine}\n\n"
        "Next:\n"
        f"  1. In the Space Settings > Variables, set PERI_ALLOWED_ORIGINS to your\n"
        "     Vercel origin, for example https://peri-peri-fries.vercel.app\n"
        "  2. Wait for the Space to finish building, then check it answers:\n"
        f"     curl -s -o /dev/null -w '%{{http_code}}' {engine}/\n"
        f"  3. vercel env add PERI_API production   # {engine}\n"
        "  4. vercel --prod\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
