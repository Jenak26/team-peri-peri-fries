"""Assemble the static bundle Vercel serves.

The console is one file and the project page is another. Vercel hosts both; the
examination engine runs elsewhere, because a serverless bundle cannot hold PyTorch.
`PERI_API` is baked in at build time so the deployed console knows where its engine is.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"

ENGINE = os.environ.get("PERI_API", "").rstrip("/")


def main() -> None:
    if PUBLIC.exists():
        shutil.rmtree(PUBLIC)
    PUBLIC.mkdir(parents=True)

    # The project page and its real examination record.
    shutil.copy2(ROOT / "site" / "index.html", PUBLIC / "index.html")
    shutil.copytree(ROOT / "site" / "demo", PUBLIC / "demo")

    # The live console, pointed at the engine.
    console = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    if ENGINE:
        console = console.replace(
            "<script>",
            f'<script>window.PERI_API = "{ENGINE}";</script>\n  <script>',
            1,
        )
    (PUBLIC / "console.html").write_text(console, encoding="utf-8")

    print(f"public/ built; engine = {ENGINE or 'not set (console will call same origin)'}")
    for path in sorted(PUBLIC.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(PUBLIC)}  {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
