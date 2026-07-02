#!/usr/bin/env python3
"""Generate publication catalog thumbnails from the first page of each PDF."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBS_PATH = ROOT / "assets/data/publications.json"
THUMB_DIR = ROOT / "assets/images/pub-thumbs"
CACHE_DIR = ROOT / ".cache/pub-pdfs"
MAGICK = shutil.which("magick") or shutil.which("convert")

if not MAGICK:
    sys.exit("ImageMagick is required. Install with: brew install imagemagick")


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:80] or "paper"


def resolve_pdf_source(link: str) -> tuple[Path | str | None, str | None]:
    if not link or link == "#":
        return None, None

    if link.endswith(".pdf"):
        if link.startswith("http"):
            return link, Path(link).name
        local_path = ROOT / link.lstrip("./")
        if local_path.exists():
            return local_path, local_path.stem
        return None, None

    match = re.search(r"arxiv\.org/abs/(\d+\.\d+)(?:v\d+)?", link)
    if match:
        paper_id = match.group(1)
        return f"https://arxiv.org/pdf/{paper_id}.pdf", paper_id.replace(".", "-")

    return None, None


def download_pdf(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "jiali-home.github.io-thumb-generator/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        dest.write_bytes(response.read())


def ensure_pdf(source: Path | str, cache_name: str) -> Path | None:
    if isinstance(source, Path):
        return source

    cached = CACHE_DIR / f"{cache_name}.pdf"
    if cached.exists() and cached.stat().st_size > 0:
        return cached

    try:
        download_pdf(source, cached)
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        print(f"  ! download failed: {error}")
        return None

    return cached


def render_thumb(pdf_path: Path, thumb_path: Path) -> bool:
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        MAGICK,
        "-density",
        "144",
        f"{pdf_path}[0]",
        "-background",
        "white",
        "-alpha",
        "remove",
        "-quality",
        "88",
        "-resize",
        "400x",
        str(thumb_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        print(f"  ! render failed: {error.stderr.strip() or error}")
        return False
    return thumb_path.exists() and thumb_path.stat().st_size > 0


def main() -> int:
    publications = json.loads(PUBS_PATH.read_text(encoding="utf-8"))
    updated = 0
    skipped = 0

    for entry in publications:
        title = entry.get("title", "paper")
        slug = slugify(title)
        thumb_rel = f"./assets/images/pub-thumbs/{slug}.jpg"
        thumb_path = ROOT / thumb_rel.lstrip("./")

        source, cache_name = resolve_pdf_source(entry.get("link", ""))
        if not source or not cache_name:
            print(f"- skip (no PDF): {title}")
            skipped += 1
            continue

        print(f"* {title}")
        pdf_path = ensure_pdf(source, cache_name)
        if not pdf_path:
            skipped += 1
            continue

        if not render_thumb(pdf_path, thumb_path):
            skipped += 1
            continue

        entry["catalogThumb"] = thumb_rel
        updated += 1
        print(f"  -> {thumb_rel}")

    PUBS_PATH.write_text(json.dumps(publications, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nDone. Generated {updated} thumbnails, skipped {skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
