#!/usr/bin/env python3
"""Fetch BibTeX citations for publications from DOI and arXiv."""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBS_PATH = ROOT / "assets/data/publications.json"
BIBTEX_PATH = ROOT / "assets/data/publications-bibtex.json"
USER_AGENT = "jiali-home.github.io-bibtex-fetcher/1.0 (mailto:JXL220096@utdallas.edu)"


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:80] or "paper"


def collect_urls(entry: dict) -> list[str]:
    urls = []
    link = entry.get("link", "")
    if link and link != "#":
        urls.append(link)
    for item in entry.get("links", []) or []:
        url = item.get("url") or item.get("href")
        if url:
            urls.append(url)
    return urls


def extract_doi(urls: list[str]) -> str | None:
    for url in urls:
        match = re.search(r"doi\.org/(.+)$", url, re.I)
        if match:
            return match.group(1).strip("/")
        match = re.search(r"doi:\s*(\S+)", url, re.I)
        if match:
            return match.group(1).strip()
    return None


def extract_arxiv_id(urls: list[str], entry: dict) -> str | None:
    if entry.get("arxivId"):
        return str(entry["arxivId"]).strip()
    for url in urls:
        match = re.search(r"arxiv\.org/abs/(\d+\.\d+)(?:v\d+)?", url, re.I)
        if match:
            return match.group(1)
    return None


def fetch_doi_bibtex(doi: str) -> str | None:
    request = urllib.request.Request(
        f"https://doi.org/{doi}",
        headers={
            "Accept": "application/x-bibtex",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8", errors="replace").strip()
            return text or None
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        print(f"    ! DOI fetch failed: {error}")
        return None


def fetch_arxiv_bibtex(arxiv_id: str) -> str | None:
    request = urllib.request.Request(
        f"https://arxiv.org/bibtex/{arxiv_id}",
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8", errors="replace").strip()
            return text or None
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        print(f"    ! arXiv fetch failed: {error}")
        return None


def format_bibtex_key(authors: list[str], year: str, title: str) -> str:
    first_author = re.sub(r"[^a-z]", "", (authors[0] if authors else "author").split()[-1].lower())
    year_part = (year or "0000")[:4]
    title_word = re.sub(r"[^a-z0-9]", "", title.lower().split()[0] if title else "paper")
    return f"{first_author}{year_part}{title_word}"


def generate_fallback_bibtex(entry: dict) -> str:
    authors = entry.get("authors", [])
    author_field = " and ".join(authors) if authors else "Unknown"
    year = (entry.get("date") or "0000")[:4]
    title = entry.get("title", "Untitled")
    tag = entry.get("tag", "misc")
    key = format_bibtex_key(authors, year, title)
    entry_type = "article" if re.search(r"journal|proceedings|cvpr|aaai|neurips|icml", tag, re.I) else "misc"
    lines = [
        f"@{entry_type}{{{key},",
        f"  title={{{title}}},",
        f"  author={{{author_field}}},",
        f"  year={{{year}}},",
        f"  note={{{tag}}},",
        "}",
    ]
    return "\n".join(lines)


def prettify_bibtex(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    text = re.sub(r"\s+", " ", text)
    text = text.replace("}, ", "},\n  ")
    text = text.replace("{", "{\n  ", 1)
    if text.endswith("}"):
        text = text[:-1] + "\n}"
    return text


def resolve_bibtex(entry: dict) -> tuple[str, str]:
    if entry.get("bibtex"):
        return "manual", entry["bibtex"].strip()

    urls = collect_urls(entry)
    doi = entry.get("doi") or extract_doi(urls)
    if doi:
        bibtex = fetch_doi_bibtex(doi)
        if bibtex:
            return "doi", prettify_bibtex(bibtex)

    arxiv_id = extract_arxiv_id(urls, entry)
    if arxiv_id:
        bibtex = fetch_arxiv_bibtex(arxiv_id)
        if bibtex:
            return "arxiv", bibtex.strip()

    return "generated", generate_fallback_bibtex(entry)


def main() -> int:
    publications = json.loads(PUBS_PATH.read_text(encoding="utf-8"))
    output: dict[str, dict] = {}

    for entry in publications:
        title = entry.get("title", "paper")
        slug = slugify(title)
        print(f"* {title}")

        source, bibtex = resolve_bibtex(entry)
        output[slug] = {
            "title": title,
            "source": source,
            "bibtex": bibtex,
        }
        print(f"  -> {source}")
        time.sleep(0.4)

    BIBTEX_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {len(output)} entries to {BIBTEX_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
