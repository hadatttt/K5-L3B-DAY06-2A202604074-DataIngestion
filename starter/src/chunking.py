"""Fixed-size (strategy A) and structure-aware paragraph (strategy B/C) chunking.

Both strategies emit the same metadata fields so that A vs B differs only in
how the text is cut. `section` is the heading in effect at the chunk start.
"""
from __future__ import annotations

import bisect
import re

from .models import Chunk, DocumentPage


def _heading_index(page: DocumentPage) -> tuple[list[int], list[str]]:
    """Offsets where each section of the unit starts, with its heading."""
    offsets, names = [], []
    pos = 0
    for heading in page.metadata.get("unit_sections", []):
        if heading == "(preamble)":
            offsets.append(0)
            names.append(heading)
            continue
        found = page.text.find(heading, pos)
        if found >= 0:
            offsets.append(found)
            names.append(heading)
            pos = found + len(heading)
    if not offsets or offsets[0] != 0:
        offsets.insert(0, 0)
        names.insert(0, names[0] if names else "(preamble)")
    return offsets, names


def _section_at(offsets: list[int], names: list[str], pos: int) -> str:
    return names[max(0, bisect.bisect_right(offsets, pos) - 1)]


def _base_meta(page: DocumentPage) -> dict:
    keep = ("pep", "title", "pep_type", "status", "created_year", "python_version")
    return {k: page.metadata[k] for k in keep if k in page.metadata}


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Character windows of `size` with `overlap`, ignoring structure."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    offsets, names = _heading_index(page)
    text, step, chunks = page.text, size - overlap, []
    for i, start in enumerate(range(0, max(len(text) - overlap, 1), step)):
        piece = text[start:start + size].strip()
        if not piece:
            continue
        chunks.append(Chunk(
            chunk_id=f"{page.document_id}:p{page.page:03d}:c{i:03d}",
            document_id=page.document_id, source=page.source, page=page.page, text=piece,
            metadata={**_base_meta(page), "section": _section_at(offsets, names, start),
                      "char_start": start, "char_end": min(start + size, len(text))}))
    return chunks


def _split_long(par: str, max_size: int) -> list[str]:
    """Split an oversized paragraph on sentence, then line, then hard boundaries."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", par)
    if len(parts) == 1:
        parts = par.split("\n")
    out, cur = [], ""
    for p in parts:
        while len(p) > max_size:  # unbreakable run (e.g. long code line)
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:max_size])
            p = p[max_size:]
        sep = "\n" if "\n" in par and "." not in p[-1:] else " "
        if cur and len(cur) + len(sep) + len(p) > max_size:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}{sep}{p}" if cur else p
    if cur:
        out.append(cur)
    return out


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group blank-line separated paragraphs up to `max_size` chars.

    A new section heading always starts a new chunk, a heading is never left
    alone at the end of a chunk, and oversized paragraphs are split on
    sentence boundaries.
    """
    offsets, names = _heading_index(page)
    heading_set = set(page.metadata.get("unit_sections", []))
    paragraphs, pos = [], 0
    for par in re.split(r"\n\s*\n", page.text):
        start = page.text.find(par, pos)
        pos = start + len(par)
        if par.strip():
            paragraphs.append((start, par.strip()))

    groups: list[tuple[int, str]] = []
    cur_start, cur = 0, ""
    for start, par in paragraphs:
        is_heading = par in heading_set
        pieces = _split_long(par, max_size) if len(par) > max_size else [par]
        for j, piece in enumerate(pieces):
            starts_section = is_heading and j == 0
            cur_is_only_heading = cur in heading_set
            if cur and (starts_section or len(cur) + 2 + len(piece) > max_size) and not cur_is_only_heading:
                groups.append((cur_start, cur))
                cur = ""
            if not cur:
                cur_start = start
            cur = f"{cur}\n\n{piece}" if cur else piece
    if cur:
        groups.append((cur_start, cur))

    return [Chunk(
        chunk_id=f"{page.document_id}:p{page.page:03d}:c{i:03d}",
        document_id=page.document_id, source=page.source, page=page.page, text=text,
        metadata={**_base_meta(page), "section": _section_at(offsets, names, start),
                  "char_start": start, "char_end": start + len(text)})
        for i, (start, text) in enumerate(groups)]
