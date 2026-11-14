"""Read and normalize PEP reStructuredText sources into traceable content units.

The corpus is plain text (no physical pages), so following GUIDE.md §1 each
document is split *in order* at section boundaries into units of at least
MIN_UNIT_CHARS normalized characters. A unit plays the role of a "page":
`page` = 1-based unit index inside the document; its section range and
character offsets are written to the manifest.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from .models import DocumentPage

MIN_UNIT_CHARS = 1500
# Sections that carry no answerable content (bibliography / boilerplate).
SKIP_SECTIONS = {"references", "references and footnotes", "copyright", "footnotes", "acknowledgements", "acknowledgments"}
HEADING_CHARS = set("=-~^\"'`#*+")

_HEADER_RE = re.compile(r"^([A-Za-z-]+):\s*(.*)$")


def normalize_text(text: str) -> str:
    """NFC-normalize, unify newlines/whitespace, strip RST inline markup.

    Paragraph breaks (blank lines) and code indentation are preserved because
    the paragraph chunker relies on them.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    text = text.replace(" ", " ").replace("​", "")
    # :pep:`634` -> PEP 634 ; :pep:`title <634>` -> title (PEP 634)
    text = re.sub(r":pep:`([^`<]+?)\s*<(\d+)[^>]*>`", r"\1 (PEP \2)", text)
    text = re.sub(r":pep:`(\d+)(#[^`]*)?`", r"PEP \1", text)
    # other roles  :ref:`x`, :class:`x` -> x  (keep link text only)
    text = re.sub(r":[a-z:]+:`([^`<]+?)\s*<[^>]+>`", r"\1", text)
    text = re.sub(r":[a-z:]+:`([^`]+)`", r"\1", text)
    # hyperlinks `text <url>`_ -> text ; `text`_ -> text
    text = re.sub(r"`([^`<]+?)\s*<[^>]+>`__?", r"\1", text)
    text = re.sub(r"`([^`]+)`__?", r"\1", text)
    # inline literals ``x`` -> x ; *emphasis* / **strong** kept as plain words
    text = re.sub(r"``([^`]+)``", r"\1", text)
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", text)
    # footnote refs [1]_ -> ""
    text = re.sub(r"\s?\[(\d+|#\w*|\*)\]_", "", text)
    # strip trailing spaces, collapse 3+ blank lines
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_header(raw: str) -> tuple[dict[str, str], str]:
    """Split the RFC-822 style PEP header from the body."""
    lines = raw.replace("\r\n", "\n").split("\n")
    header: dict[str, str] = {}
    key = None
    i = 0
    for i, line in enumerate(lines):
        if not line.strip():
            break
        m = _HEADER_RE.match(line)
        if m:
            key = m.group(1)
            header[key] = m.group(2).strip()
        elif key and line.startswith((" ", "\t")):
            header[key] += " " + line.strip()
    return header, "\n".join(lines[i + 1:])


def _is_underline(line: str, title: str) -> bool:
    s = line.strip()
    return bool(s) and len(set(s)) == 1 and s[0] in HEADING_CHARS and len(s) >= max(3, len(title.strip())) \
        and bool(title.strip()) and not title.startswith(" ")


def split_sections(body: str) -> list[tuple[str, str]]:
    """Return [(heading, text_including_heading)] in document order."""
    lines = body.split("\n")
    sections: list[tuple[str, list[str]]] = [("(preamble)", [])]
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines) and _is_underline(lines[i + 1], line) and not _is_underline(line, "x"):
            sections.append((line.strip(), [line.strip()]))
            i += 2
            continue
        sections[-1][1].append(line)
        i += 1
    return [(h, "\n".join(t)) for h, t in sections if "\n".join(t).strip()]


def _drop_directives(text: str) -> str:
    """Remove RST comments / footnote targets ('.. [1] url', '.. _x: url')."""
    out, skipping = [], False
    for line in text.split("\n"):
        if re.match(r"^\.\. (\[[^\]]+\]|_[^:]+:|$)", line):
            skipping = True
            continue
        if skipping and (line.startswith("   ") or not line.strip()):
            continue
        skipping = False
        # drop directive lines (code-block, rubric, canonical-typing-spec, ...)
        # and their ':option:' lines; the directive body (e.g. code) is kept
        if re.match(r"^\s*\.\. [\w-]+::", line) or re.match(r"^\s+:[\w-]+:( |$)", line):
            continue
        out.append(line)
    return "\n".join(out)


def load_rst(path: Path, source: str) -> tuple[list[DocumentPage], dict, list[str]]:
    """Load one PEP file -> (units, manifest_entry, log_messages)."""
    log: list[str] = []
    raw = path.read_text(encoding="utf-8")
    header, body = parse_header(raw)
    pep_no = int(header["PEP"])
    document_id = f"pep-{pep_no:04d}"
    meta = {
        "pep": pep_no,
        "title": header.get("Title", ""),
        "pep_type": header.get("Type", ""),
        "status": header.get("Status", ""),
        "created_year": int(re.search(r"(\d{4})", header.get("Created", "0000")).group(1)),
        "python_version": header.get("Python-Version", "") or "none",
    }
    sections = []
    for heading, text in split_sections(body):
        if heading.lower() in SKIP_SECTIONS:
            log.append(f"{document_id}: skipped non-content section '{heading}' ({len(text)} raw chars)")
            continue
        heading = normalize_text(heading)
        norm = normalize_text(_drop_directives(text))
        if not norm:
            log.append(f"{document_id}: empty section after normalization '{heading}'")
            continue
        sections.append((heading, norm))

    # group sections, in order, into units of >= MIN_UNIT_CHARS
    units: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    for sec in sections:
        cur.append(sec)
        if sum(len(t) for _, t in cur) >= MIN_UNIT_CHARS:
            units.append(cur)
            cur = []
    if cur:
        if units:
            units[-1].extend(cur)  # short tail merged into previous unit
        else:
            units.append(cur)
            log.append(f"{document_id}: whole document shorter than {MIN_UNIT_CHARS} chars")

    pages: list[DocumentPage] = []
    entries = []
    offset = 0
    for idx, unit in enumerate(units, start=1):
        text = "\n\n".join(t for _, t in unit)
        headings = [h for h, _ in unit]
        pages.append(DocumentPage(document_id=document_id, source=source, page=idx, text=text,
                                  metadata={**meta, "unit_sections": headings}))
        entries.append({"page": idx, "first_section": headings[0], "last_section": headings[-1],
                        "n_sections": len(headings), "char_start": offset,
                        "char_end": offset + len(text), "chars": len(text),
                        "counted": len(text) >= MIN_UNIT_CHARS})
        offset += len(text) + 2
    manifest = {
        "source_id": document_id,
        "title": meta["title"],
        "source_path_or_url": source,
        "license_or_permission": _license(raw),
        "pep_type": meta["pep_type"], "status": meta["status"],
        "created_year": meta["created_year"], "python_version": meta["python_version"],
        "raw_chars": len(raw),
        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "units": entries,
    }
    return pages, manifest, log


def _license(raw: str) -> str:
    tail = raw[-600:].lower()
    if "cc0" in tail:
        return "Public domain or CC0-1.0-Universal (stated in the document's Copyright section)"
    if "public domain" in tail:
        return "Public domain (stated in the document's Copyright section)"
    return "UNKNOWN - check manually"
