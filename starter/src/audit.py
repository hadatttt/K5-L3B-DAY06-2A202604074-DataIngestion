"""Export samples for manual audit (coherence, metadata, grounding).

Usage (from starter/, after `python -m src.run evaluate`):
    python -m src.audit

Sampling rules were fixed before looking at any score:
- Coherence: 10 source positions = middle character of unit 2 of each of the
  first 10 documents (sorted by id) that have >= 2 units; the chunk of each
  strategy that covers that position is exported. C uses B's chunks, so C's
  coherence equals B's by construction.
- Metadata + grounding: the 5 test questions x top-3 for every strategy.

The script only exports text and mechanical checks (does the chunk text occur
inside the cited unit? does pep_type match the manifest?). The 0/1/2 labels are
written by a human into results/manual_labels.json.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

from .run import RESULTS, STRATEGIES, build_chunks, load_corpus
from .evaluation import norm


def main() -> None:
    pages, manifest, _ = load_corpus()
    unit_by = {(p.document_id, p.page): p for p in pages}
    by_doc = defaultdict(list)
    for p in pages:
        by_doc[p.document_id].append(p)
    docs = [d for d in sorted(by_doc) if len(by_doc[d]) >= 2][:10]
    positions = [(d, 2, len(unit_by[(d, 2)].text) // 2) for d in docs]

    chunks = {s: build_chunks(pages, s) for s in ("A", "B")}
    with (RESULTS / "audit_coherence_samples.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "strategy", "document_id", "page", "char_pos", "chunk_id", "chunk_len",
                    "unit_text_around_pos", "chunk_text"])
        for i, (d, pg, pos) in enumerate(positions, start=1):
            unit = unit_by[(d, pg)].text
            for s in ("A", "B"):
                cover = [c for c in chunks[s] if c.document_id == d and c.page == pg
                         and c.metadata["char_start"] <= pos < c.metadata["char_end"]]
                c = cover[0]
                w.writerow([f"s{i:02d}", s, d, pg, pos, c.chunk_id, len(c.text),
                            unit[max(0, pos - 150):pos + 150], c.text])

    pep_type = {m["source_id"]: m["pep_type"] for m in manifest}
    traces = [json.loads(l) for l in (RESULTS / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    with (RESULTS / "audit_test_top3.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "id", "rank", "chunk_id", "document_id", "page", "section", "pep_type",
                    "auto_text_in_cited_unit", "auto_pep_type_matches_header", "auto_section_in_unit",
                    "applied_filter", "is_evidence_hit", "question", "expected_answer", "chunk_text"])
        for t in traces:
            if t["split"] != "test":
                continue
            for h in t["top3"]:
                m = h["metadata"]
                unit = unit_by.get((m["document_id"], m["page"]))
                w.writerow([t["strategy"], t["id"], h["rank"], h["chunk_id"], m["document_id"], m["page"],
                            m.get("section"), m.get("pep_type"),
                            bool(unit) and norm(h["text"]) in norm(unit.text),
                            m.get("pep_type") == pep_type.get(m["document_id"]),
                            bool(unit) and m.get("section") in unit.metadata["unit_sections"],
                            json.dumps(t["applied_filter"]), h["is_hit"], t["question"],
                            t["expected_answer"], h["text"]])
    print("wrote results/audit_coherence_samples.csv and results/audit_test_top3.csv")


if __name__ == "__main__":
    main()
