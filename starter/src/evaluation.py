"""Evidence-based retrieval scoring (Hit@1 / Evidence Hit@3) with full traces.

A retrieved chunk is a hit when it comes from a gold (source_id, page) pair AND
contains the *whole* gold evidence quote after whitespace/case normalization.
Coherence and grounding are NOT inferred here; they are labelled by hand.
"""
from __future__ import annotations

import re
from typing import Any


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def gold_items(question: dict[str, Any]) -> list[dict[str, Any]]:
    """Primary gold plus any documented alternatives."""
    items = [{"source_id": question["gold_source_id"], "page": question["gold_page"],
              "evidence_quote": question["evidence_quote"]}]
    items += question.get("alt_gold", [])
    return items


def is_hit(hit: dict[str, Any], question: dict[str, Any]) -> bool:
    meta, text = hit["metadata"], norm(hit["text"])
    for g in gold_items(question):
        if meta.get("document_id") == g["source_id"] and int(meta.get("page")) == int(g["page"]) \
                and norm(g["evidence_quote"]) in text:
            return True
    return False


def partial_evidence(hit: dict[str, Any], question: dict[str, Any]) -> bool:
    """Diagnostic only: right source/page but evidence split across chunks."""
    meta = hit["metadata"]
    return any(meta.get("document_id") == g["source_id"] and int(meta.get("page")) == int(g["page"])
               for g in gold_items(question)) and not is_hit(hit, question)


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]],
                   applied_filter: dict[str, Any] | None = None) -> dict[str, Any]:
    flags = [is_hit(h, question) for h in hits]
    filter_ok = None
    if applied_filter:
        filter_ok = all(all(h["metadata"].get(k) == v for k, v in applied_filter.items()) for h in hits)
    return {
        "id": question["id"], "split": question["split"], "type": question.get("type"),
        "question": question["question"], "expected_answer": question["expected_answer"],
        "gold": gold_items(question), "question_filter": question.get("filter"),
        "applied_filter": applied_filter,
        "hit_at_1": bool(flags[:1] and flags[0]), "hit_at_3": any(flags[:3]),
        "first_hit_rank": flags.index(True) + 1 if True in flags else None,
        "split_evidence_on_gold_page": any(partial_evidence(h, question) for h in hits[:3]),
        "top3_filter_compliant": filter_ok,
        "top3": [{**h, "is_hit": f} for h, f in zip(hits, flags)],
    }


def summarize(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for strategy in sorted({t["strategy"] for t in traces}):
        for split in ("dev", "test", "all"):
            sub = [t for t in traces if t["strategy"] == strategy and (split == "all" or t["split"] == split)]
            if not sub:
                continue
            filt = [t for t in sub if t["applied_filter"]]
            rows.append({
                "strategy": strategy, "split": split, "n": len(sub),
                "hit1": sum(t["hit_at_1"] for t in sub), "hit3": sum(t["hit_at_3"] for t in sub),
                "hit1_rate": round(sum(t["hit_at_1"] for t in sub) / len(sub), 3),
                "hit3_rate": round(sum(t["hit_at_3"] for t in sub) / len(sub), 3),
                "filter_questions_with_filter_applied": len(filt),
                "filter_compliant": sum(bool(t["top3_filter_compliant"]) for t in filt),
                "split_evidence_cases": sum(t["split_evidence_on_gold_page"] and not t["hit_at_3"] for t in sub),
            })
    return rows
