"""Day 06 pipeline entry point.

Usage (from starter/):
    python -m src.run prepare    # read + normalize corpus, write manifest/units, chunk stats
    python -m src.run validate   # verify every benchmark evidence quote exists on its gold unit
    python -m src.run ingest     # embed + upsert A and B into Chroma (run twice to check no dupes)
    python -m src.run evaluate   # run A/B/C on all questions, save traces + metrics
    python -m src.run all        # prepare -> validate -> ingest -> ingest -> evaluate
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .chunking import by_paragraph, fixed_size
from .evaluation import evaluate_query, norm, summarize
from .ingestion import MIN_UNIT_CHARS, load_rst
from .models import Chunk, DocumentPage

ROOT = Path(__file__).resolve().parents[1]          # starter/
REPO = ROOT.parent
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
BENCH = REPO / "benchmark" / "questions.json"
CHROMA_PATH = ROOT / "chroma_db"
SOURCE_URL = "https://raw.githubusercontent.com/python/peps/main/peps/{name}"

STRATEGIES = {
    "A": {"chunking": "fixed_size", "size": 500, "overlap": 50, "collection": "A", "use_filter": False},
    "B": {"chunking": "by_paragraph", "max_size": 1000, "collection": "B", "use_filter": False},
    "C": {"chunking": "by_paragraph", "max_size": 1000, "collection": "B", "use_filter": True},
}
TOP_K = 3


def load_corpus() -> tuple[list[DocumentPage], list[dict], list[str]]:
    pages, manifest, log = [], [], []
    files = sorted(RAW.glob("pep-*.rst"))
    if not files:
        sys.exit(f"No corpus files in {RAW}. See data/manifest.json for how to download them.")
    for f in files:
        try:
            p, m, lg = load_rst(f, SOURCE_URL.format(name=f.name))
        except Exception as exc:  # record and continue; never silently drop
            log.append(f"ERROR reading {f.name}: {exc!r}")
            continue
        m["local_path"] = f"starter/data/raw/{f.name}"
        pages += p
        manifest.append(m)
        log += lg
    return pages, manifest, log


def build_chunks(pages: list[DocumentPage], strategy: str) -> list[Chunk]:
    cfg = STRATEGIES[strategy]
    out: list[Chunk] = []
    for p in pages:
        out += fixed_size(p, cfg["size"], cfg["overlap"]) if cfg["chunking"] == "fixed_size" \
            else by_paragraph(p, cfg["max_size"])
    return out


def cmd_prepare() -> list[DocumentPage]:
    RESULTS.mkdir(exist_ok=True)
    pages, manifest, log = load_corpus()
    counted = sum(u["counted"] for m in manifest for u in m["units"])
    accessed = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    doc = {
        "corpus": "Python Enhancement Proposals (selected), reStructuredText sources",
        "repository": "https://github.com/python/peps (branch main)",
        "accessed_at": accessed,
        "page_rule": (f"Plain-text corpus without physical pages: each document is split in order at "
                      f"section boundaries into units of >= {MIN_UNIT_CHARS} normalized characters "
                      "(short tail merged into the previous unit). A unit = one 'page'. Sections "
                      "References/Copyright/Footnotes/Acknowledgements are excluded as non-content."),
        "documents": len(manifest), "content_units_counted": counted,
        "total_normalized_chars": sum(len(p.text) for p in pages),
        "sources": manifest,
    }
    (REPO / "data").mkdir(exist_ok=True)
    (REPO / "data" / "manifest.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    with (RESULTS / "units.jsonl").open("w", encoding="utf-8") as fh:
        for p in pages:
            fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    (RESULTS / "ingest_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    stats = {}
    for s in ("A", "B"):
        ch = build_chunks(pages, s)
        lens = [len(c.text) for c in ch]
        stats[s] = {"chunks": len(ch), "total_chars": sum(lens), "min": min(lens), "max": max(lens),
                    "mean": round(statistics.mean(lens), 1), "median": statistics.median(lens),
                    "p90": sorted(lens)[int(0.9 * len(lens))]}
    (RESULTS / "chunk_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"documents={len(manifest)} units={len(pages)} counted_units={counted}")
    print(json.dumps(stats, indent=2))
    if counted < 30:
        sys.exit("Corpus has fewer than 30 content units")
    return pages


def load_questions() -> list[dict]:
    return json.loads(BENCH.read_text(encoding="utf-8"))["questions"]


def cmd_validate() -> None:
    units = {(u["document_id"], u["page"]): u for u in
             map(json.loads, (RESULTS / "units.jsonl").read_text(encoding="utf-8").splitlines())}
    qs = load_questions()
    problems = []
    for q in qs:
        golds = [{"source_id": q["gold_source_id"], "page": q["gold_page"], "evidence_quote": q["evidence_quote"]}]
        golds += q.get("alt_gold", [])
        for g in golds:
            u = units.get((g["source_id"], g["page"]))
            if not u or norm(g["evidence_quote"]) not in norm(u["text"]):
                problems.append(f"{q['id']}: evidence not found on {g['source_id']} unit {g['page']}")
        if q.get("filter"):
            for k, v in q["filter"].items():
                if not any(u["metadata"].get(k) == v for u in units.values()):
                    problems.append(f"{q['id']}: filter value {k}={v!r} does not exist in corpus")
    splits = {s: sum(q["split"] == s for q in qs) for s in ("dev", "test")}
    filt = {s: sum(bool(q.get("filter")) and q["split"] == s for q in qs) for s in ("dev", "test")}
    print(f"questions={len(qs)} split={splits} with_filter={filt}")
    if problems:
        print("\n".join(problems))
        sys.exit("Benchmark validation FAILED")
    print("All evidence quotes verified on their gold units.")


def _embedder():
    from .embeddings import LocalEmbedder
    return LocalEmbedder()


def _store(emb):
    """VECTOR_STORE=pgvector (default; needs docker compose up) or chroma."""
    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    kind = os.getenv("VECTOR_STORE", "pgvector").lower()
    if kind == "chroma":
        from .store import ChromaStore
        return ChromaStore(str(CHROMA_PATH), emb.dimensions, emb.model_name)
    url = os.getenv("DATABASE_URL")
    if not url:
        sys.exit("DATABASE_URL missing: copy .env.example to .env (or set VECTOR_STORE=chroma)")
    from .store import PgVectorStore
    return PgVectorStore(url, emb.dimensions, emb.model_name)  # raises if DB is down: no silent fallback


def cmd_ingest(emb=None) -> dict:
    pages, _, _ = load_corpus()
    emb = emb or _embedder()
    store = _store(emb)
    report = {"store": type(store).__name__, "model": emb.model_name, "dimensions": emb.dimensions, "max_seq_length": emb.max_seq_length}
    for coll in ("A", "B"):
        chunks = build_chunks(pages, coll)
        t0 = time.perf_counter()
        before = store.count(coll)
        # only embed chunks that are new/changed (idempotent re-ingest)
        vectors = emb.embed_documents([c.text for c in chunks])
        t_embed = time.perf_counter() - t0
        res = store.upsert(coll, chunks, vectors)
        report[coll] = {"chunks": len(chunks), "count_before": before, "count_after": store.count(coll),
                        **res, "embed_seconds": round(t_embed, 2),
                        "total_seconds": round(time.perf_counter() - t0, 2)}
        too_long = sum(len(c.text) > emb.max_seq_length * 3 for c in chunks)
        report[coll]["chunks_possibly_truncated_by_model"] = too_long
    print(json.dumps(report, indent=2))
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "ingest_runs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), **report}) + "\n")
    return report


def cmd_evaluate(emb=None) -> None:
    emb = emb or _embedder()
    store = _store(emb)
    qs = load_questions()
    traces, timing = [], {}
    for s, cfg in STRATEGIES.items():
        t0 = time.perf_counter()
        for q in qs:
            applied = q.get("filter") if cfg["use_filter"] and q.get("filter") else None
            hits = store.search(cfg["collection"], emb.embed_query(q["question"]), TOP_K, applied)
            traces.append({"strategy": s, **evaluate_query(q, hits, applied)})
        timing[s] = round(time.perf_counter() - t0, 2)
    with (RESULTS / "traces.jsonl").open("w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    rows = summarize(traces)
    for r in rows:
        r["query_seconds_all_questions"] = timing[r["strategy"]]
    with (RESULTS / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # compact per-question table for quick reading
    with (RESULTS / "per_question.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "id", "split", "type", "filter_applied", "hit@1", "hit@3", "first_hit_rank",
                    "top1_chunk", "top2_chunk", "top3_chunk", "split_evidence_on_gold_page"])
        for t in traces:
            ids = [h["chunk_id"] for h in t["top3"]] + ["", "", ""]
            w.writerow([t["strategy"], t["id"], t["split"], t["type"], json.dumps(t["applied_filter"]),
                        int(t["hit_at_1"]), int(t["hit_at_3"]), t["first_hit_rank"], *ids[:3],
                        int(t["split_evidence_on_gold_page"])])
    print(f"{'strat':5} {'split':5} hit@1    hit@3")
    for r in rows:
        print(f"{r['strategy']:5} {r['split']:5} {r['hit1']}/{r['n']:<5} {r['hit3']}/{r['n']}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "prepare":
        cmd_prepare()
    elif cmd == "validate":
        cmd_validate()
    elif cmd == "ingest":
        cmd_ingest()
    elif cmd == "evaluate":
        cmd_evaluate()
    elif cmd == "all":
        cmd_prepare()
        cmd_validate()
        emb = _embedder()
        cmd_ingest(emb)
        cmd_ingest(emb)  # second pass must write 0 new chunks
        cmd_evaluate(emb)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
