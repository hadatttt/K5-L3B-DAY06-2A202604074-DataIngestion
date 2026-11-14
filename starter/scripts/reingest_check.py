"""Re-ingest checks on pgvector, written to results/reingest_check.json.

1. Compare top-3 chunk ids of the pgvector run with the earlier Chroma run.
2. Simulate a changed document in a scratch strategy 'Z' (B chunking of PEP 257):
   ingest v1 -> ingest same v1 again (expect 0 written) -> ingest v2 where one
   paragraph is edited and the last unit is removed (expect rewritten +
   stale rows deleted) -> drop the scratch rows.
Run from starter/:  python -m scripts.reingest_check
"""
import json
from dataclasses import replace

from src.chunking import by_paragraph
from src.embeddings import LocalEmbedder
from src.run import RESULTS, _store, load_corpus

out = {}
pg = [json.loads(l) for l in (RESULTS / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
ch = [json.loads(l) for l in (RESULTS / "chroma_run" / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
key = lambda t: (t["strategy"], t["id"])
chm = {key(t): [h["chunk_id"] for h in t["top3"]] for t in ch}
diff = [f"{s}/{i}" for (s, i), ids in ((key(t), [h["chunk_id"] for h in t["top3"]]) for t in pg) if chm[(s, i)] != ids]
out["top3_identical_to_chroma"] = {"queries": len(pg), "different": diff}

emb = LocalEmbedder()
store = _store(emb)
pages, _, _ = load_corpus()
doc = [p for p in pages if p.document_id == "pep-0257"]
v1 = [c for p in doc for c in by_paragraph(p, 1000)]
p1 = replace(doc[0], text=doc[0].text.replace("Docstring", "Doc-string", 1))
v2 = [c for p in [p1, *doc[1:-1]] for c in by_paragraph(p, 1000)]  # edit unit 1, drop last unit
steps = {}
for label, chunks in (("v1_first", v1), ("v1_again", v1), ("v2_changed", v2)):
    res = store.upsert("Z", chunks, emb.embed_documents([c.text for c in chunks]))
    steps[label] = {**res, "chunks_in_version": len(chunks), "rows_after": store.count("Z")}
with store._conn() as con:
    con.execute("DELETE FROM day06_chunks WHERE strategy = 'Z'")
out["changed_document_simulation"] = steps
(RESULTS / "reingest_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
