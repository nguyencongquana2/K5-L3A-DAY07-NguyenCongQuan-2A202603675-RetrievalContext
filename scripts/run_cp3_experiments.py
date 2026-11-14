"""Compare retrieval methods and improvement configurations on dev only."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT))

from baseline import pack_context  # noqa: E402
from src.retrieval import DenseChromaRetriever, HybridRrfRetriever, TfidfRetriever  # noqa: E402


def is_hit(chunk: dict, gold: list[dict]) -> bool:
    return any(
        chunk["source_id"] == item["source_id"]
        and chunk["page"] == item["page"]
        and item["evidence"] in chunk["text"]
        for item in gold
    )


def main() -> None:
    chunks_path = ROOT / "data" / "chunks.json"
    questions = json.loads((ROOT / "data" / "questions.json").read_text(encoding="utf-8"))
    dev = [q for q in questions if q["split"] == "dev" and q.get("answerable", True)]

    lexical = TfidfRetriever(chunks_path)
    dense = DenseChromaRetriever(chunks_path, ROOT / "artifacts" / "chroma")
    hybrid = HybridRrfRetriever(dense, lexical)
    configurations = [
        {"name": "tfidf_top3", "retriever": lexical, "top_k": 3, "budget": 3500, "filter": False,
         "role": "baseline"},
        {"name": "dense_top3", "retriever": dense, "top_k": 3, "budget": 3500, "filter": False,
         "role": "method"},
        {"name": "hybrid_rrf_top3", "retriever": hybrid, "top_k": 3, "budget": 3500, "filter": False,
         "role": "method"},
        {"name": "dense_top5_filter", "retriever": dense, "top_k": 5, "budget": 5500, "filter": True,
         "role": "improvement"},
        {"name": "hybrid_rrf_top5_filter", "retriever": hybrid, "top_k": 5, "budget": 5500, "filter": True,
         "role": "improvement"},
    ]

    output = ROOT / "results" / "cp3"
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for config in configurations:
        traces = []
        for question in dev:
            started = time.perf_counter()
            applied_filter = question.get("filter") if config["filter"] else None
            results = config["retriever"].search(
                question["question"], filters=applied_filter, top_k=config["top_k"]
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            context, context_ids = pack_context(results, budget=config["budget"])
            marked = [{**row, "hit": is_hit(row, question["gold"])} for row in results]
            traces.append(
                {
                    "query_id": question["id"],
                    "split": "dev",
                    "query": question["question"],
                    "filter": applied_filter,
                    "top_k": config["top_k"],
                    "context_budget_chars": config["budget"],
                    "gold": question["gold"],
                    "results": marked,
                    "context_ids": context_ids,
                    "context_chars": len(context),
                    "hit1": bool(marked and marked[0]["hit"]),
                    "hit3": any(row["hit"] for row in marked[:3]),
                    "context_hit": any(row["hit"] and row["id"] in context_ids for row in marked),
                    "elapsed_ms": elapsed_ms,
                }
            )
        n = len(traces)
        summary = {
            "name": config["name"],
            "role": config["role"],
            "split": "dev",
            "n": n,
            "top_k": config["top_k"],
            "context_budget_chars": config["budget"],
            "uses_question_filter": config["filter"],
            "hit1": sum(row["hit1"] for row in traces),
            "hit3": sum(row["hit3"] for row in traces),
            "context_hit": sum(row["context_hit"] for row in traces),
            "median_ms": round(statistics.median(row["elapsed_ms"] for row in traces), 3),
            "service_cost_usd": 0,
        }
        summaries.append(summary)
        (output / f"{config['name']}_traces.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    # Selection criterion declared before test: Hit@3, Hit@1, context rate, then lower latency.
    selected = max(
        summaries,
        key=lambda row: (row["hit3"], row["hit1"], row["context_hit"], -row["median_ms"]),
    )
    result = {
        "selection_criterion": "maximize hit3, then hit1, then context_hit, then minimize median_ms",
        "summaries": summaries,
        "selected_config": selected["name"],
        "test_accessed": False,
    }
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
