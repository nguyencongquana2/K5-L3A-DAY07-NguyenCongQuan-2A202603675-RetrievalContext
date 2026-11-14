"""Final held-out test run for the baseline and the config frozen in CP3."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT))

from agentic import run_agent  # noqa: E402
from baseline import pack_context  # noqa: E402
from src.local_model import LocalInstructController  # noqa: E402
from src.retrieval import DenseChromaRetriever, HybridRrfRetriever, TfidfRetriever  # noqa: E402


def is_hit(chunk: dict, gold: list[dict]) -> bool:
    return any(
        chunk["source_id"] == item["source_id"]
        and chunk["page"] == item["page"]
        and item["evidence"] in chunk["text"]
        for item in gold
    )


def retrieval_run(name: str, retriever, questions: list[dict], top_k: int, budget: int, use_filter: bool):
    traces = []
    for question in questions:
        started = time.perf_counter()
        applied_filter = question.get("filter") if use_filter else None
        results = retriever.search(question["question"], filters=applied_filter, top_k=top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        context, context_ids = pack_context(results, budget=budget)
        marked = [{**row, "hit": is_hit(row, question["gold"])} for row in results]
        traces.append(
            {
                "query_id": question["id"], "split": "test", "query": question["question"],
                "filter": applied_filter, "top_k": top_k, "context_budget_chars": budget,
                "gold": question["gold"], "results": marked, "context_ids": context_ids,
                "context_chars": len(context), "hit1": bool(marked and marked[0]["hit"]),
                "hit3": any(row["hit"] for row in marked[:3]),
                "context_hit": any(row["hit"] and row["id"] in context_ids for row in marked),
                "elapsed_ms": elapsed_ms,
            }
        )
    return traces, {
        "name": name, "split": "test", "n": len(traces), "top_k": top_k,
        "context_budget_chars": budget, "uses_question_filter": use_filter,
        "hit1": sum(row["hit1"] for row in traces),
        "hit3": sum(row["hit3"] for row in traces),
        "context_hit": sum(row["context_hit"] for row in traces),
        "median_ms": round(statistics.median(row["elapsed_ms"] for row in traces), 3),
        "service_cost_usd": 0,
    }


def agent_run(name: str, retriever, controller, questions: list[dict], top_k: int, budget: int, use_filter: bool):
    traces = []
    for number, question in enumerate(questions, start=1):
        started = time.perf_counter()
        try:
            result = run_agent(
                question["question"], controller, retriever,
                filters=question.get("filter") if use_filter else None,
                max_searches=1, top_k=top_k, budget=budget,
            )
            result.update({"query_id": question["id"], "split": "test"})
        except Exception as exc:
            result = {
                "query_id": question["id"], "split": "test", "status": "error", "answer": "",
                "citations": [], "events": [], "error": f"{type(exc).__name__}: {exc}",
            }
        result.update(
            {"model": controller.model_name, "configuration": name,
             "elapsed_seconds": round(time.perf_counter() - started, 3)}
        )
        traces.append(result)
        print(f"{name} [{number}/{len(questions)}] {question['id']}: {result['status']}", flush=True)
    return traces


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    output = ROOT / "results" / "cp4"
    output.mkdir(parents=True, exist_ok=True)
    questions = json.loads((ROOT / "data" / "questions.json").read_text(encoding="utf-8"))
    test = [question for question in questions if question["split"] == "test"]
    answerable = [question for question in test if question.get("answerable", True)]
    chunks_path = ROOT / "data" / "chunks.json"
    lexical = TfidfRetriever(chunks_path)
    dense = DenseChromaRetriever(chunks_path, ROOT / "artifacts" / "chroma")
    selected = HybridRrfRetriever(dense, lexical)

    baseline_retrieval, baseline_summary = retrieval_run(
        "tfidf_top3", lexical, answerable, top_k=3, budget=3500, use_filter=False
    )
    selected_retrieval, selected_summary = retrieval_run(
        "hybrid_rrf_top5_filter", selected, answerable, top_k=5, budget=5500, use_filter=True
    )
    write(output / "baseline_retrieval_traces.json", baseline_retrieval)
    write(output / "selected_retrieval_traces.json", selected_retrieval)
    write(output / "retrieval_summary.json", {
        "frozen_from_cp3": "hybrid_rrf_top5_filter",
        "baseline": baseline_summary,
        "selected": selected_summary,
        "test_used_for_tuning": False,
    })

    controller = LocalInstructController(min_answer_score=0.10)
    baseline_agent = agent_run(
        "tfidf_top3", lexical, controller, test, top_k=3, budget=3500, use_filter=False
    )
    controller.min_answer_score = 0.40
    selected_agent = agent_run(
        "hybrid_rrf_top5_filter", selected, controller, test, top_k=5, budget=5500, use_filter=True
    )
    write(output / "baseline_agent_traces.json", baseline_agent)
    write(output / "selected_agent_traces.json", selected_agent)
    print(json.dumps({"baseline": baseline_summary, "selected": selected_summary}, indent=2))


if __name__ == "__main__":
    main()
