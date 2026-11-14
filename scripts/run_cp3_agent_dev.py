"""Run the selected CP3 agent configuration on every dev case."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT))

from agentic import run_agent  # noqa: E402
from src.local_model import LocalInstructController  # noqa: E402
from src.retrieval import DenseChromaRetriever, HybridRrfRetriever, TfidfRetriever  # noqa: E402


def main() -> None:
    chunks_path = ROOT / "data" / "chunks.json"
    questions = json.loads((ROOT / "data" / "questions.json").read_text(encoding="utf-8"))
    dev = [question for question in questions if question["split"] == "dev"]
    dense = DenseChromaRetriever(chunks_path, ROOT / "artifacts" / "chroma")
    retriever = HybridRrfRetriever(dense, TfidfRetriever(chunks_path))
    controller = LocalInstructController()
    traces = []
    for number, question in enumerate(dev, start=1):
        started = time.perf_counter()
        try:
            result = run_agent(
                question["question"],
                controller,
                retriever,
                filters=question.get("filter"),
                max_searches=1,
                top_k=5,
                budget=5500,
            )
            result.update({"query_id": question["id"], "split": "dev"})
        except Exception as exc:
            result = {
                "query_id": question["id"],
                "split": "dev",
                "status": "error",
                "answer": "",
                "citations": [],
                "events": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        result.update(
            {
                "model": controller.model_name,
                "configuration": "hybrid_rrf_top5_filter",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        traces.append(result)
        print(f"[{number}/{len(dev)}] {question['id']}: {result['status']}", flush=True)

    output = ROOT / "results" / "cp3" / "agent_dev_traces.json"
    output.write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(traces)} traces to {output}")


if __name__ == "__main__":
    main()
