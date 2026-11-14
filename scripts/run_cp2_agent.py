"""Run the real local-model agent on one question or a safe dev subset."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT))

from agentic import run_agent  # noqa: E402
from src.local_model import LocalInstructController  # noqa: E402
from src.retrieval import DenseChromaRetriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-id", action="append", help="Dev question ID; repeat as needed")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "cp2_agent_traces.json")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=3500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = json.loads((ROOT / "data" / "questions.json").read_text(encoding="utf-8"))
    requested = set(args.question_id or ["q01", "q07", "q21"])
    selected = [q for q in questions if q["id"] in requested and q["split"] == "dev"]
    if {q["id"] for q in selected} != requested:
        raise ValueError("Only existing dev question IDs may be used in CP2")

    retriever = DenseChromaRetriever(
        ROOT / "data" / "chunks.json",
        ROOT / "artifacts" / "chroma",
    )
    controller = LocalInstructController(args.model)
    traces = []
    for question in selected:
        started = time.perf_counter()
        try:
            result = run_agent(
                question["question"],
                controller,
                retriever,
                filters=question.get("filter"),
                max_searches=1,
                top_k=args.top_k,
                budget=args.budget,
            )
            result.update({"query_id": question["id"], "split": "dev", "model": args.model})
        except Exception as exc:
            result = {
                "query_id": question["id"],
                "split": "dev",
                "model": args.model,
                "status": "error",
                "answer": "",
                "citations": [],
                "events": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        traces.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(
        [{"query_id": row["query_id"], "status": row["status"], "citations": row["citations"],
          "elapsed_seconds": row["elapsed_seconds"]} for row in traces],
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
