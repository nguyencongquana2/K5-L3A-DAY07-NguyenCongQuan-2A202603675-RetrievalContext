"""Export the selected Day 6 corpus and benchmark into the Day 7 contract.

Run this script with the Python environment from Day 6 because it reads the
persistent Chroma collection. It does not expose embeddings or secrets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import chromadb


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "python-tutorial-3.13.7"
SOURCE_PATH = "data/source/tutorial.pdf"

UNANSWERABLE_CASES = [
    {
        "id": "q21",
        "split": "dev",
        "question": "What is the default connection timeout used by pip when downloading a package?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "The corpus is the Python Tutorial 3.13.7. It mentions pip usage but does not "
            "specify pip's default network connection timeout."
        ),
    },
    {
        "id": "q22",
        "split": "dev",
        "question": "Which pytest option reruns only the tests that failed in the previous run?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "The Python Tutorial briefly discusses testing but does not document pytest "
            "command-line options."
        ),
    },
    {
        "id": "q23",
        "split": "test",
        "question": "What is NumPy's default floating-point dtype on this platform?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "NumPy implementation defaults are outside the Python Tutorial corpus, and the "
            "corpus does not provide a platform-specific NumPy dtype."
        ),
    },
    {
        "id": "q24",
        "split": "test",
        "question": "What is the default maximum pool size used by SQLAlchemy's QueuePool?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "SQLAlchemy connection-pool configuration is not covered by the Python Tutorial "
            "3.13.7 corpus."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Day 7 checkpoint 1 from Day 6")
    parser.add_argument("--day6-root", required=True, type=Path)
    parser.add_argument("--collection", default="day6_b")
    parser.add_argument(
        "--chroma-path",
        type=Path,
        help="Optional writable copy of Day 6 .chroma (useful in a read-only environment)",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_chunks(chroma_path: Path, collection_name: str) -> list[dict[str, Any]]:
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(collection_name)
    result = collection.get(include=["documents", "metadatas"])
    rows = []
    for chunk_id, text, metadata in zip(
        result["ids"], result["documents"], result["metadatas"]
    ):
        metadata = dict(metadata or {})
        rows.append(
            {
                "id": chunk_id,
                "text": text,
                "source_id": metadata.pop("document_id", SOURCE_ID),
                "source": SOURCE_PATH,
                "page": metadata.pop("page"),
                "metadata": metadata,
            }
        )
    return sorted(rows, key=lambda row: row["id"])


def exact_evidence_span(evidence: str, candidates: list[str]) -> str:
    """Return the exact chunk substring matching evidence modulo layout whitespace."""
    tokens = re.split(r"\s+", evidence.strip())
    pattern = re.compile(r"\s+".join(re.escape(token) for token in tokens), re.IGNORECASE)
    for text in candidates:
        match = pattern.search(text)
        if match:
            return match.group(0)
    raise ValueError(f"Evidence does not resolve to an exact chunk span: {evidence}")


def convert_questions(day6_root: Path, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = read_json(day6_root / "benchmark" / "questions.json")
    chunks_by_location: dict[tuple[str, int], list[str]] = {}
    for chunk in chunks:
        location = (chunk["source_id"], chunk["page"])
        chunks_by_location.setdefault(location, []).append(chunk["text"])
    converted = []
    for item in raw:
        location = (item["gold_source_id"], item["gold_page"])
        evidence = exact_evidence_span(item["evidence_quote"], chunks_by_location.get(location, []))
        question = {
            "id": item["id"],
            "split": item["split"],
            "question": item["question"],
            "answerable": True,
            "expected_answer": item["expected_answer"],
            "gold": [
                {
                    "source_id": item["gold_source_id"],
                    "page": item["gold_page"],
                    "evidence": evidence,
                }
            ],
        }
        if item.get("filter"):
            question["filter"] = item["filter"]
        converted.append(question)
    return converted + UNANSWERABLE_CASES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    day6_root = args.day6_root.resolve()
    source_pdf = day6_root / "data" / "source" / "tutorial.pdf"
    if not source_pdf.exists():
        raise FileNotFoundError(source_pdf)

    chroma_path = (args.chroma_path or day6_root / "starter" / ".chroma").resolve()
    chunks = export_chunks(chroma_path, args.collection)
    questions = convert_questions(day6_root, chunks)
    destination_pdf = ROOT / SOURCE_PATH
    destination_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_pdf, destination_pdf)

    manifest6 = read_json(day6_root / "data" / "manifest.json")
    stats6 = read_json(day6_root / "results" / "ingestion_stats.json")
    strategy = "B"
    manifest = {
        "checkpoint": 1,
        "prepared_at": date.today().isoformat(),
        "source": {
            "source_id": SOURCE_ID,
            "title": manifest6["title"],
            "release": manifest6["snapshot"]["release"],
            "path": SOURCE_PATH,
            "sha256": sha256(destination_pdf),
            "license": manifest6["license_or_permission"],
            "physical_pages": manifest6["counting_rule"]["physical_pages"],
            "content_pages": manifest6["counting_rule"]["content_page_count"],
        },
        "chunking": {
            "selected_day6_strategy": strategy,
            "method": "PDF text blocks grouped by paragraph",
            "max_characters": 1000,
            "chunk_count": len(chunks),
            "cross_page_chunks": False,
        },
        "embedding": {
            "model": stats6["model"],
            "dimensions": stats6["dimensions"],
            "normalized": True,
        },
        "vector_store": {
            "engine": stats6["vector_store"],
            "metric": stats6["metric"],
            "day6_collection": args.collection,
            "persistent_index_committed": False,
            "rebuild_input": "data/chunks.json",
            "note": "The local Day 6 Chroma index is not committed; rebuild it from chunks with the same embedding model.",
        },
        "benchmark": {
            "path": "data/questions.json",
            "gold_is_evaluation_only": True,
            "dev": {"answerable": 15, "unanswerable": 2},
            "test": {"answerable": 5, "unanswerable": 2},
        },
    }

    write_json(ROOT / "data" / "chunks.json", chunks)
    write_json(ROOT / "data" / "questions.json", questions)
    write_json(ROOT / "data" / "corpus_manifest.json", manifest)
    print(f"Exported {len(chunks)} chunks and {len(questions)} questions.")


if __name__ == "__main__":
    main()
