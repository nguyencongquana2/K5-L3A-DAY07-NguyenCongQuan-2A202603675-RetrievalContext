"""Validate checkpoint 1 without loading the model or querying the test set."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> Any:
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    chunks = load("chunks.json")
    questions = load("questions.json")
    manifest = load("corpus_manifest.json")

    require(isinstance(chunks, list) and chunks, "chunks.json must be a non-empty array")
    require(isinstance(questions, list), "questions.json must be an array")
    chunk_ids = [item.get("id") for item in chunks]
    question_ids = [item.get("id") for item in questions]
    require(len(chunk_ids) == len(set(chunk_ids)), "chunk IDs must be unique")
    require(len(question_ids) == len(set(question_ids)), "question IDs must be unique")

    required_chunk_fields = {"id", "text", "source_id", "source", "page", "metadata"}
    for chunk in chunks:
        require(required_chunk_fields <= set(chunk), f"invalid chunk schema: {chunk.get('id')}")
        require(bool(chunk["text"].strip()), f"empty chunk: {chunk['id']}")
        require(isinstance(chunk["page"], int) and chunk["page"] > 0, f"invalid page: {chunk['id']}")

    counts = Counter((item["split"], bool(item.get("answerable", True))) for item in questions)
    expected = {("dev", True): 15, ("dev", False): 2, ("test", True): 5, ("test", False): 2}
    require(dict(counts) == expected, f"benchmark split mismatch: {dict(counts)}")

    chunks_by_location: dict[tuple[str, int], list[str]] = {}
    for chunk in chunks:
        key = (chunk["source_id"], chunk["page"])
        chunks_by_location.setdefault(key, []).append(normalized(chunk["text"]))

    for question in questions:
        if question.get("answerable", True):
            require(question.get("expected_answer"), f"missing expected answer: {question['id']}")
            require(question.get("gold"), f"missing gold evidence: {question['id']}")
            for gold in question["gold"]:
                location = (gold["source_id"], gold["page"])
                page_chunks = chunks_by_location.get(location, [])
                evidence = normalized(gold["evidence"])
                require(
                    any(evidence in text for text in page_chunks),
                    f"evidence for {question['id']} is not contained in one matching chunk",
                )
        else:
            require(question.get("expected_behavior") == "search_and_abstain", f"invalid abstention case: {question['id']}")
            require(question.get("absence_reason"), f"missing absence reason: {question['id']}")
            require("gold" not in question, f"unanswerable case must not contain gold: {question['id']}")

    pdf = ROOT / manifest["source"]["path"]
    require(pdf.exists(), f"source PDF not found: {pdf}")
    require(file_sha256(pdf) == manifest["source"]["sha256"], "source PDF checksum mismatch")
    require(len(chunks) == manifest["chunking"]["chunk_count"], "manifest chunk count mismatch")

    print("CP1 validation passed")
    print(f"- chunks: {len(chunks)}")
    print("- dev: 15 answerable + 2 unanswerable")
    print("- test: 5 answerable + 2 unanswerable (structure only; not retrieved)")
    print("- all 20 gold evidence spans occur in a matching source/page chunk")


if __name__ == "__main__":
    main()
