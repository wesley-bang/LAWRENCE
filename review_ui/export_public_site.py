#!/usr/bin/env python3
"""Export a privacy-minimized static review snapshot for public hosting."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HYBRID = ROOT / "data/intermediate/google_ai/hybrid_gemma4_experiment.jsonl"
REVIEW_DB = ROOT / "data/review/indictment_reviews.sqlite3"
OUTPUT = Path(__file__).resolve().parent / "public_site" / "cases.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    with sqlite3.connect(REVIEW_DB) as connection:
        decisions = dict(connection.execute("SELECT doc_id,decision FROM reviews"))
        normalized_texts = dict(connection.execute("SELECT doc_id,normalized_text FROM cases"))

    public_cases = []
    for item in read_jsonl(HYBRID):
        if decisions.get(item["doc_id"]) == "pass":
            continue
        public_cases.append({
            "doc_id": item["doc_id"],
            "case_type": item.get("case_type", ""),
            "original_text": normalized_texts.get(item["doc_id"], ""),
            "text": item.get("text", ""),
            "crime_facts": item.get("crime_facts", ""),
            "crime_facts_summary": item.get("crime_facts_summary", []),
            "evidence": item.get("evidence", []),
            "checkpointed_at": item.get("checkpointed_at", ""),
        })

    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "privacy": "public_source_original_and_deidentified_candidate_no_notes_no_registry",
        "count": len(public_cases),
        "cases": public_cases,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    temporary.replace(OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "cases": len(public_cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
