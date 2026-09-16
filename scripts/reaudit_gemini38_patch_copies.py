#!/usr/bin/env python3
"""Bounded Gemini Flex re-audit of validated patched copies.

Dry-run is the default.  Execution is capped at two one-shot requests and
never mutates the baseline or patched-copy artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts import audit_case_pairs_with_google_ai as audit
    from scripts import hybrid_deidentify_judgments_with_google_ai as judgment_hybrid
    from scripts import hybrid_deidentify_with_google_ai as hybrid
except (ModuleNotFoundError, ImportError):
    import audit_case_pairs_with_google_ai as audit
    import hybrid_deidentify_judgments_with_google_ai as judgment_hybrid
    import hybrid_deidentify_with_google_ai as hybrid


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data/review/indictment_reviews.sqlite3"
DEFAULT_MANIFEST = ROOT / "data/raw/linked_judgments/linked_judgments_114.jsonl"
DEFAULT_COPIES = ROOT / "data/intermediate/google_ai/gemini38_validated_patch_copies.jsonl"
DEFAULT_OUTPUT = ROOT / "data/intermediate/google_ai/gemini38_patch_copy_reaudit.jsonl"
DEFAULT_FAILURES = ROOT / "data/intermediate/google_ai/gemini38_patch_copy_reaudit_failures.jsonl"
MAX_CASES = 2
MAX_OUTPUT_TOKENS = 8192
FLEX_INPUT_PER_M = 0.375
FLEX_OUTPUT_PER_M = 1.875
HARD_WORST_CASE_USD = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-audit patched copies with Gemini Flex")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--copies", type=Path, default=DEFAULT_COPIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--key-name", default="GOOGLE_STUDIO_API_KEY_4")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def raw_sources(db_path: Path, manifest_path: Path) -> tuple[dict[str, str], dict[str, dict]]:
    with sqlite3.connect(db_path) as connection:
        indictments = {
            str(doc_id): str(text or "")
            for doc_id, text in connection.execute("SELECT doc_id,normalized_text FROM cases")
        }
    return indictments, audit.load_manifest_sources(manifest_path)


def build_case(row: dict, indictments: dict[str, str], manifests: dict[str, dict]) -> dict:
    judgment_id = str(row["judgment_internal_id"])
    return {
        "pair_id": str(row["pair_id"]),
        "judgment_internal_id": judgment_id,
        "linked_indictment_doc_id": str(row["linked_indictment_doc_id"]),
        "indictment": row["indictment"],
        "judgment": row["judgment"],
        "raw_indictment": indictments[str(row["linked_indictment_doc_id"])],
        "raw_judgment": audit.judgment.normalize_judgment_text(
            str(manifests[judgment_id].get("text") or "")
        ),
    }


def failure(case: dict, error: Exception) -> dict:
    detail = str(error)
    if isinstance(error, urllib.error.HTTPError):
        detail = judgment_hybrid.error_summary(error)
    return {
        "pair_id": case["pair_id"],
        "error_type": type(error).__name__,
        "detail": detail[:1000],
        "model": "gemini-3.8-flash",
        "service_tier_requested": "flex",
        "checkpointed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    args = parse_args()
    if not 1 <= args.limit <= MAX_CASES:
        raise SystemExit(f"--limit must be between 1 and {MAX_CASES}")
    indictment_sources, manifests = raw_sources(args.db, args.manifest)
    rows = hybrid.read_jsonl(args.copies)[:args.limit]
    cases = [build_case(row, indictment_sources, manifests) for row in rows]
    prompts = [audit.audit_prompt(case) for case in cases]
    # UTF-8 bytes deliberately overestimate text tokenization.
    worst = (
        sum(len(prompt.encode("utf-8")) for prompt in prompts) / 1_000_000 * FLEX_INPUT_PER_M
        + len(prompts) * MAX_OUTPUT_TOKENS / 1_000_000 * FLEX_OUTPUT_PER_M
    )
    print(json.dumps({
        "execute": args.execute,
        "cases": len(cases),
        "requests": len(cases),
        "pessimistic_worst_case_usd": round(worst, 6),
    }, ensure_ascii=False, indent=2), flush=True)
    if worst > HARD_WORST_CASE_USD:
        raise SystemExit(
            f"worst-case cost ${worst:.6f} exceeds hard cap ${HARD_WORST_CASE_USD:.2f}"
        )
    if not args.execute:
        print("dry-run only; no API calls", flush=True)
        return

    keys = dict(hybrid.load_api_keys())
    if args.key_name not in keys:
        raise SystemExit(f"API key name not found: {args.key_name}")
    results: dict[str, dict] = {}
    failures: dict[str, dict] = {}
    for case, prompt in zip(cases, prompts):
        pair_id = case["pair_id"]
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        try:
            result, usage = hybrid.request_json(
                keys[args.key_name], "gemini-3.8-flash", prompt,
                thinking_level="medium", service_tier="flex", timeout=1200,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            result = audit.validate_audit_result(result)
            checks = audit.deterministic_checks(case)
            results[pair_id] = {
                "pair_id": pair_id,
                "model": "gemini-3.8-flash",
                "service_tier_requested": "flex",
                "service_tier_observed": usage.get("responseServiceTier"),
                "input_sha256": digest,
                "deterministic_checks": checks,
                "audit_result": result,
                **audit.derive_outcome(result, checks),
                "usage_metadata": usage,
                "checkpointed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            hybrid.write_jsonl(args.output, results, list(results))
            print(f"reaudited pair={pair_id} decision={results[pair_id]['effective_decision']}")
        except Exception as error:  # One shot: record and continue, never retry.
            failures[pair_id] = failure(case, error)
            hybrid.write_jsonl(args.failures, failures, list(failures))
            print(f"reaudit-failed pair={pair_id} type={type(error).__name__}")
    print(json.dumps({
        "completed": len(results), "failures": len(failures), "retries": 0,
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
