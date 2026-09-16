#!/usr/bin/env python3
"""Ask Gemini 3.8 Flex for validated, surgical pair-repair operations.

This is deliberately a planning experiment: it never overwrites either
de-identified corpus.  Each operation must carry a source-grounding quote and
passes local structural checks before it can be considered executable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
import unicodedata
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
DEFAULT_INDICTMENTS = ROOT / "data/intermediate/google_ai/hybrid_gemma4_experiment.jsonl"
DEFAULT_JUDGMENTS = ROOT / "data/intermediate/google_ai/hybrid_judgments_gemma4_experiment.jsonl"
DEFAULT_AUDITS = ROOT / "data/intermediate/google_ai/pair_audit_gemini38.jsonl"
DEFAULT_OUTPUT = ROOT / "data/intermediate/google_ai/gemini38_flex_patch_planner_experiment.jsonl"
DEFAULT_FAILURES = ROOT / "data/intermediate/google_ai/gemini38_flex_patch_planner_failures.jsonl"

ALLOWED_OPERATIONS = {
    "LINK_PERSON", "REGISTER_PERSON", "SET_ALIAS", "MASK_SPAN",
    "ADD_EVIDENCE", "MERGE_EVIDENCE", "RESTORE_FACT",
    "REPLACE_EVIDENCE", "REMOVE_REGISTRY_MENTION", "UPDATE_PERSON_ROLE",
    "MARK_FALSE_POSITIVE", "PATCH_RENDERER_RULE",
}
DOCUMENTS = {"indictment", "judgment", "pair"}
VALID_PERSON_ROLES = set(hybrid.ROLE_LABELS)
VALID_FIELD_ROOTS = {
    "text", "sections", "crime_facts_summary", "evidence",
    "paired_crime_facts_summary", "paired_evidence",
}
SAFE_ALIAS = re.compile(
    r"(?:被告|告訴人|被害人|證人|員警|人物|機構)[甲乙丙丁戊己庚辛壬癸]"
    r"|A\d{2,4}|〇+|○+"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemini Flex structured patch planner")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--indictments", type=Path, default=DEFAULT_INDICTMENTS)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--audits", type=Path, default=DEFAULT_AUDITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--key-name", default="GOOGLE_STUDIO_API_KEY_4")
    parser.add_argument("--model", default="gemini-3.8-flash", choices=("gemini-3.8-flash",))
    parser.add_argument("--service-tier", choices=("flex", "standard"), default="flex")
    parser.add_argument("--pair-id", action="append", dest="pair_ids")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-flex-attempts", type=int, default=3)
    parser.add_argument("--retry-seconds", type=float, default=20.0)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    return parser.parse_args()


def escape_pointer_token(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def pointer_value(record: object, pointer: str) -> object:
    if pointer == "":
        return record
    if not pointer.startswith("/"):
        raise KeyError(pointer)
    current = record
    for encoded in pointer[1:].split("/"):
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, dict):
            current = current[token]
        else:
            raise KeyError(pointer)
    return current


def collect_string_pointers(value: object, base: str) -> list[str]:
    if isinstance(value, str):
        return [base]
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            result.extend(collect_string_pointers(item, f"{base}/{index}"))
        return result
    if isinstance(value, dict):
        result = []
        for key, item in value.items():
            result.extend(collect_string_pointers(item, f"{base}/{escape_pointer_token(key)}"))
        return result
    return []


def allowed_field_paths(case: dict) -> dict[str, dict[str, list[str]]]:
    """Return only concrete, currently existing patch targets."""
    result: dict[str, dict[str, list[str]]] = {}
    for document in ("indictment", "judgment"):
        record = case[document]
        strings = []
        for root in sorted(VALID_FIELD_ROOTS):
            if root in record:
                strings.extend(collect_string_pointers(record[root], f"/{root}"))
        evidence_items = [
            f"/evidence/{index}" for index, item in enumerate(record.get("evidence", []))
            if isinstance(item, dict)
        ]
        role_fields = [
            f"/analysis_plan/persons/{index}/role"
            for index, item in enumerate((record.get("analysis_plan") or {}).get("persons", []))
            if isinstance(item, dict) and "role" in item
        ]
        result[document] = {
            "string_targets": sorted(set(strings)),
            "evidence_items": evidence_items,
            "person_role_fields": role_fields,
        }
    return result


def planner_prompt(case: dict, audit_row: dict) -> str:
    payload = audit.prompt_payload(case)
    # The paired structure is a deterministic union of the two document
    # structures and can be recomputed after patches. Omitting it materially
    # reduces Flex queue pressure without removing either source of truth.
    document_structures = payload.get("document_structures")
    if isinstance(document_structures, dict):
        document_structures.pop("paired", None)
    payload["audit_feedback"] = audit_row.get("audit_result", {})
    payload["allowed_field_paths"] = allowed_field_paths(case)
    return """你是臺灣法律資料去識別化的修正規劃器。你不得重寫全文，只能輸出可由程式驗證與執行的最小修正操作。

目標：逐項處理 audit feedback，同時維持犯罪事實、證據、人物代號及跨起訴書／判決書的一致性。人物暱稱只要不直接洩漏本名就保留，不得遮罩。

只能使用以下 op：
- LINK_PERSON：document, mentions, pair_person_id, source_quote, rationale
- REGISTER_PERSON：document, mentions, role, proposed_pair_person_id, alias, source_quote, rationale
- SET_ALIAS：document="pair", pair_person_id, alias, source_quote, rationale
- MASK_SPAN：document, field_path, old_text, replacement, source_quote, rationale
- ADD_EVIDENCE：document, evidence{name,category,quantity,proves,canonical_key}, source_quote, rationale
- REPLACE_EVIDENCE：document, field_path, evidence{name,category,quantity,proves,canonical_key}, source_quote, rationale
- MERGE_EVIDENCE：document="pair", indictment_name, judgment_name, canonical_key, indictment_source_quote, judgment_source_quote, rationale
- RESTORE_FACT：document, field_path, old_text, replacement, source_quote, rationale
- REMOVE_REGISTRY_MENTION：document, pair_person_id, mention, source_quote, rationale
- UPDATE_PERSON_ROLE：document, field_path, role, source_quote, rationale
- MARK_FALSE_POSITIVE：document, finding, rationale
- PATCH_RENDERER_RULE：document="pair", rule, source_quote, rationale（只有錯誤確定由通用 renderer 規則造成時使用）

硬性規則：
1. source_quote 必須逐字存在相應 raw source；pair 操作需要分別提供 indictment_source_quote 與 judgment_source_quote。
2. replacement 不得包含原始姓名、完整地址、電話、身分證、帳號或私人車牌。
3. LINK_PERSON 的 pair_person_id 必須已存在 registry；不存在者用 REGISTER_PERSON。
4. 同一證據跨文書只能用 MERGE_EVIDENCE，canonical_key 必須是簡短、穩定且不含個資的內容鍵。
5. 不可把不同人物或機構合併；不確定的項目放 unresolved，不得猜測。
6. 每一項 audit feedback 必須出現在 addressed_findings 或 unresolved。
7. field_path 只能逐字選用案件資料 allowed_field_paths 中對應 op 類型的 RFC 6901 JSON Pointer，不得自行組合欄位名稱。
8. 證據項內容錯誤使用 REPLACE_EVIDENCE，不可 ADD 後留下舊錯誤項；registry 含錯誤 mention 使用 REMOVE_REGISTRY_MENTION；人物角色錯誤使用 UPDATE_PERSON_ROLE。

輸出單一 JSON：
{"decision":"repairable|partially_repairable|false_positive|needs_human","addressed_findings":[{"finding":"...","operation_indexes":[0]}],"operations":[{"op":"LINK_PERSON",...}],"unresolved":[{"finding":"...","reason":"..."}]}

案件資料：
""" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def normalized_contains(container: str, quote: str) -> bool:
    return unicodedata.normalize("NFKC", quote).strip() in unicodedata.normalize(
        "NFKC", container
    )


def validate_operation(operation: object, case: dict) -> list[str]:
    if not isinstance(operation, dict):
        return ["operation_not_object"]
    errors: list[str] = []
    op = str(operation.get("op") or "")
    document = str(operation.get("document") or "")
    if op not in ALLOWED_OPERATIONS:
        errors.append("unsupported_op")
    if document not in DOCUMENTS:
        errors.append("invalid_document")
    raw = {
        "indictment": case.get("raw_indictment", ""),
        "judgment": case.get("raw_judgment", ""),
    }
    if document in raw:
        quote = str(operation.get("source_quote") or "")
        if not quote or not normalized_contains(str(raw[document]), quote):
            errors.append("source_quote_not_found")
    if op == "MERGE_EVIDENCE":
        for doc, key in (
            ("indictment", "indictment_source_quote"),
            ("judgment", "judgment_source_quote"),
        ):
            quote = str(operation.get(key) or "")
            if not quote or not normalized_contains(str(raw[doc]), quote):
                errors.append(f"{key}_not_found")
    if op == "LINK_PERSON":
        valid_ids = {
            str(item.get("pair_person_id") or item.get("indictment_group_id"))
            for item in (case["judgment"].get("pair_alias_registry") or {}).get("persons", [])
        }
        if str(operation.get("pair_person_id") or "") not in valid_ids:
            errors.append("unknown_pair_person_id")
        mentions = operation.get("mentions")
        if not isinstance(mentions, list) or not mentions:
            errors.append("missing_mentions")
        elif document in raw and any(
            not normalized_contains(str(raw[document]), str(mention)) for mention in mentions
        ):
            errors.append("mention_not_found")
    if op in {"REGISTER_PERSON", "SET_ALIAS"}:
        alias = str(operation.get("alias") or "")
        if not SAFE_ALIAS.fullmatch(alias):
            errors.append("unsafe_or_invalid_alias")
    if op == "REGISTER_PERSON":
        if str(operation.get("role") or "") not in VALID_PERSON_ROLES:
            errors.append("unsupported_person_role")
        proposed_id = str(operation.get("proposed_pair_person_id") or "")
        existing_ids = {
            str(item.get("pair_person_id") or item.get("indictment_group_id") or "")
            for item in (case["judgment"].get("pair_alias_registry") or {}).get("persons", [])
        }
        if not re.fullmatch(r"P\d{2,4}", proposed_id):
            errors.append("invalid_proposed_pair_person_id")
        elif proposed_id in existing_ids:
            errors.append("duplicate_pair_person_id")
        mentions = operation.get("mentions")
        if not isinstance(mentions, list) or not mentions:
            errors.append("missing_mentions")
        elif document in raw and any(
            not normalized_contains(str(raw[document]), str(mention)) for mention in mentions
        ):
            errors.append("mention_not_found")
    if op == "SET_ALIAS":
        valid_ids = {
            str(item.get("pair_person_id") or item.get("indictment_group_id") or "")
            for item in (case["judgment"].get("pair_alias_registry") or {}).get("persons", [])
        }
        if str(operation.get("pair_person_id") or "") not in valid_ids:
            errors.append("unknown_pair_person_id")
    paths = allowed_field_paths(case).get(document, {})
    if op in {"MASK_SPAN", "RESTORE_FACT"}:
        field_path = str(operation.get("field_path") or "")
        old_text = str(operation.get("old_text") or "")
        if not field_path or not old_text:
            errors.append("missing_patch_target")
        elif field_path not in paths.get("string_targets", []):
            errors.append("field_path_not_allowed")
        elif old_text not in str(pointer_value(case[document], field_path)):
                errors.append("old_text_not_found_in_clean_record")
        if not operation.get("replacement"):
            errors.append("missing_replacement")
    if op == "ADD_EVIDENCE" and not isinstance(operation.get("evidence"), dict):
        errors.append("missing_evidence_object")
    if op == "REPLACE_EVIDENCE":
        field_path = str(operation.get("field_path") or "")
        if field_path not in paths.get("evidence_items", []):
            errors.append("field_path_not_allowed")
        if not isinstance(operation.get("evidence"), dict):
            errors.append("missing_evidence_object")
    if op == "UPDATE_PERSON_ROLE":
        field_path = str(operation.get("field_path") or "")
        if field_path not in paths.get("person_role_fields", []):
            errors.append("field_path_not_allowed")
        if str(operation.get("role") or "") not in VALID_PERSON_ROLES:
            errors.append("unsupported_person_role")
    if op == "REMOVE_REGISTRY_MENTION":
        pair_person_id = str(operation.get("pair_person_id") or "")
        mention = str(operation.get("mention") or "")
        people = (case["judgment"].get("pair_alias_registry") or {}).get("persons", [])
        matches = [
            person for person in people
            if str(person.get("pair_person_id") or person.get("indictment_group_id") or "")
            == pair_person_id
        ]
        if not matches:
            errors.append("unknown_pair_person_id")
        elif mention not in matches[0].get("indictment_mentions", []):
            errors.append("registry_mention_not_found")
    return errors


def validate_plan(plan: object, case: dict) -> dict:
    if not isinstance(plan, dict):
        return {"pass": False, "errors": ["plan_not_object"], "operations": []}
    operations = plan.get("operations")
    if not isinstance(operations, list):
        return {"pass": False, "errors": ["operations_not_list"], "operations": []}
    results = []
    for index, operation in enumerate(operations):
        errors = validate_operation(operation, case)
        results.append({"index": index, "valid": not errors, "errors": errors})
    return {
        "pass": bool(operations) and all(item["valid"] for item in results),
        "valid_operations": sum(item["valid"] for item in results),
        "invalid_operations": sum(not item["valid"] for item in results),
        "operations": results,
    }


def failure_record(
    case: dict, error: str, service_tier: str = "flex", **details: object,
) -> dict:
    row = {
        "pair_id": case.get("pair_id"),
        "judgment_internal_id": case.get("judgment_internal_id"),
        "model": "gemini-3.8-flash",
        "service_tier_requested": service_tier,
        "error": error,
        "checkpointed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    row.update(details)
    return row


def main() -> None:
    args = parse_args()
    if args.limit < 1 or args.max_flex_attempts < 1:
        raise SystemExit("--limit and --max-flex-attempts must be at least 1")
    if not 1024 <= args.max_output_tokens <= 8192:
        raise SystemExit("--max-output-tokens must be between 1024 and 8192")
    keys = dict(hybrid.load_api_keys())
    if args.key_name not in keys:
        raise SystemExit(f"API key name not found in .env: {args.key_name}")
    with sqlite3.connect(args.db) as connection:
        indictment_sources = {
            str(doc_id): str(text or "")
            for doc_id, text in connection.execute("SELECT doc_id,normalized_text FROM cases")
        }
    cases = audit.build_cases(
        hybrid.read_jsonl(args.indictments), hybrid.read_jsonl(args.judgments),
        indictment_sources, audit.load_manifest_sources(args.manifest),
    )
    audits = {
        str(row.get("pair_id")): row for row in hybrid.read_jsonl(args.audits)
        if row.get("effective_decision") == "fail"
    }
    by_id = {case["pair_id"]: case for case in cases if case["pair_id"] in audits}
    requested_ids = args.pair_ids or list(by_id)
    selected = [by_id[pair_id] for pair_id in requested_ids if pair_id in by_id]

    existing_rows = hybrid.read_jsonl(args.output)
    failed_rows = hybrid.read_jsonl(args.failures)
    existing = {str(row.get("pair_id")): row for row in existing_rows}
    failures = {str(row.get("pair_id")): row for row in failed_rows}
    selected = [case for case in selected if case["pair_id"] not in existing][:args.limit]
    output_order = list(existing) + [case["pair_id"] for case in selected]
    failure_order = list(failures) + [case["pair_id"] for case in selected]
    print(
        f"selected={len(selected)} key={args.key_name} model={args.model} "
        f"service_tier={args.service_tier} patch_only=true", flush=True,
    )

    for index, case in enumerate(selected, 1):
        pair_id = case["pair_id"]
        prompt = planner_prompt(case, audits[pair_id])
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        plan = None
        usage: dict = {}
        detail = ""
        for attempt in range(1, args.max_flex_attempts + 1):
            try:
                plan, usage = hybrid.request_json(
                    keys[args.key_name], args.model, prompt,
                    thinking_level="medium", service_tier=args.service_tier, timeout=1200,
                    max_output_tokens=args.max_output_tokens,
                )
                break
            except urllib.error.HTTPError as error:
                detail = judgment_hybrid.error_summary(error)
                if error.code in {401, 403}:
                    raise SystemExit(
                        f"Authentication/authorization failed for {args.key_name}: "
                        f"HTTP {error.code} {detail}"
                    ) from error
                if error.code not in judgment_hybrid.TRANSIENT_HTTP_CODES or attempt == args.max_flex_attempts:
                    failures[pair_id] = failure_record(
                        case, f"{args.service_tier}_api_error",
                        service_tier=args.service_tier, error_type=f"HTTP_{error.code}",
                        detail=detail, attempts=attempt,
                    )
                    break
                wait = args.retry_seconds * (2 ** (attempt - 1))
                print(
                    f"waiting-{args.service_tier} HTTP {error.code}; "
                    f"retry in {int(wait)} seconds", flush=True,
                )
                time.sleep(wait)
            except (TimeoutError, urllib.error.URLError, ConnectionError) as error:
                detail = str(error)[:800]
                if attempt == args.max_flex_attempts:
                    failures[pair_id] = failure_record(
                        case, f"{args.service_tier}_network_error",
                        service_tier=args.service_tier, error_type=type(error).__name__,
                        detail=detail, attempts=attempt,
                    )
                    break
                wait = args.retry_seconds * (2 ** (attempt - 1))
                print(
                    f"waiting-{args.service_tier} {type(error).__name__}; "
                    f"retry in {int(wait)} seconds", flush=True,
                )
                time.sleep(wait)
            except (json.JSONDecodeError, ValueError, KeyError, StopIteration) as error:
                detail = str(error)[:800]
                failures[pair_id] = failure_record(
                    case, "invalid_model_response", service_tier=args.service_tier,
                    error_type=type(error).__name__, detail=detail,
                )
                break
        if plan is None:
            hybrid.write_jsonl(args.failures, failures, failure_order)
            print(f"skipped {index}/{len(selected)} {pair_id} {detail[:120]}", flush=True)
            continue
        validation = validate_plan(plan, case)
        row = {
            "pair_id": pair_id,
            "judgment_internal_id": case["judgment_internal_id"],
            "linked_indictment_doc_id": case["linked_indictment_doc_id"],
            "model": args.model,
            "service_tier_requested": args.service_tier,
            "service_tier_observed": usage.get("responseServiceTier"),
            "input_sha256": digest,
            "patch_plan": plan,
            "validation": validation,
            "baseline_audit": audits[pair_id].get("audit_result"),
            "read_only_experiment": True,
            "usage_metadata": usage,
            "checkpointed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        existing[pair_id] = row
        failures.pop(pair_id, None)
        hybrid.write_jsonl(args.output, existing, output_order)
        if args.failures.exists():
            hybrid.write_jsonl(args.failures, failures, failure_order)
        print(
            f"processed {index}/{len(selected)} {pair_id} "
            f"observed_tier={row['service_tier_observed']} "
            f"valid={validation.get('valid_operations', 0)} "
            f"invalid={validation.get('invalid_operations', 0)}", flush=True,
        )
    print(json.dumps({
        "requested": len(selected), "completed_total": len(existing),
        "failures": len(failures), "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
