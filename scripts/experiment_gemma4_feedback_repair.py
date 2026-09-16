#!/usr/bin/env python3
"""Repair five audit-failed judgments with Gemma 4 using Gemini feedback.

The experiment is checkpointed and isolated from production outputs.  It asks
Gemma for a corrected analysis plan, renders it through the existing
deterministic judgment pipeline, and preserves both the old audit and the
model-level repair context for human comparison.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts import deidentify_linked_judgments as judgment
    from scripts import hybrid_deidentify_judgments_with_google_ai as judgment_hybrid
    from scripts import hybrid_deidentify_with_google_ai as hybrid
    from scripts import pair_case_integration as pairing
except (ModuleNotFoundError, ImportError):
    import deidentify_linked_judgments as judgment
    import hybrid_deidentify_judgments_with_google_ai as judgment_hybrid
    import hybrid_deidentify_with_google_ai as hybrid
    import pair_case_integration as pairing


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/raw/linked_judgments/linked_judgments_114.jsonl"
DEFAULT_INDICTMENTS = ROOT / "data/intermediate/google_ai/hybrid_gemma4_experiment.jsonl"
DEFAULT_JUDGMENTS = ROOT / "data/intermediate/google_ai/hybrid_judgments_gemma4_experiment.jsonl"
DEFAULT_AUDITS = ROOT / "data/intermediate/google_ai/pair_audit_gemini38.jsonl"
DEFAULT_OUTPUT = ROOT / "data/intermediate/google_ai/gemma4_feedback_repair_experiment.jsonl"
DEFAULT_FAILURES = ROOT / "data/intermediate/google_ai/gemma4_feedback_repair_failures.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma feedback-guided judgment repair experiment")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--indictments", type=Path, default=DEFAULT_INDICTMENTS)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--audits", type=Path, default=DEFAULT_AUDITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--key-name", default="GOOGLE_STUDIO_API_KEY")
    parser.add_argument("--model", default="gemma-4-31b-it", choices=("gemma-4-31b-it",))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-transient-attempts", type=int, default=3)
    parser.add_argument("--retry-seconds", type=float, default=20.0)
    parser.add_argument(
        "--max-prompt-chars", type=int, default=28000,
        help="skip repair prompts unlikely to fit the key's per-minute input-token quota",
    )
    return parser.parse_args()


def select_failed_cases(
    audits: list[dict], judgments: list[dict], existing_ids: set[str], failure_ids: set[str]
) -> list[tuple[dict, dict]]:
    failed = {
        str(row.get("pair_id")): row
        for row in audits
        if row.get("effective_decision") == "fail"
    }
    return [
        (row, failed[str(row.get("pair_id"))])
        for row in judgments
        if str(row.get("pair_id")) in failed
        and str(row.get("pair_id")) not in existing_ids | failure_ids
    ]


def repair_prompt(source: str, baseline: dict, indictment: dict, audit: dict) -> str:
    original_prompt = judgment_hybrid.analysis_prompt(
        source, baseline.get("pair_alias_registry") or {}
    )
    context = {
        "same_case_indictment": {
            "crime_facts_summary": indictment.get("crime_facts_summary", []),
            "evidence": indictment.get("evidence", []),
            "persons": (indictment.get("analysis_plan") or {}).get("persons", []),
            "organizations": (indictment.get("analysis_plan") or {}).get("organizations", []),
        },
        "previous_gemma_judgment": {
            "analysis_plan": baseline.get("analysis_plan", {}),
        },
        "gemini_3_8_audit_feedback": audit.get("audit_result", {}),
    }
    return original_prompt + """

【修正任務；本段優先於上方初次分析指示】
上方是原始判決分析任務。下方提供同案起訴書、前次 Gemma 結果，以及 Gemini 3.8 audit 的具體錯誤。
請重新閱讀原始判決，輸出一份完整、可取代舊版的 analysis plan JSON；不得只描述修改內容。

修正要求：
1. audit 指出的每一項錯誤都要逐項修正；無法由 analysis plan 修正者，寫入 uncertainties 並具體說明。
2. 同一人物跨起訴書與判決書必須填正確 linked_indictment_group_id，且同一人的全名、遮罩名、A-code 放在同組 mentions。
3. 不同人物、商號或機構不得共用會混淆的代號；同一實體不得拆成兩個代號。
4. 證據必須涵蓋判決書所有定罪與量刑資料。同一證據若也存在起訴書，canonical_key 必須逐字沿用起訴書該筆的 canonical_key；只有確實無法對齊時才用新的穩定鍵。
5. 犯罪日期、起訖分鐘、數量、酒測值等影響犯罪事實的資訊不得自行省略或改寫；隱私欄位仍依原規則遮罩。
6. 只輸出符合上方 schema 的單一 JSON 物件，不要輸出 Markdown 或解說。

修正上下文：
""" + json.dumps(context, ensure_ascii=False, separators=(",", ":"))


def augment_registry(pair_registry: dict, plan: dict) -> dict:
    """Record explicitly linked judgment mentions in an experiment registry."""
    registry = copy.deepcopy(pair_registry)
    by_id = {
        str(item.get("pair_person_id") or item.get("indictment_group_id")): item
        for item in registry.get("persons", [])
    }
    unique = registry.setdefault("unique_mentions", {})
    for person in plan.get("persons", []):
        if not isinstance(person, dict):
            continue
        linked_id = str(person.get("linked_indictment_group_id") or "")
        target = by_id.get(linked_id)
        if not target:
            continue
        judgment_mentions = target.setdefault("judgment_mentions", [])
        for raw_mention in person.get("mentions", []):
            mention = str(raw_mention).strip()
            if not mention:
                continue
            if mention not in judgment_mentions:
                judgment_mentions.append(mention)
            normalized = pairing.normalize_person_mention(mention)
            if normalized:
                unique[normalized] = {
                    "pair_person_id": target.get("pair_person_id"),
                    "alias": target.get("alias"),
                }
    return registry


def failure_record(baseline: dict, error: str, **details: object) -> dict:
    row = {
        "pair_id": baseline.get("pair_id"),
        "judgment_internal_id": baseline.get("judgment_internal_id"),
        "model": "gemma-4-31b-it",
        "error": error,
        "checkpointed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    row.update(details)
    return row


def main() -> None:
    args = parse_args()
    if args.limit < 1 or args.max_transient_attempts < 1:
        raise SystemExit("--limit and --max-transient-attempts must be at least 1")
    keys = dict(hybrid.load_api_keys())
    if args.key_name not in keys:
        raise SystemExit(f"API key name not found in .env: {args.key_name}")

    indictments = {str(row.get("doc_id")): row for row in hybrid.read_jsonl(args.indictments)}
    manifests = {
        judgment_hybrid.judgment_internal_id(row): row
        for row in hybrid.read_jsonl(args.manifest)
    }
    existing_rows = hybrid.read_jsonl(args.output)
    failed_rows = hybrid.read_jsonl(args.failures)
    existing = {str(row.get("pair_id")): row for row in existing_rows}
    failures = {str(row.get("pair_id")): row for row in failed_rows}
    terminal_failure_ids = {
        pair_id for pair_id, row in failures.items()
        if row.get("error_type") != "HTTP_429"
    }
    candidates = select_failed_cases(
        hybrid.read_jsonl(args.audits), hybrid.read_jsonl(args.judgments),
        set(existing), terminal_failure_ids,
    )
    output_order = list(existing)
    failure_order = list(failures)
    completed = 0
    remaining_target = max(0, args.limit - len(existing))
    print(
        f"candidates={len(candidates)} target_total={args.limit} "
        f"remaining={remaining_target} key={args.key_name} "
        f"model={args.model} feedback_repair=true", flush=True,
    )

    for candidate_index, (baseline, audit) in enumerate(candidates, 1):
        if completed >= remaining_target:
            break
        pair_id = str(baseline.get("pair_id") or "")
        judgment_id = str(baseline.get("judgment_internal_id") or "")
        indictment_id = str(baseline.get("linked_indictment_doc_id") or "")
        manifest = manifests.get(judgment_id)
        indictment = indictments.get(indictment_id)
        if not manifest or not indictment:
            failures[pair_id] = failure_record(baseline, "missing_pair_source")
            if pair_id not in failure_order:
                failure_order.append(pair_id)
            hybrid.write_jsonl(args.failures, failures, failure_order)
            continue

        source = judgment.normalize_judgment_text(str(manifest.get("text") or ""))
        prompt = repair_prompt(source, baseline, indictment, audit)
        if len(prompt) > args.max_prompt_chars:
            print(
                f"skipped-oversize candidate={candidate_index} {pair_id} "
                f"prompt_chars={len(prompt)}", flush=True,
            )
            continue
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        plan = None
        usage: dict = {}
        last_detail = ""
        transient_attempts = 0
        while plan is None:
            try:
                plan, usage = hybrid.request_json(
                    keys[args.key_name], args.model, prompt, timeout=900
                )
                break
            except urllib.error.HTTPError as error:
                last_detail = judgment_hybrid.error_summary(error)
                if error.code in {401, 403}:
                    raise SystemExit(
                        f"Authentication/authorization failed for {args.key_name}: "
                        f"HTTP {error.code} {last_detail}"
                    ) from error
                if error.code == 429:
                    retry_match = re.search(r"retry in ([0-9.]+)s", last_detail, re.I)
                    wait = float(retry_match.group(1)) if retry_match else args.retry_seconds
                    wait = max(wait + 2.0, args.retry_seconds)
                    print(
                        f"waiting HTTP 429; retry same case in {int(wait)} seconds",
                        flush=True,
                    )
                    time.sleep(wait)
                    continue
                if error.code not in judgment_hybrid.TRANSIENT_HTTP_CODES:
                    failures[pair_id] = failure_record(
                        baseline, "api_error", error_type=f"HTTP_{error.code}",
                        detail=last_detail, attempts=1,
                    )
                    break
                transient_attempts += 1
                if transient_attempts >= args.max_transient_attempts:
                    failures[pair_id] = failure_record(
                        baseline, "transient_api_error", error_type=f"HTTP_{error.code}",
                        detail=last_detail, attempts=transient_attempts,
                    )
                    break
                wait = args.retry_seconds * (2 ** (transient_attempts - 1))
                print(f"waiting HTTP {error.code}; retry in {int(wait)} seconds", flush=True)
                time.sleep(wait)
            except (TimeoutError, urllib.error.URLError, ConnectionError) as error:
                last_detail = str(error)[:800]
                transient_attempts += 1
                if transient_attempts >= args.max_transient_attempts:
                    failures[pair_id] = failure_record(
                        baseline, "network_error", error_type=type(error).__name__,
                        detail=last_detail, attempts=transient_attempts,
                    )
                    break
                wait = args.retry_seconds * (2 ** (transient_attempts - 1))
                print(f"waiting {type(error).__name__}; retry in {int(wait)} seconds", flush=True)
                time.sleep(wait)
            except (json.JSONDecodeError, ValueError, KeyError, StopIteration) as error:
                last_detail = str(error)[:800]
                failures[pair_id] = failure_record(
                    baseline, "invalid_model_response", error_type=type(error).__name__,
                    detail=last_detail, attempts=1,
                )
                break
        if plan is None:
            if pair_id not in failure_order:
                failure_order.append(pair_id)
            hybrid.write_jsonl(args.failures, failures, failure_order)
            print(f"skipped candidate={candidate_index} {pair_id} {last_detail[:120]}", flush=True)
            continue

        repaired_registry = augment_registry(baseline.get("pair_alias_registry") or {}, plan)
        case = {
            "judgment_internal_id": judgment_id,
            "manifest_record": manifest,
            "indictment_record": indictment,
        }
        record = judgment_hybrid.render_record(
            case, plan, usage, digest, args.model, repaired_registry, baseline,
        )
        record.update({
            "experiment": "gemma4_gemini38_feedback_repair",
            "baseline_model": baseline.get("model"),
            "baseline_doc_id": baseline.get("doc_id"),
            "baseline_audit": audit.get("audit_result"),
            "baseline_effective_decision": audit.get("effective_decision"),
            "feedback_included": True,
            "registry_augmented_from_explicit_links": True,
        })
        existing[pair_id] = record
        failures.pop(pair_id, None)
        if pair_id not in output_order:
            output_order.append(pair_id)
        completed += 1
        hybrid.write_jsonl(args.output, existing, output_order)
        if args.failures.exists():
            hybrid.write_jsonl(args.failures, failures, failure_order)
        print(
            f"processed total={len(existing)}/{args.limit} {pair_id} "
            f"replacements={record.get('applied_replacements')} "
            f"evidence={len(record.get('evidence') or [])}", flush=True,
        )

    print(json.dumps({
        "target_total": args.limit,
        "completed_this_run": completed,
        "completed_total": len(existing),
        "failures": len(failures),
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
