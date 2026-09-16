#!/usr/bin/env python3
"""Audit paired indictment/judgment de-identification with an independent LLM.

The auditor is read-only: it compares raw sources, rendered texts, structured
facts/evidence, and the pair alias registry, then writes restricted findings.
It never rewrites the source pipeline outputs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import sqlite3
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts import deidentify_judgments as deterministic
    from scripts import deidentify_linked_judgments as judgment
    from scripts import hybrid_deidentify_with_google_ai as hybrid
    from scripts import hybrid_deidentify_judgments_with_google_ai as judgment_hybrid
except (ModuleNotFoundError, ImportError):  # Direct execution adds scripts/.
    import deidentify_judgments as deterministic
    import deidentify_linked_judgments as judgment
    import hybrid_deidentify_with_google_ai as hybrid
    import hybrid_deidentify_judgments_with_google_ai as judgment_hybrid


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data/review/indictment_reviews.sqlite3"
DEFAULT_MANIFEST = ROOT / "data/raw/linked_judgments/linked_judgments_114.jsonl"
DEFAULT_INDICTMENTS = ROOT / "data/intermediate/google_ai/hybrid_gemma4_experiment.jsonl"
DEFAULT_JUDGMENTS = ROOT / "data/intermediate/google_ai/hybrid_judgments_gemma4_experiment.jsonl"
DEFAULT_OUTPUT = ROOT / "data/intermediate/google_ai/pair_audit_gemini38.jsonl"
DEFAULT_FAILURES = ROOT / "data/intermediate/google_ai/pair_audit_gemini38_failures.jsonl"
SUPPORTED_AUDIT_MODELS = {
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
}
HIGH_RISK_REGEX_KINDS = {"TW_ID", "EMAIL", "IP", "URL", "MOBILE", "LANDLINE", "PLATE"}
VALID_SECTION_STATUS = {"pass", "warning", "fail"}
VALID_DECISIONS = {"pass", "review", "fail"}
VALID_SEVERITIES = {"none", "low", "medium", "high"}
SAFE_ROLE_CODES = re.compile(r"(?:A\d+|[甲乙丙丁戊己庚辛壬癸]+[男女]?)", re.I)
MASK_ONLY = re.compile(r"[○〇ＯOo]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent paired de-identification and coverage audit"
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--indictments", type=Path, default=DEFAULT_INDICTMENTS)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--model", default="gemini-3.8-flash", choices=sorted(SUPPORTED_AUDIT_MODELS))
    parser.add_argument(
        "--key-name", dest="key_names", action="append",
        help="use only this .env key; repeat for multiple fixed workers (default: all)",
    )
    parser.add_argument("--scope", choices=("pending", "errors", "all"), default="pending")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--delay-seconds", type=float, default=8.0)
    parser.add_argument(
        "--thinking-level", choices=("low", "medium", "high"), default="medium",
        help="Gemini 3.8 audit reasoning level (minimal is not supported)",
    )
    parser.add_argument(
        "--retry-429-seconds", type=float, default=60.0,
        help="fallback wait for 429 when Google does not return a retry hint",
    )
    parser.add_argument(
        "--retry-transient-seconds", type=float, default=60.0,
        help="initial wait for 5xx/network errors; doubles separately up to 10 minutes",
    )
    parser.add_argument("--max-transient-attempts", type=int, default=4)
    parser.add_argument(
        "--max-prompt-chars", type=int, default=160000,
        help="defer unusually large paired prompts rather than truncate legal context",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="load and inspect eligible pairs without loading keys or calling the API",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_indictment_sources(db_path: Path) -> dict[str, str]:
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute("SELECT doc_id,normalized_text FROM cases").fetchall()
    return {str(doc_id): str(text or "") for doc_id, text in rows}


def load_manifest_sources(path: Path) -> dict[str, dict]:
    return {
        judgment_hybrid.judgment_internal_id(row): row
        for row in hybrid.read_jsonl(path)
    }


def build_cases(
    indictment_rows: list[dict],
    judgment_rows: list[dict],
    indictment_sources: dict[str, str],
    judgment_sources: dict[str, dict],
) -> list[dict]:
    indictments = {str(row.get("doc_id")): row for row in indictment_rows}
    cases = []
    for judgment_row in judgment_rows:
        indictment_id = str(judgment_row.get("linked_indictment_doc_id") or "")
        judgment_id = str(judgment_row.get("judgment_internal_id") or "")
        pair_id = str(judgment_row.get("pair_id") or "")
        if not all((indictment_id, judgment_id, pair_id)):
            continue
        if indictment_id not in indictments or indictment_id not in indictment_sources:
            continue
        if judgment_id not in judgment_sources:
            continue
        raw_judgment = judgment.normalize_judgment_text(
            str(judgment_sources[judgment_id].get("text") or "")
        )
        if not raw_judgment:
            continue
        cases.append({
            "pair_id": pair_id,
            "judgment_internal_id": judgment_id,
            "linked_indictment_doc_id": indictment_id,
            "indictment": indictments[indictment_id],
            "judgment": judgment_row,
            "raw_indictment": indictment_sources[indictment_id],
            "raw_judgment": raw_judgment,
        })
    return cases


def select_cases(
    cases: list[dict], existing_ids: set[str], error_ids: set[str], scope: str, limit: int
) -> list[dict]:
    if scope == "pending":
        cases = [case for case in cases if case["pair_id"] not in existing_ids | error_ids]
    elif scope == "errors":
        cases = [case for case in cases if case["pair_id"] in error_ids]
    return cases[:limit]


def _person_mentions(case: dict) -> list[tuple[str, str]]:
    mentions: list[tuple[str, str]] = []
    for document_name in ("indictment", "judgment"):
        plan = case[document_name].get("analysis_plan") or {}
        for person in plan.get("persons", []):
            if not isinstance(person, dict):
                continue
            for mention in person.get("mentions", []):
                value = str(mention).strip()
                if value:
                    mentions.append((document_name, value))
    return mentions


def is_intrinsically_safe_person_code(value: str) -> bool:
    return bool(SAFE_ROLE_CODES.fullmatch(value) or MASK_ONLY.fullmatch(value))


def is_masked_identifier(value: str) -> bool:
    compact = re.sub(r"[-－\s()]", "", value)
    return bool(compact) and not (set(compact) - set("0Oo○〇Ｏ"))


def deterministic_checks(case: dict) -> dict:
    """Find objective leaks/collisions before asking the model for judgment."""
    clean_documents = {
        "indictment": str(case["indictment"].get("text") or ""),
        "judgment": str(case["judgment"].get("text") or ""),
    }
    exact_leaks = []
    for source_document, mention in _person_mentions(case):
        if is_intrinsically_safe_person_code(mention):
            continue
        for clean_document, text in clean_documents.items():
            if mention in text:
                exact_leaks.append({
                    "document": clean_document,
                    "span": mention,
                    "source_plan": source_document,
                    "severity": "high",
                })

    regex_hits = []
    for document_name, text in clean_documents.items():
        for kind, pattern in deterministic.LEAK_SCAN_PATTERNS.items():
            for match in pattern.finditer(text):
                span = match.group(0)
                digits = re.sub(r"[^0-9]", "", span)
                if is_masked_identifier(span):
                    continue
                # The shared broad plate candidate regex also matches ROC dates
                # such as 113-05. A Taiwanese plate candidate must include a letter.
                if kind == "PLATE" and not re.search(r"[A-Za-z]", span):
                    continue
                if kind in {"PLATE", "LANDLINE", "MOBILE"} and digits and set(digits) == {"0"}:
                    continue
                regex_hits.append({
                    "document": document_name,
                    "kind": kind,
                    "span": span,
                    # Other decisions cited in a judgment may legitimately retain a case number.
                    "severity": "high" if kind in HIGH_RISK_REGEX_KINDS else "medium",
                })

    aliases: dict[str, dict[str, set[str]]] = {}
    for person in (case["judgment"].get("pair_alias_registry") or {}).get("persons", []):
        alias = str(person.get("alias") or "").strip()
        person_id = str(person.get("pair_person_id") or person.get("indictment_group_id") or "").strip()
        if alias and person_id:
            entry = aliases.setdefault(alias, {"person_ids": set(), "roles": set()})
            entry["person_ids"].add(person_id)
            entry["roles"].add(str(person.get("role") or "OTHER"))
    alias_collisions = [
        {
            "alias": alias,
            "pair_person_ids": sorted(entry["person_ids"]),
            "severity": "high",
        }
        for alias, entry in sorted(aliases.items())
        if len(entry["person_ids"]) > 1
        and not MASK_ONLY.fullmatch(alias)
        and not entry["roles"].issubset(set(hybrid.PROFESSIONAL_ROLES))
    ]
    high_risk = bool(exact_leaks or alias_collisions) or any(
        item["severity"] == "high" for item in regex_hits
    )
    return {
        "pass": not high_risk,
        "exact_person_mention_leaks": exact_leaks,
        "identifier_pattern_hits": regex_hits,
        "alias_collisions": alias_collisions,
    }


def prompt_payload(case: dict) -> dict:
    indictment = case["indictment"]
    decision = case["judgment"]
    return {
        "raw_sources": {
            "indictment": case["raw_indictment"],
            "judgment": case["raw_judgment"],
        },
        "deidentified_documents": {
            "indictment": indictment.get("text", ""),
            "judgment": decision.get("text", ""),
        },
        "document_structures": {
            "indictment": {
                "crime_facts_summary": indictment.get("crime_facts_summary", []),
                "evidence": indictment.get("evidence", []),
            },
            "judgment": {
                "crime_facts_summary": decision.get("crime_facts_summary", []),
                "evidence": decision.get("evidence", []),
            },
            "paired": {
                "crime_facts_summary": decision.get("paired_crime_facts_summary", []),
                "evidence": decision.get("paired_evidence", []),
            },
        },
        "restricted_alias_registry": decision.get("pair_alias_registry", {}),
    }


def audit_prompt(case: dict) -> str:
    data = json.dumps(prompt_payload(case), ensure_ascii=False, separators=(",", ":"))
    return f"""你是臺灣刑事案件資料集的獨立品質稽核員。比較同一案件的原始起訴書、原始判決書、兩份去識別化全文、結構化犯罪事實、證據清單與跨文書人物代號。你只能回報問題，不得改寫任何全文。只輸出一個合法 JSON 物件，不要 Markdown。

重要判準：
1. raw_sources 是比對基準，含原始個資本身不算缺失；只有個資仍出現在 deidentified_documents 或 document_structures 才是 privacy leak。
2. 犯罪事實必須忠於原文，不得遺漏會影響犯罪成立、主觀犯意、行為人、被害人、時間順序、金額、數量、結果、抗辯、罪名或判決結果的資訊，也不得加入原文沒有的內容。
3. 證據「存在」僅指原文有明確提及。不得假稱已看到未提供的卷宗、附件或實體物。每個結構化證據都要指出 indictment、judgment 或 both 的原文支持；推論或原文未提及者列 unsupported_items。
4. 分別確認起訴書與判決書提到的每一種供述、證詞、筆錄、書證、照片、影音／數位資料、搜索扣押文件、實體物證、鑑定／測試報告都進入各自 evidence，且 paired evidence 是兩者完整聯集。合理去重不算遺漏。
5. privacy leak 包含未遮罩本名、部分姓名、證件、電話、Email、帳號、車牌、精確私人地址、本案案號／來源 URL，以及足以直接回查個人的識別資訊。
6. 純暱稱、綽號、網路代稱或原法院 A01/A02 代碼，只要沒有直接洩漏本名，就保留且不得因暱稱本身報 privacy leak。若字串仍含真實姓氏或本名則不是安全暱稱。
7. 只回報會影響裁判理解的代號混淆：例如把不同被告合併、同一人跨文書換代號、行為／證據／刑責綁錯人。純文句不順、重複職稱但不影響人物辨識者不要報。
8. 犯罪事實、證據的 name/proves、兩份全文與 pair alias registry 中，同一人物必須使用同一代號。
9. finding 的 quote 必須是輸入中可逐字找到的最短片段；沒有片段時填 null。不得在 explanation 或 suggestion 重述完整本名、電話、地址等敏感值。

固定輸出格式：
{{
  "crime_facts": {{
    "status": "pass|warning|fail",
    "accuracy_issues": [{{"document":"indictment|judgment|paired","kind":"unsupported|missing|distorted|wrong_binding","severity":"low|medium|high","quote":"最短片段或null","explanation":"說明"}}],
    "privacy_leaks": [{{"document":"indictment|judgment|paired","kind":"類型","severity":"low|medium|high","quote":"最短片段","explanation":"說明"}}]
  }},
  "evidence": {{
    "status": "pass|warning|fail",
    "unsupported_items": [{{"document":"indictment|judgment|paired","item":"證據名稱","severity":"low|medium|high","explanation":"說明"}}],
    "missing_items": [{{"source_document":"indictment|judgment","missing_from":"indictment_record|judgment_record|paired_record","item":"證據名稱","severity":"low|medium|high","source_quote":"最短原文"}}],
    "privacy_leaks": [{{"document":"indictment|judgment|paired","kind":"類型","severity":"low|medium|high","quote":"最短片段","explanation":"說明"}}]
  }},
  "aliases": {{
    "status": "pass|warning|fail",
    "confusing_aliases": [{{"severity":"low|medium|high","alias_or_pair":"代號","affected_legal_binding":"行為、證據或刑責如何綁錯","quote":"最短片段或null"}}],
    "cross_document_inconsistencies": [{{"severity":"low|medium|high","pair_person_id":"代號或null","explanation":"說明","quote":"最短片段或null"}}],
    "privacy_leaks": [{{"document":"indictment|judgment|paired","kind":"類型","severity":"low|medium|high","quote":"最短片段","explanation":"說明"}}],
    "safe_nicknames_observed": ["可安全保留的暱稱；沒有則空陣列"]
  }},
  "cross_component_consistency": {{
    "status": "pass|warning|fail",
    "issues": [{{"severity":"low|medium|high","components":["元件名稱"],"explanation":"說明","quote":"最短片段或null"}}]
  }},
  "overall": {{
    "decision": "pass|review|fail",
    "highest_severity": "none|low|medium|high",
    "reasons": ["簡短理由"]
  }}
}}

待稽核資料：
{data}
"""


def _ensure_list(value: object, field: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def validate_audit_result(result: dict) -> dict:
    """Validate the control fields while preserving detailed model findings."""
    section_lists = {
        "crime_facts": ("accuracy_issues", "privacy_leaks"),
        "evidence": ("unsupported_items", "missing_items", "privacy_leaks"),
        "aliases": (
            "confusing_aliases", "cross_document_inconsistencies",
            "privacy_leaks", "safe_nicknames_observed",
        ),
        "cross_component_consistency": ("issues",),
    }
    for section_name, list_fields in section_lists.items():
        section = result.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"{section_name} must be an object")
        if section.get("status") not in VALID_SECTION_STATUS:
            raise ValueError(f"invalid {section_name}.status")
        for field in list_fields:
            _ensure_list(section.get(field), f"{section_name}.{field}")
    overall = result.get("overall")
    if not isinstance(overall, dict):
        raise ValueError("overall must be an object")
    if overall.get("decision") not in VALID_DECISIONS:
        raise ValueError("invalid overall.decision")
    if overall.get("highest_severity") not in VALID_SEVERITIES:
        raise ValueError("invalid overall.highest_severity")
    _ensure_list(overall.get("reasons"), "overall.reasons")
    return result


def derive_outcome(result: dict, checks: dict) -> dict:
    section_pass = all(
        result[name]["status"] == "pass"
        for name in ("crime_facts", "evidence", "aliases", "cross_component_consistency")
    )
    audit_pass = (
        result["overall"]["decision"] == "pass"
        and section_pass
        and checks["pass"]
    )
    return {
        "audit_pass": audit_pass,
        "requires_manual_review": not audit_pass,
        "effective_decision": "pass" if audit_pass else (
            "fail" if result["overall"]["decision"] == "fail" else "review"
        ),
    }


def failure_record(case: dict, model: str, error: str, **details: object) -> dict:
    row = {
        "pair_id": case["pair_id"],
        "judgment_internal_id": case["judgment_internal_id"],
        "linked_indictment_doc_id": case["linked_indictment_doc_id"],
        "model": model,
        "error": error,
        "checkpointed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    row.update(details)
    return row


def retry_after_seconds(error: urllib.error.HTTPError, detail: str, fallback: float) -> float:
    """Honor Google's 429 hint without coupling it to transient backoff."""
    header = error.headers.get("Retry-After") if error.headers else None
    if header:
        try:
            return max(5.0, float(header) + 3.0)
        except ValueError:
            pass
    match = re.search(r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*s", detail, re.I)
    if match:
        return max(5.0, float(match.group(1)) + 3.0)
    return max(5.0, fallback)


def request_audit(
    case: dict,
    key_name: str,
    key: str,
    args: argparse.Namespace,
    prior: dict | None,
    last_request: list[float | None],
) -> tuple[str, dict | None, dict, str]:
    prompt = audit_prompt(case)
    digest_source = json.dumps({
        "model": args.model,
        "thinking_level": args.thinking_level,
        "prompt": prompt,
    }, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()
    if prior and not args.force and prior.get("input_sha256") == digest:
        return "cached", prior.get("audit_result"), prior.get("usage_metadata", {}), digest
    if len(prompt) > args.max_prompt_chars:
        return "oversize", None, {"prompt_chars": len(prompt)}, digest

    transient_attempts = 0
    while True:
        if last_request[0] is not None:
            time.sleep(max(0, args.delay_seconds - (time.monotonic() - last_request[0])))
        last_request[0] = time.monotonic()
        try:
            result, usage = hybrid.request_json(
                key, args.model, prompt, thinking_level=args.thinking_level
            )
            return "success", validate_audit_result(result), usage, digest
        except hybrid.EmptyModelResponseError as error:
            return "empty", None, {"response": error.envelope}, digest
        except (json.JSONDecodeError, ValueError, KeyError, StopIteration) as error:
            return "invalid", None, {
                "error_type": type(error).__name__, "detail": str(error)[:800]
            }, digest
        except urllib.error.HTTPError as error:
            detail = judgment_hybrid.error_summary(error)
            if error.code == 429:
                wait_seconds = retry_after_seconds(error, detail, args.retry_429_seconds)
                print(
                    f"waiting: {key_name} HTTP 429; retry in {int(wait_seconds)} seconds "
                    f"for {case['pair_id']}", flush=True,
                )
            elif error.code in judgment_hybrid.TRANSIENT_HTTP_CODES:
                transient_attempts += 1
                if transient_attempts >= args.max_transient_attempts:
                    return "transient", None, {
                        "error_type": f"HTTP_{error.code}", "detail": detail,
                        "attempts": transient_attempts,
                    }, digest
                wait_seconds = min(
                    max(5.0, args.retry_transient_seconds)
                    * (2 ** (transient_attempts - 1)),
                    600.0,
                )
                print(
                    f"waiting: {key_name} HTTP {error.code}; retry in {int(wait_seconds)} "
                    f"seconds for {case['pair_id']}", flush=True,
                )
            else:
                return "api", None, {
                    "error_type": f"HTTP_{error.code}", "detail": detail,
                    "attempts": 1,
                }, digest
            if detail:
                print(f"api-detail: {key_name} {detail}", flush=True)
        except (TimeoutError, urllib.error.URLError, ConnectionError) as error:
            transient_attempts += 1
            if transient_attempts >= args.max_transient_attempts:
                return "transient", None, {
                    "error_type": type(error).__name__, "detail": str(error)[:800],
                    "attempts": transient_attempts,
                }, digest
            wait_seconds = min(
                max(5.0, args.retry_transient_seconds)
                * (2 ** (transient_attempts - 1)),
                600.0,
            )
            print(
                f"waiting: {key_name} {type(error).__name__}; retry in "
                f"{int(wait_seconds)} seconds for {case['pair_id']}", flush=True,
            )
        time.sleep(wait_seconds)


def render_audit_record(
    case: dict, result: dict, checks: dict, usage: dict, digest: str, model: str
) -> dict:
    return {
        "pair_id": case["pair_id"],
        "judgment_internal_id": case["judgment_internal_id"],
        "linked_indictment_doc_id": case["linked_indictment_doc_id"],
        "model": model,
        "input_sha256": digest,
        "restricted": True,
        "read_only_audit": True,
        "deterministic_checks": checks,
        "audit_result": result,
        **derive_outcome(result, checks),
        "usage_metadata": usage,
        "checkpointed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    args = parse_args()
    if args.limit < 1 or args.max_transient_attempts < 1:
        raise SystemExit("--limit and --max-transient-attempts must be at least 1")
    cases = build_cases(
        hybrid.read_jsonl(args.indictments),
        hybrid.read_jsonl(args.judgments),
        load_indictment_sources(args.db),
        load_manifest_sources(args.manifest),
    )
    existing = {str(row.get("pair_id")): row for row in hybrid.read_jsonl(args.output)}
    failures = {str(row.get("pair_id")): row for row in hybrid.read_jsonl(args.failures)}
    selected = select_cases(cases, set(existing), set(failures), args.scope, args.limit)
    if args.dry_run:
        prompt_lengths = [len(audit_prompt(case)) for case in selected]
        deterministic_alerts = sum(not deterministic_checks(case)["pass"] for case in selected)
        print(json.dumps({
            "dry_run": True,
            "eligible_pairs": len(cases),
            "selected": len(selected),
            "existing_audits": len(existing),
            "existing_failures": len(failures),
            "deterministic_alert_pairs": deterministic_alerts,
            "prompt_chars": {
                "minimum": min(prompt_lengths, default=0),
                "maximum": max(prompt_lengths, default=0),
                "over_limit": sum(length > args.max_prompt_chars for length in prompt_lengths),
            },
        }, ensure_ascii=False, indent=2))
        return

    keys = hybrid.load_api_keys()
    if args.key_names:
        requested = set(args.key_names)
        available = {name for name, _ in keys}
        missing = requested - available
        if missing:
            raise SystemExit(f"API key name not found in .env: {','.join(sorted(missing))}")
        keys = [item for item in keys if item[0] in requested]

    output_order = list(existing) + [
        case["pair_id"] for case in selected if case["pair_id"] not in existing
    ]
    failure_order = list(failures) + [
        case["pair_id"] for case in selected if case["pair_id"] not in failures
    ]
    worker_count = min(len(keys), len(selected))
    print(
        f"eligible={len(cases)} selected={len(selected)} workers={worker_count} "
        f"keys={','.join(name for name, _ in keys)} model={args.model}", flush=True,
    )

    case_iter = iter(enumerate(selected, 1))
    states: list[list[float | None]] = [[None] for _ in keys]
    active: dict[concurrent.futures.Future, tuple[int, int, dict, dict, dict | None]] = {}

    def submit_next(executor: concurrent.futures.ThreadPoolExecutor, worker_index: int) -> None:
        try:
            index, case = next(case_iter)
        except StopIteration:
            return
        checks = deterministic_checks(case)
        prior = existing.get(case["pair_id"])
        key_name, key = keys[worker_index]
        future = executor.submit(
            request_audit, case, key_name, key, args, prior, states[worker_index]
        )
        active[future] = (worker_index, index, case, checks, prior)

    if worker_count:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            for worker_index in range(worker_count):
                submit_next(executor, worker_index)
            while active:
                done, _ = concurrent.futures.wait(
                    active, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    worker_index, index, case, checks, prior = active.pop(future)
                    key_name = keys[worker_index][0]
                    status, result, detail, digest = future.result()
                    if result is None:
                        error_name = {
                            "oversize": "prompt_too_long",
                            "empty": "empty_model_response",
                            "invalid": "invalid_model_response",
                            "transient": "transient_api_error",
                            "api": "api_request_error",
                        }[status]
                        failures[case["pair_id"]] = failure_record(
                            case, args.model, error_name, key_name=key_name,
                            input_sha256=digest, **detail,
                        )
                        hybrid.write_jsonl(args.failures, failures, failure_order)
                        print(
                            f"skipped-{status} {index}/{len(selected)} {case['pair_id']} "
                            f"key={key_name}", flush=True,
                        )
                    else:
                        row = render_audit_record(
                            case, result, checks, detail, digest, args.model
                        )
                        existing[case["pair_id"]] = row
                        failures.pop(case["pair_id"], None)
                        hybrid.write_jsonl(args.output, existing, output_order)
                        if args.failures.exists():
                            hybrid.write_jsonl(args.failures, failures, failure_order)
                        print(
                            f"{'cached' if status == 'cached' else 'audited'} "
                            f"{index}/{len(selected)} {case['pair_id']} key={key_name} "
                            f"decision={row['effective_decision']}", flush=True,
                        )
                    submit_next(executor, worker_index)

    print(json.dumps({
        "eligible_pairs": len(cases),
        "audits": len(existing),
        "selected": len(selected),
        "scope": args.scope,
        "model": args.model,
        "workers": worker_count,
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
