#!/usr/bin/env python3
"""Checkpointed LLM analysis and deterministic rendering for paired judgments."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import sqlite3
import time
import urllib.error
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts import deidentify_linked_judgments as judgment
    from scripts import hybrid_deidentify_with_google_ai as hybrid
    from scripts import pair_case_integration as pairing
except (ModuleNotFoundError, ImportError):  # Direct execution adds scripts/ rather than repo root.
    import deidentify_linked_judgments as judgment
    import hybrid_deidentify_with_google_ai as hybrid
    import pair_case_integration as pairing


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/raw/linked_judgments/linked_judgments_114.jsonl"
DEFAULT_DB = ROOT / "data/review/indictment_reviews.sqlite3"
DEFAULT_INDICTMENTS = ROOT / "data/intermediate/google_ai/hybrid_gemma4_experiment.jsonl"
DEFAULT_OUTPUT = ROOT / "data/intermediate/google_ai/hybrid_judgments_gemma4_experiment.jsonl"
DEFAULT_FAILURES = ROOT / "data/intermediate/google_ai/hybrid_judgments_gemma4_failures.jsonl"
TRANSIENT_HTTP_CODES = {500, 502, 503, 504}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired judgment LLM analysis plus deterministic rendering"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--indictments", type=Path, default=DEFAULT_INDICTMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--model", default="gemma-4-31b-it", choices=sorted(hybrid.FREE_ONLY_MODELS))
    parser.add_argument(
        "--key-name", dest="key_names", action="append",
        help="use only this .env key name; repeat to select multiple keys (default: all)",
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--scope", choices=("pending", "errors", "rendered"), default="pending")
    parser.add_argument("--delay-seconds", type=float, default=8.0)
    parser.add_argument("--max-source-chars", type=int, default=24000)
    parser.add_argument("--retry-429-seconds", type=float, default=300.0)
    parser.add_argument("--max-transient-attempts", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def linked_source_ids(record: dict) -> list[str]:
    return [
        f"MOJ-PROSECUTION:{str(item.get('barcode') or '').strip()}"
        for item in record.get("linked_indictments", [])
        if str(item.get("barcode") or "").strip()
    ]


def load_indictment_sources(db_path: Path) -> dict[str, dict]:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT doc_id,source_id,normalized_text,entities_json FROM cases"
        ).fetchall()
    return {str(row["source_id"]): dict(row) for row in rows}


def judgment_internal_id(record: dict) -> str:
    return judgment.adapt_linked_judgment(record)["internal_doc_id"]


def select_cases(
    manifest: Path,
    indictment_records: dict[str, dict],
    indictment_sources: dict[str, dict],
    existing_ids: set[str],
    error_ids: set[str],
    completed_ids: set[str],
    scope: str,
    limit: int,
) -> list[dict]:
    selected = []
    for record in hybrid.read_jsonl(manifest):
        matches = [source_id for source_id in linked_source_ids(record) if source_id in indictment_records]
        if len(matches) != 1 or matches[0] not in indictment_sources:
            continue
        internal_id = judgment_internal_id(record)
        if scope == "errors" and internal_id not in error_ids:
            continue
        if scope == "rendered" and internal_id not in completed_ids:
            continue
        if scope == "pending" and internal_id in existing_ids | error_ids:
            continue
        selected.append({
            "judgment_internal_id": internal_id,
            "manifest_record": record,
            "indictment_record": indictment_records[matches[0]],
            "indictment_source": indictment_sources[matches[0]],
        })
    return selected[:limit]


def analysis_prompt(source: str, pair_registry: dict) -> str:
    anchors = [
        {
            "indictment_group_id": item.get("indictment_group_id"),
            "role": item.get("role"),
            "alias": item.get("alias"),
            "indictment_mentions": item.get("indictment_mentions", []),
        }
        for item in pair_registry.get("persons", [])
    ]
    return f"""你是臺灣刑事判決書的法律資訊分析員。你只建立結構化替換計畫、犯罪事實摘要與證據清單，不得重寫全文。只輸出合法 JSON 物件，不要 Markdown。

跨文書人物規則：
1. 下方「起訴書人物錨點」來自同一案件。判決書中的人物若與錨點是同一人，persons 必須填 linked_indictment_group_id；即使姓名呈現為全名、OO、○○或既有 A01 代號，也應依上下文連結。
2. 無法確定時 linked_indictment_group_id 填 null，不得猜測。同名不同人必須分組。
3. mentions 只能逐字抄錄判決書中的姓名或人物代號本身，不含前置職稱。A01、A02、A03 等若能依同一份判決的全名、附件或角色明確連到起訴書錨點，必須與該全名放在同一 persons 組並填 linked_indictment_group_id，使後續統一為同一代號；無法確定對應的既有代號才保留，並寫入 uncertainties。甲男、乙女等泛稱不要列入 persons。
4. role 只能是 DEFENDANT、WITNESS、COMPLAINANT、VICTIM、INVESTOR、PROSECUTOR、CLERK、JUDGE、DEFENSE_COUNSEL、COMPLAINANT_COUNSEL、PRIVATE_PROSECUTOR_COUNSEL、POLICE、APPELLANT、PETITIONER、RESPONDENT、SENTENCED_PERSON、LEGAL_REPRESENTATIVE、PRIVATE_PROSECUTOR、OTHER。

機構與識別規則：
1. 涉案私人公司、商號、團體及其簡稱 action=MASK；政府機關、法院、地檢署及一般金融機構 action=KEEP。
2. identifiers 只列私人電話、Email、身分證、非全零帳號／車牌、真正未遮罩私人地址，以及本案案號。含○／〇／Ｏ的既有遮罩資料 KEEP；引用其他裁判的案號 KEEP。

事實與證據規則：
1. crime_facts_summary 整理判決認定或引用的犯罪構成事實，不加入推測。
2. evidence 窮盡判決提及的供述、證詞、筆錄、書證、照片、數位資料、搜索扣押文件、實體物及鑑定報告。每一種證據各列一項，保留原文明示數量與證明事項。
3. 判決若引用附件起訴書，仍照原文列出；後續程式會與起訴書證據做保守去重並保留來源。

輸出格式：
{{
  "persons":[{{"group_id":"J01","linked_indictment_group_id":"P01或null","role":"DEFENDANT","mentions":["原文姓名"],"same_person_reason":"依據"}}],
  "organizations":[{{"group_id":"O01","mentions":["原文名稱"],"action":"MASK|KEEP","reason":"理由"}}],
  "identifiers":[{{"mention":"原文字串","category":"PHONE|EMAIL|ID|ACCOUNT|PLATE|ADDRESS|CASE_NO","replacement":"[電話]等","reason":"理由"}}],
  "crime_facts_summary":["事實句"],
  "evidence":[{{"canonical_key":"簡短穩定內容鍵或null","name":"證據名稱","category":"供述|證詞|書證|照片|數位證據|扣押文件|實體物證|鑑定報告|其他","quantity":"原文數量或null","proves":"證明事項"}}],
  "uncertainties":[]
}}

起訴書人物錨點：
{json.dumps(anchors, ensure_ascii=False)}

原始正規化判決書：
{source}
"""


def error_summary(error: urllib.error.HTTPError) -> str:
    body = error.read().decode("utf-8", errors="replace")
    try:
        info = json.loads(body).get("error", {})
        return re.sub(
            r"\s+", " ", f"{info.get('status', '')}: {info.get('message', '')}"
        ).strip()[:800]
    except json.JSONDecodeError:
        return re.sub(r"\s+", " ", body).strip()[:800]


def failure_record(case: dict, model: str, error: str, **details: object) -> dict:
    record = {
        "judgment_internal_id": case["judgment_internal_id"],
        "linked_indictment_doc_id": case["indictment_record"]["doc_id"],
        "model": model,
        "error": error,
        "checkpointed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    record.update(details)
    return record


def request_plan(
    case: dict,
    source: str,
    registry: dict,
    key_name: str,
    key: str,
    args: argparse.Namespace,
    existing: dict | None,
    last_request: list[float | None],
) -> tuple[str, dict | None, dict, str]:
    prompt = analysis_prompt(source, registry)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if existing and not args.force and existing.get("input_sha256") == digest:
        return "cached", existing.get("analysis_plan"), existing.get("usage_metadata", {}), digest
    if len(source) > args.max_source_chars:
        return "oversize", None, {"source_chars": len(source)}, digest

    retry_wait = max(60.0, args.retry_429_seconds)
    attempts = 0
    while True:
        if last_request[0] is not None:
            time.sleep(max(0, args.delay_seconds - (time.monotonic() - last_request[0])))
        last_request[0] = time.monotonic()
        try:
            plan, usage = hybrid.request_json(key, args.model, prompt)
            return "success", plan, usage, digest
        except hybrid.EmptyModelResponseError as error:
            return "empty", None, {"response": error.envelope}, digest
        except (json.JSONDecodeError, ValueError, KeyError, StopIteration) as error:
            return "invalid", None, {
                "error_type": type(error).__name__, "detail": str(error)[:800]
            }, digest
        except urllib.error.HTTPError as error:
            detail = error_summary(error)
            if error.code == 429:
                print(
                    f"waiting: {key_name} HTTP 429; retry in {int(retry_wait)} seconds "
                    f"for {case['judgment_internal_id']}", flush=True,
                )
            elif error.code in TRANSIENT_HTTP_CODES:
                attempts += 1
                if attempts >= args.max_transient_attempts:
                    return "transient", None, {
                        "error_type": f"HTTP_{error.code}", "detail": detail, "attempts": attempts
                    }, digest
                print(
                    f"waiting: {key_name} HTTP {error.code}; retry in {int(retry_wait)} "
                    f"seconds for {case['judgment_internal_id']}", flush=True,
                )
            else:
                raise
            if detail:
                print(f"api-detail: {key_name} {detail}", flush=True)
        except (TimeoutError, urllib.error.URLError, ConnectionError) as error:
            attempts += 1
            if attempts >= args.max_transient_attempts:
                return "transient", None, {
                    "error_type": type(error).__name__, "detail": str(error)[:800],
                    "attempts": attempts,
                }, digest
            print(
                f"waiting: {key_name} {type(error).__name__}; retry in "
                f"{int(retry_wait)} seconds for {case['judgment_internal_id']}", flush=True,
            )
        time.sleep(retry_wait)
        retry_wait = min(retry_wait * 2, 3600.0)


def render_record(
    case: dict,
    plan: dict,
    usage: dict,
    digest: str,
    model: str,
    pair_registry: dict,
    prior: dict | None,
) -> dict:
    clean_id = prior.get("doc_id") if prior else uuid.uuid4().hex
    pair_id = prior.get("pair_id") if prior else uuid.uuid4().hex
    rendered = judgment.prepare_rendered_record(
        case["manifest_record"], plan, pair_alias_registry=pair_registry, clean_id=clean_id
    )
    indictment = case["indictment_record"]
    rendered.update({
        "experiment": "paired_judgment_llm_analysis_deterministic_renderer",
        "restricted": True,
        "judgment_internal_id": case["judgment_internal_id"],
        "pair_id": pair_id,
        "linked_indictment_doc_id": indictment["doc_id"],
        "model": model,
        "input_sha256": digest,
        "analysis_plan": plan,
        "pair_alias_registry": pair_registry,
        "paired_crime_facts_summary": pairing.merge_pair_facts(
            indictment.get("crime_facts_summary", []), rendered.get("crime_facts_summary", [])
        ),
        "paired_evidence": pairing.merge_pair_evidence(
            indictment.get("evidence", []), rendered.get("evidence", [])
        ),
        "usage_metadata": usage,
        "checkpointed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return rendered


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.max_transient_attempts < 1:
        raise SystemExit("--max-transient-attempts must be at least 1")
    keys = hybrid.load_api_keys()
    if args.key_names:
        requested = set(args.key_names)
        available = {name for name, _ in keys}
        missing = requested - available
        if missing:
            raise SystemExit(f"API key name not found in .env: {','.join(sorted(missing))}")
        keys = [item for item in keys if item[0] in requested]

    indictment_records = {
        row["source_id"]: row for row in hybrid.read_jsonl(args.indictments)
    }
    indictment_sources = load_indictment_sources(args.db)
    existing = {
        row["judgment_internal_id"]: row for row in hybrid.read_jsonl(args.output)
    }
    failures = {
        row["judgment_internal_id"]: row for row in hybrid.read_jsonl(args.failures)
    }
    cases = select_cases(
        args.manifest, indictment_records, indictment_sources,
        set(existing), set(failures), set(existing), args.scope, args.limit,
    )
    output_order = list(existing)
    output_order.extend(
        case["judgment_internal_id"] for case in cases
        if case["judgment_internal_id"] not in existing
    )
    failure_order = list(failures)
    failure_order.extend(
        case["judgment_internal_id"] for case in cases
        if case["judgment_internal_id"] not in failures
    )
    worker_count = min(len(keys), len(cases))
    print(
        f"selected={len(cases)} workers={worker_count} "
        f"keys={','.join(name for name, _ in keys)} "
        f"linked_indictments={len(indictment_records)}",
        flush=True,
    )
    case_iter = iter(enumerate(cases, 1))
    active: dict[concurrent.futures.Future, tuple[int, int, dict, dict, dict | None]] = {}
    states: list[list[float | None]] = [[None] for _ in keys]

    def submit_next(executor: concurrent.futures.ThreadPoolExecutor, worker_index: int) -> None:
        try:
            index, case = next(case_iter)
        except StopIteration:
            return
        source_row = case["indictment_source"]
        indictment = case["indictment_record"]
        pair_registry = pairing.build_pair_alias_registry(
            source_row["normalized_text"], indictment["analysis_plan"],
            json.loads(source_row["entities_json"]),
        )
        source = judgment.normalize_judgment_text(case["manifest_record"]["text"])
        prior = existing.get(case["judgment_internal_id"])
        key_name, key = keys[worker_index]
        future = executor.submit(
            request_plan, case, source, pair_registry, key_name, key, args, prior,
            states[worker_index],
        )
        active[future] = (worker_index, index, case, pair_registry, prior)

    if worker_count:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            for worker_index in range(worker_count):
                submit_next(executor, worker_index)
            while active:
                done, _ = concurrent.futures.wait(
                    active, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    worker_index, index, case, pair_registry, prior = active.pop(future)
                    key_name = keys[worker_index][0]
                    status, plan, detail, digest = future.result()
                    if plan is None:
                        error_name = {
                            "oversize": "source_too_long_for_free_api",
                            "empty": "empty_model_response",
                            "invalid": "invalid_model_response",
                            "transient": "transient_api_error",
                        }[status]
                        failures[case["judgment_internal_id"]] = failure_record(
                            case, args.model, error_name, key_name=key_name,
                            input_sha256=digest, **detail,
                        )
                        hybrid.write_jsonl(args.failures, failures, failure_order)
                        print(
                            f"skipped-{status} {index}/{len(cases)} "
                            f"{case['judgment_internal_id']} key={key_name}", flush=True,
                        )
                    else:
                        record = render_record(
                            case, plan, detail, digest, args.model, pair_registry, prior
                        )
                        existing[case["judgment_internal_id"]] = record
                        failures.pop(case["judgment_internal_id"], None)
                        hybrid.write_jsonl(args.output, existing, output_order)
                        if args.failures.exists():
                            hybrid.write_jsonl(args.failures, failures, failure_order)
                        print(
                            f"{'rerendered' if status == 'cached' else 'processed'} "
                            f"{index}/{len(cases)} {case['judgment_internal_id']} key={key_name} "
                            f"replacements={record['applied_replacements']} "
                            f"evidence={len(record['evidence'])} "
                            f"paired_evidence={len(record['paired_evidence'])}", flush=True,
                        )
                    submit_next(executor, worker_index)
    print(json.dumps({
        "documents": len(existing), "selected": len(cases), "scope": args.scope,
        "workers": worker_count, "keys": [name for name, _ in keys],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
