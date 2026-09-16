#!/usr/bin/env python3
"""Generate bounded, evidence-grounded prosecutor/defense/judge reasoning.

Only paired Gemma 4 deidentification checkpoints are accepted as source data.
The three calls have deliberately different visibility and are sequential:
prosecutor -> defense -> judge.  Outputs are experimental and restricted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts import hybrid_deidentify_with_google_ai as hybrid
except (ModuleNotFoundError, ImportError):
    import hybrid_deidentify_with_google_ai as hybrid


ROOT = Path(__file__).resolve().parents[1]
API_ROOT = "https://openrouter.ai/api/v1"
MODEL = "deepseek/deepseek-v4-flash-0731"
PROVIDER = "open-inference"
PROVIDER_DISPLAY = "OpenInference"
QUANTIZATION = "fp8"
PROMPT_PRICE_PER_M = 0.04
COMPLETION_PRICE_PER_M = 0.10
MAX_OUTPUT_TOKENS = 6144
MAX_PAIRS = 3
MAX_WORST_CASE_USD = 0.10

DEFAULT_INDICTMENTS = ROOT / "data/intermediate/google_ai/hybrid_gemma4_experiment.jsonl"
DEFAULT_JUDGMENTS = (
    ROOT / "data/intermediate/google_ai/hybrid_judgments_gemma4_experiment.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data/intermediate/openrouter/deepseek_v4_flash_0731_three_role_experiment.jsonl"
)
DEFAULT_FAILURES = (
    ROOT
    / "data/intermediate/openrouter/deepseek_v4_flash_0731_three_role_failures.jsonl"
)

ROLE_REQUIREMENTS = {
    "prosecutor": ("issues", "reasoning_trace", "prosecution_brief"),
    "defense": ("issues", "reasoning_trace", "defense_brief"),
    "judge": ("issues", "reasoning_trace", "decision"),
}


class InvalidRoleResponseError(ValueError):
    """A billed response arrived but its visible content was not valid JSON."""

    def __init__(self, cause: Exception, usage: dict, answer: str):
        super().__init__(f"invalid model JSON: {cause}; visible_chars={len(answer)}")
        self.usage = usage
        self.answer_preview = answer[:2000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded Gemma-only three-role legal-reasoning experiment"
    )
    parser.add_argument("--indictments", type=Path, default=DEFAULT_INDICTMENTS)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument(
        "--pair-id", action="append", default=[],
        help="specific Gemma pair_id; may be repeated (otherwise selects outcome coverage)",
    )
    parser.add_argument("--execute", action="store_true", help="send paid OpenRouter calls")
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse locally validated role outputs from a matching partial checkpoint",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_openrouter_key(env_path: Path = ROOT / ".env") -> str:
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name, value = line.split("=", 1)
        if name.strip().strip("\"'") == "OPENROUTER_API_KEY":
            key = hybrid.parse_dotenv_value(value)
            if key:
                return key
    raise RuntimeError("OPENROUTER_API_KEY is missing or empty in .env")


def api_json(
    url: str, key: str | None = None, payload: dict | None = None, timeout: float = 900
) -> dict:
    headers = {
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/lawrence-legal-reasoning",
        "X-Title": "LAWRENCE bounded legal reasoning experiment",
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        url,
        data=(json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None),
        headers=headers,
        method="POST" if payload else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def verify_key(key: str) -> dict:
    return api_json(f"{API_ROOT}/auth/key", key)


def verify_route() -> dict:
    author, slug = MODEL.split("/", 1)
    envelope = api_json(f"{API_ROOT}/models/{author}/{slug}/endpoints")
    matches = [
        endpoint
        for endpoint in envelope.get("data", {}).get("endpoints", [])
        if endpoint.get("provider_name") == PROVIDER_DISPLAY
        and str(endpoint.get("quantization") or "").lower() == QUANTIZATION
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {PROVIDER_DISPLAY} {QUANTIZATION} endpoint, found {len(matches)}"
        )
    endpoint = matches[0]
    prompt_price = float(endpoint["pricing"]["prompt"]) * 1_000_000
    completion_price = float(endpoint["pricing"]["completion"]) * 1_000_000
    if prompt_price > PROMPT_PRICE_PER_M or completion_price > COMPLETION_PRICE_PER_M:
        raise RuntimeError(
            f"route price exceeds cap: prompt={prompt_price}, completion={completion_price}"
        )
    if "response_format" not in endpoint.get("supported_parameters", []):
        raise RuntimeError("selected endpoint does not support JSON response_format")
    return endpoint


def is_gemma(record: dict) -> bool:
    return str(record.get("model") or "").lower() == "gemma-4-31b-it"


def judgment_holding(record: dict) -> str:
    sections = record.get("sections") or {}
    holding = str(sections.get("主文") or "").strip()
    if not holding:
        raise ValueError(f"judgment {record.get('doc_id')} has no 主文 section")
    return holding


def outcome_class(holding: str) -> str:
    has_acquittal = "無罪" in holding
    has_conviction = bool(re.search(r"(?:犯.{0,30}罪|處(?:有期徒刑|拘役|罰金)|應執行)", holding))
    if has_acquittal and has_conviction:
        return "mixed"
    if has_acquittal:
        return "acquittal"
    return "conviction"


def load_pairs(indictment_path: Path, judgment_path: Path) -> list[dict]:
    indictments = {
        str(row.get("doc_id")): row
        for row in hybrid.read_jsonl(indictment_path)
        if is_gemma(row)
    }
    pairs = []
    seen: set[str] = set()
    for judgment in hybrid.read_jsonl(judgment_path):
        if not is_gemma(judgment):
            continue
        pair_id = str(judgment.get("pair_id") or "")
        indictment_id = str(judgment.get("linked_indictment_doc_id") or "")
        indictment = indictments.get(indictment_id)
        if not pair_id or not indictment or pair_id in seen:
            continue
        try:
            holding = judgment_holding(judgment)
        except ValueError:
            # A linked pair without an extractable dispositive section cannot
            # provide the judge stage's required target outcome.
            continue
        seen.add(pair_id)
        pairs.append({
            "pair_id": pair_id,
            "indictment": indictment,
            "judgment": judgment,
            "target_holding": holding,
            "outcome_class": outcome_class(holding),
        })
    return pairs


def pair_source_chars(pair: dict) -> int:
    return len(str(pair["indictment"].get("text") or "")) + len(
        str(pair["judgment"].get("text") or "")
    )


def select_pairs(pairs: list[dict], limit: int, requested_ids: list[str]) -> list[dict]:
    if requested_ids:
        wanted = requested_ids[:limit]
        by_id = {pair["pair_id"]: pair for pair in pairs}
        missing = [pair_id for pair_id in wanted if pair_id not in by_id]
        if missing:
            raise ValueError(f"not valid paired Gemma pair_id(s): {', '.join(missing)}")
        return [by_id[pair_id] for pair_id in wanted]

    # Short complete cases reduce experimental cost.  The first two deliberately
    # cover conviction and acquittal; a third, if requested, covers a mixed result.
    chosen: list[dict] = []
    for result_type in ("conviction", "acquittal", "mixed"):
        candidates = [pair for pair in pairs if pair["outcome_class"] == result_type]
        candidates.sort(key=lambda pair: (pair_source_chars(pair), pair["pair_id"]))
        if candidates:
            chosen.append(candidates[0])
        if len(chosen) == limit:
            return chosen
    remaining = sorted(
        (pair for pair in pairs if pair not in chosen),
        key=lambda pair: (pair_source_chars(pair), pair["pair_id"]),
    )
    return (chosen + remaining)[:limit]


def evidence_catalog(record: dict, prefix: str, source: str) -> list[dict]:
    catalog = []
    for index, item in enumerate(record.get("evidence") or [], 1):
        evidence = dict(item) if isinstance(item, dict) else {"description": str(item)}
        catalog.append({
            "evidence_id": f"{prefix}-E{index:03d}",
            "source_document": source,
            "record": evidence,
        })
    return catalog


def case_materials(pair: dict) -> dict:
    indictment = pair["indictment"]
    judgment = pair["judgment"]
    indictment_evidence = evidence_catalog(indictment, "I", "indictment")
    judgment_evidence = evidence_catalog(judgment, "J", "judgment")
    return {
        "pair_id": pair["pair_id"],
        "indictment_text": str(indictment.get("text") or ""),
        "indictment_crime_facts": indictment.get("crime_facts_summary") or [],
        "judgment_crime_facts": judgment.get("crime_facts_summary") or [],
        "indictment_evidence": indictment_evidence,
        "all_evidence": indictment_evidence + judgment_evidence,
        "target_holding": pair["target_holding"],
        "outcome_class": pair["outcome_class"],
    }


def grounding_sources(materials: dict, prosecutor_only: bool) -> list[str]:
    evidence = (
        materials["indictment_evidence"] if prosecutor_only else materials["all_evidence"]
    )
    facts: Any = materials["indictment_crime_facts"]
    if not prosecutor_only:
        facts = [materials["indictment_crime_facts"], materials["judgment_crime_facts"]]
    return [
        materials["indictment_text"],
        json.dumps(facts, ensure_ascii=False),
        json.dumps(evidence, ensure_ascii=False),
    ]


COMMON_RULES = """
你正在建立臺灣刑事案件的可稽核法律推理資料。請輸出 JSON object，不得輸出 markdown。
這不是要求揭露模型的隱藏內在思考；reasoning_trace 只寫可供人工核對的精簡法律理由。
絕對禁止新增、補寫或暗示材料中不存在的證據、證詞、鑑定、程序或事實。
每一項事實主張、證據評價與推論都必須列 evidence_ids，且只能使用輸入提供的 ID。
reasoning_trace 每一步還必須列 support_snippets，逐字摘錄案件材料中支持該步驟的短句。
你只看到文字與證據清單摘要，沒有親自閱覽影片、照片或卷證本體；除非案件材料明文記載，
不得聲稱影像「顯示／未顯示」某人事物，也不得以清單未列某證據推論該證據不存在。
法律常識可以用於解釋或推論，但不可偽裝為本案證據。證據不足時必須明說不足。
人物代號必須沿用輸入，不得猜測真名。new_evidence_claimed 必須是空陣列。
""".strip()


def prompt_for_prosecutor(materials: dict) -> str:
    payload = {
        "indictment_text": materials["indictment_text"],
        "crime_facts": materials["indictment_crime_facts"],
        "evidence_catalog": materials["indictment_evidence"],
    }
    shape = {
        "role": "prosecutor",
        "issues": ["爭點"],
        "reasoning_trace": [{
            "step_id": "P01", "proposition": "待證命題", "evidence_ids": ["I-E001"],
            "support_snippets": ["案件材料中的逐字短句"],
            "evidence_interpretation": "證據如何支持命題", "inference": "有限推論",
            "uncertainty": "限制或空字串",
        }],
        "prosecution_brief": {
            "theory_of_case": {"statement": "案件理論", "evidence_ids": ["I-E001"]},
            "facts_asserted": [{"statement": "事實", "evidence_ids": ["I-E001"]}],
            "evidence_arguments": [{"argument": "證據論證", "evidence_ids": ["I-E001"]}],
            "legal_position": ["法律主張"],
            "requested_disposition": "檢方請求",
        },
        "limitations": ["資料限制"],
        "new_evidence_claimed": [],
    }
    return (
        COMMON_RULES
        + "\n\n角色：檢方。只能看起訴書、起訴犯罪事實與起訴書證據，形成與起訴思路相符、但不誇大的論證。"
        + "\n輸出結構範例（內容須依本案）：\n"
        + json.dumps(shape, ensure_ascii=False)
        + "\n\n案件材料：\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def prompt_for_defense(materials: dict, prosecutor: dict) -> str:
    payload = {
        "indictment_text": materials["indictment_text"],
        "prosecution_brief": prosecutor["prosecution_brief"],
        "crime_facts": {
            "indictment": materials["indictment_crime_facts"],
            "judgment_extracted": materials["judgment_crime_facts"],
        },
        "evidence_catalog": materials["all_evidence"],
    }
    shape = {
        "role": "defense",
        "issues": ["爭點"],
        "reasoning_trace": [{
            "step_id": "D01", "proposition": "辯方命題", "evidence_ids": ["I-E001"],
            "support_snippets": ["案件材料中的逐字短句"],
            "evidence_interpretation": "證據限制或反向解釋", "inference": "有限推論",
            "uncertainty": "限制或空字串",
        }],
        "defense_brief": {
            "defense_theory": "答辯方向",
            "admissions": [{"statement": "不爭執事項", "evidence_ids": ["I-E001"]}],
            "disputes": [{"statement": "爭執事項", "evidence_ids": ["J-E001"]}],
            "evidence_arguments": [{"argument": "證據評價", "evidence_ids": ["J-E001"]}],
            "legal_position": ["法律主張"],
            "requested_disposition": "辯方請求",
        },
        "hypotheticals": [{"possibility": "僅供檢驗的可能性", "is_asserted_fact": False}],
        "limitations": ["資料限制"],
        "new_evidence_claimed": [],
    }
    return (
        COMMON_RULES
        + "\n\n角色：辯方。你不知道判決主文或判決理由。只能就現有資料提出可信的證據解釋、證明力攻防與法律答辯。"
        + "任何未被證據支持的替代可能都只能放在 hypotheticals，且 is_asserted_fact 必須為 false；不得把假設寫成已發生事實。"
        + "\n輸出結構範例（內容須依本案）：\n"
        + json.dumps(shape, ensure_ascii=False)
        + "\n\n案件材料：\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def prompt_for_judge(materials: dict, prosecutor: dict, defense: dict) -> str:
    payload = {
        "indictment_text": materials["indictment_text"],
        "prosecution_brief": prosecutor["prosecution_brief"],
        "defense_brief": defense["defense_brief"],
        "crime_facts": {
            "indictment": materials["indictment_crime_facts"],
            "judgment_extracted": materials["judgment_crime_facts"],
        },
        "evidence_catalog": materials["all_evidence"],
        "actual_target_holding": materials["target_holding"],
    }
    shape = {
        "role": "judge",
        "issues": ["爭點"],
        "reasoning_trace": [{
            "step_id": "J01", "proposition": "裁判命題", "evidence_ids": ["J-E001"],
            "support_snippets": ["案件材料中的逐字短句"],
            "evidence_interpretation": "證據取捨", "inference": "有限推論",
            "uncertainty": "限制或空字串",
        }],
        "decision": {
            "target_holding": "逐字複製 actual_target_holding",
            "issue_findings": [{"issue": "爭點", "finding": "認定", "evidence_ids": ["J-E001"]}],
            "response_to_prosecution": [{"response": "回應", "evidence_ids": ["I-E001"]}],
            "response_to_defense": [{"response": "回應", "evidence_ids": ["J-E001"]}],
            "rationale": [{"statement": "與主文一致的精簡理由", "evidence_ids": ["J-E001"]}],
        },
        "limitations": ["資料限制"],
        "new_evidence_claimed": [],
    }
    return (
        COMMON_RULES
        + "\n\n角色：法官。依起訴、檢辯主張及全部證據形成中立理由，結論必須與 actual_target_holding 一致。"
        + "不得倒填判決書中未出現在案件材料的理由；若材料不足以完整支持主文，須在 limitations 明確標示。"
        + "decision.target_holding 必須逐字複製 actual_target_holding。"
        + "\n輸出結構範例（內容須依本案）：\n"
        + json.dumps(shape, ensure_ascii=False)
        + "\n\n案件材料：\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def collect_evidence_ids(value: Any, found: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "evidence_ids":
                if not isinstance(child, list) or not child:
                    raise ValueError("every evidence_ids field must be a non-empty list")
                if not all(isinstance(item, str) and item for item in child):
                    raise ValueError("evidence_ids must contain non-empty strings")
                found.extend(child)
            else:
                collect_evidence_ids(child, found)
    elif isinstance(value, list):
        for child in value:
            collect_evidence_ids(child, found)


def normalize_support_text(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def validate_role_output(
    output: dict,
    role: str,
    allowed_evidence_ids: set[str],
    support_sources: list[str],
    target_holding: str | None = None,
) -> None:
    if not isinstance(output, dict) or output.get("role") != role:
        raise ValueError(f"expected role={role}")
    for field in ROLE_REQUIREMENTS[role]:
        if field not in output or not output[field]:
            raise ValueError(f"{role} output missing non-empty {field}")
    if output.get("new_evidence_claimed") != []:
        raise ValueError(f"{role} claimed new evidence")
    trace = output.get("reasoning_trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError(f"{role} reasoning_trace must be a non-empty list")
    for index, step in enumerate(trace, 1):
        if not isinstance(step, dict) or not step.get("evidence_ids"):
            raise ValueError(f"{role} reasoning step {index} has no evidence_ids")
        snippets = step.get("support_snippets")
        if not isinstance(snippets, list) or not snippets:
            raise ValueError(f"{role} reasoning step {index} has no support_snippets")
        for snippet in snippets:
            if not isinstance(snippet, str) or len(snippet.strip()) < 3:
                raise ValueError(f"{role} reasoning step {index} has an invalid support snippet")
            normalized_snippet = normalize_support_text(snippet.strip())
            if not any(
                normalized_snippet in normalize_support_text(source) for source in support_sources
            ):
                raise ValueError(
                    f"{role} reasoning step {index} quoted text absent from visible materials"
                )

    cited_collections: list[tuple[str, Any]] = []
    if role == "prosecutor":
        brief = output.get("prosecution_brief") or {}
        cited_collections.extend([
            ("prosecution theory", [brief.get("theory_of_case")]),
            ("prosecution facts", brief.get("facts_asserted")),
            ("prosecution evidence arguments", brief.get("evidence_arguments")),
        ])
    elif role == "defense":
        brief = output.get("defense_brief") or {}
        cited_collections.extend([
            ("defense admissions", brief.get("admissions")),
            ("defense disputes", brief.get("disputes")),
            ("defense evidence arguments", brief.get("evidence_arguments")),
        ])
    else:
        decision = output.get("decision") or {}
        cited_collections.extend([
            ("judge issue findings", decision.get("issue_findings")),
            ("judge response to prosecution", decision.get("response_to_prosecution")),
            ("judge response to defense", decision.get("response_to_defense")),
            ("judge rationale", decision.get("rationale")),
        ])
    for label, items in cited_collections:
        if items is None or not isinstance(items, list):
            raise ValueError(f"{label} must be a list")
        for index, item in enumerate(items, 1):
            if not isinstance(item, dict) or not item.get("evidence_ids"):
                raise ValueError(f"{label} item {index} has no evidence_ids")
    cited: list[str] = []
    collect_evidence_ids(output, cited)
    unknown = sorted(set(cited) - allowed_evidence_ids)
    if unknown:
        raise ValueError(f"{role} cited unknown evidence IDs: {', '.join(unknown)}")
    if not cited:
        raise ValueError(f"{role} cited no evidence")
    if role == "defense":
        for item in output.get("hypotheticals") or []:
            if not isinstance(item, dict) or item.get("is_asserted_fact") is not False:
                raise ValueError("defense hypothetical must set is_asserted_fact=false")
    if role == "judge":
        actual = (output.get("decision") or {}).get("target_holding")
        if actual != target_holding:
            raise ValueError("judge did not copy the exact target holding")


def extract_json_object(answer: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip())
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object, got {type(value).__name__}")
    return value


def request_role(key: str, prompt: str) -> tuple[dict, dict]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
        "reasoning": {"effort": "medium", "exclude": True},
        "provider": {
            "only": [PROVIDER],
            "quantizations": [QUANTIZATION],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "max_price": {
                "prompt": PROMPT_PRICE_PER_M,
                "completion": COMPLETION_PRICE_PER_M,
            },
        },
    }
    envelope = api_json(f"{API_ROOT}/chat/completions", key, payload)
    observed_provider = str(envelope.get("provider") or "")
    if observed_provider.lower().replace(" ", "") != PROVIDER_DISPLAY.lower():
        raise RuntimeError(f"unexpected provider: {observed_provider!r}")
    choices = envelope.get("choices") or []
    if not choices:
        raise RuntimeError("OpenRouter response contained no choices")
    answer = str(choices[0].get("message", {}).get("content") or "")
    usage = dict(envelope.get("usage") or {})
    usage.update({
        "openrouter_generation_id": envelope.get("id"),
        "provider": observed_provider,
        "requested_quantization": QUANTIZATION,
    })
    try:
        output = extract_json_object(answer)
    except (json.JSONDecodeError, ValueError) as error:
        raise InvalidRoleResponseError(error, usage, answer) from error
    return output, usage


def conservative_cost(prompts: list[str], calls: int) -> float:
    # UTF-8 bytes deliberately overestimate input tokens for CJK text.  Upstream
    # model outputs are additionally budgeted as full-size input to later roles.
    static_input = sum(len(prompt.encode("utf-8")) for prompt in prompts)
    upstream_output_tokens = calls * MAX_OUTPUT_TOKENS
    return (
        (static_input + upstream_output_tokens) / 1_000_000 * PROMPT_PRICE_PER_M
        + calls * MAX_OUTPUT_TOKENS / 1_000_000 * COMPLETION_PRICE_PER_M
    )


def write_records(path: Path, records: list[dict]) -> None:
    by_id = {str(row["pair_id"]): row for row in records}
    hybrid.write_jsonl(path, by_id, [str(row["pair_id"]) for row in records])


def upsert_record(path: Path, record: dict) -> None:
    existing = hybrid.read_jsonl(path)
    by_id = {str(row["pair_id"]): row for row in existing}
    order = [str(row["pair_id"]) for row in existing]
    pair_id = str(record["pair_id"])
    if pair_id not in by_id:
        order.append(pair_id)
    by_id[pair_id] = record
    hybrid.write_jsonl(path, by_id, order)


def safe_http_detail(error: Exception) -> str:
    if isinstance(error, urllib.error.HTTPError):
        return error.read().decode("utf-8", errors="replace")[:1200]
    return str(error)[:1200]


def main() -> None:
    args = parse_args()
    if not 1 <= args.limit <= MAX_PAIRS:
        raise SystemExit(f"--limit must be between 1 and {MAX_PAIRS}")
    if len(args.pair_id) > args.limit:
        raise SystemExit("number of --pair-id values cannot exceed --limit")

    all_pairs = load_pairs(args.indictments, args.judgments)
    selected = select_pairs(all_pairs, args.limit, args.pair_id)
    if len(selected) != args.limit:
        raise SystemExit(f"requested {args.limit} pairs but found {len(selected)}")
    materials = [case_materials(pair) for pair in selected]
    route = verify_route()

    # Only prompts knowable before execution are used here.  The cost function
    # separately reserves full upstream-output allowances for subsequent calls.
    initial_prompts = [prompt_for_prosecutor(item) for item in materials]
    initial_prompts += [
        json.dumps({
            "indictment": item["indictment_text"],
            "facts": [item["indictment_crime_facts"], item["judgment_crime_facts"]],
            "evidence": item["all_evidence"],
            "holding": item["target_holding"],
        }, ensure_ascii=False)
        for item in materials for _ in range(2)
    ]
    calls = len(materials) * 3
    ceiling = conservative_cost(initial_prompts, calls)
    summary = {
        "execute": args.execute,
        "gemma_only_pairs_available": len(all_pairs),
        "selection_strategy": "shortest conviction + shortest acquittal (+ shortest mixed)",
        "pairs": len(selected),
        "calls": calls,
        "pair_ids": [pair["pair_id"] for pair in selected],
        "outcomes": [pair["outcome_class"] for pair in selected],
        "model": MODEL,
        "provider": route.get("provider_name"),
        "quantization": route.get("quantization"),
        "conservative_worst_case_usd": round(ceiling, 6),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if ceiling > MAX_WORST_CASE_USD:
        raise SystemExit(
            f"worst-case cost ${ceiling:.6f} exceeds hard cap ${MAX_WORST_CASE_USD}"
        )
    if not args.execute:
        print("dry-run only; pass --execute to send at most 3 calls per pair", flush=True)
        return

    key = load_openrouter_key()
    try:
        verify_key(key)
    except urllib.error.HTTPError as error:
        raise SystemExit(
            f"OpenRouter key preflight failed with HTTP {error.code}; no model request sent"
        ) from None

    completed: list[dict] = []
    failures: list[dict] = []
    all_request_usage: list[dict] = []
    resumable = {
        str(row.get("pair_id")): row.get("partial_record")
        for row in hybrid.read_jsonl(args.failures)
        if isinstance(row.get("partial_record"), dict)
    } if args.resume else {}
    for item in materials:
        pair_id = item["pair_id"]
        allowed_i = {row["evidence_id"] for row in item["indictment_evidence"]}
        allowed_all = {row["evidence_id"] for row in item["all_evidence"]}
        fresh_record = {
            "experiment": "gemma_only_three_role_legal_reasoning",
            "restricted": True,
            "pair_id": pair_id,
            "model": MODEL,
            "provider_requested": PROVIDER,
            "quantization_requested": QUANTIZATION,
            "source_contract": {
                "indictment": "hybrid_gemma4_experiment.jsonl",
                "judgment": "hybrid_judgments_gemma4_experiment.jsonl",
                "both_models_verified": "gemma-4-31b-it",
                "material_sha256": sha256_json(item),
            },
            "visibility_contract": {
                "prosecutor": ["indictment_text", "indictment_crime_facts", "indictment_evidence"],
                "defense": ["indictment_text", "prosecution_brief", "all_crime_facts", "all_evidence"],
                "judge": ["indictment_text", "prosecution_brief", "defense_brief", "all_crime_facts", "all_evidence", "target_holding"],
            },
            "outcome_class": item["outcome_class"],
            "target_holding": item["target_holding"],
            "evidence_catalog": item["all_evidence"],
            "roles": {},
            "usage": {},
        }
        record = fresh_record
        prior = resumable.get(pair_id)
        if (
            isinstance(prior, dict)
            and prior.get("model") == MODEL
            and (prior.get("source_contract") or {}).get("material_sha256")
            == fresh_record["source_contract"]["material_sha256"]
        ):
            record = prior
            print(f"resuming pair={pair_id} saved_roles={sorted(record.get('roles') or {})}", flush=True)
        try:
            prosecutor = (record.get("roles") or {}).get("prosecutor")
            if prosecutor is None:
                prosecutor, usage = request_role(key, prompt_for_prosecutor(item))
                record["roles"]["prosecutor"] = prosecutor
                record["usage"]["prosecutor"] = usage
                all_request_usage.append(usage)
            validate_role_output(
                prosecutor, "prosecutor", allowed_i, grounding_sources(item, True)
            )

            defense = (record.get("roles") or {}).get("defense")
            if defense is None:
                defense, usage = request_role(key, prompt_for_defense(item, prosecutor))
                record["roles"]["defense"] = defense
                record["usage"]["defense"] = usage
                all_request_usage.append(usage)
            validate_role_output(
                defense, "defense", allowed_all, grounding_sources(item, False)
            )

            judge = (record.get("roles") or {}).get("judge")
            if judge is None:
                judge, usage = request_role(key, prompt_for_judge(item, prosecutor, defense))
                record["roles"]["judge"] = judge
                record["usage"]["judge"] = usage
                all_request_usage.append(usage)
            validate_role_output(
                judge,
                "judge",
                allowed_all,
                grounding_sources(item, False),
                item["target_holding"],
            )
            record["checkpointed_at"] = utc_now()
            completed.append(record)
            upsert_record(args.output, record)
            print(f"completed pair={pair_id} roles=3", flush=True)
        except Exception as error:
            unrecorded_usage = getattr(error, "usage", None)
            if isinstance(unrecorded_usage, dict):
                all_request_usage.append(unrecorded_usage)
            failure_record = {
                "experiment": "gemma_only_three_role_legal_reasoning",
                "restricted": True,
                "pair_id": pair_id,
                "completed_roles": sorted(record["roles"]),
                "error_type": type(error).__name__,
                "detail": safe_http_detail(error),
                "failed_response_preview": getattr(error, "answer_preview", ""),
                "partial_record": record,
                "checkpointed_at": utc_now(),
            }
            failures.append(failure_record)
            upsert_record(args.failures, failure_record)
            print(
                f"failed pair={pair_id} after={sorted(record['roles'])} type={type(error).__name__}",
                flush=True,
            )
        time.sleep(0.5)

    total_cost = sum(float(usage.get("cost") or 0) for usage in all_request_usage)
    print(json.dumps({
        "completed_pairs": len(completed),
        "failures": len(failures),
        "reported_cost_usd": round(total_cost, 8),
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
