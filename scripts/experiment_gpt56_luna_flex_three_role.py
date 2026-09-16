#!/usr/bin/env python3
"""One-pair GPT-5.6 Luna Flex comparison using the three-role pipeline."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

try:
    from scripts import generate_three_role_legal_reasoning as base
except (ModuleNotFoundError, ImportError):
    import generate_three_role_legal_reasoning as base


MODEL = "openai/gpt-5.6-luna"
PROVIDER = "openai"
PROVIDER_DISPLAY = "OpenAI"
SERVICE_TIER = "flex"
PROMPT_PRICE_PER_M = 0.10
COMPLETION_PRICE_PER_M = 0.60
MAX_OUTPUT_TOKENS = 6144
MAX_WORST_CASE_USD = 0.02
PAIR_ID = "f0ef9188c5324c13baf4309235abab6a"

DEFAULT_OUTPUT = (
    base.ROOT / "data/intermediate/openrouter/gpt56_luna_flex_three_role_experiment.jsonl"
)
DEFAULT_FAILURES = (
    base.ROOT / "data/intermediate/openrouter/gpt56_luna_flex_three_role_failures.jsonl"
)


class InvalidFlexResponseError(ValueError):
    def __init__(self, cause: Exception, usage: dict, answer: str):
        super().__init__(f"invalid model JSON: {cause}; visible_chars={len(answer)}")
        self.usage = usage
        self.answer_preview = answer[:2000]


def verify_flex_route() -> dict:
    author, slug = MODEL.split("/", 1)
    envelope = base.api_json(f"{base.API_ROOT}/models/{author}/{slug}/endpoints")
    matches = [
        endpoint
        for endpoint in envelope.get("data", {}).get("endpoints", [])
        if endpoint.get("provider_name") == PROVIDER_DISPLAY
        and endpoint.get("tag") == "openai/flex"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one OpenAI Flex endpoint, found {len(matches)}")
    endpoint = matches[0]
    prompt_price = float(endpoint["pricing"]["prompt"]) * 1_000_000
    completion_price = float(endpoint["pricing"]["completion"]) * 1_000_000
    if prompt_price > PROMPT_PRICE_PER_M or completion_price > COMPLETION_PRICE_PER_M:
        raise RuntimeError(
            f"Flex route price exceeds cap: prompt={prompt_price}, completion={completion_price}"
        )
    required = {"response_format", "reasoning", "max_tokens"}
    missing = required - set(endpoint.get("supported_parameters", []))
    if missing:
        raise RuntimeError(f"OpenAI Flex endpoint lacks parameters: {sorted(missing)}")
    return endpoint


def request_role(key: str, prompt: str) -> tuple[dict, dict]:
    payload = {
        "model": MODEL,
        "service_tier": SERVICE_TIER,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
        "reasoning": {"effort": "medium", "exclude": True},
        "provider": {
            "only": [PROVIDER],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "max_price": {
                "prompt": PROMPT_PRICE_PER_M,
                "completion": COMPLETION_PRICE_PER_M,
            },
        },
    }
    envelope = base.api_json(f"{base.API_ROOT}/chat/completions", key, payload)
    provider = str(envelope.get("provider") or "")
    tier = str(envelope.get("service_tier") or "")
    if provider.lower() != PROVIDER_DISPLAY.lower():
        raise RuntimeError(f"unexpected provider: {provider!r}")
    if tier.lower() != SERVICE_TIER:
        raise RuntimeError(f"requested Flex but response reported service_tier={tier!r}")
    choices = envelope.get("choices") or []
    if not choices:
        raise RuntimeError("OpenRouter response contained no choices")
    answer = str(choices[0].get("message", {}).get("content") or "")
    usage = dict(envelope.get("usage") or {})
    usage.update({
        "openrouter_generation_id": envelope.get("id"),
        "provider": provider,
        "service_tier": tier,
    })
    try:
        output = base.extract_json_object(answer)
    except (json.JSONDecodeError, ValueError) as error:
        raise InvalidFlexResponseError(error, usage, answer) from error
    return output, usage


def conservative_cost(materials: dict) -> float:
    static_prompts = [
        base.prompt_for_prosecutor(materials),
        json.dumps({
            "indictment": materials["indictment_text"],
            "facts": [
                materials["indictment_crime_facts"],
                materials["judgment_crime_facts"],
            ],
            "evidence": materials["all_evidence"],
            "holding": materials["target_holding"],
        }, ensure_ascii=False),
    ] * 2
    static_bytes = sum(len(prompt.encode("utf-8")) for prompt in static_prompts)
    upstream_tokens = 3 * MAX_OUTPUT_TOKENS
    return (
        (static_bytes + upstream_tokens) / 1_000_000 * PROMPT_PRICE_PER_M
        + upstream_tokens / 1_000_000 * COMPLETION_PRICE_PER_M
    )


def main() -> None:
    route = verify_flex_route()
    pairs = base.load_pairs(base.DEFAULT_INDICTMENTS, base.DEFAULT_JUDGMENTS)
    selected = base.select_pairs(pairs, 1, [PAIR_ID])
    materials = base.case_materials(selected[0])
    ceiling = conservative_cost(materials)
    print(json.dumps({
        "pair_id": PAIR_ID,
        "model": MODEL,
        "provider": route.get("provider_name"),
        "endpoint_tag": route.get("tag"),
        "service_tier": SERVICE_TIER,
        "calls": 3,
        "worst_case_usd": round(ceiling, 6),
    }, ensure_ascii=False, indent=2), flush=True)
    if ceiling > MAX_WORST_CASE_USD:
        raise SystemExit(
            f"worst-case cost ${ceiling:.6f} exceeds hard cap ${MAX_WORST_CASE_USD}"
        )

    key = base.load_openrouter_key()
    try:
        base.verify_key(key)
    except urllib.error.HTTPError as error:
        raise SystemExit(
            f"OpenRouter key preflight failed with HTTP {error.code}; no model request sent"
        ) from None

    allowed_i = {row["evidence_id"] for row in materials["indictment_evidence"]}
    allowed_all = {row["evidence_id"] for row in materials["all_evidence"]}
    record = {
        "experiment": "gpt56_luna_flex_three_role_comparison",
        "restricted": True,
        "pair_id": PAIR_ID,
        "model": MODEL,
        "provider_requested": PROVIDER,
        "service_tier_requested": SERVICE_TIER,
        "source_contract": {
            "indictment": "hybrid_gemma4_experiment.jsonl",
            "judgment": "hybrid_judgments_gemma4_experiment.jsonl",
            "both_models_verified": "gemma-4-31b-it",
            "material_sha256": base.sha256_json(materials),
        },
        "outcome_class": materials["outcome_class"],
        "target_holding": materials["target_holding"],
        "evidence_catalog": materials["all_evidence"],
        "roles": {},
        "usage": {},
    }
    request_usage: list[dict] = []
    try:
        prosecutor, usage = request_role(key, base.prompt_for_prosecutor(materials))
        record["roles"]["prosecutor"] = prosecutor
        record["usage"]["prosecutor"] = usage
        request_usage.append(usage)
        base.validate_role_output(
            prosecutor, "prosecutor", allowed_i, base.grounding_sources(materials, True)
        )
        print("prosecutor passed", flush=True)

        defense, usage = request_role(key, base.prompt_for_defense(materials, prosecutor))
        record["roles"]["defense"] = defense
        record["usage"]["defense"] = usage
        request_usage.append(usage)
        base.validate_role_output(
            defense, "defense", allowed_all, base.grounding_sources(materials, False)
        )
        print("defense passed", flush=True)

        judge, usage = request_role(key, base.prompt_for_judge(materials, prosecutor, defense))
        record["roles"]["judge"] = judge
        record["usage"]["judge"] = usage
        request_usage.append(usage)
        base.validate_role_output(
            judge,
            "judge",
            allowed_all,
            base.grounding_sources(materials, False) + [
                json.dumps(prosecutor, ensure_ascii=False),
                json.dumps(defense, ensure_ascii=False),
            ],
            materials["target_holding"],
        )
        record["checkpointed_at"] = base.utc_now()
        base.upsert_record(DEFAULT_OUTPUT, record)
        print("judge passed", flush=True)
    except Exception as error:
        usage = getattr(error, "usage", None)
        if isinstance(usage, dict):
            request_usage.append(usage)
        failure = {
            "experiment": record["experiment"],
            "restricted": True,
            "pair_id": PAIR_ID,
            "completed_roles": sorted(record["roles"]),
            "error_type": type(error).__name__,
            "detail": base.safe_http_detail(error),
            "failed_response_preview": getattr(error, "answer_preview", ""),
            "partial_record": record,
            "checkpointed_at": base.utc_now(),
        }
        base.upsert_record(DEFAULT_FAILURES, failure)
        print(f"failed after={sorted(record['roles'])} type={type(error).__name__}", flush=True)

    print(json.dumps({
        "roles_returned": sorted(record["roles"]),
        "reported_cost_usd": round(
            sum(float(usage.get("cost") or 0) for usage in request_usage), 8
        ),
        "output": str(DEFAULT_OUTPUT.resolve()),
        "failures": str(DEFAULT_FAILURES.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
