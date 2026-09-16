#!/usr/bin/env python3
"""Checkpointed bulk GPT-5.6 Luna Flex three-role data generation.

The source is restricted to paired Gemma 4 deidentification checkpoints.  Each
role is persisted immediately.  Requests are never retried automatically; a
rejected or blocked request is recorded and the worker moves to the next pair.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import json
import threading
import unicodedata
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts import experiment_gpt56_luna_flex_three_role as luna
    from scripts import generate_three_role_legal_reasoning as base
    from scripts import hybrid_deidentify_with_google_ai as hybrid
except (ModuleNotFoundError, ImportError):
    import experiment_gpt56_luna_flex_three_role as luna
    import generate_three_role_legal_reasoning as base
    import hybrid_deidentify_with_google_ai as hybrid


MODEL = luna.MODEL
SERVICE_TIER = luna.SERVICE_TIER
MAX_OUTPUT_TOKENS = luna.MAX_OUTPUT_TOKENS
DEFAULT_OUTPUT = (
    base.ROOT / "data/intermediate/openrouter/gpt56_luna_flex_three_role_bulk.jsonl"
)
DEFAULT_CHECKPOINTS = (
    base.ROOT / "data/intermediate/openrouter/gpt56_luna_flex_three_role_checkpoints.jsonl"
)
DEFAULT_FAILURES = (
    base.ROOT / "data/intermediate/openrouter/gpt56_luna_flex_three_role_bulk_failures.jsonl"
)
DEFAULT_PROGRESS = (
    base.ROOT / "data/intermediate/openrouter/gpt56_luna_flex_three_role_progress.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk Gemma-only GPT-5.6 Luna Flex pipeline")
    parser.add_argument("--limit", type=int, default=0, help="0 means all eligible pairs")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-total-cost-usd", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def string_schema() -> dict:
    return {"type": "string"}


def evidence_ids_schema() -> dict:
    return {
        "type": "array",
        "items": {"type": "string", "pattern": "^[IJ]-E[0-9]{3}$"},
        "minItems": 1,
    }


def cited_object_schema(text_fields: list[str]) -> dict:
    properties = {field: string_schema() for field in text_fields}
    properties["evidence_ids"] = evidence_ids_schema()
    return {
        "type": "object",
        "properties": properties,
        "required": [*text_fields, "evidence_ids"],
        "additionalProperties": False,
    }


def trace_schema(max_items: int) -> dict:
    fields = [
        "step_id", "claim_type", "proposition", "evidence_interpretation",
        "inference", "uncertainty",
    ]
    properties = {field: string_schema() for field in fields}
    properties["claim_type"] = {
        "type": "string",
        "enum": [
            "fact", "evidence_observation", "inference", "legal_rule",
            "legal_application", "hypothetical", "conclusion",
        ],
    }
    properties["evidence_ids"] = evidence_ids_schema()
    properties["support_snippets"] = {
        "type": "array", "items": string_schema(), "minItems": 1, "maxItems": 5,
    }
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": properties,
            "required": [*fields, "evidence_ids", "support_snippets"],
            "additionalProperties": False,
        },
        "minItems": 1,
        "maxItems": max_items,
    }


def common_properties(role: str, max_steps: int) -> dict:
    return {
        "role": {"type": "string", "const": role},
        "issues": {
            "type": "array", "items": string_schema(), "minItems": 1, "maxItems": 6,
        },
        "reasoning_trace": trace_schema(max_steps),
        "limitations": {"type": "array", "items": string_schema(), "maxItems": 8},
        "new_evidence_claimed": {
            "type": "array", "items": string_schema(), "maxItems": 0,
        },
    }


def role_schema(role: str) -> dict:
    max_steps = {"prosecutor": 6, "defense": 7, "judge": 6}[role]
    properties = common_properties(role, max_steps)
    required = ["role", "issues", "reasoning_trace", "limitations", "new_evidence_claimed"]
    if role == "prosecutor":
        properties["prosecution_brief"] = {
            "type": "object",
            "properties": {
                "theory_of_case": cited_object_schema(["statement"]),
                "facts_asserted": {
                    "type": "array", "items": cited_object_schema(["statement"]), "maxItems": 10,
                },
                "evidence_arguments": {
                    "type": "array", "items": cited_object_schema(["argument"]), "maxItems": 10,
                },
                "legal_position": {"type": "array", "items": string_schema(), "maxItems": 8},
                "requested_disposition": string_schema(),
            },
            "required": [
                "theory_of_case", "facts_asserted", "evidence_arguments",
                "legal_position", "requested_disposition",
            ],
            "additionalProperties": False,
        }
        required.append("prosecution_brief")
    elif role == "defense":
        properties["defense_brief"] = {
            "type": "object",
            "properties": {
                "defense_theory": string_schema(),
                "admissions": {
                    "type": "array", "items": cited_object_schema(["statement"]), "maxItems": 10,
                },
                "disputes": {
                    "type": "array", "items": cited_object_schema(["statement"]), "maxItems": 10,
                },
                "evidence_arguments": {
                    "type": "array", "items": cited_object_schema(["argument"]), "maxItems": 12,
                },
                "legal_position": {"type": "array", "items": string_schema(), "maxItems": 8},
                "requested_disposition": string_schema(),
            },
            "required": [
                "defense_theory", "admissions", "disputes", "evidence_arguments",
                "legal_position", "requested_disposition",
            ],
            "additionalProperties": False,
        }
        properties["hypotheticals"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "possibility": string_schema(),
                    "is_asserted_fact": {"type": "boolean", "const": False},
                },
                "required": ["possibility", "is_asserted_fact"],
                "additionalProperties": False,
            },
            "maxItems": 5,
        }
        required.extend(["defense_brief", "hypotheticals"])
    else:
        properties["decision"] = {
            "type": "object",
            "properties": {
                "target_holding": string_schema(),
                "issue_findings": {
                    "type": "array",
                    "items": cited_object_schema(["issue", "finding"]),
                    "minItems": 1,
                    "maxItems": 8,
                },
                "response_to_prosecution": {
                    "type": "array",
                    "items": cited_object_schema(["response"]),
                    "maxItems": 6,
                },
                "response_to_defense": {
                    "type": "array",
                    "items": cited_object_schema(["response"]),
                    "maxItems": 6,
                },
                "rationale": {
                    "type": "array",
                    "items": cited_object_schema(["statement"]),
                    "minItems": 1,
                    "maxItems": 8,
                },
            },
            "required": [
                "target_holding", "issue_findings", "response_to_prosecution",
                "response_to_defense", "rationale",
            ],
            "additionalProperties": False,
        }
        required.append("decision")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def request_role(key: str, prompt: str, role: str) -> tuple[dict, dict]:
    payload = {
        "model": MODEL,
        "service_tier": SERVICE_TIER,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": f"lawrence_{role}_reasoning",
                "strict": True,
                "schema": role_schema(role),
            },
        },
        "reasoning": {"effort": "medium", "exclude": True},
        "provider": {
            "only": [luna.PROVIDER],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "max_price": {
                "prompt": luna.PROMPT_PRICE_PER_M,
                "completion": luna.COMPLETION_PRICE_PER_M,
            },
        },
    }
    envelope = base.api_json(f"{base.API_ROOT}/chat/completions", key, payload)
    provider = str(envelope.get("provider") or "")
    tier = str(envelope.get("service_tier") or "")
    if provider.lower() != luna.PROVIDER_DISPLAY.lower():
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
        wrapped = luna.InvalidFlexResponseError(error, usage, answer)
        raise wrapped from error
    return output, usage


def append_jsonl(path: Path, row: dict, lock: threading.Lock) -> None:
    encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="") as stream:
            stream.write(encoded)
            stream.flush()


def latest_checkpoints(path: Path) -> dict[tuple[str, str], dict]:
    latest: dict[tuple[str, str], dict] = {}
    for row in hybrid.read_jsonl(path):
        pair_id = str(row.get("pair_id") or "")
        role = str(row.get("role") or "")
        if pair_id and role:
            latest[(pair_id, role)] = row
    return latest


def completed_ids(path: Path) -> set[str]:
    return {
        str(row.get("pair_id"))
        for row in hybrid.read_jsonl(path)
        if row.get("status") == "complete" and len(row.get("roles") or {}) == 3
    }


def failed_ids(path: Path) -> set[str]:
    return {str(row.get("pair_id")) for row in hybrid.read_jsonl(path)}


def request_ceiling(prompt: str) -> float:
    return (
        # Flex cache writes are priced at $0.125/M, slightly above ordinary input.
        len(prompt.encode("utf-8")) / 1_000_000 * 0.125
        + MAX_OUTPUT_TOKENS / 1_000_000 * luna.COMPLETION_PRICE_PER_M
    )


def normalized_with_source_map(value: str) -> tuple[str, list[int]]:
    characters: list[str] = []
    source_indexes: list[int] = []
    for source_index, character in enumerate(value):
        for normalized in unicodedata.normalize("NFKC", character):
            if not normalized.isspace():
                characters.append(normalized)
                source_indexes.append(source_index)
    return "".join(characters), source_indexes


def closest_exact_source_quote(snippet: str, sources: list[str]) -> str | None:
    needle = base.normalize_support_text(snippet)
    if len(needle) < 8:
        return None
    best: tuple[float, str] | None = None
    anchors = [(offset, needle[offset:offset + 4]) for offset in range(0, len(needle) - 3, 4)]
    for source in sources:
        normalized, index_map = normalized_with_source_map(source)
        candidates: set[int] = set()
        for offset, anchor in anchors:
            start = 0
            while True:
                position = normalized.find(anchor, start)
                if position < 0:
                    break
                candidates.add(max(0, position - offset))
                start = position + 1
        for candidate_start in candidates:
            for delta in range(-3, 4):
                start = max(0, candidate_start + delta)
                end = min(len(normalized), start + len(needle))
                candidate = normalized[start:end]
                score = difflib.SequenceMatcher(None, needle, candidate).ratio()
                if score < 0.90 or not candidate or end <= start:
                    continue
                original_start = index_map[start]
                original_end = index_map[end - 1] + 1
                exact = source[original_start:original_end]
                if best is None or score > best[0]:
                    best = (score, exact)
    return best[1] if best else None


def repair_support_snippets(output: dict, sources: list[str]) -> list[dict]:
    repairs: list[dict] = []
    for step in output.get("reasoning_trace") or []:
        snippets = step.get("support_snippets") or []
        for index, snippet in enumerate(list(snippets)):
            if any(
                base.normalize_support_text(snippet) in base.normalize_support_text(source)
                for source in sources
            ):
                continue
            replacement = closest_exact_source_quote(snippet, sources)
            if replacement is None:
                continue
            snippets[index] = replacement
            repairs.append({
                "step_id": step.get("step_id"),
                "action": "replace_with_exact_source_quote",
                "original": snippet,
                "replacement": replacement,
            })
        valid = [
            snippet for snippet in snippets
            if any(
                base.normalize_support_text(snippet) in base.normalize_support_text(source)
                for source in sources
            )
        ]
        if valid:
            for snippet in snippets:
                if snippet not in valid:
                    repairs.append({
                        "step_id": step.get("step_id"),
                        "action": "drop_unverifiable_extra_quote",
                        "original": snippet,
                        "replacement": None,
                    })
            step["support_snippets"] = valid
    return repairs


class RunState:
    def __init__(self, total: int, max_cost: float, lock: threading.Lock, progress: Path):
        self.total = total
        self.max_cost = max_cost
        self.lock = lock
        self.progress_path = progress
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.roles = 0
        self.actual_cost = 0.0
        self.reserved_cost = 0.0
        self.stop = False

    def reserve(self, amount: float) -> bool:
        with self.lock:
            if self.stop or self.actual_cost + self.reserved_cost + amount > self.max_cost:
                self.stop = True
                self._write_progress_locked()
                return False
            self.reserved_cost += amount
            return True

    def settle(self, reserved: float, usage: dict | None) -> None:
        with self.lock:
            self.reserved_cost = max(0.0, self.reserved_cost - reserved)
            if usage:
                self.actual_cost += float(usage.get("cost") or 0)
            self._write_progress_locked()

    def mark(self, kind: str) -> None:
        with self.lock:
            setattr(self, kind, getattr(self, kind) + 1)
            self._write_progress_locked()

    def snapshot(self) -> dict:
        return {
            "total_scheduled": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "roles_checkpointed": self.roles,
            "actual_cost_usd": round(self.actual_cost, 8),
            "reserved_cost_usd": round(self.reserved_cost, 8),
            "max_total_cost_usd": self.max_cost,
            "stopped_by_cost_cap": self.stop,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def _write_progress_locked(self) -> None:
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.progress_path.with_suffix(self.progress_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.progress_path)


def error_detail(error: Exception) -> tuple[str, int | None]:
    if isinstance(error, urllib.error.HTTPError):
        body = error.read().decode("utf-8", errors="replace")[:4000]
        return body, error.code
    return str(error)[:4000], None


def prompt_for(role: str, materials: dict, roles: dict) -> str:
    suffix = (
        "\n\n格式補充：reasoning_trace 每一步都必須填 claim_type，且只能是 "
        "fact、evidence_observation、inference、legal_rule、legal_application、"
        "hypothetical、conclusion 之一。請避免重複步驟。"
    )
    if role == "prosecutor":
        return base.prompt_for_prosecutor(materials) + suffix
    if role == "defense":
        return base.prompt_for_defense(materials, roles["prosecutor"]) + suffix
    return (
        base.prompt_for_judge(materials, roles["prosecutor"], roles["defense"])
        + suffix
    )


def validate(role: str, output: dict, materials: dict, roles: dict) -> None:
    allowed_i = {row["evidence_id"] for row in materials["indictment_evidence"]}
    allowed_all = {row["evidence_id"] for row in materials["all_evidence"]}
    if role == "prosecutor":
        base.validate_role_output(
            output, role, allowed_i, base.grounding_sources(materials, True)
        )
    elif role == "defense":
        base.validate_role_output(
            output, role, allowed_all, base.grounding_sources(materials, False)
        )
    else:
        base.validate_role_output(
            output,
            role,
            allowed_all,
            base.grounding_sources(materials, False) + [
                json.dumps(roles["prosecutor"], ensure_ascii=False),
                json.dumps(roles["defense"], ensure_ascii=False),
            ],
            materials["target_holding"],
        )


def support_sources(role: str, materials: dict, roles: dict) -> list[str]:
    if role == "prosecutor":
        return base.grounding_sources(materials, True)
    sources = base.grounding_sources(materials, False)
    if role == "judge":
        sources += [
            json.dumps(roles["prosecutor"], ensure_ascii=False),
            json.dumps(roles["defense"], ensure_ascii=False),
        ]
    return sources


def process_pair(
    pair: dict,
    key: str,
    checkpoints: dict[tuple[str, str], dict],
    args: argparse.Namespace,
    state: RunState,
    file_lock: threading.Lock,
) -> None:
    pair_id = pair["pair_id"]
    materials = base.case_materials(pair)
    material_hash = base.sha256_json(materials)
    roles: dict[str, dict] = {}
    usage_by_role: dict[str, dict] = {}
    for role in ("prosecutor", "defense", "judge"):
        saved = checkpoints.get((pair_id, role))
        if (
            saved
            and saved.get("status") == "validated"
            and saved.get("material_sha256") == material_hash
        ):
            roles[role] = saved["output"]
            usage_by_role[role] = saved.get("usage") or {}
            validate(role, roles[role], materials, roles)
            continue

        prompt = prompt_for(role, materials, roles)
        reservation = request_ceiling(prompt)
        if not state.reserve(reservation):
            state.mark("skipped")
            return
        usage: dict | None = None
        try:
            output, usage = request_role(key, prompt, role)
            roles[role] = output
            usage_by_role[role] = usage
            repairs = repair_support_snippets(output, support_sources(role, materials, roles))
            validate(role, output, materials, roles)
            append_jsonl(args.checkpoints, {
                "pair_id": pair_id,
                "role": role,
                "status": "validated",
                "material_sha256": material_hash,
                "model": MODEL,
                "service_tier": SERVICE_TIER,
                "output": output,
                "deterministic_support_repairs": repairs,
                "usage": usage,
                "checkpointed_at": base.utc_now(),
            }, file_lock)
            state.mark("roles")
            print(f"role_ok pair={pair_id} role={role}", flush=True)
        except Exception as error:
            failed_usage = getattr(error, "usage", None)
            if isinstance(failed_usage, dict):
                usage = failed_usage
            detail, http_status = error_detail(error)
            append_jsonl(args.failures, {
                "pair_id": pair_id,
                "failed_role": role,
                "completed_roles": sorted(roles),
                "model": MODEL,
                "service_tier": SERVICE_TIER,
                "http_status": http_status,
                "terminal_for_this_run": True,
                "no_automatic_retry": True,
                "error_type": type(error).__name__,
                "detail": detail,
                "usage": usage or {},
                "returned_output": roles.get(role),
                "failed_response_preview": getattr(error, "answer_preview", ""),
                "checkpointed_at": base.utc_now(),
            }, file_lock)
            state.mark("failed")
            print(
                f"pair_skip pair={pair_id} role={role} http={http_status} type={type(error).__name__}",
                flush=True,
            )
            return
        finally:
            state.settle(reservation, usage)

    append_jsonl(args.output, {
        "schema_version": "1.0",
        "experiment": "gpt56_luna_flex_three_role_bulk",
        "restricted": True,
        "status": "complete",
        "pair_id": pair_id,
        "model": MODEL,
        "provider": luna.PROVIDER_DISPLAY,
        "service_tier": SERVICE_TIER,
        "source_contract": {
            "indictment": "hybrid_gemma4_experiment.jsonl",
            "judgment": "hybrid_judgments_gemma4_experiment.jsonl",
            "both_models_verified": "gemma-4-31b-it",
            "material_sha256": material_hash,
        },
        "outcome_class": materials["outcome_class"],
        "target_holding": materials["target_holding"],
        "evidence_catalog": materials["all_evidence"],
        "roles": roles,
        "usage": usage_by_role,
        "checkpointed_at": base.utc_now(),
    }, file_lock)
    state.mark("completed")
    print(f"pair_complete pair={pair_id}", flush=True)


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit cannot be negative")
    if not 1 <= args.workers <= 16:
        raise SystemExit("--workers must be between 1 and 16")
    if args.max_total_cost_usd <= 0:
        raise SystemExit("--max-total-cost-usd must be positive")

    route = luna.verify_flex_route()
    all_pairs = base.load_pairs(base.DEFAULT_INDICTMENTS, base.DEFAULT_JUDGMENTS)
    all_pairs.sort(key=lambda pair: pair["pair_id"])
    done = completed_ids(args.output)
    prior_failures = set() if args.retry_failures else failed_ids(args.failures)
    pending = [
        pair for pair in all_pairs
        if pair["pair_id"] not in done and pair["pair_id"] not in prior_failures
    ]
    if args.limit:
        pending = pending[:args.limit]
    checkpoints = latest_checkpoints(args.checkpoints)

    print(json.dumps({
        "eligible_gemma_pairs": len(all_pairs),
        "already_complete": len(done),
        "prior_failures_skipped": len(prior_failures),
        "scheduled": len(pending),
        "workers": args.workers,
        "model": MODEL,
        "provider": route.get("provider_name"),
        "endpoint_tag": route.get("tag"),
        "service_tier": SERVICE_TIER,
        "max_total_cost_usd": args.max_total_cost_usd,
        "automatic_request_retries": 0,
    }, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run or not pending:
        return

    key = base.load_openrouter_key()
    try:
        base.verify_key(key)
    except urllib.error.HTTPError as error:
        raise SystemExit(
            f"OpenRouter key preflight failed with HTTP {error.code}; no model request sent"
        ) from None

    file_lock = threading.Lock()
    state = RunState(len(pending), args.max_total_cost_usd, threading.Lock(), args.progress)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_pair, pair, key, checkpoints, args, state, file_lock
            )
            for pair in pending
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                state.mark("failed")
                print(f"worker_error type={type(error).__name__} detail={str(error)[:300]}", flush=True)

    print(json.dumps(state.snapshot(), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
