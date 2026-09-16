#!/usr/bin/env python3
"""Bounded DeepInfra/FP8 comparison on the two prior Gemini Flex cases.

The experiment reuses the byte-identical Gemma 4 prompts and the existing
deterministic renderers.  It never overwrites production/baseline artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts import deidentify_linked_judgments as judgment
    from scripts import hybrid_deidentify_judgments_with_google_ai as judgment_hybrid
    from scripts import hybrid_deidentify_with_google_ai as hybrid
except (ModuleNotFoundError, ImportError):
    import deidentify_linked_judgments as judgment
    import hybrid_deidentify_judgments_with_google_ai as judgment_hybrid
    import hybrid_deidentify_with_google_ai as hybrid


ROOT = Path(__file__).resolve().parents[1]
API_ROOT = "https://openrouter.ai/api/v1"
MODEL = "deepseek/deepseek-v4.1-flash"
PROVIDER = "deepinfra"
QUANTIZATION = "fp8"
PROMPT_PRICE_PER_M = 0.20
COMPLETION_PRICE_PER_M = 0.60
MAX_OUTPUT_TOKENS = 8192
MAX_PAIRS = 2
MAX_WORST_CASE_USD = 0.05

DEFAULT_DB = ROOT / "data/review/indictment_reviews.sqlite3"
DEFAULT_MANIFEST = ROOT / "data/raw/linked_judgments/linked_judgments_114.jsonl"
DEFAULT_INDICTMENTS = ROOT / "data/intermediate/google_ai/hybrid_gemma4_experiment.jsonl"
DEFAULT_JUDGMENTS = ROOT / "data/intermediate/google_ai/hybrid_judgments_gemma4_experiment.jsonl"
DEFAULT_FLEX = ROOT / "data/intermediate/google_ai/gemini38_flex_judgment_experiment.jsonl"
DEFAULT_INDICTMENT_OUTPUT = (
    ROOT / "data/intermediate/openrouter/deepinfra_fp8_indictment_experiment.jsonl"
)
DEFAULT_JUDGMENT_OUTPUT = (
    ROOT / "data/intermediate/openrouter/deepinfra_fp8_judgment_experiment.jsonl"
)
DEFAULT_FAILURES = ROOT / "data/intermediate/openrouter/deepinfra_fp8_failures.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-pair OpenRouter DeepInfra FP8 experiment")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--indictments", type=Path, default=DEFAULT_INDICTMENTS)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--flex-cases", type=Path, default=DEFAULT_FLEX)
    parser.add_argument("--indictment-output", type=Path, default=DEFAULT_INDICTMENT_OUTPUT)
    parser.add_argument("--judgment-output", type=Path, default=DEFAULT_JUDGMENT_OUTPUT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--execute", action="store_true", help="send the paid requests")
    return parser.parse_args()


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


def api_json(url: str, key: str, payload: dict | None = None, timeout: float = 900) -> dict:
    request = urllib.request.Request(
        url,
        data=(json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/lawrence-deidentification-experiment",
            "X-Title": "LAWRENCE bounded deidentification experiment",
        },
        method="POST" if payload else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def verify_route(key: str) -> dict:
    author, slug = MODEL.split("/", 1)
    envelope = api_json(f"{API_ROOT}/models/{author}/{slug}/endpoints", key)
    matches = [
        endpoint for endpoint in envelope.get("data", {}).get("endpoints", [])
        if endpoint.get("provider_name") == "DeepInfra"
        and endpoint.get("quantization") == QUANTIZATION
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one DeepInfra FP8 endpoint, found {len(matches)}")
    endpoint = matches[0]
    prompt_price = float(endpoint["pricing"]["prompt"]) * 1_000_000
    completion_price = float(endpoint["pricing"]["completion"]) * 1_000_000
    if prompt_price > PROMPT_PRICE_PER_M or completion_price > COMPLETION_PRICE_PER_M:
        raise RuntimeError(
            f"route price exceeds cap: prompt={prompt_price}, completion={completion_price}"
        )
    if "response_format" not in endpoint.get("supported_parameters", []):
        raise RuntimeError("DeepInfra FP8 endpoint does not support JSON response_format")
    return endpoint


def request_plan(key: str, prompt: str) -> tuple[dict, dict]:
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
    if str(envelope.get("provider") or "").lower() != "deepinfra":
        raise RuntimeError(f"unexpected provider: {envelope.get('provider')!r}")
    choices = envelope.get("choices") or []
    if not choices:
        raise RuntimeError("OpenRouter response contained no choices")
    answer = choices[0].get("message", {}).get("content") or ""
    answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(answer).strip())
    plan = json.loads(answer)
    if isinstance(plan, list) and len(plan) == 1 and isinstance(plan[0], dict):
        plan = plan[0]
    if not isinstance(plan, dict):
        raise ValueError(f"expected JSON object, got {type(plan).__name__}")
    usage = dict(envelope.get("usage") or {})
    usage.update({
        "openrouter_generation_id": envelope.get("id"),
        "provider": envelope.get("provider"),
        "requested_quantization": QUANTIZATION,
    })
    return plan, usage


def load_case(db_path: Path, doc_id: str) -> dict:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT c.doc_id,c.source_id,c.case_type,c.normalized_text,c.entities_json,"
            "coalesce(r.notes,'') notes,r.decision "
            "FROM cases c LEFT JOIN reviews r USING(doc_id) WHERE c.doc_id=?",
            (doc_id,),
        ).fetchone()
    if row is None:
        raise RuntimeError(f"indictment source not found: {doc_id}")
    return dict(row)


def selected_pairs(args: argparse.Namespace) -> list[dict]:
    flex_pair_ids = [str(row.get("pair_id")) for row in hybrid.read_jsonl(args.flex_cases)]
    baselines = {str(row.get("pair_id")): row for row in hybrid.read_jsonl(args.judgments)}
    selected = []
    for pair_id in flex_pair_ids:
        if pair_id in baselines and pair_id not in {row["pair_id"] for row in selected}:
            selected.append({"pair_id": pair_id, "judgment": baselines[pair_id]})
        if len(selected) >= args.limit:
            break
    return selected


def prompts_for_pairs(args: argparse.Namespace, pairs: list[dict]) -> list[dict]:
    indictment_baselines = {
        str(row.get("doc_id")): row for row in hybrid.read_jsonl(args.indictments)
    }
    manifests = {
        judgment_hybrid.judgment_internal_id(row): row
        for row in hybrid.read_jsonl(args.manifest)
    }
    work = []
    for item in pairs:
        pair_id = item["pair_id"]
        judgment_baseline = item["judgment"]
        indictment_id = str(judgment_baseline.get("linked_indictment_doc_id") or "")
        indictment_baseline = indictment_baselines[indictment_id]
        case = load_case(args.db, indictment_id)
        indictment_prompt = hybrid.analysis_prompt(case)
        indictment_digest = hashlib.sha256(indictment_prompt.encode("utf-8")).hexdigest()
        if indictment_digest != indictment_baseline.get("input_sha256"):
            raise RuntimeError(f"Gemma indictment prompt changed for pair {pair_id}")

        manifest = manifests[str(judgment_baseline.get("judgment_internal_id") or "")]
        source = judgment.normalize_judgment_text(str(manifest.get("text") or ""))
        registry = judgment_baseline.get("pair_alias_registry") or {}
        judgment_prompt = judgment_hybrid.analysis_prompt(source, registry)
        judgment_digest = hashlib.sha256(judgment_prompt.encode("utf-8")).hexdigest()
        if judgment_digest != judgment_baseline.get("input_sha256"):
            raise RuntimeError(f"Gemma judgment prompt changed for pair {pair_id}")
        work.append({
            "pair_id": pair_id,
            "case": case,
            "indictment_baseline": indictment_baseline,
            "indictment_prompt": indictment_prompt,
            "indictment_digest": indictment_digest,
            "manifest": manifest,
            "judgment_source": source,
            "judgment_baseline": judgment_baseline,
            "judgment_prompt": judgment_prompt,
            "judgment_digest": judgment_digest,
            "pair_registry": registry,
        })
    return work


def worst_case_cost(work: list[dict]) -> float:
    # UTF-8 byte count is intentionally much more conservative than token count.
    prompt_bytes = sum(
        len(item[field].encode("utf-8"))
        for item in work for field in ("indictment_prompt", "judgment_prompt")
    )
    requests = len(work) * 2
    return (
        prompt_bytes / 1_000_000 * PROMPT_PRICE_PER_M
        + requests * MAX_OUTPUT_TOKENS / 1_000_000 * COMPLETION_PRICE_PER_M
    )


def failure(pair_id: str, document_type: str, error: Exception) -> dict:
    detail = str(error)
    if isinstance(error, urllib.error.HTTPError):
        detail = error.read().decode("utf-8", errors="replace")[:1200]
    return {
        "pair_id": pair_id,
        "document_type": document_type,
        "model": MODEL,
        "provider": PROVIDER,
        "quantization": QUANTIZATION,
        "error_type": type(error).__name__,
        "detail": detail[:1200],
        "checkpointed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    args = parse_args()
    if not 1 <= args.limit <= MAX_PAIRS:
        raise SystemExit(f"--limit must be between 1 and {MAX_PAIRS}")
    key = load_openrouter_key()
    endpoint = verify_route(key)
    pairs = selected_pairs(args)
    if len(pairs) != args.limit:
        raise SystemExit(f"requested {args.limit} pairs but found {len(pairs)} prior Flex cases")
    work = prompts_for_pairs(args, pairs)
    ceiling = worst_case_cost(work)
    print(json.dumps({
        "execute": args.execute,
        "pairs": len(work),
        "requests": len(work) * 2,
        "pair_ids": [item["pair_id"] for item in work],
        "model": MODEL,
        "provider": endpoint.get("provider_name"),
        "quantization": endpoint.get("quantization"),
        "max_output_tokens_per_request": MAX_OUTPUT_TOKENS,
        "conservative_worst_case_usd": round(ceiling, 6),
    }, ensure_ascii=False, indent=2), flush=True)
    if ceiling > MAX_WORST_CASE_USD:
        raise SystemExit(f"worst-case cost ${ceiling:.6f} exceeds hard cap ${MAX_WORST_CASE_USD}")
    if not args.execute:
        print("dry-run only; pass --execute to send requests", flush=True)
        return

    indictment_records: dict[str, dict] = {}
    judgment_records: dict[str, dict] = {}
    failures: list[dict] = []
    for item in work:
        pair_id = item["pair_id"]
        indictment_plan = None
        try:
            indictment_plan, usage = request_plan(key, item["indictment_prompt"])
            rendered, _, _ = hybrid.render_checkpoint(
                item["case"], indictment_plan, usage, item["indictment_digest"], MODEL, None
            )
            rendered.update({
                "experiment": "openrouter_deepinfra_fp8_exact_gemma_prompt",
                "pair_id": pair_id,
                "provider_requested": PROVIDER,
                "quantization_requested": QUANTIZATION,
                "exact_gemma_prompt_verified": True,
            })
            indictment_records[pair_id] = rendered
            hybrid.write_jsonl(args.indictment_output, indictment_records, list(indictment_records))
            print(f"processed indictment pair={pair_id}", flush=True)
        except Exception as error:  # Preserve the other document for model comparison.
            failures.append(failure(pair_id, "indictment", error))
            print(f"failed indictment pair={pair_id} type={type(error).__name__}", flush=True)

        try:
            plan, usage = request_plan(key, item["judgment_prompt"])
            case = {
                "judgment_internal_id": item["judgment_baseline"]["judgment_internal_id"],
                "manifest_record": item["manifest"],
                "indictment_record": item["indictment_baseline"],
            }
            rendered = judgment_hybrid.render_record(
                case, plan, usage, item["judgment_digest"], MODEL,
                item["pair_registry"], item["judgment_baseline"],
            )
            rendered.update({
                "experiment": "openrouter_deepinfra_fp8_exact_gemma_prompt",
                "provider_requested": PROVIDER,
                "quantization_requested": QUANTIZATION,
                "exact_gemma_prompt_verified": True,
            })
            judgment_records[pair_id] = rendered
            hybrid.write_jsonl(args.judgment_output, judgment_records, list(judgment_records))
            print(f"processed judgment pair={pair_id}", flush=True)
        except Exception as error:
            failures.append(failure(pair_id, "judgment", error))
            print(f"failed judgment pair={pair_id} type={type(error).__name__}", flush=True)

    if failures:
        args.failures.parent.mkdir(parents=True, exist_ok=True)
        args.failures.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
            encoding="utf-8",
        )
    total_cost = sum(
        float(record.get("usage_metadata", {}).get("cost") or 0)
        for record in [*indictment_records.values(), *judgment_records.values()]
    )
    print(json.dumps({
        "indictments": len(indictment_records),
        "judgments": len(judgment_records),
        "failures": len(failures),
        "reported_cost_usd": round(total_cost, 8),
        "indictment_output": str(args.indictment_output.resolve()),
        "judgment_output": str(args.judgment_output.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
