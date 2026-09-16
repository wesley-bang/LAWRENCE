#!/usr/bin/env python3
"""Re-run audit-failed judgment de-identification through Gemini 3.8 Flex.

This experiment deliberately reuses the exact prompt stored for the Gemma 4
baseline and the same deterministic renderer. Results are written separately
and never replace the baseline judgment corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
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
DEFAULT_MANIFEST = ROOT / "data/raw/linked_judgments/linked_judgments_114.jsonl"
DEFAULT_DB = ROOT / "data/review/indictment_reviews.sqlite3"
DEFAULT_INDICTMENTS = ROOT / "data/intermediate/google_ai/hybrid_gemma4_experiment.jsonl"
DEFAULT_JUDGMENTS = ROOT / "data/intermediate/google_ai/hybrid_judgments_gemma4_experiment.jsonl"
DEFAULT_AUDITS = ROOT / "data/intermediate/google_ai/pair_audit_gemini38.jsonl"
DEFAULT_OUTPUT = ROOT / "data/intermediate/google_ai/gemini38_flex_judgment_experiment.jsonl"
DEFAULT_FAILURES = ROOT / "data/intermediate/google_ai/gemini38_flex_judgment_failures.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-model judgment de-identification comparison")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--indictments", type=Path, default=DEFAULT_INDICTMENTS)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--audits", type=Path, default=DEFAULT_AUDITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--key-name", default="GOOGLE_STUDIO_API_KEY_4")
    parser.add_argument("--model", default="gemini-3.8-flash", choices=("gemini-3.8-flash",))
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-flex-attempts", type=int, default=3)
    parser.add_argument("--retry-seconds", type=float, default=20.0)
    return parser.parse_args()


def select_failed_cases(
    audits: list[dict], judgments: list[dict], existing_ids: set[str], failure_ids: set[str]
) -> list[tuple[dict, dict]]:
    failed_audits = {
        str(row.get("pair_id")): row for row in audits
        if row.get("effective_decision") == "fail"
    }
    selected = []
    for baseline in judgments:
        pair_id = str(baseline.get("pair_id") or "")
        if pair_id in failed_audits and pair_id not in existing_ids | failure_ids:
            selected.append((baseline, failed_audits[pair_id]))
    return selected


def exact_baseline_prompt(baseline: dict, manifest: dict) -> tuple[str, str, str]:
    source = judgment.normalize_judgment_text(str(manifest.get("text") or ""))
    prompt = judgment_hybrid.analysis_prompt(source, baseline.get("pair_alias_registry") or {})
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return source, prompt, digest


def api_key_without_inline_comment(raw_key: str) -> str:
    """Remove only an explicit whitespace-delimited dotenv comment.

    Authorization keys (including the new ``AQ.`` form) are otherwise passed
    through unchanged.  Silently truncating arbitrary whitespace would make a
    malformed credential much harder to diagnose.
    """
    parts = raw_key.split(maxsplit=1)
    if len(parts) == 1:
        return raw_key
    key, suffix = parts
    if suffix.startswith("#"):
        # load_api_keys already removes the opening quote from a dotenv value;
        # when an inline comment follows, its closing quote is no longer at the
        # end of the full string and therefore must be removed here.
        return key.rstrip("\"'")
    raise ValueError("API key contains unexpected whitespace-delimited content")


def failure_row(baseline: dict, error: str, **details: object) -> dict:
    row = {
        "pair_id": baseline.get("pair_id"),
        "judgment_internal_id": baseline.get("judgment_internal_id"),
        "error": error,
        "model": "gemini-3.8-flash",
        "service_tier_requested": "flex",
        "checkpointed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    row.update(details)
    return row


def main() -> None:
    args = parse_args()
    if args.limit < 1 or args.max_flex_attempts < 1:
        raise SystemExit("--limit and --max-flex-attempts must be at least 1")
    keys = dict(hybrid.load_api_keys())
    if args.key_name not in keys:
        raise SystemExit(f"API key name not found in .env: {args.key_name}")

    indictments = {str(row.get("doc_id")): row for row in hybrid.read_jsonl(args.indictments)}
    manifests = {
        judgment_hybrid.judgment_internal_id(row): row for row in hybrid.read_jsonl(args.manifest)
    }
    existing_rows = hybrid.read_jsonl(args.output)
    failure_rows = hybrid.read_jsonl(args.failures)
    existing = {str(row.get("pair_id")): row for row in existing_rows}
    failures = {str(row.get("pair_id")): row for row in failure_rows}
    candidates = select_failed_cases(
        hybrid.read_jsonl(args.audits), hybrid.read_jsonl(args.judgments),
        set(existing), set(failures),
    )
    output_order = list(existing)
    failure_order = list(failures)
    try:
        api_key = api_key_without_inline_comment(keys[args.key_name])
    except ValueError as error:
        raise SystemExit(f"Invalid {args.key_name}: {error}") from error
    print(
        f"candidates={len(candidates)} requested={args.limit} "
        f"key={args.key_name} model={args.model} "
        "service_tier=flex exact_prompt_required=true",
        flush=True,
    )

    completed_this_run = 0
    for index, (baseline, prior_audit) in enumerate(candidates, 1):
        if completed_this_run >= args.limit:
            break
        pair_id = str(baseline.get("pair_id"))
        if pair_id not in output_order:
            output_order.append(pair_id)
        if pair_id not in failure_order:
            failure_order.append(pair_id)
        internal_id = str(baseline.get("judgment_internal_id") or "")
        manifest = manifests.get(internal_id)
        indictment = indictments.get(str(baseline.get("linked_indictment_doc_id") or ""))
        if not manifest or not indictment:
            failures[pair_id] = failure_row(baseline, "missing_pair_source")
            hybrid.write_jsonl(args.failures, failures, failure_order)
            continue
        source, prompt, digest = exact_baseline_prompt(baseline, manifest)
        if digest != baseline.get("input_sha256"):
            failures[pair_id] = failure_row(
                baseline, "baseline_prompt_mismatch", generated_sha256=digest,
                baseline_sha256=baseline.get("input_sha256"),
            )
            hybrid.write_jsonl(args.failures, failures, failure_order)
            print(f"skipped-input-mismatch candidate={index} {pair_id}", flush=True)
            continue

        plan = None
        usage = {}
        last_detail = ""
        for attempt in range(1, args.max_flex_attempts + 1):
            try:
                plan, usage = hybrid.request_json(
                    api_key, args.model, prompt,
                    thinking_level="medium", service_tier="flex", timeout=1200,
                )
                break
            except urllib.error.HTTPError as error:
                last_detail = judgment_hybrid.error_summary(error)
                if error.code in {401, 403}:
                    raise SystemExit(
                        f"Authentication/authorization failed for {args.key_name}: "
                        f"HTTP {error.code} {last_detail}"
                    ) from error
                if error.code not in judgment_hybrid.TRANSIENT_HTTP_CODES or attempt == args.max_flex_attempts:
                    failures[pair_id] = failure_row(
                        baseline, "flex_api_error", error_type=f"HTTP_{error.code}",
                        detail=last_detail, attempts=attempt,
                    )
                    break
                wait = args.retry_seconds * (2 ** (attempt - 1))
                print(f"waiting-flex HTTP {error.code}; retry in {int(wait)} seconds", flush=True)
                time.sleep(wait)
            except (TimeoutError, urllib.error.URLError, ConnectionError) as error:
                last_detail = str(error)[:800]
                if attempt == args.max_flex_attempts:
                    failures[pair_id] = failure_row(
                        baseline, "flex_network_error", error_type=type(error).__name__,
                        detail=last_detail, attempts=attempt,
                    )
                    break
                wait = args.retry_seconds * (2 ** (attempt - 1))
                print(f"waiting-flex {type(error).__name__}; retry in {int(wait)} seconds", flush=True)
                time.sleep(wait)
        if plan is None:
            hybrid.write_jsonl(args.failures, failures, failure_order)
            print(f"skipped-flex candidate={index} {pair_id} {last_detail[:120]}", flush=True)
            continue

        case = {
            "judgment_internal_id": internal_id,
            "manifest_record": manifest,
            "indictment_record": indictment,
        }
        record = judgment_hybrid.render_record(
            case, plan, usage, digest, args.model,
            baseline.get("pair_alias_registry") or {}, baseline,
        )
        record.update({
            "experiment": "gemini38_flex_exact_gemma_prompt_comparison",
            "baseline_model": baseline.get("model"),
            "baseline_doc_id": baseline.get("doc_id"),
            "baseline_input_sha256": baseline.get("input_sha256"),
            "exact_baseline_prompt_verified": True,
            "service_tier_requested": "flex",
            "service_tier_observed": usage.get("responseServiceTier"),
            "baseline_audit": prior_audit.get("audit_result"),
        })
        existing[pair_id] = record
        completed_this_run += 1
        failures.pop(pair_id, None)
        hybrid.write_jsonl(args.output, existing, output_order)
        if args.failures.exists():
            hybrid.write_jsonl(args.failures, failures, failure_order)
        print(
            f"processed-flex {completed_this_run}/{args.limit} {pair_id} "
            f"observed_tier={record.get('service_tier_observed')} "
            f"replacements={record.get('applied_replacements')}",
            flush=True,
        )

    print(json.dumps({
        "requested": args.limit, "completed_this_run": completed_this_run,
        "completed_total": len(existing), "failures": len(failures),
        "output": str(args.output.resolve()), "service_tier": "flex",
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
