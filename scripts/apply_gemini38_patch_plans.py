#!/usr/bin/env python3
"""Apply validated Gemini patch plans to copies and run deterministic checks.

No API calls are made here.  Baseline corpora are read-only and patched
records are written to a separate restricted experiment artifact.
"""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts import audit_case_pairs_with_google_ai as audit
    from scripts import deidentify_linked_judgments as judgment
    from scripts import experiment_gemini38_flex_patch_planner as planner
    from scripts import hybrid_deidentify_judgments_with_google_ai as judgment_hybrid
    from scripts import hybrid_deidentify_with_google_ai as hybrid
    from scripts import pair_case_integration as pairing
except (ModuleNotFoundError, ImportError):
    import audit_case_pairs_with_google_ai as audit
    import deidentify_linked_judgments as judgment
    import experiment_gemini38_flex_patch_planner as planner
    import hybrid_deidentify_judgments_with_google_ai as judgment_hybrid
    import hybrid_deidentify_with_google_ai as hybrid
    import pair_case_integration as pairing


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data/review/indictment_reviews.sqlite3"
DEFAULT_MANIFEST = ROOT / "data/raw/linked_judgments/linked_judgments_114.jsonl"
DEFAULT_INDICTMENTS = ROOT / "data/intermediate/google_ai/hybrid_gemma4_experiment.jsonl"
DEFAULT_JUDGMENTS = ROOT / "data/intermediate/google_ai/hybrid_judgments_gemma4_experiment.jsonl"
DEFAULT_PLANS = ROOT / "data/intermediate/google_ai/gemini38_flex_patch_planner_experiment.jsonl"
DEFAULT_OUTPUT = ROOT / "data/intermediate/google_ai/gemini38_validated_patch_copies.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply validated patch plans to copies")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--indictments", type=Path, default=DEFAULT_INDICTMENTS)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=3)
    return parser.parse_args()


def pointer_parent(record: object, pointer: str) -> tuple[object, str]:
    tokens = pointer[1:].split("/") if pointer.startswith("/") else []
    if not tokens:
        raise KeyError(pointer)
    current = record
    for encoded in tokens[:-1]:
        token = encoded.replace("~1", "/").replace("~0", "~")
        current = current[int(token)] if isinstance(current, list) else current[token]
    return current, tokens[-1].replace("~1", "/").replace("~0", "~")


def set_pointer(record: object, pointer: str, value: object) -> None:
    parent, token = pointer_parent(record, pointer)
    if isinstance(parent, list):
        parent[int(token)] = value
    else:
        parent[token] = value


def person_id(person: dict) -> str:
    return str(person.get("pair_person_id") or person.get("indictment_group_id") or "")


def find_plan_person(record: dict, mentions: list[str]) -> dict:
    wanted = {str(value) for value in mentions}
    matches = [
        item for item in (record.get("analysis_plan") or {}).get("persons", [])
        if isinstance(item, dict) and wanted.intersection(map(str, item.get("mentions", [])))
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one plan person for {sorted(wanted)}, found {len(matches)}")
    return matches[0]


def registry_person(registry: dict, pair_person_id: str) -> dict:
    matches = [item for item in registry.get("persons", []) if person_id(item) == pair_person_id]
    if len(matches) != 1:
        raise ValueError(f"expected one registry person {pair_person_id}, found {len(matches)}")
    return matches[0]


def replace_alias_in_value(value: object, old: str, new: str) -> object:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [replace_alias_in_value(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: replace_alias_in_value(item, old, new) for key, item in value.items()}
    return value


def add_registry_person(registry: dict, operation: dict) -> None:
    pair_id = str(operation["proposed_pair_person_id"])
    if any(person_id(item) == pair_id for item in registry.get("persons", [])):
        raise ValueError(f"duplicate pair person id: {pair_id}")
    mentions = [str(value) for value in operation["mentions"]]
    entry = {
        "pair_person_id": pair_id,
        "indictment_group_id": None,
        "role": operation["role"],
        "alias": operation["alias"],
        "indictment_mentions": mentions if operation["document"] == "indictment" else [],
        "judgment_mentions": mentions if operation["document"] == "judgment" else [],
    }
    registry.setdefault("persons", []).append(entry)
    unique = registry.setdefault("unique_mentions", {})
    for mention in mentions:
        unique[mention] = {"pair_person_id": pair_id, "alias": operation["alias"]}


def apply_structural_operation(case: dict, operation: dict) -> tuple[bool, bool]:
    """Apply plan/registry changes; return indictment/judgment rerender flags."""
    op = operation["op"]
    document = operation.get("document")
    indictment_changed = judgment_changed = False
    registry = case["judgment"].setdefault("pair_alias_registry", {})
    if op == "LINK_PERSON":
        target = find_plan_person(case[document], operation["mentions"])
        target["linked_indictment_group_id"] = operation["pair_person_id"]
        judgment_changed = document == "judgment"
        indictment_changed = document == "indictment"
    elif op == "REGISTER_PERSON":
        record = case[document]
        people = record.setdefault("analysis_plan", {}).setdefault("persons", [])
        group_prefix = "J" if document == "judgment" else "P"
        people.append({
            "group_id": f"{group_prefix}{len(people) + 1:02d}",
            "linked_indictment_group_id": None,
            "role": operation["role"],
            "mentions": list(operation["mentions"]),
            "same_person_reason": operation.get("rationale", "validated patch"),
        })
        add_registry_person(registry, operation)
        judgment_changed = document == "judgment"
        indictment_changed = document == "indictment"
    elif op == "SET_ALIAS":
        target = registry_person(registry, str(operation["pair_person_id"]))
        old_alias = str(target.get("alias") or "")
        new_alias = str(operation["alias"])
        target["alias"] = new_alias
        for mention, item in registry.get("unique_mentions", {}).items():
            if str(item.get("pair_person_id")) == str(operation["pair_person_id"]):
                item["alias"] = new_alias
        if old_alias and old_alias != new_alias:
            case["indictment"] = replace_alias_in_value(case["indictment"], old_alias, new_alias)
            case["judgment"] = replace_alias_in_value(case["judgment"], old_alias, new_alias)
    elif op == "REMOVE_REGISTRY_MENTION":
        target = registry_person(registry, str(operation["pair_person_id"]))
        mention = str(operation["mention"])
        target["indictment_mentions"] = [
            value for value in target.get("indictment_mentions", []) if value != mention
        ]
        target["judgment_mentions"] = [
            value for value in target.get("judgment_mentions", []) if value != mention
        ]
        registry.get("unique_mentions", {}).pop(mention, None)
    elif op == "UPDATE_PERSON_ROLE":
        set_pointer(case[document], operation["field_path"], operation["role"])
        indictment_changed = document == "indictment"
        judgment_changed = document == "judgment"
    return indictment_changed, judgment_changed


def apply_content_operation(case: dict, operation: dict) -> None:
    op = operation["op"]
    document = operation.get("document")
    if op in {"MASK_SPAN", "RESTORE_FACT"}:
        current = planner.pointer_value(case[document], operation["field_path"])
        set_pointer(
            case[document], operation["field_path"],
            current.replace(operation["old_text"], operation["replacement"]),
        )
    elif op == "ADD_EVIDENCE":
        case[document].setdefault("evidence", []).append(copy.deepcopy(operation["evidence"]))
    elif op == "REPLACE_EVIDENCE":
        set_pointer(case[document], operation["field_path"], copy.deepcopy(operation["evidence"]))
    elif op == "MERGE_EVIDENCE":
        for doc, name_key in (("indictment", "indictment_name"), ("judgment", "judgment_name")):
            matches = [
                item for item in case[doc].get("evidence", [])
                if isinstance(item, dict) and item.get("name") == operation.get(name_key)
            ]
            if len(matches) != 1:
                raise ValueError(f"expected one {doc} evidence named {operation.get(name_key)!r}")
            matches[0]["canonical_key"] = operation["canonical_key"]


def load_indictment_case(db_path: Path, doc_id: str) -> dict:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT c.doc_id,c.source_id,c.case_type,c.normalized_text,c.entities_json,"
            "coalesce(r.notes,'') notes,r.decision "
            "FROM cases c LEFT JOIN reviews r USING(doc_id) WHERE c.doc_id=?", (doc_id,),
        ).fetchone()
    if row is None:
        raise KeyError(doc_id)
    return dict(row)


def rerender(case: dict, db_path: Path, manifests: dict, indictment_changed: bool,
             judgment_changed: bool) -> None:
    if indictment_changed:
        source = load_indictment_case(db_path, case["linked_indictment_doc_id"])
        rendered, _, _ = hybrid.render_checkpoint(
            source, case["indictment"]["analysis_plan"],
            case["indictment"].get("usage_metadata", {}),
            case["indictment"].get("input_sha256", ""),
            case["indictment"].get("model", "patched"), None,
        )
        rendered["pair_id"] = case["pair_id"]
        case["indictment"] = rendered
    if judgment_changed:
        baseline = case["judgment"]
        manifest = manifests[case["judgment_internal_id"]]
        render_case = {
            "judgment_internal_id": case["judgment_internal_id"],
            "manifest_record": manifest,
            "indictment_record": case["indictment"],
        }
        case["judgment"] = judgment_hybrid.render_record(
            render_case, baseline["analysis_plan"], baseline.get("usage_metadata", {}),
            baseline.get("input_sha256", ""), baseline.get("model", "patched"),
            baseline.get("pair_alias_registry", {}), baseline,
        )


def recompute_pair_fields(case: dict) -> None:
    case["judgment"]["paired_crime_facts_summary"] = pairing.merge_pair_facts(
        case["indictment"].get("crime_facts_summary", []),
        case["judgment"].get("crime_facts_summary", []),
    )
    case["judgment"]["paired_evidence"] = pairing.merge_pair_evidence(
        case["indictment"].get("evidence", []), case["judgment"].get("evidence", [])
    )


def apply_plan(case: dict, patch_plan: dict, db_path: Path, manifests: dict) -> dict:
    patched = copy.deepcopy(case)
    applied = []
    already_satisfied = []
    skipped = []
    structural = {"LINK_PERSON", "REGISTER_PERSON", "SET_ALIAS",
                  "REMOVE_REGISTRY_MENTION", "UPDATE_PERSON_ROLE"}
    indictment_changed = judgment_changed = False
    operations = patch_plan.get("operations", [])
    for index, operation in enumerate(operations):
        errors = planner.validate_operation(operation, patched)
        if errors:
            skipped.append({"index": index, "errors": errors})
            continue
        if operation["op"] in structural:
            try:
                changed_i, changed_j = apply_structural_operation(patched, operation)
                indictment_changed |= changed_i
                judgment_changed |= changed_j
                applied.append(index)
            except (KeyError, ValueError, TypeError) as error:
                skipped.append({"index": index, "errors": [f"execution:{error}"]})
    rerender(patched, db_path, manifests, indictment_changed, judgment_changed)
    for index, operation in enumerate(operations):
        if index in applied or operation.get("op") in structural:
            continue
        if operation.get("op") in {"MASK_SPAN", "RESTORE_FACT"}:
            try:
                current = str(planner.pointer_value(
                    patched[operation["document"]], operation["field_path"]
                ))
                if (
                    str(operation.get("old_text") or "") not in current
                    and str(operation.get("replacement") or "") in current
                ):
                    already_satisfied.append(index)
                    continue
            except (KeyError, IndexError, TypeError, ValueError):
                pass
        errors = planner.validate_operation(operation, patched)
        if errors:
            if not any(item["index"] == index for item in skipped):
                skipped.append({"index": index, "errors": errors})
            continue
        if operation["op"] in {"MARK_FALSE_POSITIVE", "PATCH_RENDERER_RULE"}:
            skipped.append({"index": index, "errors": ["non_mutating_or_global_operation"]})
            continue
        try:
            apply_content_operation(patched, operation)
            applied.append(index)
        except (KeyError, ValueError, TypeError) as error:
            skipped.append({"index": index, "errors": [f"execution:{error}"]})
    recompute_pair_fields(patched)
    return {
        "pair_id": patched["pair_id"],
        "judgment_internal_id": patched["judgment_internal_id"],
        "linked_indictment_doc_id": patched["linked_indictment_doc_id"],
        "read_only_experiment": True,
        "patch_applied_to_copy": True,
        "applied_operation_indexes": applied,
        "already_satisfied_operation_indexes": already_satisfied,
        "skipped_operations": skipped,
        "deterministic_checks_before": audit.deterministic_checks(case),
        "deterministic_checks_after": audit.deterministic_checks(patched),
        "indictment": patched["indictment"],
        "judgment": patched["judgment"],
        "checkpointed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    args = parse_args()
    sources = audit.load_indictment_sources(args.db)
    manifests = audit.load_manifest_sources(args.manifest)
    cases = audit.build_cases(
        hybrid.read_jsonl(args.indictments), hybrid.read_jsonl(args.judgments), sources, manifests,
    )
    by_id = {case["pair_id"]: case for case in cases}
    plans = hybrid.read_jsonl(args.plans)[:args.limit]
    results = {}
    for row in plans:
        pair_id = str(row.get("pair_id") or "")
        if pair_id not in by_id:
            continue
        result = apply_plan(by_id[pair_id], row.get("patch_plan") or {}, args.db, manifests)
        results[pair_id] = result
        print(
            f"patched-copy pair={pair_id} applied={len(result['applied_operation_indexes'])} "
            f"skipped={len(result['skipped_operations'])} "
            f"scanner_pass={result['deterministic_checks_after']['pass']}", flush=True,
        )
    hybrid.write_jsonl(args.output, results, list(results))
    print(json.dumps({
        "pairs": len(results), "api_calls": 0, "output": str(args.output.resolve())
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
