#!/usr/bin/env python3
"""Deterministically repair and quality-tier GPT-5.6 Luna three-role records.

The paid generation file is immutable input.  This script writes a new JSONL,
keeps an audit trail for every mutation, and never calls an external API.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/intermediate/openrouter/gpt56_luna_flex_three_role_bulk.jsonl"
DEFAULT_INDICTMENTS = ROOT / "data/intermediate/google_ai/hybrid_gemma4_experiment.jsonl"
DEFAULT_JUDGMENTS = ROOT / "data/intermediate/google_ai/hybrid_judgments_gemma4_experiment.jsonl"
DEFAULT_OUTPUT = ROOT / "data/intermediate/openrouter/gpt56_luna_flex_three_role_cleaned.jsonl"
DEFAULT_RECOMMENDED = ROOT / "data/intermediate/openrouter/gpt56_luna_flex_three_role_recommended.jsonl"
DEFAULT_REVIEW = ROOT / "data/intermediate/openrouter/gpt56_luna_flex_three_role_review.jsonl"
DEFAULT_REJECTED = ROOT / "data/intermediate/openrouter/gpt56_luna_flex_three_role_rejected.jsonl"
DEFAULT_REPORT = ROOT / "data/intermediate/openrouter/gpt56_luna_flex_three_role_cleaning_report.json"

CLEANING_VERSION = "gpt56-luna-clean-v1"
PERSON_LETTERS = "甲乙丙丁戊己庚辛壬癸"
MASK_CHARS = "○〇ＯOo"
ROLE_LABELS = {
    "DEFENDANT": "被告", "WITNESS": "證人", "COMPLAINANT": "告訴人",
    "VICTIM": "被害人", "INVESTOR": "投資人", "PROSECUTOR": "檢察官",
    "CLERK": "書記官", "JUDGE": "法官", "DEFENSE_COUNSEL": "辯護人",
    "COMPLAINANT_COUNSEL": "告訴代理人",
    "PRIVATE_PROSECUTOR_COUNSEL": "自訴代理人", "POLICE": "員警",
    "LEGAL_REPRESENTATIVE": "法定代理人", "OTHER": "人物",
}
PROFESSIONAL_ROLES = {
    "PROSECUTOR", "CLERK", "JUDGE", "DEFENSE_COUNSEL",
    "COMPLAINANT_COUNSEL", "PRIVATE_PROSECUTOR_COUNSEL",
}
SUFFIXES = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥"
PREFIXED_MASK_RE = re.compile(
    rf"(被告|告訴人|證人|被害人|法定代理人)([{PERSON_LETTERS}])[{MASK_CHARS}]{{2,}}"
)
STANDALONE_MASK_RE = re.compile(rf"([{PERSON_LETTERS}])[{MASK_CHARS}]{{2,}}")
HOLDING_TAIL_RE = re.compile(
    r"(?<=[。；;])\s*(?:犯罪事實及證據名稱|犯罪事實及證據|"
    r"犯罪事實|事實及證據名稱|事實及理由|事實|理由)(?=\s*[^，,。；;])"
)
CONVICTION_RE = re.compile(r"(?:犯[^。；]{0,45}罪|處(?:有期徒刑|拘役|罰金)|應執行)")
PROCEDURAL_RE = re.compile(r"公訴不受理|本件免訴|本案免訴|管轄錯誤|上訴不受理")
INTERLOCUTORY_RE = re.compile(r"停止審判|再開辯論|限制出境|限制出海")
CONFLICT_RE = re.compile(r"不一致|相互矛盾|矛盾|記載差異|有待(?:法院)?釐清|同時出現")
CIRCULAR_RE = re.compile(
    r"依本案主文(?:所示|範圍|作成|宣告)|本案指定之.{0,12}結論|"
    r"為與.{0,12}主文.{0,8}一致|僅依.{0,12}主文"
)
GENERIC_DEFENSE_RE = re.compile(
    r"未(?:親自)?(?:提供|閱覽).{0,18}(?:原文|全文|本體|原件|完整卷證)|"
    r"現有(?:資料|材料|文字|摘要).{0,18}(?:不足|無法)|"
    r"仍須.{0,18}(?:法院|完整卷證).{0,12}(?:判斷|審查|認定)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--indictments", type=Path, default=DEFAULT_INDICTMENTS)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recommended-output", type=Path, default=DEFAULT_RECOMMENDED)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--rejected-output", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            records.append(value)
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalized(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value)))


def clean_holding(value: str) -> tuple[str, bool]:
    """Cut a polluted 主文 at the first explicit post-holding section marker."""
    holding = re.sub(r"\s+", " ", str(value or "")).strip()
    match = HOLDING_TAIL_RE.search(holding)
    if match:
        cleaned = holding[: match.start()].strip()
        return cleaned, cleaned != holding
    return holding, False


def classify_outcome(holding: str) -> str:
    groups = []
    if CONVICTION_RE.search(holding):
        groups.append("conviction")
    if "無罪" in holding:
        groups.append("acquittal")
    if PROCEDURAL_RE.search(holding):
        groups.append("procedural")
    if INTERLOCUTORY_RE.search(holding):
        groups.append("interlocutory")
    if len(groups) > 1:
        return "mixed"
    return groups[0] if groups else "unknown"


def transform_strings(value: Any, transform: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, list):
        return [transform_strings(item, transform) for item in value]
    if isinstance(value, dict):
        return {key: transform_strings(item, transform) for key, item in value.items()}
    return value


def is_likely_private_person_name(mention: str, reason: str = "") -> bool:
    mention = str(mention).strip()
    reason = str(reason)
    if not mention or re.search(rf"[{MASK_CHARS}\[\]0-9A-Za-z]", mention):
        return False
    if re.search(r"暱稱|綽號|網路代稱|匿名", reason):
        return False
    if mention in {"被告", "證人", "告訴人", "被害人", "法定代理人", "不詳男子", "不詳女子"}:
        return False
    if re.fullmatch(rf"(?:被告|證人|告訴人|被害人|人物|法定代理人)?[{PERSON_LETTERS}男女]", mention):
        return False
    if re.fullmatch(r"[\u3400-\u9fff]姓(?:友人|男子|女子)", mention):
        return False
    # Preserve obvious nicknames even when an older analysis plan labelled the
    # person but did not explicitly put 暱稱 in its explanation.
    if len(mention) <= 3 and (
        mention.startswith("阿")
        or mention.endswith(("哥", "姐", "仔", "董", "少", "毛", "爺"))
        or (len(mention) == 2 and mention[0] == mention[1])
    ):
        return False
    return bool(re.fullmatch(r"[\u3400-\u9fff]{2,4}(?:教授|律師|醫師)?", mention))


def private_replacement_map(indictment: dict, judgment: dict) -> dict[str, str]:
    """Rebuild stable aliases for leaked structured fields across both documents."""
    replacements: dict[str, str] = {}
    role_counts: Counter[str] = Counter()
    groups: list[dict] = []
    for source in (indictment, judgment):
        for person in (source.get("analysis_plan") or {}).get("persons") or []:
            if isinstance(person, dict):
                groups.append(person)

    for person in groups:
        reason = str(person.get("same_person_reason") or "")
        mentions = [
            str(value).strip() for value in person.get("mentions") or []
            if (
                is_likely_private_person_name(str(value), reason)
                or bool(re.fullmatch(rf"[\u3400-\u9fff][{MASK_CHARS}]{{2,}}", str(value).strip()))
            )
        ]
        if not mentions:
            continue
        existing = next((replacements[item] for item in mentions if item in replacements), "")
        role = str(person.get("role") or "OTHER")
        label = ROLE_LABELS.get(role, "人物")
        if existing:
            alias = existing
        elif role in PROFESSIONAL_ROLES:
            alias = "〇〇〇"
        else:
            index = role_counts[label]
            suffix = SUFFIXES[index] if index < len(SUFFIXES) else f"{index + 1:02d}"
            alias = f"{label}{suffix}"
            role_counts[label] += 1
        for mention in mentions:
            replacements.setdefault(mention, alias)

    for source in (indictment, judgment):
        for identifier in (source.get("analysis_plan") or {}).get("identifiers") or []:
            if not isinstance(identifier, dict):
                continue
            mention = str(identifier.get("mention") or "").strip()
            if not mention or re.search(rf"[{MASK_CHARS}\[\]]", mention):
                continue
            compact = re.sub(r"[-－\s]", "", mention)
            if compact and set(compact) <= {"0"}:
                continue
            category = str(identifier.get("category") or "")
            default = {
                "PHONE": "[電話]", "EMAIL": "[Email]", "ID": "[身分證字號]",
                "ACCOUNT": "[銀行帳號]", "PLATE": "[車牌]", "ADDRESS": "[地址]",
                "CASE_NO": "[案號]",
            }.get(category, "[敏感資訊]")
            replacements.setdefault(mention, str(identifier.get("replacement") or default))
    return replacements


def replace_private_literals(value: Any, replacements: dict[str, str]) -> tuple[Any, int]:
    changed_strings = 0
    ordered = sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)

    def replace(text: str) -> str:
        nonlocal changed_strings
        original = text
        for mention, alias in ordered:
            text = text.replace(mention, alias)
        if text != original:
            changed_strings += 1
        return text

    return transform_strings(value, replace), changed_strings


def source_role_map(source_text: str) -> dict[str, str]:
    """Resolve masked standalone tokens only when source context has one role."""
    mapping: dict[str, str] = {}
    roles = ("被告", "告訴人", "證人", "被害人", "法定代理人")
    for token in set(STANDALONE_MASK_RE.findall(source_text)):
        # findall returns the letter because the regex has one capture group
        masked = re.search(rf"{re.escape(token)}[{MASK_CHARS}]{{2,}}", source_text)
        if not masked:
            continue
        literal = masked.group(0)
        observed = {role for role in roles if f"{role}{literal}" in source_text}
        if len(observed) == 1:
            role = next(iter(observed))
            mapping[literal] = f"{role}{token}"
    return mapping


def normalize_aliases(value: Any, source_text: str) -> tuple[Any, list[dict]]:
    role_map = source_role_map(source_text)
    repairs: list[dict] = []

    def replace(text: str) -> str:
        original = text
        text = PREFIXED_MASK_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}", text)
        for literal, alias in role_map.items():
            # Prefix replacements above have already consumed safe prefixed forms.
            text = re.sub(
                rf"(?<!被告)(?<!告訴人)(?<!證人)(?<!被害人)(?<!法定代理人){re.escape(literal)}",
                alias,
                text,
            )
        if text != original:
            repairs.append({"kind": "alias_normalization", "before": original, "after": text})
        return text

    return transform_strings(value, replace), repairs


def source_documents(indictment: dict, judgment: dict) -> dict[str, str]:
    return {
        "indictment": json.dumps(indictment, ensure_ascii=False, separators=(",", ":")),
        "judgment": json.dumps(judgment, ensure_ascii=False, separators=(",", ":")),
    }


def migrate_legal_rules(roles: dict, documents: dict[str, str]) -> list[dict]:
    repairs: list[dict] = []
    normalized_docs = {name: normalized(text) for name, text in documents.items()}
    for role_name, role in roles.items():
        if not isinstance(role, dict):
            continue
        for step in role.get("reasoning_trace") or []:
            if not isinstance(step, dict) or step.get("claim_type") != "legal_rule":
                continue
            old_ids = list(step.get("evidence_ids") or [])
            legal_sources: set[str] = set()
            for snippet in step.get("support_snippets") or []:
                needle = normalized(snippet)
                for document_name, document_text in normalized_docs.items():
                    if needle and needle in document_text:
                        legal_sources.add(document_name)
            if not legal_sources:
                legal_sources.add("role_context")
            step["evidence_ids"] = []
            step["legal_sources"] = sorted(legal_sources)
            repairs.append({
                "kind": "legal_rule_citation_migration",
                "role": role_name,
                "step_id": step.get("step_id"),
                "removed_evidence_ids": old_ids,
                "legal_sources": sorted(legal_sources),
            })
    return repairs


def collect_evidence_ids(value: Any, found: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "evidence_ids" and isinstance(child, list):
                found.extend(str(item) for item in child)
            else:
                collect_evidence_ids(child, found)
    elif isinstance(value, list):
        for child in value:
            collect_evidence_ids(child, found)


def original_private_literals(*sources: dict) -> set[str]:
    literals: set[str] = set()
    for source in sources:
        plan = source.get("analysis_plan") or {}
        for person in plan.get("persons") or []:
            if not isinstance(person, dict):
                continue
            for mention in person.get("mentions") or []:
                mention = str(mention).strip()
                reason = str(person.get("same_person_reason") or "")
                if is_likely_private_person_name(mention, reason):
                    literals.add(mention)
        for identifier in plan.get("identifiers") or []:
            if isinstance(identifier, dict):
                mention = str(identifier.get("mention") or "").strip()
                reason = str(identifier.get("reason") or "")
                compact = re.sub(r"[-－\s]", "", mention)
                already_masked = bool(re.search(rf"[{MASK_CHARS}\[\]]", mention))
                all_zero_placeholder = bool(compact) and set(compact) <= {"0"}
                explicitly_kept = "KEEP" in reason.upper() or (
                    str(identifier.get("category") or "") == "CASE_NO"
                    and "引用其他裁判" in reason
                )
                if (
                    len(mention) >= 3
                    and not already_masked
                    and not all_zero_placeholder
                    and not explicitly_kept
                ):
                    literals.add(mention)
    return literals


def role_text(role: dict) -> str:
    return json.dumps(role, ensure_ascii=False, separators=(",", ":"))


def validate_and_flag(record: dict, indictment: dict, judgment: dict) -> tuple[list[str], list[str]]:
    hard: list[str] = []
    review: list[str] = []
    roles = record.get("roles") or {}
    required_roles = {"prosecutor", "defense", "judge"}
    if not required_roles.issubset(roles):
        hard.append("missing_role")
        return hard, review

    catalog_ids = {str(item.get("evidence_id")) for item in record.get("evidence_catalog") or []}
    cited: list[str] = []
    collect_evidence_ids(roles, cited)
    if set(cited) - catalog_ids:
        hard.append("unknown_evidence_id")
    if any((roles[name].get("new_evidence_claimed") or []) for name in required_roles):
        hard.append("new_evidence_claimed")
    prosecutor_ids: list[str] = []
    collect_evidence_ids(roles["prosecutor"], prosecutor_ids)
    if any(item.startswith("J-E") for item in prosecutor_ids):
        hard.append("prosecutor_judgment_evidence_leak")

    judge_decision = (roles["judge"].get("decision") or {})
    if judge_decision.get("target_holding") != record.get("target_holding"):
        hard.append("judge_holding_mismatch")
    if record.get("outcome_class") == "unknown":
        hard.append("unknown_outcome")
    if record.get("outcome_class") == "interlocutory":
        hard.append("non_final_interlocutory_order")
    if "actual_target_holding" in role_text(roles):
        hard.append("unremoved_prompt_artifact")

    private_literals = original_private_literals(indictment, judgment)
    rendered_roles = role_text(roles)
    leaks = sorted(literal for literal in private_literals if literal in rendered_roles)
    if leaks:
        hard.append("raw_private_literal_leak")
        record["quality"]["private_literal_leak_count"] = len(leaks)

    residual_aliases = sorted(set(STANDALONE_MASK_RE.findall(rendered_roles)))
    if residual_aliases:
        review.append("unresolved_masked_person_alias")
        record["quality"]["residual_mask_letters"] = residual_aliases

    judge_text = role_text(roles["judge"])
    if CONFLICT_RE.search(judge_text):
        review.append("cross_document_fact_conflict")
    catalog_size = max(1, len(catalog_ids))
    broad_steps = 0
    for role in roles.values():
        for step in role.get("reasoning_trace") or []:
            ids = set(step.get("evidence_ids") or [])
            claim_type = step.get("claim_type")
            overly_broad_observation = (
                claim_type in {"fact", "evidence_observation"}
                and len(ids) >= 8
                and len(ids) / catalog_size >= 0.65
            )
            overly_broad_reasoning = (
                claim_type in {"inference", "legal_application"}
                and len(ids) >= 12
                and len(ids) / catalog_size >= 0.75
            )
            if overly_broad_observation or overly_broad_reasoning:
                broad_steps += 1
    if broad_steps:
        review.append("overbroad_evidence_citation")
        record["quality"]["overbroad_trace_steps"] = broad_steps

    generic_steps = 0
    for step in roles["defense"].get("reasoning_trace") or []:
        text = " ".join(str(step.get(key) or "") for key in ("proposition", "inference"))
        if GENERIC_DEFENSE_RE.search(text):
            generic_steps += 1
    if generic_steps >= 4:
        review.append("defense_boilerplate_heavy")
        record["quality"]["generic_defense_steps"] = generic_steps

    return sorted(set(hard)), sorted(set(review))


def clean_record(record: dict, indictment: dict, judgment: dict) -> dict:
    cleaned = copy.deepcopy(record)
    input_digest = hashlib.sha256(
        json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    repairs: list[dict] = []
    old_schema = cleaned.get("schema_version")
    cleaned["schema_version"] = CLEANING_VERSION

    source_holding = str((judgment.get("sections") or {}).get("主文") or cleaned.get("target_holding") or "")
    holding, holding_changed = clean_holding(source_holding)
    if holding_changed or holding != cleaned.get("target_holding"):
        repairs.append({
            "kind": "holding_reextraction",
            "before_length": len(str(cleaned.get("target_holding") or "")),
            "after_length": len(holding),
        })
    cleaned["target_holding"] = holding
    judge_decision = ((cleaned.get("roles") or {}).get("judge") or {}).get("decision") or {}
    if judge_decision:
        judge_decision["target_holding"] = holding

    old_outcome = str(cleaned.get("outcome_class") or "")
    new_outcome = classify_outcome(holding)
    cleaned["outcome_class"] = new_outcome
    if new_outcome != old_outcome:
        repairs.append({"kind": "outcome_reclassification", "before": old_outcome, "after": new_outcome})

    replacements = private_replacement_map(indictment, judgment)
    cleaned["evidence_catalog"], evidence_private_repairs = replace_private_literals(
        cleaned.get("evidence_catalog") or [], replacements
    )
    cleaned["roles"], role_private_repairs = replace_private_literals(
        cleaned.get("roles") or {}, replacements
    )
    if evidence_private_repairs or role_private_repairs:
        repairs.append({
            "kind": "private_literal_replacement",
            "evidence_strings_changed": evidence_private_repairs,
            "role_strings_changed": role_private_repairs,
        })

    roles = cleaned.get("roles") or {}
    prompt_hits = role_text(roles).count("actual_target_holding")
    if prompt_hits:
        cleaned["roles"] = transform_strings(
            roles, lambda text: text.replace("actual_target_holding", "本案主文")
        )
        repairs.append({"kind": "prompt_artifact_replacement", "count": prompt_hits})
        roles = cleaned["roles"]

    source_text = source_documents(indictment, judgment)
    cleaned["roles"], alias_repairs = normalize_aliases(roles, "\n".join(source_text.values()))
    if alias_repairs:
        repairs.append({"kind": "alias_normalization", "changed_strings": len(alias_repairs)})
    repairs.extend(migrate_legal_rules(cleaned["roles"], source_text))

    cleaned["quality"] = {
        "status": "pending",
        "hard_flags": [],
        "review_flags": [],
        "info_flags": [],
    }
    hard, review = validate_and_flag(cleaned, indictment, judgment)
    info: list[str] = []
    if holding_changed:
        info.append("holding_content_reextracted")
    if prompt_hits:
        info.append("prompt_artifact_repaired")
    if CIRCULAR_RE.search(role_text(cleaned["roles"]["judge"])):
        info.append("target_conditioned_language_present")
    hard = sorted(set(hard))
    review = sorted(set(review))
    status = "rejected" if hard else ("needs_review" if review else "recommended")
    cleaned["quality"].update({
        "status": status,
        "hard_flags": hard,
        "review_flags": review,
        "info_flags": sorted(set(info)),
        "training_eligible": not hard,
        "recommended_for_training": status == "recommended",
    })
    cleaned["cleaning"] = {
        "version": CLEANING_VERSION,
        "original_schema_version": old_schema,
        "source_record_sha256": input_digest,
        "repairs_applied": repairs,
        "cleaned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return cleaned


def main() -> int:
    args = parse_args()
    source_records = read_jsonl(args.input)
    indictment_by_id = {str(item.get("doc_id")): item for item in read_jsonl(args.indictments)}
    judgment_by_pair = {
        str(item.get("pair_id")): item
        for item in read_jsonl(args.judgments)
        if item.get("pair_id")
    }

    cleaned_records: list[dict] = []
    missing_sources: list[str] = []
    for record in source_records:
        pair_id = str(record.get("pair_id") or "")
        judgment = judgment_by_pair.get(pair_id)
        indictment = indictment_by_id.get(str((judgment or {}).get("linked_indictment_doc_id") or ""))
        if not judgment or not indictment:
            missing_sources.append(pair_id)
            continue
        cleaned_records.append(clean_record(record, indictment, judgment))

    write_jsonl(args.output, cleaned_records)
    recommended_records = [item for item in cleaned_records if item["quality"]["status"] == "recommended"]
    review_records = [item for item in cleaned_records if item["quality"]["status"] == "needs_review"]
    rejected_records = [item for item in cleaned_records if item["quality"]["status"] == "rejected"]
    write_jsonl(args.recommended_output, recommended_records)
    write_jsonl(args.review_output, review_records)
    write_jsonl(args.rejected_output, rejected_records)

    status_counts = Counter(item["quality"]["status"] for item in cleaned_records)
    hard_counts = Counter(
        flag for item in cleaned_records for flag in item["quality"]["hard_flags"]
    )
    review_counts = Counter(
        flag for item in cleaned_records for flag in item["quality"]["review_flags"]
    )
    info_counts = Counter(
        flag for item in cleaned_records for flag in item["quality"]["info_flags"]
    )
    repair_counts = Counter(
        repair["kind"]
        for item in cleaned_records
        for repair in item["cleaning"]["repairs_applied"]
    )
    report = {
        "cleaning_version": CLEANING_VERSION,
        "input": str(args.input),
        "output": str(args.output),
        "recommended_output": str(args.recommended_output),
        "review_output": str(args.review_output),
        "rejected_output": str(args.rejected_output),
        "input_records": len(source_records),
        "output_records": len(cleaned_records),
        "missing_source_pairs": missing_sources,
        "status_counts": dict(status_counts),
        "hard_flag_counts": dict(hard_counts),
        "review_flag_counts": dict(review_counts),
        "info_flag_counts": dict(info_counts),
        "repair_counts": dict(repair_counts),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
