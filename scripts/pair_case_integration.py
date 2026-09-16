#!/usr/bin/env python3
"""Cross-document aliases and conservative evidence integration for case pairs.

This module performs no API calls. It anchors aliases in the indictment plan,
resolves judgment people against that anchor, and merges already de-identified
facts/evidence without dropping source provenance.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from collections import Counter

try:
    from scripts import hybrid_deidentify_with_google_ai as hybrid
except (ModuleNotFoundError, ImportError):  # Direct execution adds scripts/ rather than repo root.
    import hybrid_deidentify_with_google_ai as hybrid


ALIAS_LABELS = tuple(sorted({
    *hybrid.ROLE_LABELS.values(),
    "上訴人", "聲請人", "相對人", "受刑人", "法定代理人", "自訴人", "人物",
}, key=len, reverse=True))


def normalize_person_mention(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value))
    value = value.replace("Ｏ", "○").replace("〇", "○")
    # O/o is used as a mask in Chinese names but is an ordinary letter in
    # romanized names, so normalize it only when a CJK character is present.
    if re.search(r"[\u3400-\u9fff]", value):
        value = value.replace("O", "○").replace("o", "○")
    return re.sub(r"[\s·‧．。]", "", value).strip()


def alias_label(alias: str) -> str:
    return next((label for label in ALIAS_LABELS if alias.startswith(label)), "人物")


def build_pair_alias_registry(
    indictment_text: str,
    indictment_plan: dict,
    deterministic_registry: dict | None = None,
) -> dict:
    """Create the serializable alias anchor used by the paired judgment."""
    replacements = hybrid.build_replacements(
        indictment_text, indictment_plan, registry=deterministic_registry
    )
    alias_by_literal = {mention: alias for mention, alias, _ in replacements}
    people = []
    key_candidates: dict[str, list[tuple[str, str]]] = {}
    for index, item in enumerate(indictment_plan.get("persons", []), 1):
        if not isinstance(item, dict):
            continue
        mentions = [
            str(value).strip() for value in item.get("mentions", [])
            if str(value).strip() in indictment_text
        ]
        aliases = [alias_by_literal[value] for value in mentions if value in alias_by_literal]
        if not aliases:
            continue
        alias = Counter(aliases).most_common(1)[0][0]
        group_id = str(item.get("group_id") or f"P{index:02d}")
        person = {
            "pair_person_id": group_id,
            "indictment_group_id": group_id,
            "role": str(item.get("role") or "OTHER"),
            "alias": alias,
            "indictment_mentions": mentions,
        }
        people.append(person)
        for mention in mentions:
            key = normalize_person_mention(mention)
            if key:
                key_candidates.setdefault(key, []).append((group_id, alias))

    unique_mentions = {}
    ambiguous_mentions = []
    for key, candidates in key_candidates.items():
        unique = sorted(set(candidates))
        if len(unique) == 1:
            unique_mentions[key] = {
                "pair_person_id": unique[0][0], "alias": unique[0][1]
            }
        else:
            ambiguous_mentions.append(key)
    return {
        "version": "pair-alias-v1",
        "restricted": True,
        "persons": people,
        "unique_mentions": unique_mentions,
        "ambiguous_mentions": sorted(ambiguous_mentions),
        "manual_review_required": bool(ambiguous_mentions),
    }


def resolve_pair_alias(judgment_person: dict, pair_registry: dict) -> dict | None:
    """Resolve explicitly linked groups first, then unique normalized mentions."""
    explicit = str(
        judgment_person.get("linked_indictment_group_id")
        or judgment_person.get("pair_person_id") or ""
    )
    if explicit:
        for person in pair_registry.get("persons", []):
            if explicit in {str(person.get("pair_person_id")), str(person.get("indictment_group_id"))}:
                return {
                    "pair_person_id": person["pair_person_id"], "alias": person["alias"],
                    "method": "explicit_group_link",
                }
    candidates = set()
    for mention in judgment_person.get("mentions", []):
        match = pair_registry.get("unique_mentions", {}).get(normalize_person_mention(mention))
        if match:
            candidates.add((match["pair_person_id"], match["alias"]))
    if len(candidates) == 1:
        pair_person_id, alias = candidates.pop()
        return {
            "pair_person_id": pair_person_id, "alias": alias,
            "method": "unique_normalized_mention",
        }
    return None


def _normalized_content_key(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).lower()
    value = re.sub(r"各?\s*\d+\s*(?:份|張|件|支|顆|只|冊|片)", "", value)
    value = re.sub(r"(?:在|附)?卷(?:內)?(?:可稽|可佐|可參|足憑|足佐)?", "", value)
    return re.sub(r"[^0-9a-z\u3400-\u9fff\[\]]", "", value)


def evidence_key(item: dict) -> str:
    explicit = str(item.get("canonical_key") or "").strip()
    if explicit:
        return f"explicit:{explicit}"
    category = _normalized_content_key(item.get("category") or item.get("type") or "")
    name = _normalized_content_key(
        item.get("name") or item.get("description") or item.get("evidence") or ""
    )
    return f"content:{category}:{name}"


def merge_pair_evidence(indictment_evidence: list, judgment_evidence: list) -> list[dict]:
    """Conservatively union evidence, merging only equal canonicalized items."""
    merged: dict[str, dict] = {}
    order: list[str] = []
    for source, items in (("indictment", indictment_evidence), ("judgment", judgment_evidence)):
        for value in items or []:
            item = dict(value) if isinstance(value, dict) else {"name": str(value)}
            key = evidence_key(item)
            if not key.rsplit(":", 1)[-1]:
                key = f"empty:{source}:{len(order)}"
            if key not in merged:
                order.append(key)
                merged[key] = {
                    "evidence_id": f"E{len(order):03d}",
                    "name": item.get("name") or item.get("description") or item.get("evidence"),
                    "category": item.get("category") or item.get("type"),
                    "proves": [],
                    "provenance": [],
                    "source_records": [],
                }
            target = merged[key]
            if source not in target["provenance"]:
                target["provenance"].append(source)
            proves = item.get("proves")
            proves_values = proves if isinstance(proves, list) else ([proves] if proves else [])
            for statement in proves_values:
                if statement not in target["proves"]:
                    target["proves"].append(statement)
            target["source_records"].append({"source": source, "record": item})
    return [merged[key] for key in order]


def merge_pair_facts(indictment_facts: list, judgment_facts: list) -> list[dict]:
    merged: dict[str, dict] = {}
    order = []
    for source, items in (("indictment", indictment_facts), ("judgment", judgment_facts)):
        for value in items or []:
            text = str(value.get("text") if isinstance(value, dict) else value).strip()
            if not text:
                continue
            key = _normalized_content_key(text)
            if key not in merged:
                order.append(key)
                merged[key] = {"text": text, "provenance": []}
            if source not in merged[key]["provenance"]:
                merged[key]["provenance"].append(source)
    return [merged[key] for key in order]


def integrate_pair_records(
    indictment_record: dict,
    judgment_record: dict,
    pair_registry: dict,
    pair_id: str | None = None,
) -> dict:
    """Build one training bundle without exposing either restricted source ID."""
    # Random like the existing clean document IDs; callers must preserve it in
    # a restricted mapping rather than derive a reversible source fingerprint.
    pair_id = pair_id or uuid.uuid4().hex
    return {
        "pair_id": pair_id,
        "documents": {
            "indictment": indictment_record.get("text", ""),
            "judgment": judgment_record.get("text", ""),
        },
        # Raw cross-document mentions remain only in the restricted registry.
        "pair_entities": [
            {
                "pair_person_id": person.get("pair_person_id"),
                "role": person.get("role"),
                "alias": person.get("alias"),
            }
            for person in pair_registry.get("persons", [])
        ],
        "crime_facts": merge_pair_facts(
            indictment_record.get("crime_facts_summary", []),
            judgment_record.get("crime_facts_summary", []),
        ),
        "evidence": merge_pair_evidence(
            indictment_record.get("evidence", []), judgment_record.get("evidence", [])
        ),
        "privacy": {
            "method": "paired_llm_analysis_deterministic_render",
            "manual_review_required": True,
        },
    }


if __name__ == "__main__":
    raise SystemExit("This library performs no API calls and has no standalone batch yet.")
