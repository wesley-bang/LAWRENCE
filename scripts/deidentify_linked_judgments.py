#!/usr/bin/env python3
"""Deterministic rendering primitives for linked criminal court decisions.

This module intentionally performs no API calls and writes no corpus files. It
is the judgment-specific rendering half of a future LLM-analysis + deterministic
replacement pipeline.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from collections import defaultdict

try:
    from scripts import deidentify_judgments as base
    from scripts import hybrid_deidentify_with_google_ai as hybrid
except (ModuleNotFoundError, ImportError):  # Direct execution adds scripts/ rather than repo root.
    import deidentify_judgments as base
    import hybrid_deidentify_with_google_ai as hybrid


JUDGMENT_ROLE_LABELS = {
    "APPELLANT": "上訴人",
    "PETITIONER": "聲請人",
    "RESPONDENT": "相對人",
    "SENTENCED_PERSON": "受刑人",
    "LEGAL_REPRESENTATIVE": "法定代理人",
    "PRIVATE_PROSECUTOR": "自訴人",
}

SECTION_HEADINGS = (
    "主文", "犯罪事實及理由", "犯罪事實", "事實及理由", "事實", "理由", "證據及理由",
    "論罪科刑", "沒收", "不另為無罪之諭知", "附錄", "附件",
)

JUDICIAL_PAGE_BODY = re.compile(
    r"(?:臺灣|福建)[^\n]{1,60}?(?:法院|法庭)\s*刑事[^\n]{0,16}?(?:判決|裁定)"
)
JUDGMENT_HEADER_CASE_NUMBERS = re.compile(
    r"(?<!\d)\d{2,3}\s*年\s*度\s*[\u3400-\u9fffA-Za-z]{1,16}\s*字\s*"
    r"第?\s*\d+(?:\s*[、,，]\s*\d+)*\s*號"
    r"(?:\s*第?\s*\d+\s*號)*"
)


def adapt_linked_judgment(record: dict) -> dict:
    """Adapt one downloader-manifest record without exposing its source URL."""
    metadata = record.get("judgment_metadata") or {}
    detail_url = str(record.get("detail_url") or "").strip()
    text = str(record.get("text") or "").strip()
    if not detail_url or not text or not isinstance(metadata, dict):
        raise ValueError("Each linked judgment requires detail_url, text, and judgment_metadata")
    year_roc = int(metadata.get("year") or 0)
    if not year_roc:
        raise ValueError("judgment_metadata.year is required")
    source_digest = hashlib.sha256(detail_url.encode("utf-8")).hexdigest()
    title = str(record.get("title") or "")
    document_type = "刑事裁定" if "裁定" in title else "刑事判決"
    return {
        "internal_doc_id": hashlib.sha256(f"JUDICIAL:{source_digest}".encode()).hexdigest(),
        "source_id": f"JUDICIAL:{source_digest}",
        "raw_text": text,
        "document_type": document_type,
        "issuing_level": "court",
        "court_code": str(metadata.get("court_code") or ""),
        "court_name": str(metadata.get("court_name") or ""),
        "source_year_roc": year_roc,
        "source_year_ad": year_roc + 1911,
        "judgment_metadata": {
            key: metadata.get(key)
            for key in ("court_code", "court_name", "sys", "year", "case_word", "number", "date")
        },
    }


def own_case_number_patterns(metadata: dict) -> tuple[re.Pattern[str], ...]:
    """Return exact patterns for the current decision's docket number."""
    year = str(metadata.get("year") or "").strip()
    word = str(metadata.get("case_word") or "").strip()
    number = str(metadata.get("number") or "").strip()
    if not year or not word or not number:
        raise ValueError("year, case_word, and number are required to mask the own-case number")
    return (
        re.compile(
            rf"(?<!\d){re.escape(year)}\s*年\s*度\s*{re.escape(word)}\s*字\s*"
            rf"第?\s*{re.escape(number)}\s*號"
        ),
        re.compile(
            rf"(?<!\d){re.escape(year)}\s*[,，]\s*{re.escape(word)}\s*[,，]\s*"
            rf"{re.escape(number)}(?!\d)"
        ),
    )


def mask_own_case_number(text: str, metadata: dict) -> str:
    """Mask the primary docket everywhere, before preserving cited precedents."""
    # A consolidated decision can have a heading such as
    # "114年度易字第484號 第663號 第821號" while its metadata exposes only
    # one of those numbers. Everything before the first party label is the
    # document header, not a cited precedent.
    party = re.search(r"公訴人|上訴人|聲請人|相對人|受刑人|自訴人|被告", text)
    first_line_end = text.find("\n")
    header_end = party.start() if party else (
        first_line_end if first_line_end >= 0 else min(len(text), 300)
    )
    header = JUDGMENT_HEADER_CASE_NUMBERS.sub("[本案案號]", text[:header_end])
    text = header + text[header_end:]
    for pattern in own_case_number_patterns(metadata):
        text = pattern.sub("[本案案號]", text)
    return text


def normalize_judgment_text(text: str) -> str:
    """Normalize court HTML text and remove the Judicial Yuan page chrome."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\uf6b0", "\n").replace("\uf6af", "\n")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"[ \t\xa0]+", " ", text)

    # The extracted page begins with sharing/search controls and repeats the
    # public metadata. Start at the actual signed judicial document.
    body = JUDICIAL_PAGE_BODY.search(text)
    if body:
        text = text[body.start():]

    spaced_terms = (
        "中華民國", "公訴人", "聲請人", "相對人", "受刑人", "上訴人",
        "自訴人", "被告", "被害人", "告訴人", "證人", "辯護人",
        "代理人", "法定代理人", "審判長法官", "檢察官", "法官", "書記官",
        *SECTION_HEADINGS,
    )
    for term in sorted(set(spaced_terms), key=len, reverse=True):
        text = re.sub(r"[ \t\n]*".join(map(re.escape, term)), term, text)

    heading_pattern = "|".join(map(re.escape, sorted(SECTION_HEADINGS, key=len, reverse=True)))

    def heading_line(match: re.Match[str]) -> str:
        return f"\n{match.group('heading')}\n"

    # Most HTML block boundaries survive as newlines, but the heading and its
    # first sentence are commonly flattened onto the same line.
    text = re.sub(
        rf"(?m)^[ \t]*(?P<heading>{heading_pattern})[ \t]*[:：]?[ \t]*",
        heading_line,
        text,
    )
    text = re.sub(
        r"(?P<intro>(?:判決|裁定)[^。；\n]{0,24}?如下[:：])[ \t]*(?:主\s*文)[ \t]*",
        lambda match: f"{match.group('intro')}\n主文\n",
        text,
    )
    # Some traditional layouts flatten a new section after the prior full stop.
    text = re.sub(
        rf"(?<=[。；:：])[ \t]+(?P<heading>{heading_pattern})[ \t]*[:：]?[ \t]*"
        rf"(?=[一二三四五六七八九十壹貳參肆伍陸柒捌玖拾0-9、(（])",
        heading_line,
        text,
    )
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_judgment_replacements(
    source: str,
    plan: dict,
    registry: dict | None = None,
    pair_alias_registry: dict | None = None,
) -> list[tuple[str, str, str]]:
    """Build replacements, extending the indictment roles with court roles."""
    common_plan = dict(plan)
    special_people = [
        item for item in plan.get("persons", [])
        if isinstance(item, dict) and str(item.get("role")) in JUDGMENT_ROLE_LABELS
    ]
    common_plan["persons"] = [
        item for item in plan.get("persons", [])
        if not isinstance(item, dict) or str(item.get("role")) not in JUDGMENT_ROLE_LABELS
    ]
    replacements = hybrid.build_replacements(source, common_plan, registry=registry)
    priority_mentions: set[str] = set()
    role_counts: defaultdict[str, int] = defaultdict(int)
    groups = []
    for item in special_people:
        mentions = sorted({
            str(value).strip() for value in item.get("mentions", [])
            if len(str(value).strip()) >= 2 and str(value).strip() in source
        }, key=len, reverse=True)
        if mentions:
            groups.append((min(source.find(value) for value in mentions), item, mentions))
    for _, item, mentions in sorted(groups, key=lambda value: value[0]):
        label = JUDGMENT_ROLE_LABELS[str(item["role"])]
        alias = f"{label}{hybrid.suffix(role_counts[label])}"
        role_counts[label] += 1
        for mention in mentions:
            replacements.append((mention, alias, label))
            priority_mentions.add(mention)

    if pair_alias_registry:
        try:
            from scripts import pair_case_integration as pairing
        except (ModuleNotFoundError, ImportError):
            import pair_case_integration as pairing
        for item in plan.get("persons", []):
            if not isinstance(item, dict):
                continue
            resolved = pairing.resolve_pair_alias(item, pair_alias_registry)
            if not resolved:
                continue
            alias = resolved["alias"]
            label = pairing.alias_label(alias)
            for value in item.get("mentions", []):
                mention = str(value).strip()
                if len(mention) >= 2 and mention in source:
                    replacements.append((mention, alias, label))
                    priority_mentions.add(mention)

    # An explicit LLM judgment role takes precedence over a broad registry
    # supplement for the same mention.
    deduplicated: dict[str, tuple[str, str, str]] = {}
    for replacement in replacements:
        mention = replacement[0]
        if mention not in deduplicated or mention in priority_mentions:
            deduplicated[mention] = replacement
    return list(deduplicated.values())


def render_judgment_text(
    text: str,
    replacements: list[tuple[str, str, str]],
    metadata: dict,
) -> str:
    """Render an LLM plan while masking this decision's docket deterministically."""
    # Protect the primary docket before a broad LLM CASE_NO replacement can
    # collapse it into the less informative generic [案號].
    text = mask_own_case_number(text, metadata)
    text = hybrid.render_text(text, replacements)
    suffixes = re.escape("".join(hybrid.SUFFIXES))
    for label in JUDGMENT_ROLE_LABELS.values():
        text = re.sub(
            rf"{re.escape(label)}\s*({re.escape(label)}(?:[{suffixes}]|\d{{2}}))",
            r"\1",
            text,
        )
    text = base.strip_source_identifiers(text)
    text = mask_own_case_number(text, metadata)
    # The generic pass can now preserve genuine cited decisions without
    # accidentally preserving the already-masked primary docket.
    text = hybrid.generalize_hybrid_text(text)
    return base.final_normalize(text)


def render_judgment_value(
    value: object,
    replacements: list[tuple[str, str, str]],
    metadata: dict,
) -> object:
    if isinstance(value, str):
        return render_judgment_text(value, replacements, metadata)
    if isinstance(value, list):
        return [render_judgment_value(item, replacements, metadata) for item in value]
    if isinstance(value, dict):
        return {
            key: render_judgment_value(item, replacements, metadata)
            for key, item in value.items()
        }
    return value


def extract_judgment_sections(text: str) -> dict[str, str]:
    """Split normalized decisions without assuming an indictment-only layout."""
    text = normalize_judgment_text(text)
    heading_pattern = "|".join(map(re.escape, sorted(SECTION_HEADINGS, key=len, reverse=True)))
    matches = list(re.finditer(rf"(?m)^\s*(?P<heading>{heading_pattern})\s*[:：]?\s*$", text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[match.end():end].strip()
        if content:
            prior = sections.get(match.group("heading"))
            sections[match.group("heading")] = f"{prior}\n{content}" if prior else content
    return sections


def prepare_rendered_record(
    record: dict,
    plan: dict,
    registry: dict | None = None,
    pair_alias_registry: dict | None = None,
    clean_id: str | None = None,
) -> dict:
    """Return a clean candidate from an existing plan; this never calls an LLM."""
    adapted = adapt_linked_judgment(record)
    normalized = normalize_judgment_text(adapted["raw_text"])
    metadata = adapted["judgment_metadata"]
    replacements = build_judgment_replacements(
        normalized, plan, registry, pair_alias_registry
    )
    clean_text = render_judgment_text(normalized, replacements, metadata)
    sections = extract_judgment_sections(clean_text)
    return {
        "doc_id": clean_id or uuid.uuid4().hex,
        "document_type": adapted["document_type"],
        "issuing_level": adapted["issuing_level"],
        "court_code": adapted["court_code"],
        "year_roc": adapted["source_year_roc"],
        "text": clean_text,
        "sections": sections,
        "crime_facts_summary": render_judgment_value(
            plan.get("crime_facts_summary", []), replacements, metadata
        ),
        "evidence": render_judgment_value(plan.get("evidence", []), replacements, metadata),
        "applied_replacements": len(replacements),
        "privacy": {
            "method": "llm_analysis_deterministic_render",
            "own_case_number_masked": "[本案案號]" in clean_text,
            "manual_review_required": True,
        },
    }


if __name__ == "__main__":
    raise SystemExit(
        "This module only provides deterministic judgment rendering; it performs no API calls."
    )
