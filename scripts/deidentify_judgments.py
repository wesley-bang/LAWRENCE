#!/usr/bin/env python3
"""Normalize and deterministically de-identify the raw legal-document corpus.

This is the MVP privacy pass described in judgment_corpus_deidentification_pipeline.md.
Raw data is never modified. Restricted mappings and entity registries are written
separately from the clean training corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
import unicodedata
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable


TAIPEI_TZ = timezone(timedelta(hours=8))
VERSION = "deid-indictment-v1"
CJK_NAME = r"[\u3400-\u9fff\uF900-\uFAFF○Ｏ〇Oo·‧]{2,5}"
ALIAS_MARKS = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥"
COMMON_SURNAME = (
    r"(?:歐陽|司馬|上官|諸葛|夏侯|皇甫|尉遲|公孫|慕容|司徒|司空|"
    r"趙|錢|孫|李|周|吳|鄭|王|馮|陳|蔣|沈|韓|楊|朱|秦|尤|許|何|呂|施|張|"
    r"曹|嚴|華|金|魏|陶|姜|謝|鄒|蘇|潘|葛|范|彭|魯|韋|馬|苗|方|俞|任|袁|"
    r"柳|史|唐|薛|雷|賀|倪|湯|殷|羅|畢|郝|安|常|樂|于|傅|齊|康|伍|余|顧|"
    r"孟|平|黃|蕭|尹|姚|邵|汪|祁|毛|狄|米|貝|戴|宋|龐|熊|紀|舒|屈|項|祝|"
    r"董|梁|杜|阮|藍|閔|席|季|賈|路|江|童|顏|郭|梅|盛|林|鍾|徐|邱|高|夏|"
    r"蔡|田|樊|胡|凌|霍|萬|柯|盧|莫|房|裘|解|應|丁|鄧|洪|包|左|石|崔|龔|"
    r"程|邢|裴|陸|翁|牛|侯|全|白|賴|卓|池|喬|溫|莊|廖|簡|饒|曾|連|游|葉|"
    r"劉|詹|黎|費|席|章|歐|龍|康|龔|芮|巫|沙|向|古|易|辛|阮)"
)
PROSE_NAME = rf"{COMMON_SURNAME}[\u3400-\u9fff\uF900-\uFAFF○Ｏ〇Oo]{{1,2}}"
LATIN_PERSON_NAME = r"[A-Z][A-Z'-]{1,20}(?:\s+[A-Z][A-Z'-]{1,20}){1,5}"
LATIN_FAMILY = r"(?:NGUYEN|TRAN|PHAM|LE|VO|BUI|DANG|MAI|THAI|HAN|HO|LUONG)"
LATIN_ROSTER_BOUNDARY = r"(?:NGUYEN|PHAM)"
NON_PERSON_LATIN = {
    "LINE PAY", "APPLE PAY", "GOOGLE PAY", "GOOGLE MAP", "QR CODE",
    "PRO MAX", "OPPO RENO", "SAMSUNG GALAXY", "IPHONE XR", "IPHONE SE",
    "CARGO RECEIPT", "STATEMENT OF FACTS", "MADE IN VIETNAM", "ALL IN",
    "THE NORTH FACE", "UNDER ARMOUR", "LINE VOOM", "VPLUS GOLF",
    "EFFERALGAN CODEINE", "EFFERALGAN PARACETAMOL", "ULTRA COMPACT PRO",
    "TEREA MENTHOL", "TEREA PURPLE MENTHOL", "MI-NE ORIGINAL",
    "DOUBLE HAPPINESS VIRGINIA",
}

ROLE_LABELS = {
    "同案被告": "DEFENDANT",
    "共同被告": "DEFENDANT",
    "被告": "DEFENDANT",
    "原告": "PLAINTIFF",
    "自訴人": "PRIVATE_PROSECUTOR",
    "自訴代理人": "PRIVATE_PROSECUTOR_COUNSEL",
    "被害人": "VICTIM",
    "幼童": "CO_OFFENDER",
    "少年": "CO_OFFENDER",
    "告訴人": "COMPLAINANT",
    "告發人": "COMPLAINANT",
    "證人": "WITNESS",
    "共犯": "CO_OFFENDER",
    "法定代理人": "LEGAL_REPRESENTATIVE",
    "告訴代理人": "COMPLAINANT_COUNSEL",
    "選任辯護人": "DEFENSE_COUNSEL",
    "指定辯護人": "DEFENSE_COUNSEL",
    "辯護人": "DEFENSE_COUNSEL",
    "代理人": "REPRESENTATIVE",
    "檢察官": "PROSECUTOR",
    "審判長法官": "JUDGE",
    "法官": "JUDGE",
    "書記官": "CLERK",
    "司法警察官": "POLICE",
    "員警": "POLICE",
    "本案詐欺集團成員": "CO_OFFENDER",
    "詐欺集團成員": "CO_OFFENDER",
    "集團成員": "CO_OFFENDER",
    "人資主管": "CO_OFFENDER",
    "車手": "CO_OFFENDER",
    "收水": "CO_OFFENDER",
}

# Longer labels first so 被告 does not consume 同案被告, etc.
ROLE_ALTERNATION = "|".join(sorted(map(re.escape, ROLE_LABELS), key=len, reverse=True))
ROLE_ONLY = {
    "PROSECUTOR", "JUDGE", "CLERK", "DEFENSE_COUNSEL", "COMPLAINANT_COUNSEL",
    "PRIVATE_PROSECUTOR_COUNSEL", "POLICE",
}
ROLE_DISPLAY = {
    "DEFENDANT": "被告",
    "PLAINTIFF": "原告",
    "PRIVATE_PROSECUTOR": "自訴人",
    "PRIVATE_PROSECUTOR_COUNSEL": "自訴代理人",
    "VICTIM": "被害人",
    "COMPLAINANT": "告訴人",
    "WITNESS": "證人",
    "CO_OFFENDER": "共犯",
    "LEGAL_REPRESENTATIVE": "法定代理人",
    "REPRESENTATIVE": "代理人",
    "DEFENSE_COUNSEL": "辯護人",
    "COMPLAINANT_COUNSEL": "告訴代理人",
    "PROSECUTOR": "檢察官",
    "JUDGE": "法官",
    "CLERK": "書記官",
    "POLICE": "員警",
}

BAD_NAMES = {
    "上列", "本件", "本案", "其等", "渠等", "因而", "因此", "部分", "方面",
    "所為", "犯意", "罪嫌", "前科", "年籍", "住居", "坦承", "自白", "辯稱",
    "供稱", "聲稱", "表示", "主張", "陳稱", "及其", "與其", "於本院", "在本院",
    "之行為", "之犯行", "之供述", "之證述", "之姓名", "之身分", "經本院",
    "到庭", "提起", "偵辦", "執行", "聲請", "起訴", "移送", "依法", "係犯",
    "所有", "所騎", "所駕", "所受", "遭受", "犯竊盜", "犯詐欺", "未到庭",
    "被告", "被害人", "告訴人", "證人", "受刑人", "檢察官", "法官", "書記官",
    "施用前", "應依累", "詳下述",
    "以上", "提起公訴並", "提起公訴及", "追加起訴並", "辯護人", "選任辯護人",
    "律師", "附錄", "簡易程", "簡易庭", "方法", "詐騙方式", "匯款時間",
    "匯款金額", "匯入帳戶", "編號", "證據名稱",
    "身分傳喚", "身分傳喚甲", "身分傳喚乙", "身分傳喚丙", "童頭部", "凌虐",
}

CITY_NAMES = (
    "臺北市", "台北市", "新北市", "桃園市", "臺中市", "台中市", "臺南市", "台南市",
    "高雄市", "基隆市", "新竹市", "嘉義市", "新竹縣", "苗栗縣", "彰化縣", "南投縣",
    "雲林縣", "嘉義縣", "屏東縣", "宜蘭縣", "花蓮縣", "臺東縣", "台東縣", "澎湖縣",
    "金門縣", "連江縣",
)
CITY_ALT = "|".join(map(re.escape, CITY_NAMES))

IDENTIFIER_PATTERNS = {
    "TW_ID": re.compile(r"(?<![A-Za-z0-9])[A-Z][12]\d{8}(?!\d)"),
    "EMAIL": re.compile(r"(?i)(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?!\w)"),
    "IP": re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    # Restrict URLs to URI characters.  Using \S+ is unsafe in Chinese prose:
    # without an intervening space it can consume the following evidence or
    # account label (for example "https://example.test銀行帳號000...").
    "URL": re.compile(
        r"(?i)(?<![A-Za-z0-9])(?:https?://|www\.)"
        r"[A-Za-z0-9][A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]*"
    ),
    "MOBILE": re.compile(r"(?<!\d)09\d{2}[- ]?\d{3}[- ]?\d{3}(?!\d)"),
    "LANDLINE": re.compile(r"(?<!\d)(?:\(0\d{1,2}\)|0\d{1,2})[- ]?\d{6,8}(?:#\d+)?(?!\d)"),
    "PLATE": re.compile(r"(?<![A-Za-z0-9])[A-Z0-9]{2,4}[-－][A-Z0-9]{2,4}(?![A-Za-z0-9])", re.I),
    "CASE_NO": re.compile(r"(?<!\d)\d{2,3}\s*年度?\s*[\u3400-\u9fffA-Za-z]{1,14}\s*字\s*第?\s*[0-9A-Za-z-]+\s*號"),
}

LEAK_SCAN_PATTERNS = {
    key: pattern for key, pattern in IDENTIFIER_PATTERNS.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/indictments/indictments_114_source.jsonl"),
    )
    parser.add_argument("--year-roc", type=int, default=114)
    parser.add_argument("--normalized", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--clean", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    stem = f"indictments_{args.year_roc}"
    args.normalized = args.normalized or Path(f"data/intermediate/normalized/{stem}_normalized.jsonl")
    args.registry = args.registry or Path(f"data/intermediate/entity_registry/{stem}_entities.jsonl")
    args.audit = args.audit or Path(f"data/intermediate/audit/{stem}_audit.jsonl")
    args.mapping = args.mapping or Path(f"data/raw/metadata/{stem}_source_mapping.jsonl")
    args.clean = args.clean or Path(f"data/clean/indictments/{stem}_clean.jsonl")
    args.manifest = args.manifest or Path(f"data/clean/indictments/{stem}_clean_manifest.json")
    return args


def adapt_source_record(record: dict[str, object]) -> dict[str, object]:
    """Convert the Ministry of Justice indictment schema to pipeline fields."""
    barcode = str(record.get("barcode") or "").strip()
    text = str(record.get("text") or "").strip()
    investigation = record.get("investigation") or {}
    charge = record.get("charge") or {}
    agency = record.get("agency") or {}
    if not barcode or not text or not isinstance(investigation, dict):
        raise ValueError("Each source row requires barcode, text, and investigation")
    year_roc = int(investigation.get("year_roc") or 0)
    source_id = f"MOJ-PROSECUTION:{barcode}"
    return {
        "internal_doc_id": hashlib.sha256(source_id.encode("utf-8")).hexdigest(),
        "source_id": source_id,
        "raw_text": text,
        "document_type": str(record.get("doc_type") or "起訴書"),
        "case_type": str(charge.get("normalized") or charge.get("raw") or "unknown")
            if isinstance(charge, dict) else "unknown",
        "agency_group": str(agency.get("short") or agency.get("name") or "unknown")
            if isinstance(agency, dict) else "unknown",
        "source_year_roc": year_roc,
        "source_year_ad": int(investigation.get("year_ad") or year_roc + 1911),
    }


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\uf6b0", "\n").replace("\uf6af", "\n")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"[ \t\xa0]+", " ", text)

    # Rejoin role labels and formal date wording that was visually letter-spaced.
    spaced_terms = (
        "中華民國", "公訴人", "聲請人", "自訴人", "被告", "被害人", "告訴人",
        "證人", "辯護人", "代理人", "法定代理人", "選任辯護人", "指定辯護人",
        "審判長法官", "檢察官", "法官", "書記官", "主文", "理由",
    )
    for term in spaced_terms:
        pattern = r"\s*".join(map(re.escape, term))
        text = re.sub(pattern, term, text)

    # Recover useful structure lost when adjacent HTML div elements were flattened.
    section_terms = "主文|事實及理由|犯罪事實|證據並所犯法條|論罪科刑|理由|附錄|附件[:：]?"
    text = re.sub(rf"\s*(?=({section_terms}))", "\n", text)
    text = text.replace("事實及\n理由", "事實及理由")
    text = re.sub(r"(?<!\n)(中華民國\s*\d{2,3}\s*年)", r"\n\1", text)
    text = re.sub(r"(?<!\n)(?:以上正本|以上正本係|以上正本證明)", lambda m: "\n" + m.group(0), text)
    text = re.sub(r"\n[ ]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Social-media exhibit tables sometimes flatten an item number directly
    # into a ROC year ("Threads14" + "113年" -> "Threads14113年").
    text = re.sub(r"(?i)(Threads)(\d{1,2}?)(1\d{2}年)", r"\1\n\2. \3", text)
    return text.strip()


def normalized_role_text(text: str) -> str:
    # normalize_text already rejoins common labels; cover compound labels here.
    for label in sorted(ROLE_LABELS, key=len, reverse=True):
        text = re.sub(r"\s*".join(map(re.escape, label)), label, text)
    return text


def valid_person_name(name: str) -> bool:
    compact = re.sub(r"\s+", "", name).replace("Ｏ", "○").replace("〇", "○").replace("O", "○").replace("o", "○")
    if compact in BAD_NAMES or len(compact) < 2 or len(compact) > 5:
        return False
    if compact.startswith(("本", "該", "其", "上", "下", "前", "同", "之", "或", "及", "並", "另", "再", "均")) and "○" not in compact:
        return False
    # Purely masked names are already anonymous and need no registry entry.
    if set(compact) <= {"○", "某", "甲", "乙", "丙", "丁"}:
        return False
    return bool(re.fullmatch(CJK_NAME, compact))


def extract_people(text: str) -> list[dict[str, object]]:
    text = normalized_role_text(text)
    found: dict[str, dict[str, object]] = {}

    def add_as(role: str, raw_name: str) -> None:
        raw_mention = raw_name.strip()
        name = re.sub(r"\s+", "", raw_name).replace("Ｏ", "○").replace("〇", "○").replace("O", "○").replace("o", "○")
        name = name.removesuffix("律師")
        name = re.sub(r"(?:上列|本件)$", "", name)
        if not valid_person_name(name):
            return
        entry = found.setdefault(name, {"canonical_name": name, "roles": [], "mentions": [name]})
        if raw_mention != name and raw_mention not in entry["mentions"]:
            entry["mentions"].append(raw_mention)
        if role not in entry["roles"]:
            entry["roles"].append(role)

    def add_latin_as(role: str, raw_name: str) -> None:
        name = re.sub(r"\s+", " ", raw_name).strip()
        # All-capital multi-token strings in these documents are normally
        # romanized personal names. Exclude the few common product acronyms.
        if (
            not re.fullmatch(LATIN_PERSON_NAME, name)
            or name in NON_PERSON_LATIN
        ):
            return
        entry = found.setdefault(name, {"canonical_name": name, "roles": [], "mentions": [name]})
        if role not in entry["roles"]:
            entry["roles"].append(role)

    def add(label: str, raw_name: str) -> None:
        add_as(ROLE_LABELS[label], raw_name)

    # Header/table-like entries: a role followed by whitespace and a short name.
    lazy_name = r"[\u3400-\u9fff\uF900-\uFAFF○Ｏ〇Oo·‧]{2,5}?"
    header = re.compile(
        rf"(?P<label>{ROLE_ALTERNATION})\s+(?P<name>{lazy_name})(?:律師)?"
        rf"(?=\s|上列|住|居|籍|男|女|民國|中華民國|選任|指定|以上|上正本|正本|附錄|附件|提起|到庭|如不服|本件|[（(]|(?P<next>{ROLE_ALTERNATION})|$)"
    )
    for match in header.finditer(text):
        candidate = match.group("name")
        # Some prosecution sources omit 列 in "上列被告", producing
        # "陳元龍上被告". The final 上 is prose, not part of the name.
        if candidate.endswith("上") and text[match.end():].lstrip().startswith("被告"):
            candidate = candidate[:-1]
        add(match.group("label"), candidate)

    # Prosecution headers may list multiple defendants after a single label,
    # including uncommon surnames not present in COMMON_SURNAME.
    header_roster = re.compile(
        rf"(?P<label>被告|共同被告|同案被告)\s+"
        rf"(?P<block>(?:{CJK_NAME}\s+){{1,12}}?{CJK_NAME})(?=上列)"
    )
    for match in header_roster.finditer(text):
        for candidate in re.findall(CJK_NAME, match.group("block")):
            add(match.group("label"), candidate)

    # Prose first mentions normally have no separating space: 被告王小明於…
    prose_follow = (
        "於|在|因|與|騎|駕|搭|甫從|甫於|前往|基於|明知|意圖|持|將|向|遂|旋即|"
        "遭|受有|陳稱|表示|主張|指述|證稱|坦承|供稱|辯稱|犯|涉|」|”|"
        "提起|到庭|執行|偵辦|聲請|具結|所有|所騎|所駕|所受|撤回|律師|開庭|請|交易|車輛|"
        "警詢|偵訊|偵查|之供述|之證述|之指訴|共同|共犯|等|[（(]"
    )
    prose = re.compile(
        rf"(?P<label>{ROLE_ALTERNATION})\s*(?P<name>{PROSE_NAME})(?={prose_follow})"
    )
    for match in prose.finditer(text):
        add(match.group("label"), match.group("name"))

    # Strong document/formula boundaries allow names outside the common-surname
    # list without opening the broad prose rule to ordinary legal phrases.
    strong_follow = "律師|到庭|提起|上列|本件|上正本|以上正本|正本證明|如不服|中華民國"
    strong = re.compile(
        rf"(?P<label>{ROLE_ALTERNATION})\s*(?P<name>{lazy_name})(?={strong_follow})"
    )
    for match in strong.finditer(text):
        add(match.group("label"), match.group("name"))

    # Multi-party headers often state a role once followed by several names.
    block_end = rf"上列|本件(?:正本|證明)|中華民國|{ROLE_ALTERNATION}"
    header_block = re.compile(
        rf"(?P<label>{ROLE_ALTERNATION})\s+"
        rf"(?P<block>(?:{PROSE_NAME}(?:律師)?(?:\s+|[、,，])){{1,8}}{PROSE_NAME}(?:律師)?)"
        rf"(?={block_end})"
    )
    for match in header_block.finditer(text):
        role = ROLE_LABELS[match.group("label")]
        for name_match in re.finditer(PROSE_NAME, match.group("block")):
            add_as(role, name_match.group(0))

    # Expand comma/、 separated party lists from a role-seeded person.
    for _ in range(3):
        snapshot = list(found.items())
        for known, entity in snapshot:
            role = choose_primary_role(entity["roles"])
            after = re.compile(rf"{re.escape(known)}\s*[、,，及與]\s*({PROSE_NAME})")
            before = re.compile(rf"({PROSE_NAME})\s*[、,，及與]\s*{re.escape(known)}")
            for match in after.finditer(text):
                add_as(role, match.group(1))
            for match in before.finditer(text):
                add_as(role, match.group(1))

    # A name used as a formal signature/byline is identifying. Pure nicknames
    # are retained because they do not disclose a real-world identity by
    # themselves (e.g. 暱稱「金魚」).
    introduced = re.compile(rf"(?:署名|名為)\s*[「\"']({PROSE_NAME})[」\"']")
    for match in introduced.finditer(text):
        add_as("CO_OFFENDER", match.group(1))
    for match in re.finditer(rf"({PROSE_NAME})(?=律師)", text):
        add_as("DEFENSE_COUNSEL", match.group(1))

    # Relationship and occupational introductions are strong person cues even
    # when the source does not repeat a procedural role label.
    related_roles = {
        "前妻": "CO_OFFENDER", "前夫": "CO_OFFENDER", "妻子": "CO_OFFENDER",
        "丈夫": "CO_OFFENDER", "友人": "CO_OFFENDER", "負責人": "CO_OFFENDER",
        "法官": "JUDGE", "證人": "WITNESS", "被害人": "VICTIM", "告訴人": "COMPLAINANT",
    }
    relation = re.compile(rf"(?P<label>{'|'.join(related_roles)})\s*(?P<name>{PROSE_NAME})")
    for match in relation.finditer(text):
        add_as(related_roles[match.group("label")], match.group("name"))
    for match in re.finditer(rf"(?P<name>{PROSE_NAME})於[^，。\n]{{0,35}}?受雇於", text):
        add_as("VICTIM", match.group("name"))
    for match in re.finditer(rf"(?P<name>{PROSE_NAME})則為", text):
        add_as("CO_OFFENDER", match.group("name"))
    for match in re.finditer(rf"(?:向|未經)(?P<name>{PROSE_NAME})(?:承租|同意)", text):
        add_as("VICTIM", match.group("name"))
    # Repeat list expansion because a relationship cue may have seeded the
    # first person only (e.g. "證人林甲、林乙於警詢...").
    for known, entity in list(found.items()):
        role = choose_primary_role(entity["roles"])
        for match in re.finditer(rf"{re.escape(known)}\s*[、,，及與]\s*({PROSE_NAME})", text):
            add_as(role, match.group(1))

    # Foreign parties are commonly printed as all-capital romanized names.
    # Require a party/relationship cue; product names and English exhibit text
    # are also all-capital and must not be treated as people globally.
    latin_role = re.compile(rf"(?P<label>{ROLE_ALTERNATION})\s*(?P<name>{LATIN_PERSON_NAME})")
    for match in latin_role.finditer(text):
        add_latin_as(ROLE_LABELS[match.group("label")], match.group("name"))
    latin_context = re.compile(rf"(?:搭載|案經|友人|前妻|前夫)\s*(?P<name>{LATIN_PERSON_NAME})")
    for match in latin_context.finditer(text):
        add_latin_as("CO_OFFENDER", match.group("name"))
    for match in re.finditer(rf"(?P<name>{LATIN_PERSON_NAME})\s*[（(](?:越南|泰國|印尼|菲律賓|馬來西亞)籍[）)]", text):
        add_latin_as("CO_OFFENDER", match.group("name"))
    for match in re.finditer(rf"(?P<name>{LATIN_PERSON_NAME})\s*[（(](?:下稱|中文名|中文姓名|真實姓名)", text):
        add_latin_as("CO_OFFENDER", match.group("name"))
    # Split whitespace-only Vietnamese rosters at the next family name.
    vietnamese_name = re.compile(
        rf"\b(?P<name>{LATIN_FAMILY}(?:\s+(?!{LATIN_ROSTER_BOUNDARY}\b)[A-Z][A-Z'-]{{1,20}}){{1,4}})"
    )
    for match in vietnamese_name.finditer(text):
        add_latin_as("CO_OFFENDER", match.group("name"))
    # The remaining all-capital multi-token spans are overwhelmingly foreign
    # names in this prosecution corpus (including personnel tables). Preserve a
    # small explicit product/document whitelist to avoid semantic damage.
    for match in re.finditer(LATIN_PERSON_NAME, text):
        add_latin_as("CO_OFFENDER", match.group(0))

    # Surname-bearing partial designations remain identifying enough to receive
    # an alias (e.g. 曾蜜小姐, 徐先生). A no-surname court code still stays.
    titled = re.compile(rf"(?P<base>{COMMON_SURNAME}[\u3400-\u9fff]{{0,2}})(?P<title>先生|小姐)")
    for match in titled.finditer(text):
        base = match.group("base")
        full = match.group(0)
        if len(base) >= 2:
            add_as("CO_OFFENDER", base)
            entry = found.get(base)
        else:
            entry = found.setdefault(full, {"canonical_name": full, "roles": [], "mentions": [full]})
            if "CO_OFFENDER" not in entry["roles"]:
                entry["roles"].append("CO_OFFENDER")
        if entry and full not in entry["mentions"]:
            entry["mentions"].append(full)

    # Parenthetical references such as 被害人（真名） or 證人A即王小明.
    parenthetical = re.compile(
        rf"(?P<label>{ROLE_ALTERNATION})[（(](?P<name>{CJK_NAME})[）)]"
    )
    for match in parenthetical.finditer(text):
        add(match.group("label"), match.group("name"))

    # Already partly masked names still need stable case-level aliases. This
    # covers prose/table forms such as "告訴人鍾○○、林○○證述".
    masked_party_list = re.compile(
        rf"(?P<label>{ROLE_ALTERNATION})\s*"
        rf"(?P<names>{COMMON_SURNAME}[○Ｏ〇Oo]{{2}}(?:\s*[、,，及與]\s*{COMMON_SURNAME}[○Ｏ〇Oo]{{2}}){{0,12}})"
    )
    for match in masked_party_list.finditer(text):
        role = ROLE_LABELS[match.group("label")]
        for name in re.findall(rf"{COMMON_SURNAME}[○Ｏ〇Oo]{{2}}", match.group("names")):
            add_as(role, name)

    # Signatures sometimes insert spaces between every character of a name.
    spaced_signature = re.compile(
        rf"(?P<label>審判長法官|法官|書記官|檢察官)\s+"
        rf"(?P<name>{COMMON_SURNAME}(?:\s*[㐀-鿿豈-﫿]){{1,2}})"
        rf"(?=\s*(?:{ROLE_ALTERNATION}|以上正本|上列正本|中華民國|所犯法條|$))"
    )
    for match in spaced_signature.finditer(text):
        add(match.group("label"), match.group("name"))

    # A partly redacted name still carries a surname and must not pass through
    # merely because a court omitted the role label at that occurrence.
    for match in re.finditer(rf"{COMMON_SURNAME}[○Ｏ〇Oo]{{2}}", text):
        add_as("CO_OFFENDER", match.group(0))

    people = list(found.values())
    role_counts: Counter[str] = Counter()
    for person in people:
        primary = choose_primary_role(person["roles"])
        role_counts[primary] += 1
        number = role_counts[primary]
        base = ROLE_DISPLAY[primary]
        if primary in ROLE_ONLY:
            # Preserve the source job title and use one neutral name mask for
            # court, legal, and law-enforcement professionals.
            person["alias"] = "〇〇〇"
        else:
            mark = ALIAS_MARKS[len([p for p in people[: people.index(person)] if choose_primary_role(p["roles"]) not in ROLE_ONLY]) % len(ALIAS_MARKS)]
            # Keep the case-level identity neutral. The surrounding source role
            # then naturally yields 被告甲 / 告訴人甲 without duplicated or
            # conflicting labels when one person has multiple procedural roles.
            person["alias"] = mark
        person["role"] = primary
    return people


def choose_primary_role(roles: Iterable[str]) -> str:
    priority = (
        "DEFENDANT", "PLAINTIFF", "PRIVATE_PROSECUTOR", "VICTIM", "COMPLAINANT", "WITNESS", "CO_OFFENDER",
        "LEGAL_REPRESENTATIVE", "REPRESENTATIVE", "DEFENSE_COUNSEL",
        "COMPLAINANT_COUNSEL", "PRIVATE_PROSECUTOR_COUNSEL", "PROSECUTOR", "JUDGE", "CLERK", "POLICE",
    )
    role_set = set(roles)
    return next((role for role in priority if role in role_set), next(iter(role_set)))


def extract_organizations(text: str) -> list[dict[str, str]]:
    organizations: dict[str, str] = {}
    jurisdiction = (
        "臺北|台北|士林|新北|宜蘭|基隆|桃園|新竹|苗栗|臺中|台中|彰化|南投|"
        "雲林|嘉義|臺南|台南|高雄|橋頭|花蓮|臺東|台東|屏東|澎湖|金門|連江"
    )
    fixed_patterns = [
        (rf"(?:臺灣|福建)(?:{jurisdiction})地方法院", "某地方法院"),
        (r"臺灣高等法院(?:臺中|臺南|高雄|花蓮)分院|福建高等法院金門分院|臺灣高等法院|最高法院", "某上級法院"),
        (rf"(?:臺灣|福建)(?:{jurisdiction})地方檢察署", "某地方檢察署"),
        (r"臺灣高等檢察署(?:臺中|臺南|高雄|花蓮)檢察分署|福建高等檢察署金門檢察分署|臺灣高等檢察署|最高檢察署", "某上級檢察署"),
        (rf"(?:{CITY_ALT})(?:政府)?警察局(?:[\u3400-\u9fff]{{1,8}}分局)?(?:[\u3400-\u9fff]{{1,8}}(?:分隊|派出所))?", "某警察機關"),
        (r"法務部矯正署[\u3400-\u9fff]{1,12}(?:監獄|看守所|戒治所)", "某矯正機關"),
        (r"[\u3400-\u9fff]{1,8}簡易庭", "某簡易庭"),
    ]
    for pattern, alias in fixed_patterns:
        for match in re.finditer(pattern, text):
            organizations.setdefault(match.group(0), alias)

    suffix = (
        "股份有限公司|有限公司|銀行|醫院|診所|大學|高級中學|高中|國民中學|國中|"
        "國民小學|國小|幼兒園|酒店|旅館|餐廳|便利商店|超商[\u3400-\u9fff]{0,8}門市|"
        "[\u3400-\u9fff]{1,10}(?:門市|分店)|莊園|會館|夜店|KTV|商行|企業社|工作室|基金會|協會|工會"
    )
    generic = re.compile(rf"(?:^|[於在至向由為及與、，。;；:：()（）號「」『』])([\u3400-\u9fffA-Za-z0-9·○Ｏ〇]{{2,16}}(?:{suffix}))")
    action_split = re.compile(
        r"委託不知情之|及不知情之|名下所有之|所申設之|行車路線自|欲駛入|前往|離開|設於|搭載|"
        r"明知|委託|送往|出具|"
        r"附載|應與|使用|郵寄|旁之|復有|另有|並有|名下|至|往|於|在|為|與|及|之"
    )
    bad_org_fragments = (
        "被告", "原告", "告訴人", "本案", "前揭", "上開", "該銀行", "自用", "普通",
        "小客車", "機車", "帳戶", "密碼", "犯罪", "判決", "行車路線", "居所",
        "自陳", "自述", "警詢", "審理", "等人", "所示", "金融機構帳號", "提款卡",
        "申請", "客服", "之前有跟", "自己向", "附表",
        "證人", "證述",
    )
    generic_org_terms = {
        "網路銀行", "連結銀行", "銀行", "醫院", "大學", "高中", "國中", "國小",
        "餐廳", "酒店", "旅館", "公司", "有限公司", "股份有限公司", "便利商店",
        "家股份有限公司",
    }

    def clean_org_candidate(raw_name: str) -> str | None:
        parts = action_split.split(raw_name)
        name = re.sub(r"^\d+", "", parts[-1] if len(parts) > 1 else raw_name)
        name = re.sub(r"^(?:[一二三四五六七八九十]+|[.．、])+\s*", "", name)
        name = name.removeprefix("樓").removeprefix("被告")
        # Court-assigned person codes may sit directly beside prose and a bank
        # name ("嗣經A02經星展銀行", "A05以網路銀行").  They are not part
        # of an organization and must never be swallowed by its replacement.
        name = re.sub(r"^.*?A\d{2,}(?:經|以|之|所)?", "", name, flags=re.I)
        name = name.removeprefix("下稱")
        if (
            len(name) < 2
            or len(name) > 24
            or name in generic_org_terms
            or any(token in name for token in bad_org_fragments)
        ):
            return None
        return name

    generic_index = 0
    for match in generic.finditer(text):
        name = clean_org_candidate(match.group(1))
        if not name:
            continue
        if name in organizations:
            continue
        generic_index += 1
        organizations[name] = f"機構{ALIAS_MARKS[(generic_index - 1) % len(ALIAS_MARKS)]}"
    # A frequent source form is a masked street number immediately followed by
    # a venue name ("00號某某酒店"). Capture it independently of the address.
    after_number = re.compile(rf"(?<=號)([\u3400-\u9fffA-Za-z0-9·○Ｏ〇]{{2,20}}(?:{suffix}))")
    for match in after_number.finditer(text):
        name = clean_org_candidate(match.group(1))
        if not name:
            continue
        if name not in organizations:
            generic_index += 1
            organizations[name] = f"機構{ALIAS_MARKS[(generic_index - 1) % len(ALIAS_MARKS)]}"

    # Catch organizations after strong grammatical cues. Keep these patterns
    # narrow: a generic suffix scan can consume substantive Chinese prose.
    contextual_named = re.compile(
        rf"(?:委託不知情之|明知)(?P<name>[\u3400-\u9fffA-Za-z0-9·○Ｏ〇]{{2,28}}(?:{suffix}))"
    )
    for match in contextual_named.finditer(text):
        name = clean_org_candidate(match.group("name"))
        if name and name not in organizations:
            generic_index += 1
            organizations[name] = f"機構{ALIAS_MARKS[(generic_index - 1) % len(ALIAS_MARKS)]}"
    partly_masked_org = re.compile(
        rf"[\u3400-\u9fff]{{1,4}}[○Ｏ〇]{{1,2}}[\u3400-\u9fff]{{0,12}}(?:{suffix})"
    )
    for match in partly_masked_org.finditer(text):
        name = clean_org_candidate(match.group(0))
        if name and name not in organizations:
            generic_index += 1
            organizations[name] = f"機構{ALIAS_MARKS[(generic_index - 1) % len(ALIAS_MARKS)]}"
    named_hospital = re.compile(
        r"(?:國立|敏盛|佛教|長庚|三軍|衛生福利部|中國醫藥|高雄醫學|奇美|馬偕|亞東|彰化基督教)"
        r"(?:(?![及與、，])\s*[\u3400-\u9fff]){1,24}?(?:醫院|診所)"
    )
    for match in named_hospital.finditer(text):
        name = clean_org_candidate(match.group(0))
        if name and name not in organizations:
            generic_index += 1
            organizations[name] = f"機構{ALIAS_MARKS[(generic_index - 1) % len(ALIAS_MARKS)]}"
    defendant_bank = re.compile(r"(?<=被告)[\u3400-\u9fff]{2,16}(?:商業銀行|銀行)")
    for match in defendant_bank.finditer(text):
        name = clean_org_candidate(match.group(0))
        if name and name not in organizations:
            generic_index += 1
            organizations[name] = f"機構{ALIAS_MARKS[(generic_index - 1) % len(ALIAS_MARKS)]}"
    named_bank = re.compile(
        r"(?:中國信託|台北富邦|臺北富邦|國泰世華|合作金庫|彰化|華南|兆豐|臺灣|台灣|"
        r"新光|永豐|元大|聯邦|遠東|星展|渣打|滙豐|匯豐|凱基|台新|臺新|陽信|板信|上海)"
        r"(?:商業銀行|銀行)"
    )
    for match in named_bank.finditer(text):
        name = clean_org_candidate(match.group(0))
        if name and name not in organizations:
            generic_index += 1
            organizations[name] = f"機構{ALIAS_MARKS[(generic_index - 1) % len(ALIAS_MARKS)]}"

    # Store/branch names can disclose a precise location even after the street
    # address itself has been generalized.
    venue_context = re.compile(
        r"(?:號|之[「『]?|[「『])"
        r"([\u3400-\u9fffA-Za-z0-9·○Ｏ〇]{2,20}(?:門市|分店|店|酒店|旅館|餐廳|會館|夜店|KTV))"
    )
    for match in venue_context.finditer(text):
        name = clean_org_candidate(match.group(1))
        if name and name not in organizations:
            generic_index += 1
            organizations[name] = f"機構{ALIAS_MARKS[(generic_index - 1) % len(ALIAS_MARKS)]}"

    # Bank names frequently occur without punctuation before them.
    bank_pattern = re.compile(
        r"(?:(?<=復有)|(?<=另有)|(?<=並有)|(?<=有)|(?<=[、，,]))"
        r"[\u3400-\u9fff]{2,12}(?:商業銀行|銀行)"
    )
    for match in bank_pattern.finditer(text):
        name = clean_org_candidate(match.group(0))
        if name and name not in organizations:
            generic_index += 1
            organizations[name] = f"機構{ALIAS_MARKS[(generic_index - 1) % len(ALIAS_MARKS)]}"
    # Ensure a parenthetical short name resolves to the same alias as its full
    # organization name: 全名(下稱短名) -> 機構甲(下稱機構甲).
    short_form = re.compile(
        r"(?P<full>[\u3400-\u9fffA-Za-z0-9·○Ｏ〇]{2,40})"
        r"[（(]下稱(?P<short>[\u3400-\u9fffA-Za-z0-9·○Ｏ〇]{2,24})[）)]"
    )
    for match in short_form.finditer(text):
        candidates = [
            name for name in organizations
            if name != match.group("short") and match.group("full").endswith(name)
        ]
        full = max(candidates, key=len, default=None)
        if full:
            organizations[match.group("short")] = organizations[full]
    return [{"canonical_name": name, "alias": alias, "type": "ORGANIZATION"} for name, alias in organizations.items()]


def replace_entities(text: str, people: list[dict[str, object]], organizations: list[dict[str, str]]) -> str:
    replacements: list[tuple[str, str]] = []
    for person in people:
        replacements.extend((str(mention), str(person["alias"])) for mention in person["mentions"])
    replacements.extend((org["canonical_name"], org["alias"]) for org in organizations)
    replacements.sort(key=lambda item: len(item[0]), reverse=True)
    for source, target in replacements:
        text = text.replace(source, target)
    return text


def generalize_birth_dates(text: str) -> str:
    pattern = re.compile(r"(?:民國)?(?P<year>\d{2,3})\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日\s*生")
    def repl(match: re.Match[str]) -> str:
        roc_year = int(match.group("year"))
        if roc_year == 0:
            return match.group(0)
        ce_year = roc_year + 1911
        decade = (ce_year // 10) * 10
        return f"約西元{decade}年代出生"
    return pattern.sub(repl, text)


def generalize_addresses(text: str) -> str:
    # Require a street-like component directly after the locality. This avoids
    # treating phrases such as "彰化縣警察局道路交通事故..." as an address.
    pattern = re.compile(
        rf"(?P<city>{CITY_ALT})(?!政府|警察局|地方)"
        rf"(?:[\u3400-\u9fff○〇0-9]{{1,8}}(?:區|鄉|鎮|市|村|里)){{0,3}}"
        rf"[\u3400-\u9fff○〇0-9]{{1,14}}?(?:路|街|大道)"
        rf"[^，。;；\n]{{0,35}}?(?:交岔路口處|交岔路口|巷口|路口|\d+\s*(?:號|樓)|[○〇0]+\s*號|前|內|旁|附近)"
    )
    def replacement(match: re.Match[str]) -> str:
        # Judicial source text commonly publishes an already-masked address
        # such as "高雄市○○區○○○路00○0號".  It no longer identifies a precise
        # location, so preserve the source signal instead of masking it again.
        if re.search(r"[○〇Ｏ]", match.group(0)):
            return match.group(0)
        return f"{match.group('city')}某處"

    return pattern.sub(replacement, text)


def mask_deterministic_pii(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    protected: dict[str, str] = {}

    def protection_token(value: str, counter_name: str) -> str:
        token = f"\n\ue000{chr(0xE100 + len(protected))}\ue001\n"
        protected[token] = value
        counts[counter_name] += 1
        return token

    # Protect source-supplied all-zero values before telephone/address rules.
    # The temporary newlines also prevent an address pattern from consuming a
    # later masked plate in the same sentence.
    zero_plate = re.compile(
        r"(?P<prefix>(?:原)?車牌(?:號碼)?\s*)"
        r"(?P<value>0{2,4}[-－]0{2,4})"
    )
    text = zero_plate.sub(
        lambda match: match.group("prefix")
        + protection_token(match.group("value"), "SOURCE_MASKED_PLATE_PRESERVED"),
        text,
    )
    zero_account = re.compile(
        r"(?P<prefix>(?:帳\s*(?:戶|號|號碼)|金融帳戶)\s*[:：]?\s*)"
        r"(?P<value>0(?:[ \-]*0){5,19})"
    )
    text = zero_account.sub(
        lambda match: match.group("prefix")
        + protection_token(match.group("value"), "SOURCE_MASKED_ACCOUNT_PRESERVED"),
        text,
    )
    # A01/A02-style labels are anonymization codes already assigned by the
    # publishing court.  Protect them from phone, plate and social-ID regexes.
    court_code = re.compile(r"(?<![A-Za-z0-9-])A\d{2,}(?![A-Za-z0-9])", re.I)
    text = court_code.sub(
        lambda match: protection_token(match.group(0), "COURT_CODE_PRESERVED"),
        text,
    )
    replacements = {
        "TW_ID": "[身分證字號]", "EMAIL": "[Email]", "IP": "[IP]", "URL": "[URL]",
        "MOBILE": "[電話]", "LANDLINE": "[電話]", "PLATE": "[車牌]", "CASE_NO": "[案號]",
    }
    for kind, pattern in IDENTIFIER_PATTERNS.items():
        def identifier_replacement(match: re.Match[str], current_kind: str = kind) -> str:
            value = match.group(0)
            # Judicial publications conventionally use zeros for values the
            # source has already masked. Preserve that visible source signal.
            if current_kind == "PLATE" and set(re.sub(r"[-－]", "", value)) <= {"0"}:
                return value
            return replacements[current_kind]

        text, count = pattern.subn(identifier_replacement, text)
        if kind == "PLATE":
            count -= sum(
                set(re.sub(r"[-－]", "", match.group(0))) <= {"0"}
                for match in pattern.finditer(text)
            )
        counts[kind] += count

    account = re.compile(r"(?P<label>(?:銀行)?帳(?:戶|號|號碼)|金融帳戶)\s*[:：]?\s*(?P<value>[0-9-]{6,20})")

    def account_replacement(match: re.Match[str]) -> str:
        digits = match.group("value").replace("-", "")
        if digits and set(digits) <= {"0"}:
            return match.group(0)
        return f"{match.group('label')}[銀行帳號]"

    original_account_matches = list(account.finditer(text))
    text = account.sub(account_replacement, text)
    count = sum(set(match.group("value").replace("-", "")) != {"0"} for match in original_account_matches)
    counts["BANK_ACCOUNT"] += count

    social = re.compile(
        r"(?i)(?P<label>LINE|微信|WeChat|Telegram|IG|Instagram)"
        r"(?:\s*(?:ID|帳號)\s*[:：]?\s*|\s*[:：]\s*)"
        r"[A-Za-z0-9_.-]{4,30}"
    )
    text, count = social.subn(lambda m: f"{m.group('label')}帳號[社群帳號]", text)
    counts["SOCIAL_ID"] += count

    zero_birth = re.compile(r"(?:民國\s*)?0{2,3}\s*年\s*0{1,2}\s*月\s*0{1,2}\s*日\s*生")
    text = zero_birth.sub(
        lambda match: protection_token(match.group(0), "SOURCE_MASKED_BIRTH_PRESERVED"),
        text,
    )
    text = generalize_birth_dates(text)
    text = generalize_addresses(text)

    # Day-level dates and minute/second precision are indirect identifiers.
    roc_date = re.compile(r"民國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*\d{1,2}\s*日")
    text, count = roc_date.subn(r"民國\1年\2月某日", text)
    counts["EXACT_DATE"] += count
    bare_roc_date = re.compile(r"(?<!\d)(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*\d{1,2}\s*日")
    text, count = bare_roc_date.subn(r"\1年\2月某日", text)
    counts["EXACT_DATE"] += count
    ce_date = re.compile(r"(?<!\d)(20\d{2})[年./-](\d{1,2})[月./-]\d{1,2}日?")
    text, count = ce_date.subn(r"\1年\2月某日", text)
    counts["EXACT_DATE"] += count
    exact_time = re.compile(r"(凌晨|清晨|上午|中午|下午|晚間|晚上|夜間)?\s*(\d{1,2})\s*時\s*\d{1,2}\s*分(?:\s*\d{1,2}\s*秒)?(?:許)?")
    text, count = exact_time.subn(lambda m: f"{m.group(1) or ''}約{m.group(2)}時", text)
    counts["EXACT_TIME"] += count
    relative_day = re.compile(r"(翌日?|同日?)[（(]\d{1,2}[）)]日?")
    text, count = relative_day.subn(lambda m: "翌日" if m.group(1).startswith("翌") else "同日", text)
    counts["EXACT_DATE"] += count
    for token, original in protected.items():
        text = text.replace(token, original)
    return text, counts


def strip_source_identifiers(text: str) -> str:
    # Any literal JID or source link is forbidden in clean text.
    text = re.sub(r"[A-Z]{3,6},\d+,[^,\s]+,[^,\s]+,\d{8},\d+", "[案號]", text)
    text = re.sub(r"(?i)https?://(?:judgment|data)\.judicial\.gov\.tw\S*", "", text)
    return text


VOLUME_CITATION = re.compile(
    r"[（(]\s*(?:詳?見|參見)[^()（）\n]{0,240}?(?:卷|筆錄)"
    r"[^()（）\n]{0,160}?(?:頁|記載|所示)?\s*[）)]"
)
BARE_VOLUME_CITATION = re.compile(
    r"[（(]\s*[^()（）\n]{0,80}?(?:本院卷|院卷|偵卷|警卷|簡字卷|審理卷)"
    r"[^()（）\n]{0,40}?第(?:\[車牌\]|\d+)"
    r"(?:\s*[、,，至\-－]\s*(?:\[車牌\]|\d+))*頁[^()（）\n]{0,20}?[）)]"
)


def strip_volume_references(text: str) -> tuple[str, int]:
    """Remove non-semantic docket/page pointers while retaining evidence names."""
    text, count = VOLUME_CITATION.subn("", text)
    text, bare_count = BARE_VOLUME_CITATION.subn("", text)
    count += bare_count
    bracket_citation = re.compile(r"【\s*(?:詳?見|參見)[^】\n]{0,500}?(?:卷|筆錄)[^】\n]{0,300}?】")
    text, bracket_count = bracket_citation.subn("", text)
    count += bracket_count
    inline = re.compile(
        r"(?:詳?見|參見)(?P<description>[^，。;；()（）\n]{0,180}?)"
        r"(?:本院卷[一二三四五六七八九十]?|院卷[一二三四五六七八九十]?|"
        r"偵查卷|刑案偵查卷|偵卷(?:[A-Z]\d+)?|警卷)"
        r"第(?:\[車牌\]|\d+)(?:\s*[、,，至\-－]\s*(?:\[車牌\]|\d+))*頁"
    )

    def inline_replacement(match: re.Match[str]) -> str:
        description = match.group("description").strip(" —-:：")
        evidence_words = ("筆錄", "照片", "紀錄", "報告", "鑑定", "證明", "對話")
        if any(word in description for word in evidence_words) and "字第" not in description:
            return description
        return ""

    text, inline_count = inline.subn(inline_replacement, text)
    count += inline_count
    # Some courts cite a volume without 見, or define a short volume label.
    # Remove the locator itself while retaining any surrounding substantive text.
    residual_volume = re.compile(
        r"(?:詳?見|參見|參|依)?(?:本院卷|院卷|警\d*卷|警卷|偵卷(?:[A-Z]\d+)?|審理卷|簡字卷)"
        r"(?:[一二三四五六七八九十])?"
        r"(?:第?(?:\[車牌\]|\d+)(?:\s*(?:[、,，至\-－~]|之)\s*(?:\[車牌\]|\d+))*頁?)?"
    )
    text, residual_count = residual_volume.subn("", text)
    count += residual_count
    text = re.sub(r"[（(]\s*(?:下稱|附於|參)?\s*[）)]", "", text)
    text = re.sub(r"各?\s*\d+\s*(?:份|張|件)(?=\s*(?:在|附)?卷)", "", text)
    text = re.sub(
        r"(?:在|附)(?:本案)?卷(?:內)?(?:可稽|可佐|可參|足憑|足佐|可考)?",
        "可佐",
        text,
    )
    text = re.sub(r"(?<=疾病)等一切情狀", "", text)
    text = re.sub(r"[（(]\s*如附件\s*[）)]", "", text)
    text = text.replace("《附件》", "")
    return text, count


def normalize_account_phrasing(text: str) -> str:
    """Keep the bank-account fact while removing the identifying number."""
    marks = re.escape(ALIAS_MARKS)
    text = re.sub(rf"(機構[{marks}])(?:銀行)?帳號\[銀行帳號\]號", r"\1銀行帳號", text)
    text = re.sub(r"(?:銀行)?帳號\[銀行帳號\]號", "銀行帳號", text)
    return text


def extract_evidence(text: str) -> list[dict[str, object]]:
    """Extract de-identified evidence descriptions cited by the legal document.

    Public judgment pages do not expose the investigation or court dossier.
    Content therefore stays null unless a future authorized source supplies it.
    """
    cues = re.compile(
        r"(?<![領具所含享])(?:並有|復有|另有|有)(?P<name>[^。；\n]{2,600}?)"
        r"(?:可佐|可稽|可參|足憑|足佐|足資證明)"
    )
    evidence: list[dict[str, object]] = []
    seen: set[str] = set()
    for match in cues.finditer(text):
        name = match.group("name").strip(" ,，、;；")
        # If the prose introduces the actual document after a comma, discard
        # the preceding sentencing/factual clause.
        name = re.split(r"[,，](?:並|復|另)?有", name)[-1]
        name = re.sub(r"各?\s*\d+\s*(?:份|張|件)", "", name)
        name = re.sub(r"\s+", " ", name).strip(" ,，、;；")
        if len(name) < 2 or len(name) > 500 or name in seen:
            continue
        seen.add(name)
        evidence.append({
            "evidence_id": f"E{len(evidence) + 1}",
            "name": name,
            "content": None,
            "content_status": "not_publicly_available",
            "source": "legal_document_text_reference",
        })
    return evidence


def final_normalize(text: str) -> str:
    text = re.sub(rf"審判長法官\s*法官([{re.escape(ALIAS_MARKS)}]?)", r"審判長法官\1", text)
    role_names = sorted(set(ROLE_DISPLAY.values()), key=len, reverse=True)
    marks = re.escape(ALIAS_MARKS)
    for role in role_names:
        # Replacement acts on the name span, while the role label remains in
        # the source sentence. Collapse "被告 被告甲" to "被告甲".
        text = re.sub(rf"{re.escape(role)}\s*({re.escape(role)}[{marks}]?)", r"\1", text)
        text = re.sub(rf"{re.escape(role)}\s+([{marks}])", rf"{role}\1", text)
    # Preserve professional roles but neutralize their names, including old
    # role-letter aliases and visually letter-spaced signature names.
    professional_titles = (
        "審判長法官", "法官", "檢察官", "書記官", "選任辯護人", "指定辯護人",
        "辯護人", "告訴代理人", "自訴代理人", "司法警察官", "員警",
    )
    for title in professional_titles:
        text = re.sub(rf"{re.escape(title)}\s*[{marks}]", f"{title}〇〇〇", text)
        text = re.sub(rf"{re.escape(title)}\s*〇〇〇", f"{title}〇〇〇", text)
    signature_name = rf"{COMMON_SURNAME}(?:\s*[㐀-鿿豈-﫿]){{1,2}}"
    text = re.sub(
        rf"(?P<title>檢察官|書記官)\s*(?:[{marks}]|{signature_name})"
        rf"(?=\s*(?:本件正本|以上正本|附錄|參考法條|所犯法條|中華民國|\n|$))",
        lambda match: f"{match.group('title')}〇〇〇",
        text,
    )
    text = re.sub(
        rf"(?P<title>(?:審判長)?法官)\s*(?:[{marks}]|{signature_name})"
        rf"(?=\s*(?:以上正本|中華民國|\n|$))",
        lambda match: f"{match.group('title')}〇〇〇",
        text,
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def semantic_check(before: str, after: str) -> tuple[bool, dict[str, object]]:
    citation = re.compile(r"第\s*\d+(?:\s*之\s*\d+)?\s*條")
    before_citations = Counter(re.sub(r"\s+", "", x) for x in citation.findall(before))
    after_citations = Counter(re.sub(r"\s+", "", x) for x in citation.findall(after))
    disposition_terms = ("有期徒刑", "拘役", "罰金", "無罪", "不受理", "緩刑", "沒收")
    term_loss = [term for term in disposition_terms if term in before and term not in after]
    passed = before_citations == after_citations and not term_loss and len(after) >= len(before) * 0.45
    return passed, {
        "legal_citations_preserved": before_citations == after_citations,
        "disposition_term_loss": term_loss,
        "length_ratio": round(len(after) / max(1, len(before)), 4),
    }


def leakage_scan(text: str, people: list[dict[str, object]], organizations: list[dict[str, str]]) -> dict[str, object]:
    exact_mentions = []
    for person in people:
        exact_mentions.extend(m for m in person["mentions"] if str(m) in text)
    for org in organizations:
        if org["canonical_name"] in text:
            exact_mentions.append(org["canonical_name"])
    regex_hits: dict[str, int] = {}
    for kind, pattern in LEAK_SCAN_PATTERNS.items():
        matches = list(pattern.finditer(text))
        if kind in {"PLATE", "LANDLINE", "MOBILE"}:
            matches = [
                match for match in matches
                if set(re.sub(r"[^0-9]", "", match.group(0))) != {"0"}
            ]
        regex_hits[kind] = len(matches)
    regex_hits = {kind: count for kind, count in regex_hits.items() if count}

    # Detect an unaliased short name immediately following a sensitive role.
    residual_role_names = []
    role_name = re.compile(rf"(?:{ROLE_ALTERNATION})\s*({PROSE_NAME})(?={prose_follow_for_audit()})")
    for match in role_name.finditer(text):
        value = match.group(1)
        if value[0] not in ALIAS_MARKS and valid_person_name(value):
            residual_role_names.append(value)
    broad_strong = re.compile(
        rf"(?:{ROLE_ALTERNATION})\s*({CJK_NAME}?)(?=律師|到庭|提起|上列|本件|上正本|以上正本|正本證明|如不服|中華民國)"
    )
    for match in broad_strong.finditer(text):
        value = match.group(1)
        alias_tokens = [rf"{re.escape(role)}[{re.escape(ALIAS_MARKS)}]?" for role in set(ROLE_DISPLAY.values())]
        alias_value = any(re.fullmatch(token, value) for token in alias_tokens)
        alias_value = alias_value or any(re.search(token, value) for token in alias_tokens)
        alias_value = alias_value or value in {org["alias"] for org in organizations}
        if value and not alias_value and value[0] not in ALIAS_MARKS and valid_person_name(value):
            residual_role_names.append(value)
    residual_role_names = sorted(set(residual_role_names))
    passed = not exact_mentions and not regex_hits and not residual_role_names
    return {
        "pass": passed,
        "original_mentions": sorted(set(map(str, exact_mentions))),
        "regex_hits": regex_hits,
        "residual_role_names": residual_role_names,
    }


def prose_follow_for_audit() -> str:
    return (
        "於|在|因|與|騎|駕|搭|甫從|甫於|前往|基於|明知|意圖|持|將|向|遂|旋即|"
        "遭|受有|陳稱|表示|主張|指述|證稱|坦承|供稱|辯稱|犯|涉|」|”|"
        "提起|到庭|執行|偵辦|聲請|具結|所有|所騎|所駕|所受|撤回|律師"
    )


def deidentify(record: dict[str, object], clean_id: str | None = None) -> tuple[dict, dict, dict, dict, dict]:
    normalized = normalize_text(str(record["raw_text"]))
    people = extract_people(normalized)
    organizations = extract_organizations(normalized)
    clean_text = replace_entities(normalized, people, organizations)
    clean_text = strip_source_identifiers(clean_text)
    clean_text, volume_citations_removed = strip_volume_references(clean_text)
    clean_text, mask_counts = mask_deterministic_pii(clean_text)
    clean_text = normalize_account_phrasing(clean_text)
    clean_text = final_normalize(clean_text)
    evidence = extract_evidence(clean_text)

    leakage = leakage_scan(clean_text, people, organizations)
    semantic_pass, semantic_details = semantic_check(normalized, clean_text)
    # Random and intentionally unrelated to the source identifier/text. The restricted
    # mapping file is the only place that links this ID back to the source.
    clean_id = clean_id or uuid.uuid4().hex
    year = int(record["source_year_ad"])

    normalized_record = {
        "internal_doc_id": record["internal_doc_id"],
        "source_id": record["source_id"],
        "normalization_version": "normalize-v1",
        "normalized_text": normalized,
        "normalized_hash": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    }
    registry_record = {
        "case_internal_id": record["internal_doc_id"],
        "source_id": record["source_id"],
        "persons": people,
        "organizations": organizations,
    }
    audit_record = {
        "case_internal_id": record["internal_doc_id"],
        "doc_id": clean_id,
        "leakage": leakage,
        "semantic": {"pass": semantic_pass, **semantic_details},
        "mask_counts": dict(mask_counts),
        "volume_citations_removed": volume_citations_removed,
        "evidence_count": len(evidence),
        "llm_audit": "not_run",
        "manual_review_required": True,
    }
    clean_record = {
        "doc_id": clean_id,
        "document_type": record["document_type"],
        "issuing_level": "district_prosecutors_office",
        "case_domain": "criminal",
        "year_roc": record["source_year_roc"],
        "year_bucket": f"{(year // 10) * 10}s",
        "case_type": record["case_type"],
        "text": clean_text,
        "evidence": evidence,
        "privacy": {
            "version": VERSION,
            "entity_count": len(people) + len(organizations),
            "audit_pass": leakage["pass"],
            "regex_pass": not leakage["regex_hits"],
            "semantic_pass": semantic_pass,
            "llm_audit": False,
            "manual_review_required": True,
        },
    }
    mapping_record = {"internal_clean_doc_id": clean_id, "source_id": record["source_id"]}
    return normalized_record, registry_record, audit_record, clean_record, mapping_record


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    source_rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    selected_rows = [
        row for row in source_rows
        if int((row.get("investigation") or {}).get("year_roc") or 0) == args.year_roc
    ]
    records = [adapt_source_record(row) for row in selected_rows]
    source_ids = [str(row["source_id"]) for row in records]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("Duplicate source identifiers remain after year filtering")
    # Preserve random clean IDs across reruns so existing human-review records
    # remain attached to the same source case.
    existing_ids: dict[str, str] = {}
    if args.mapping.exists():
        for line in args.mapping.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                source_id = row.get("source_id") or row.get("source_jid")
                if source_id:
                    existing_ids[str(source_id)] = str(row["internal_clean_doc_id"])
    normalized_rows, registry_rows, audit_rows, clean_rows, mapping_rows = [], [], [], [], []
    for index, record in enumerate(records, 1):
        outputs = deidentify(record, existing_ids.get(str(record["source_id"])))
        for bucket, value in zip(
            (normalized_rows, registry_rows, audit_rows, clean_rows, mapping_rows), outputs
        ):
            bucket.append(value)
        if index % 100 == 0:
            print(f"processed {index}/{len(records)}", flush=True)

    write_jsonl(args.normalized, normalized_rows)
    write_jsonl(args.registry, registry_rows)
    write_jsonl(args.audit, audit_rows)
    write_jsonl(args.clean, clean_rows)
    write_jsonl(args.mapping, mapping_rows)

    manifest = {
        "created_at": datetime.now(TAIPEI_TZ).isoformat(timespec="seconds"),
        "version": VERSION,
        "selection": {
            "source_file": str(args.input.resolve()),
            "requested_year_roc": args.year_roc,
            "source_row_count": len(source_rows),
            "excluded_other_year_count": len(source_rows) - len(selected_rows),
            "selected_count": len(selected_rows),
            "strategy": "strict year filter; retain the existing agency-by-charge stratified sample",
            "agency_distribution": dict(sorted(Counter(row["agency_group"] for row in records).items())),
            "charge_distribution": dict(Counter(row["case_type"] for row in records).most_common()),
        },
        "input_count": len(records),
        "output_count": len(clean_rows),
        "unique_doc_ids": len({row["doc_id"] for row in clean_rows}),
        "unique_clean_text_count": len({hashlib.sha256(row["text"].encode("utf-8")).hexdigest() for row in clean_rows}),
        "leakage_pass_count": sum(row["leakage"]["pass"] for row in audit_rows),
        "semantic_pass_count": sum(row["semantic"]["pass"] for row in audit_rows),
        "manual_review_required_count": len(clean_rows),
        "llm_audit_run_count": 0,
        "total_people": sum(len(row["persons"]) for row in registry_rows),
        "total_organizations": sum(len(row["organizations"]) for row in registry_rows),
        "total_evidence_items": sum(len(row["evidence"]) for row in clean_rows),
        "volume_citations_removed": sum(row["volume_citations_removed"] for row in audit_rows),
        "mask_counts": dict(sum((Counter(row["mask_counts"]) for row in audit_rows), Counter())),
        "outputs": {
            "normalized": str(args.normalized.resolve()),
            "entity_registry_restricted": str(args.registry.resolve()),
            "audit": str(args.audit.resolve()),
            "clean": str(args.clean.resolve()),
            "source_mapping_restricted": str(args.mapping.resolve()),
        },
        "limitations": [
            "Deterministic pipeline only; no cloud LLM received raw or normalized text.",
            "Chinese NER/LLM privacy audit and complete human review have not yet been run.",
            "Investigation dossiers and original evidence are not exposed by the public indictment page; evidence content is null when only cited by name.",
            "Clean records remain marked manual_review_required and are not release-ready.",
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
