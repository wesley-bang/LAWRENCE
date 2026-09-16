#!/usr/bin/env python3
"""One-call LLM analysis plus deterministic rendering for new pilot cases."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import sqlite3
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.deidentify_judgments import generalize_addresses, generalize_birth_dates
except ModuleNotFoundError:  # Direct execution adds scripts/ rather than repo root.
    from deidentify_judgments import generalize_addresses, generalize_birth_dates


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data/review/indictment_reviews.sqlite3"
DEFAULT_OUTPUT = ROOT / "data/intermediate/google_ai/hybrid_gemma4_experiment.jsonl"
DEFAULT_FAILURES = ROOT / "data/intermediate/google_ai/hybrid_gemma4_failures.jsonl"
TWO_PASS_OUTPUT = ROOT / "data/intermediate/google_ai/two_pass_gemma4_experiment.jsonl"
FREE_ONLY_MODELS = {"gemma-4-31b-it", "gemma-4-26b-a4b-it"}
CASE_NUMBER_PATTERN = re.compile(
    r"(?<!\d)(?P<year>\d{2,3})\s*(?:年度?)?\s*(?P<word>[\u3400-\u9fffA-Za-z]{1,14})\s*字\s*第?\s*[0-9A-Za-z-]+\s*號"
)
SUFFIXES = list("甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥")
ROLE_LABELS = {
    "DEFENDANT": "被告", "WITNESS": "證人", "COMPLAINANT": "告訴人",
    "VICTIM": "被害人", "INVESTOR": "投資人", "PROSECUTOR": "檢察官",
    "CLERK": "書記官", "JUDGE": "法官", "DEFENSE_COUNSEL": "辯護人",
    "COMPLAINANT_COUNSEL": "告訴代理人", "PRIVATE_PROSECUTOR_COUNSEL": "自訴代理人",
    "POLICE": "員警", "OTHER": "人物",
}
PROFESSIONAL_ROLES = (
    "PROSECUTOR", "CLERK", "JUDGE", "DEFENSE_COUNSEL",
    "COMPLAINANT_COUNSEL", "PRIVATE_PROSECUTOR_COUNSEL", "POLICE",
)


class EmptyModelResponseError(RuntimeError):
    """The API returned successfully but supplied no candidate content."""

    def __init__(self, envelope: dict):
        self.envelope = envelope
        super().__init__(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpointed hybrid free-only Gemma 4 pipeline")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--model", default="gemma-4-31b-it", choices=sorted(FREE_ONLY_MODELS))
    parser.add_argument(
        "--key-name",
        help="use only this .env key name, for example GOOGLE_STUDIO_API_KEY",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--scope", choices=("pending", "failed", "errors", "rendered"), default="pending",
        help="pending handles new documents; failed rerenders reviewed failures; errors retries empty/transient responses; rendered rerenders all checkpoints without new requests",
    )
    parser.add_argument("--delay-seconds", type=float, default=8.0)
    parser.add_argument(
        "--max-source-chars", type=int, default=24000,
        help="skip documents that cannot fit the free API's per-request token allowance",
    )
    parser.add_argument(
        "--retry-429-seconds", type=float, default=300.0,
        help="initial wait after HTTP 429/temporary errors; doubles up to one hour",
    )
    parser.add_argument(
        "--max-transient-attempts", type=int, default=4,
        help="defer one case after this many 5xx/network failures so its worker can continue",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, records: dict[str, dict], order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(
        json.dumps(records[doc_id], ensure_ascii=False, separators=(",", ":")) + "\n"
        for doc_id in order if doc_id in records
    ), encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            # Windows scanners/editors can briefly hold the destination open.
            # Retrying the local atomic rename does not repeat an API request.
            time.sleep(0.2 * (attempt + 1))


def parse_dotenv_value(value: str) -> str:
    """Parse one simple dotenv value, including quoted values with comments."""
    value = value.strip()
    if not value:
        return ""
    if value[0] in {"\"", "'"}:
        quote = value[0]
        closing = value.find(quote, 1)
        if closing < 0:
            raise ValueError("unterminated quoted dotenv value")
        remainder = value[closing + 1:].strip()
        if remainder and not remainder.startswith("#"):
            raise ValueError("unexpected content after quoted dotenv value")
        return value[1:closing]
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def load_api_keys(env_path: Path | None = None) -> list[tuple[str, str]]:
    env_path = env_path or ROOT / ".env"
    keys: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name, value = line.split("=", 1)
        name = name.strip().strip("\"'")
        if re.fullmatch(r"GOOGLE_STUDIO_API_KEY(?:_\d+)?", name):
            key = parse_dotenv_value(value)
            if key:
                keys[name] = key
    if not keys:
        raise RuntimeError("GOOGLE_STUDIO_API_KEY is missing from .env")

    def key_order(item: tuple[str, str]) -> tuple[int, int]:
        name = item[0]
        if name == "GOOGLE_STUDIO_API_KEY":
            return (0, 0)
        return (1, int(name.rsplit("_", 1)[1]))

    return sorted(keys.items(), key=key_order)


def load_api_key() -> str:
    """Backward-compatible accessor for callers that only need the primary key."""
    return load_api_keys()[0][1]


def request_json(
    key: str,
    model: str,
    prompt: str,
    thinking_level: str = "minimal",
    service_tier: str | None = None,
    timeout: float = 900,
    max_output_tokens: int = 32768,
) -> tuple[dict, dict]:
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "maxOutputTokens": max_output_tokens,
            "thinkingConfig": {"thinkingLevel": thinking_level},
        },
    }
    if service_tier:
        payload["service_tier"] = service_tier
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        envelope = json.load(response)
        observed_service_tier = response.headers.get("x-gemini-service-tier")
    if not envelope.get("candidates"):
        raise EmptyModelResponseError(envelope)
    parts = envelope["candidates"][0]["content"]["parts"]
    answer = next(part["text"] for part in reversed(parts) if part.get("text"))
    answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip())
    parsed = json.loads(answer)
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected one JSON object, got {type(parsed).__name__}")
    usage = dict(envelope.get("usageMetadata", {}))
    if observed_service_tier:
        usage["responseServiceTier"] = observed_service_tier
    return parsed, usage


def crime_facts_section(text: str) -> str:
    match = re.search(
        r"(?:^|\n)[ \t]*犯罪事實[ \t]*(?P<body>一、.*?)(?=\n[ \t]*(?:證據並所犯法條|證據及所犯法條|證據清單|所犯法條)|(?<=[。；])[ \t]*(?:證據並所犯法條|證據及所犯法條|證據清單|所犯法條))",
        text,
        re.S,
    )
    return match.group("body").strip() if match else ""


def select_cases(
    db_path: Path,
    scope: str,
    excluded: set[str],
    error_ids: set[str],
    completed_ids: set[str],
    limit: int,
) -> list[dict]:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT c.doc_id,c.source_id,c.case_type,c.normalized_text,c.entities_json,"
            "coalesce(r.notes,'') notes,r.decision "
            "FROM cases c LEFT JOIN reviews r USING(doc_id) "
            "ORDER BY CASE WHEN r.decision='fail' THEN 0 WHEN c.in_sample=1 THEN 1 ELSE 2 END,"
            "r.updated_at DESC,c.sample_rank,c.doc_id"
        ).fetchall()
    selected = [dict(row) for row in rows]
    if scope == "failed":
        selected = [row for row in selected if row.get("decision") == "fail"]
    elif scope == "errors":
        selected = [row for row in selected if row["doc_id"] in error_ids]
    elif scope == "rendered":
        selected = [row for row in selected if row["doc_id"] in completed_ids]
    else:
        selected = [row for row in selected if row["doc_id"] not in excluded]
    return selected[:limit]


def analysis_prompt(case: dict) -> str:
    registry = json.loads(case["entities_json"])
    person_candidates = [item["canonical_name"] for item in registry.get("persons", [])]
    organization_candidates = [item["canonical_name"] for item in registry.get("organizations", [])]
    return f"""你是臺灣刑事起訴書的法律資訊分析員。你只負責建立結構化替換計畫與證據清單，不得重寫全文。只輸出合法 JSON 物件，不要 Markdown。

人物規則：
1. persons 對每一個自然人建立一組；同一人的全名、部分遮罩名、異體字及誤植放在同一組。mentions 只能逐字抄錄原文中的姓名本身，不包含前面的「被告／證人」字樣。
2. role 只能是 DEFENDANT、WITNESS、COMPLAINANT、VICTIM、INVESTOR、PROSECUTOR、CLERK、JUDGE、DEFENSE_COUNSEL、COMPLAINANT_COUNSEL、PRIVATE_PROSECUTOR_COUNSEL、POLICE、OTHER。被告務必標 DEFENDANT；檢察官、書記官、法官、辯護人／律師、訴訟代理人、員警及簽名欄姓名也要列出並標正確職務。
3. A01/A02、甲男、乙女、純甲乙丙及純暱稱保留，不列入 persons；朱OO、林○○等仍含姓氏者要列入。

機構與識別規則：
1. 涉案私人公司、商號、團體及其簡稱 action=MASK；政府機關、法院、地檢署與一般金融機構 action=KEEP。不要把普通法律句子當機構。
2. identifiers 只列必須處理的私人電話、Email、身分證、非全零帳號／車牌、真正未遮罩私人地址，以及案件抬頭或本案偵查案號。
3. 含「○／〇／Ｏ」的地址、公文文號或其他來源既有遮罩字串一律 KEEP，不可列入 identifiers；全零帳號、全零車牌保留。引用其他法院判決的案號保留。

證據規則：
1. evidence 要窮盡全文證據，包含供述、筆錄、名冊、契約、申請書、匯款／交易資料、每一種照片、對話紀錄、數位檔案、搜索扣押文件、實體物及鑑定／測試報告。
2. 每種證據各自一項，保留原文明示數量；name、proves 可先沿用原文姓名與公司名稱，後續確定性程式會依計畫替換。
3. crime_facts_summary 只整理犯罪構成事實，不加入推測。

輸出格式：
{{
  "persons":[{{"group_id":"P01","role":"DEFENDANT","mentions":["原文姓名"],"same_person_reason":"依據"}}],
  "organizations":[{{"group_id":"O01","mentions":["原文名稱","原文簡稱"],"action":"MASK|KEEP","reason":"理由"}}],
  "identifiers":[{{"mention":"原文字串","category":"PHONE|EMAIL|ID|ACCOUNT|PLATE|ADDRESS|CASE_NO","replacement":"[電話]等","reason":"理由"}}],
  "crime_facts_summary":["事實句"],
  "evidence":[{{"name":"證據名稱","category":"供述|書證|照片|數位證據|扣押文件|實體物證|鑑定報告|其他","quantity":"原文數量或null","proves":"證明事項"}}],
  "uncertainties":[]
}}

既有規則候選人物（可能含誤判，須由你核對原文）：{json.dumps(person_candidates, ensure_ascii=False)}
既有規則候選機構（可能含誤判，須由你核對原文）：{json.dumps(organization_candidates, ensure_ascii=False)}
人工備註：{case.get('notes') or '無'}

原始正規化起訴書：
{case['normalized_text']}
"""


def suffix(index: int) -> str:
    return SUFFIXES[index] if index < len(SUFFIXES) else f"{index + 1:02d}"


def build_replacements(
    source: str,
    plan: dict,
    notes: str = "",
    registry: dict | None = None,
) -> list[tuple[str, str, str]]:
    replacements: list[tuple[str, str, str]] = []
    role_counts: defaultdict[str, int] = defaultdict(int)
    registry_roles: defaultdict[str, set[str]] = defaultdict(set)
    for person in (registry or {}).get("persons", []):
        roles = {str(role) for role in person.get("roles", [])}
        if person.get("role"):
            roles.add(str(person["role"]))
        for mention in [person.get("canonical_name", ""), *person.get("mentions", [])]:
            if mention:
                registry_roles[str(mention).strip()].update(roles)
    groups = []
    for item in plan.get("persons", []):
        if not isinstance(item, dict):
            continue
        mentions = sorted({str(x).strip() for x in item.get("mentions", []) if len(str(x).strip()) >= 2}, key=len, reverse=True)
        mentions = [x for x in mentions if x in source and not re.fullmatch(r"A\d{2,}|[甲乙丙丁戊己庚辛壬癸]", x)]
        if mentions:
            groups.append((min(source.find(x) for x in mentions), item, mentions))
    for _, item, mentions in sorted(groups, key=lambda value: value[0]):
        role = str(item.get("role", "OTHER"))
        known_roles = set().union(*(registry_roles.get(mention, set()) for mention in mentions))
        known_professional_roles = [value for value in PROFESSIONAL_ROLES if value in known_roles]
        if known_professional_roles:
            role = known_professional_roles[0]
        label = ROLE_LABELS.get(role, "人物")
        # LLMs sometimes call a separately investigated co-offender a defendant.
        # The deterministic registry and explicit 共犯 context are more reliable
        # for this high-impact distinction.
        if role == "DEFENDANT" and "DEFENDANT" not in known_roles and known_roles:
            if any(re.search(rf"共犯\s*{re.escape(mention)}", source) for mention in mentions):
                label = "共犯"
            elif "WITNESS" in known_roles:
                label = "證人"
            elif "VICTIM" in known_roles:
                label = "被害人"
            elif "COMPLAINANT" in known_roles:
                label = "告訴人"
            else:
                label = "人物"
        alias = "〇〇〇" if role in PROFESSIONAL_ROLES else f"{label}{suffix(role_counts[label])}"
        if role not in PROFESSIONAL_ROLES:
            role_counts[label] += 1
        for mention in mentions:
            replacements.append((mention, alias, label))

    # Gemma occasionally omits a source-redacted name such as 林○浚 or 朱OO.
    # It is still identifying because the surname remains, so supplement only
    # these partial names from the deterministic registry.  Do not import all
    # registry people here because its broad extraction intentionally favors
    # recall and can contain false positives.
    covered_mentions = {mention for mention, _, _ in replacements}
    for person in (registry or {}).get("persons", []):
        candidates = [person.get("canonical_name", ""), *person.get("mentions", [])]
        mentions = sorted({
            str(value).strip() for value in candidates
            if value and re.search(r"[○〇ＯOo]", str(value)) and str(value).strip() in source
        }, key=len, reverse=True)
        mentions = [mention for mention in mentions if mention not in covered_mentions]
        if not mentions:
            continue
        role = str(person.get("role") or next(iter(person.get("roles", [])), "OTHER"))
        label = ROLE_LABELS.get(role, "人物")
        if role == "CO_OFFENDER" and any(
            re.search(rf"少年\s*{re.escape(mention)}", source) for mention in mentions
        ):
            label = "少年"
        alias = "〇〇〇" if role in PROFESSIONAL_ROLES else f"{label}{suffix(role_counts[label])}"
        if role not in PROFESSIONAL_ROLES:
            role_counts[label] += 1
        for mention in mentions:
            replacements.append((mention, alias, label))
            covered_mentions.add(mention)

    forced_organizations: dict[str, str] = {}
    forced_pattern = re.compile(
        r'["「]([^"」()（）]+)[(（]下稱([^"」()（）]+)[)）]["」]'
        r'應該遮罩成["「](機構[^"」()（）]+)[(（]下稱[^"」()（）]+[)）]["」]'
    )
    for full_name, short_name, alias in forced_pattern.findall(notes or ""):
        forced_organizations[full_name] = alias
        forced_organizations[short_name] = alias

    organization_index = 0
    used_organization_aliases = set(forced_organizations.values())
    for item in plan.get("organizations", []):
        if not isinstance(item, dict):
            continue
        mentions = sorted({str(x).strip() for x in item.get("mentions", [])}, key=len, reverse=True)
        joined = " ".join(mentions)
        is_public = bool(re.search(
            r"政府|法院|檢察署|警察局|警政署|公路局|公所|行政院|司法院|立法院|監察院|考試院|"
            r"內政部|外交部|國防部|財政部|教育部|法務部|經濟部|交通部|勞動部|農業部|"
            r"衛生福利部|環境部|文化部|數位發展部|國家發展委員會|國立.+大學|市立.+醫院|縣立.+醫院",
            joined,
        ))
        is_public_financial = any(token in joined for token in ("銀行", "郵局", "郵政"))
        if item.get("action") != "MASK" and (is_public or is_public_financial) and not any(
            mention in forced_organizations for mention in mentions
        ):
            continue
        alias = next((forced_organizations[x] for x in mentions if x in forced_organizations), "")
        if not alias:
            while f"機構{suffix(organization_index)}" in used_organization_aliases:
                organization_index += 1
            alias = f"機構{suffix(organization_index)}"
            used_organization_aliases.add(alias)
            organization_index += 1
        for mention in mentions:
            if len(mention) >= 2 and mention in source:
                replacements.append((mention, alias, "機構"))

    for item in plan.get("identifiers", []):
        if not isinstance(item, dict):
            continue
        mention = str(item.get("mention", "")).strip()
        category = str(item.get("category", ""))
        if not mention or mention not in source:
            continue
        if re.search(r"[○〇Ｏ]", mention):
            continue
        compact = re.sub(r"[-－\s]", "", mention)
        if compact and set(compact) <= {"0"}:
            continue
        default = {
            "PHONE": "[電話]", "EMAIL": "[Email]", "ID": "[身分證字號]",
            "ACCOUNT": "[銀行帳號]", "PLATE": "[車牌]", "ADDRESS": "[地址]",
            "CASE_NO": "[案號]",
        }.get(category, "[敏感資訊]")
        replacement = str(item.get("replacement") or default)
        replacements.append((mention, replacement, "識別"))
    return replacements


def render_text(text: str, replacements: list[tuple[str, str, str]]) -> str:
    tokens: dict[str, str] = {}
    for index, (mention, alias, label) in enumerate(sorted(replacements, key=lambda value: len(value[0]), reverse=True)):
        token = f"\ue200{index:04d}\ue201"
        if label in {"被告", "證人", "告訴人", "被害人", "投資人", "少年", "共犯"}:
            text = re.sub(rf"{label}\s*{re.escape(mention)}", token, text)
        text = text.replace(mention, token)
        tokens[token] = alias
    for token, alias in tokens.items():
        text = text.replace(token, alias)
    text = re.sub(r"證人\s*共犯", "證人即共犯", text)
    text = re.sub(r"被告\s*共犯", "共犯", text)
    professional_titles = "檢察官|書記官|審判長法官|法官|選任辯護人|指定辯護人|辯護人|告訴代理人|自訴代理人|司法警察官|員警"
    text = re.sub(rf"({professional_titles})\s*〇〇〇", r"\1〇〇〇", text)
    alias_prefixes = "被告|證人|告訴人|被害人|投資人|檢察官|書記官|少年|共犯|人物|機構"
    text = re.sub(
        rf"((?:{alias_prefixes})[甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥])\s+(?=[\u3400-\u9fff])",
        r"\1",
        text,
    )
    text = re.sub(
        rf"(?<=[\u3400-\u9fff])\s+(?=(?:{alias_prefixes})[甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥])",
        "",
        text,
    )
    return text


def render_value(value: object, replacements: list[tuple[str, str, str]]) -> object:
    if isinstance(value, str):
        return render_text(value, replacements)
    if isinstance(value, list):
        return [render_value(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: render_value(item, replacements) for key, item in value.items()}
    return value


def deterministic_mask_value(value: object) -> object:
    if isinstance(value, str):
        return generalize_hybrid_text(value)
    if isinstance(value, list):
        return [deterministic_mask_value(item) for item in value]
    return value


def mask_case_numbers_value(value: object) -> object:
    if isinstance(value, str):
        return mask_uncited_case_numbers(value)
    if isinstance(value, list):
        return [mask_case_numbers_value(item) for item in value]
    if isinstance(value, dict):
        return {key: mask_case_numbers_value(item) for key, item in value.items()}
    return value


def generalize_hybrid_text(text: str) -> str:
    protected: dict[str, str] = {}

    def protect(match: re.Match[str]) -> str:
        token = f"\ue300{len(protected):04d}\ue301"
        protected[token] = match.group(0)
        return token

    text = re.sub(r"(?:民國\s*)?0{2,3}\s*年\s*0{1,2}\s*月\s*0{1,2}\s*日\s*生", protect, text)
    text = generalize_birth_dates(text)
    text = generalize_addresses(text)
    text = mask_uncited_case_numbers(text)
    text = re.sub(r"民國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*\d{1,2}\s*日", r"民國\1年\2月某日", text)
    text = re.sub(r"(?<!\d)(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*\d{1,2}\s*日", r"\1年\2月某日", text)
    text = re.sub(
        r"(凌晨|清晨|上午|中午|下午|晚間|晚上|夜間)?\s*(\d{1,2})\s*時\s*\d{1,2}\s*分(?:\s*\d{1,2}\s*秒)?(?:許)?",
        lambda match: f"{match.group(1) or ''}約{match.group(2)}時",
        text,
    )
    for token, value in protected.items():
        text = text.replace(token, value)
    return text


def mask_uncited_case_numbers(text: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        word = match.group("word")
        prefix = text[max(0, match.start() - 35):match.start()]
        if re.search(r"台上|臺上|台抗|臺抗|判例|裁判|見解|意旨", word + prefix):
            return match.group(0)
        if re.search(r"(?:最高|高等|地方法院).{0,20}$", prefix) and re.search(r"判決|裁定", text[match.end():match.end() + 20]):
            return match.group(0)
        return "[案號]"

    return CASE_NUMBER_PATTERN.sub(replacement, text)


def request_case(
    case: dict,
    existing_row: dict | None,
    args: argparse.Namespace,
    key_name: str,
    key: str,
    worker_state: dict[str, float | None],
) -> dict:
    """Fetch one analysis. This runs in a worker and never writes checkpoints."""
    prompt = analysis_prompt(case)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cached = (
        args.scope == "rendered" and existing_row is not None
    ) or (
        not args.force and existing_row is not None
        and existing_row.get("input_sha256") == digest
    )
    if cached:
        return {
            "kind": "success", "cached": True, "digest": digest,
            "plan": existing_row["analysis_plan"],
            "usage": existing_row.get("usage_metadata", {}),
        }
    if len(case["normalized_text"]) > args.max_source_chars:
        return {
            "kind": "failure", "reason": "oversize",
            "record": {
                "doc_id": case["doc_id"],
                "source_id": case["source_id"],
                "case_type": case["case_type"],
                "model": args.model,
                "input_sha256": digest,
                "error": "source_too_long_for_free_api",
                "source_chars": len(case["normalized_text"]),
                "max_source_chars": args.max_source_chars,
                "checkpointed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        }

    retry_wait = max(60.0, args.retry_429_seconds)
    transient_attempts = 0

    def transient_failure(error_type: str, detail: str) -> dict:
        return {
            "kind": "failure", "reason": "transient",
            "record": {
                "doc_id": case["doc_id"],
                "source_id": case["source_id"],
                "case_type": case["case_type"],
                "model": args.model,
                "input_sha256": digest,
                "error": "transient_api_error",
                "transient_error_type": error_type,
                "transient_error_detail": detail[:800],
                "attempts": transient_attempts,
                "key_name": key_name,
                "checkpointed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        }

    while True:
        last_request = worker_state["last_request"]
        if last_request is not None:
            time.sleep(max(0, args.delay_seconds - (time.monotonic() - last_request)))
        worker_state["last_request"] = time.monotonic()
        try:
            plan, usage = request_json(key, args.model, prompt)
            return {
                "kind": "success", "cached": False, "digest": digest,
                "plan": plan, "usage": usage,
            }
        except EmptyModelResponseError as error:
            return {
                "kind": "failure", "reason": "empty",
                "record": {
                    "doc_id": case["doc_id"],
                    "source_id": case["source_id"],
                    "case_type": case["case_type"],
                    "model": args.model,
                    "input_sha256": digest,
                    "error": "empty_model_response",
                    "response": error.envelope,
                    "checkpointed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            }
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504}:
                raise
            error_body = error.read().decode("utf-8", errors="replace")
            try:
                error_payload = json.loads(error_body)
                error_info = error_payload.get("error", {})
                error_summary = re.sub(
                    r"\s+", " ",
                    f"{error_info.get('status', '')}: {error_info.get('message', '')}",
                ).strip()[:800]
            except json.JSONDecodeError:
                error_summary = re.sub(r"\s+", " ", error_body).strip()[:800]
            if error.code != 429:
                transient_attempts += 1
                if transient_attempts >= args.max_transient_attempts:
                    print(
                        f"deferred: {key_name} HTTP {error.code} after "
                        f"{transient_attempts} attempts for {case['doc_id']}", flush=True,
                    )
                    return transient_failure(f"HTTP_{error.code}", error_summary)
            header_wait = error.headers.get("Retry-After")
            wait_seconds = retry_wait
            if header_wait and header_wait.isdigit():
                wait_seconds = max(wait_seconds, float(header_wait))
            print(
                f"waiting: {key_name} HTTP {error.code}; retry in {int(wait_seconds)} "
                f"seconds for {case['doc_id']}", flush=True,
            )
            if error_summary:
                print(f"rate-limit-detail: {key_name} {error_summary}", flush=True)
        except (TimeoutError, urllib.error.URLError, ConnectionError) as error:
            transient_attempts += 1
            if transient_attempts >= args.max_transient_attempts:
                print(
                    f"deferred: {key_name} {type(error).__name__} after "
                    f"{transient_attempts} attempts for {case['doc_id']}", flush=True,
                )
                return transient_failure(type(error).__name__, str(error))
            wait_seconds = retry_wait
            print(
                f"waiting: {key_name} {type(error).__name__}; retry in "
                f"{int(wait_seconds)} seconds for {case['doc_id']}", flush=True,
            )
        time.sleep(wait_seconds)
        retry_wait = min(retry_wait * 2, 3600.0)


def render_checkpoint(
    case: dict, plan: dict, usage: dict, digest: str, model: str, prior: dict | None,
) -> tuple[dict, int, int]:
    registry = json.loads(case["entities_json"])
    replacements = build_replacements(
        case["normalized_text"], plan, case.get("notes") or "", registry
    )
    final_text = generalize_hybrid_text(render_text(case["normalized_text"], replacements))
    evidence = mask_case_numbers_value(render_value(plan.get("evidence", []), replacements))
    evidence_source = "hybrid_analysis"
    if prior:
        prior_final = prior.get("repair") or prior.get("reviewed") or {}
        prior_evidence = prior_final.get("evidence", [])
        if len(prior_evidence) > len(evidence):
            evidence = mask_case_numbers_value(render_value(prior_evidence, replacements))
            evidence_source = "previous_human_reviewed_best"
    summaries = deterministic_mask_value(
        render_value(plan.get("crime_facts_summary", []), replacements)
    )
    record = {
        "experiment": "llm_analysis_deterministic_render",
        "restricted": True,
        "doc_id": case["doc_id"],
        "source_id": case["source_id"],
        "case_type": case["case_type"],
        "model": model,
        "input_sha256": digest,
        "text": final_text,
        "crime_facts": crime_facts_section(final_text),
        "crime_facts_summary": summaries,
        "evidence": evidence,
        "evidence_source": evidence_source,
        "analysis_plan": plan,
        "applied_replacements": len(replacements),
        "usage_metadata": usage,
        "checkpointed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checks": {
            "length_ratio": round(len(final_text) / max(1, len(case["normalized_text"])), 4),
            "duplicate_defendant_label": "被告被告" in final_text,
            "evidence_count": len(evidence),
            "deterministic_generalization": "birth/date/time and precise unmasked address only",
        },
    }
    return record, len(replacements), len(evidence)


def main() -> None:
    args = parse_args()
    if args.limit < 1 or args.limit > 1000:
        raise SystemExit("--limit must be between 1 and 1000")
    if args.max_transient_attempts < 1:
        raise SystemExit("--max-transient-attempts must be at least 1")
    existing = {row["doc_id"]: row for row in read_jsonl(args.output)}
    failures = {row["doc_id"]: row for row in read_jsonl(args.failures)}
    two_pass = {row["doc_id"]: row for row in read_jsonl(TWO_PASS_OUTPUT)}
    excluded = set(existing) | set(failures)
    cases = select_cases(args.db, args.scope, excluded, set(failures), set(existing), args.limit)
    order = list(existing)
    order.extend(case["doc_id"] for case in cases if case["doc_id"] not in existing)
    failure_order = list(failures)
    failure_order.extend(case["doc_id"] for case in cases if case["doc_id"] not in failures)
    keys = load_api_keys()
    if args.key_name:
        keys = [item for item in keys if item[0] == args.key_name]
        if not keys:
            raise SystemExit(f"API key name not found in .env: {args.key_name}")
    worker_count = min(len(keys), len(cases))
    print(f"workers={worker_count} keys={','.join(name for name, _ in keys)}", flush=True)

    case_iter = iter(enumerate(cases, 1))
    active: dict[concurrent.futures.Future, tuple[int, int, dict]] = {}
    states = [{"last_request": None} for _ in keys]

    def submit_next(executor: concurrent.futures.ThreadPoolExecutor, worker_index: int) -> None:
        try:
            index, case = next(case_iter)
        except StopIteration:
            return
        key_name, key = keys[worker_index]
        future = executor.submit(
            request_case, case, existing.get(case["doc_id"]), args,
            key_name, key, states[worker_index],
        )
        active[future] = (worker_index, index, case)

    if worker_count:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            for worker_index in range(worker_count):
                submit_next(executor, worker_index)
            while active:
                done, _ = concurrent.futures.wait(
                    active, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    worker_index, index, case = active.pop(future)
                    result = future.result()
                    if result["kind"] == "failure":
                        failures[case["doc_id"]] = result["record"]
                        write_jsonl(args.failures, failures, failure_order)
                        reason = result["reason"]
                        detail = (
                            f" chars={len(case['normalized_text'])}" if reason == "oversize" else ""
                        )
                        print(
                            f"skipped-{reason} {index}/{len(cases)} {case['doc_id']}{detail}",
                            flush=True,
                        )
                    else:
                        record, replacement_count, evidence_count = render_checkpoint(
                            case, result["plan"], result["usage"], result["digest"],
                            args.model, two_pass.get(case["doc_id"]),
                        )
                        existing[case["doc_id"]] = record
                        failures.pop(case["doc_id"], None)
                        write_jsonl(args.output, existing, order)
                        if args.failures.exists():
                            write_jsonl(args.failures, failures, failure_order)
                        action = "rerendered" if result["cached"] else "processed"
                        print(
                            f"{action} {index}/{len(cases)} {case['doc_id']} "
                            f"replacements={replacement_count} evidence={evidence_count}",
                            flush=True,
                        )
                    submit_next(executor, worker_index)
    print(json.dumps({
        "documents": len(existing), "selected": len(cases), "scope": args.scope,
        "workers": worker_count, "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
