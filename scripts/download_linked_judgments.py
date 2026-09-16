#!/usr/bin/env python3
"""Download Judicial Yuan judgments linked by the 114/115 indictment corpus."""

from __future__ import annotations

import argparse
import hashlib
import html as html_module
import json
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from lxml import html


ROOT = Path(__file__).resolve().parents[1]
RAW_INDICTMENTS = ROOT / "data/raw/indictments"
OUTPUT_ROOT = ROOT / "data/raw/linked_judgments"
USER_AGENT = "Mozilla/5.0 (compatible; LAWRENCE academic corpus downloader; +https://homepage.ntu.edu.tw/)"
SSL_CONTEXT = ssl.create_default_context()
# The Judicial Yuan chain currently lacks an extension required only by
# OpenSSL's strict mode. Keep CA and hostname verification enabled.
SSL_CONTEXT.verify_flags &= ~ssl.VERIFY_X509_STRICT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpointed downloader for linked Judicial Yuan judgments")
    parser.add_argument("--years", nargs="+", type=int, default=[114, 115], choices=[114, 115])
    parser.add_argument("--limit", type=int, default=0, help="0 downloads every pending unique URL")
    parser.add_argument("--delay-seconds", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def source_groups(year: int) -> list[dict]:
    path = RAW_INDICTMENTS / f"indictments_{year}_source.jsonl"
    grouped: dict[str, dict] = {}
    for item in read_jsonl(path):
        judgment = item.get("judgment") or {}
        query_url = str(judgment.get("url") or "").strip()
        if not query_url:
            continue
        group = grouped.setdefault(query_url, {
            "source_year_roc": year,
            "query_url": query_url,
            "judgment_metadata": judgment,
            "linked_indictments": [],
        })
        group["linked_indictments"].append({
            "barcode": item.get("barcode"),
            "investigation_case_no": (item.get("investigation") or {}).get("case_no"),
            "investigation_case_no_detail": (item.get("investigation") or {}).get("case_no_detail"),
        })
    return list(grouped.values())


def fetch(url: str, timeout: float) -> tuple[bytes, str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "zh-TW,zh;q=0.9"})
    with urllib.request.urlopen(request, timeout=timeout, context=SSL_CONTEXT) as response:
        return response.read(), response.geturl(), response.headers.get_content_type()


def detail_urls(body: bytes, final_url: str) -> list[str]:
    decoded = body.decode("utf-8", errors="replace")
    decoded = html_module.unescape(decoded).replace("\\/", "/")
    candidates: list[str] = []
    if re.search(r"/(?:data|printData)\.aspx\?", final_url, re.I):
        candidates.append(final_url)
    patterns = (
        r"(?:https?://judgment\.judicial\.gov\.tw)?/FJUD/(?:data|printData)\.aspx\?[^\"'<>\s]+",
        r"(?:data|printData)\.aspx\?[^\"'<>\s]+",
    )
    for pattern in patterns:
        for match in re.findall(pattern, decoded, re.I):
            candidates.append(urllib.parse.urljoin(final_url, match.rstrip("),;")))
    unique: list[str] = []
    for url in candidates:
        if url not in unique:
            unique.append(url)
    return unique


def result_list_url(body: bytes, final_url: str) -> str | None:
    """Return the Judicial Yuan result-list iframe URL, when present."""
    document = html.fromstring(body)
    sources = document.xpath("//iframe[contains(@src,'qryresultlst.aspx')]/@src")
    return urllib.parse.urljoin(final_url, sources[0]) if sources else None


def choose_detail_url(urls: list[str], judgment_metadata: dict) -> str:
    """Choose the exact decision date because one case number can have many rulings."""
    expected = re.sub(r"\D", "", str(judgment_metadata.get("date") or ""))
    if len(expected) == 8:
        dated = [url for url in urls if expected in urllib.parse.unquote(url)]
        if len(dated) == 1:
            return dated[0]
        if len(dated) > 1:
            return dated[0]
    if len(urls) == 1:
        return urls[0]
    raise ValueError(
        f"Could not uniquely match decision date {judgment_metadata.get('date')!r}; "
        f"found {len(urls)} decisions"
    )


def extract_text(body: bytes) -> tuple[str, str]:
    document = html.fromstring(body)
    for node in document.xpath("//script|//style|//noscript"):
        node.drop_tree()
    title = " ".join(document.xpath("//title//text()")[:1]).strip()
    main = document.xpath("//*[@id='jud']|//*[@id='jud_content']|//div[contains(@class,'judgment')] | //body")
    target = main[0] if main else document
    lines = [re.sub(r"[\t\u3000 ]+", " ", line).strip() for line in target.text_content().splitlines()]
    text = "\n".join(line for line in lines if line)
    return title, text


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.2 * (attempt + 1))


def atomic_json(path: Path, record: dict) -> None:
    atomic_write(path, json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def build_manifest(year: int) -> None:
    records = []
    for path in sorted((OUTPUT_ROOT / str(year) / "metadata").glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    content = "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in records)
    atomic_write(OUTPUT_ROOT / f"linked_judgments_{year}.jsonl", content.encode("utf-8"))


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit cannot be negative")
    pending = []
    for year in args.years:
        for group in source_groups(year):
            key = hashlib.sha256(group["query_url"].encode("utf-8")).hexdigest()[:24]
            metadata_path = OUTPUT_ROOT / str(year) / "metadata" / f"{key}.json"
            if not metadata_path.exists():
                pending.append((year, key, metadata_path, group))
    if args.limit:
        pending = pending[:args.limit]
    print(json.dumps({"pending_selected": len(pending), "years": args.years}, ensure_ascii=False), flush=True)

    last_request = None
    completed_years: set[int] = set()
    for index, (year, key, metadata_path, group) in enumerate(pending, 1):
        if last_request is not None:
            time.sleep(max(0, args.delay_seconds - (time.monotonic() - last_request)))
        last_request = time.monotonic()
        try:
            query_body, query_final_url, _ = fetch(group["query_url"], args.timeout)
            list_url = result_list_url(query_body, query_final_url)
            if list_url:
                time.sleep(max(0, args.delay_seconds))
                list_body, list_final_url, list_content_type = fetch(list_url, args.timeout)
                if list_content_type not in {"text/html", "application/xhtml+xml"}:
                    raise ValueError(f"Unexpected result-list content type: {list_content_type}")
                urls = detail_urls(list_body, list_final_url)
            else:
                urls = detail_urls(query_body, query_final_url)
            if not urls:
                raise ValueError("No data.aspx or printData.aspx judgment URL found")
            detail_url = choose_detail_url(urls, group["judgment_metadata"])
            if detail_url == query_final_url:
                judgment_body = query_body
            else:
                time.sleep(max(0, args.delay_seconds))
                judgment_body, detail_url, content_type = fetch(detail_url, args.timeout)
                if content_type not in {"text/html", "application/xhtml+xml"}:
                    raise ValueError(f"Unexpected judgment content type: {content_type}")
            title, text = extract_text(judgment_body)
            html_path = OUTPUT_ROOT / str(year) / "html" / f"{key}.html"
            atomic_write(html_path, judgment_body)
            record = {
                **group,
                "detail_url": detail_url,
                "title": title,
                "text": text,
                "text_len": len(text),
                "html_path": html_path.relative_to(ROOT).as_posix(),
                "html_sha256": hashlib.sha256(judgment_body).hexdigest(),
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "status": "complete",
            }
            atomic_json(metadata_path, record)
            error_path = OUTPUT_ROOT / str(year) / "errors" / f"{key}.json"
            if error_path.exists():
                error_path.unlink()
            completed_years.add(year)
            print(f"downloaded {index}/{len(pending)} year={year} key={key} chars={len(text)}", flush=True)
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            error_path = OUTPUT_ROOT / str(year) / "errors" / f"{key}.json"
            atomic_json(error_path, {
                **group, "status": "error", "error": type(error).__name__, "message": str(error),
                "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            print(f"error {index}/{len(pending)} year={year} key={key}: {error}", flush=True)
    for year in set(args.years) | completed_years:
        build_manifest(year)
    print(json.dumps({"selected": len(pending), "output": str(OUTPUT_ROOT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
