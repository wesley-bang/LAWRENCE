#!/usr/bin/env python3
"""Local review UI for paired indictment and judgment de-identification."""

from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import sqlite3
import sys
import threading
import webbrowser
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import deidentify_linked_judgments as judgment  # noqa: E402


STATIC_DIR = Path(__file__).resolve().parent / "pair_static"
INDICTMENTS = ROOT / "data/intermediate/google_ai/hybrid_gemma4_experiment.jsonl"
JUDGMENTS = ROOT / "data/intermediate/google_ai/hybrid_judgments_gemma4_experiment.jsonl"
JUDGMENT_MANIFEST = ROOT / "data/raw/linked_judgments/linked_judgments_114.jsonl"
AUDITS = ROOT / "data/intermediate/google_ai/pair_audit_gemini38.jsonl"
AUDIT_FAILURES = ROOT / "data/intermediate/google_ai/pair_audit_gemini38_failures.jsonl"
INDICTMENT_DB = ROOT / "data/review/indictment_reviews.sqlite3"
TAIPEI = timezone(timedelta(hours=8))
DECISIONS = {"pass", "fail", "follow_up"}
SEVERITIES = {"none", "minor", "major", "critical"}
ISSUES = {
    "indictment_privacy", "judgment_privacy", "inconsistent_alias",
    "facts_mismatch", "evidence_missing", "evidence_bad_merge",
    "over_redaction", "semantic_loss", "formatting", "other",
}


def now_iso() -> str:
    return datetime.now(TAIPEI).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def init_database(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS pair_cases (
                pair_id TEXT PRIMARY KEY,
                judgment_internal_id TEXT NOT NULL UNIQUE,
                indictment_doc_id TEXT NOT NULL,
                judgment_doc_id TEXT NOT NULL,
                case_type TEXT NOT NULL,
                court_name TEXT NOT NULL,
                court_case_no TEXT NOT NULL,
                document_type TEXT NOT NULL,
                indictment_raw TEXT NOT NULL,
                indictment_clean TEXT NOT NULL,
                judgment_raw TEXT NOT NULL,
                judgment_clean TEXT NOT NULL,
                pair_entities_json TEXT NOT NULL,
                crime_facts_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                indictment_facts TEXT NOT NULL,
                judgment_sections_json TEXT NOT NULL,
                evidence_count INTEGER NOT NULL,
                fact_count INTEGER NOT NULL,
                source_updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pair_reviews (
                pair_id TEXT PRIMARY KEY REFERENCES pair_cases(pair_id),
                decision TEXT NOT NULL,
                severity TEXT NOT NULL,
                issues_json TEXT NOT NULL,
                notes TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pair_review_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair_id TEXT NOT NULL REFERENCES pair_cases(pair_id),
                decision TEXT NOT NULL,
                severity TEXT NOT NULL,
                issues_json TEXT NOT NULL,
                notes TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pair_audits (
                pair_id TEXT PRIMARY KEY REFERENCES pair_cases(pair_id),
                status TEXT NOT NULL,
                highest_severity TEXT NOT NULL,
                model TEXT NOT NULL,
                audit_pass INTEGER NOT NULL,
                requires_manual_review INTEGER NOT NULL,
                result_json TEXT,
                checks_json TEXT,
                error TEXT NOT NULL,
                error_detail TEXT NOT NULL,
                checkpointed_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pair_cases_type ON pair_cases(case_type);
            CREATE INDEX IF NOT EXISTS idx_pair_reviews_decision ON pair_reviews(decision);
            CREATE INDEX IF NOT EXISTS idx_pair_audits_status ON pair_audits(status);
            """
        )


def load_indictment_sources() -> dict[str, dict]:
    if not INDICTMENT_DB.exists():
        return {}
    with sqlite3.connect(INDICTMENT_DB) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT doc_id,normalized_text FROM cases"
        ).fetchall()
    return {str(row["doc_id"]): dict(row) for row in rows}


def review_pair_entities(judgment_row: dict) -> dict:
    pair_registry = judgment_row.get("pair_alias_registry", {})
    return {
        "persons": [
            {
                "pair_person_id": item.get("pair_person_id"),
                "role": item.get("role"),
                "alias": item.get("alias"),
            }
            for item in pair_registry.get("persons", [])
        ],
        "indictment_mentions": sorted({
            str(mention)
            for item in pair_registry.get("persons", [])
            for mention in item.get("indictment_mentions", []) if str(mention)
        }),
        "judgment_mentions": sorted({
            str(mention)
            for item in judgment_row.get("analysis_plan", {}).get("persons", [])
            for mention in item.get("mentions", []) if str(mention)
        }),
    }


def sync_pair_audits(connection: sqlite3.Connection) -> dict[str, int]:
    """Refresh the read-only audit index without touching human reviews."""
    available_pairs = {
        str(row[0]) for row in connection.execute("SELECT pair_id FROM pair_cases")
    }
    successful = {
        str(row.get("pair_id")): row for row in read_jsonl(AUDITS)
        if str(row.get("pair_id") or "") in available_pairs
    }
    failures = {
        str(row.get("pair_id")): row for row in read_jsonl(AUDIT_FAILURES)
        if str(row.get("pair_id") or "") in available_pairs
        and str(row.get("pair_id") or "") not in successful
    }
    connection.execute("DELETE FROM pair_audits")
    for pair_id, row in successful.items():
        result = row.get("audit_result") or {}
        overall = result.get("overall") or {}
        connection.execute(
            """
            INSERT INTO pair_audits(
                pair_id,status,highest_severity,model,audit_pass,
                requires_manual_review,result_json,checks_json,error,error_detail,
                checkpointed_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                pair_id, str(row.get("effective_decision") or "review"),
                str(overall.get("highest_severity") or "none"),
                str(row.get("model") or ""), int(bool(row.get("audit_pass"))),
                int(bool(row.get("requires_manual_review"))),
                json.dumps(result, ensure_ascii=False),
                json.dumps(row.get("deterministic_checks") or {}, ensure_ascii=False),
                "", "", str(row.get("checkpointed_at") or ""),
            ),
        )
    for pair_id, row in failures.items():
        detail = row.get("detail")
        if not detail and row.get("response"):
            detail = json.dumps(row["response"], ensure_ascii=False)
        connection.execute(
            """
            INSERT INTO pair_audits(
                pair_id,status,highest_severity,model,audit_pass,
                requires_manual_review,result_json,checks_json,error,error_detail,
                checkpointed_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                pair_id, "api_failure", "none", str(row.get("model") or ""), 0, 1,
                None, None, str(row.get("error") or "unknown_error"),
                str(detail or "")[:4000], str(row.get("checkpointed_at") or ""),
            ),
        )
    return {"audit_synced": len(successful), "audit_failures_synced": len(failures)}


def sync_pair_cases(db_path: Path) -> dict[str, int]:
    indictment_rows = {row["doc_id"]: row for row in read_jsonl(INDICTMENTS)}
    indictment_sources = load_indictment_sources()
    manifest_rows = {}
    for row in read_jsonl(JUDGMENT_MANIFEST):
        try:
            manifest_rows[judgment.adapt_linked_judgment(row)["internal_doc_id"]] = row
        except ValueError:
            continue
    judgment_rows = read_jsonl(JUDGMENTS)
    synced = 0
    skipped = 0
    with connect(db_path) as connection:
        for judgment_row in judgment_rows:
            indictment_id = str(judgment_row.get("linked_indictment_doc_id") or "")
            internal_id = str(judgment_row.get("judgment_internal_id") or "")
            indictment_row = indictment_rows.get(indictment_id)
            indictment_source = indictment_sources.get(indictment_id)
            manifest = manifest_rows.get(internal_id)
            if not indictment_row or not indictment_source or not manifest:
                skipped += 1
                continue
            metadata = manifest.get("judgment_metadata") or {}
            facts = judgment_row.get("paired_crime_facts_summary", [])
            evidence = judgment_row.get("paired_evidence", [])
            pair_id = str(judgment_row.get("pair_id") or "")
            if not pair_id:
                skipped += 1
                continue
            connection.execute(
                """
                INSERT INTO pair_cases (
                    pair_id,judgment_internal_id,indictment_doc_id,judgment_doc_id,
                    case_type,court_name,court_case_no,document_type,
                    indictment_raw,indictment_clean,judgment_raw,judgment_clean,
                    pair_entities_json,crime_facts_json,evidence_json,indictment_facts,
                    judgment_sections_json,evidence_count,fact_count,source_updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(pair_id) DO UPDATE SET
                    judgment_internal_id=excluded.judgment_internal_id,
                    indictment_doc_id=excluded.indictment_doc_id,
                    judgment_doc_id=excluded.judgment_doc_id,
                    case_type=excluded.case_type,court_name=excluded.court_name,
                    court_case_no=excluded.court_case_no,document_type=excluded.document_type,
                    indictment_raw=excluded.indictment_raw,
                    indictment_clean=excluded.indictment_clean,
                    judgment_raw=excluded.judgment_raw,judgment_clean=excluded.judgment_clean,
                    pair_entities_json=excluded.pair_entities_json,
                    crime_facts_json=excluded.crime_facts_json,
                    evidence_json=excluded.evidence_json,
                    indictment_facts=excluded.indictment_facts,
                    judgment_sections_json=excluded.judgment_sections_json,
                    evidence_count=excluded.evidence_count,fact_count=excluded.fact_count,
                    source_updated_at=excluded.source_updated_at
                """,
                (
                    pair_id, internal_id, indictment_id, judgment_row.get("doc_id", ""),
                    indictment_row.get("case_type", ""), metadata.get("court_name", ""),
                    metadata.get("case_no", ""), judgment_row.get("document_type", ""),
                    indictment_source.get("normalized_text", ""), indictment_row.get("text", ""),
                    judgment.normalize_judgment_text(manifest.get("text", "")),
                    judgment_row.get("text", ""),
                    json.dumps(review_pair_entities(judgment_row), ensure_ascii=False),
                    json.dumps(facts, ensure_ascii=False), json.dumps(evidence, ensure_ascii=False),
                    indictment_row.get("crime_facts", ""),
                    json.dumps(judgment_row.get("sections", {}), ensure_ascii=False),
                    len(evidence), len(facts), judgment_row.get("checkpointed_at", ""),
                ),
            )
            synced += 1
        audit_counts = sync_pair_audits(connection)
    return {
        "available": len(judgment_rows), "synced": synced, "skipped": skipped,
        **audit_counts,
    }


class PairReviewHandler(BaseHTTPRequestHandler):
    server_version = "PairReview/1.0"

    @property
    def db_path(self) -> Path:
        return self.server.db_path  # type: ignore[attr-defined]

    @property
    def sync_lock(self) -> threading.Lock:
        return self.server.sync_lock  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_json(self, data: object, status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_error_json(self, message: str, status: int = 400) -> None:
        self.send_json({"error": message}, status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/stats":
            return self.get_stats()
        if parsed.path == "/api/cases":
            return self.get_cases(parse_qs(parsed.query))
        if parsed.path.startswith("/api/cases/"):
            return self.get_case(unquote(parsed.path.removeprefix("/api/cases/")))
        if parsed.path == "/api/export":
            return self.export_reviews(parse_qs(parsed.query))
        self.serve_static(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.send_error_json("Invalid Content-Length")
        if length > 65536:
            return self.send_error_json("Request body is too large", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self.send_error_json("Invalid JSON")
        if parsed.path == "/api/sync":
            with self.sync_lock:
                result = sync_pair_cases(self.db_path)
            return self.send_json({"ok": True, **result})
        if parsed.path.startswith("/api/reviews/"):
            return self.save_review(unquote(parsed.path.removeprefix("/api/reviews/")), body)
        self.send_error_json("Not found", HTTPStatus.NOT_FOUND)

    def serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        target = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in target.parents and target != STATIC_DIR.resolve():
            return self.send_error_json("Not found", HTTPStatus.NOT_FOUND)
        if not target.is_file():
            return self.send_error_json("Not found", HTTPStatus.NOT_FOUND)
        payload = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def get_stats(self) -> None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) total,
                       SUM(CASE WHEN r.pair_id IS NULL THEN 1 ELSE 0 END) pending,
                       SUM(CASE WHEN r.decision='pass' THEN 1 ELSE 0 END) passed,
                       SUM(CASE WHEN r.decision='fail' THEN 1 ELSE 0 END) failed,
                       SUM(CASE WHEN r.decision='follow_up' THEN 1 ELSE 0 END) follow_up,
                       COALESCE(SUM(c.evidence_count),0) evidence_count,
                       SUM(CASE WHEN a.status='pass' THEN 1 ELSE 0 END) audit_pass,
                       SUM(CASE WHEN a.status='review' THEN 1 ELSE 0 END) audit_review,
                       SUM(CASE WHEN a.status='fail' THEN 1 ELSE 0 END) audit_fail,
                       SUM(CASE WHEN a.status='api_failure' THEN 1 ELSE 0 END) audit_api_failure,
                       SUM(CASE WHEN a.pair_id IS NULL THEN 1 ELSE 0 END) audit_unaudited
                FROM pair_cases c
                LEFT JOIN pair_reviews r USING(pair_id)
                LEFT JOIN pair_audits a USING(pair_id)
                """
            ).fetchone()
        data = {key: (value or 0) for key, value in dict(row).items()}
        self.send_json(data)

    def get_cases(self, query: dict[str, list[str]]) -> None:
        status = query.get("status", ["pending"])[0]
        audit_status = query.get("audit", ["all"])[0]
        case_type = query.get("case_type", [""])[0]
        search = query.get("q", [""])[0].strip()
        try:
            limit = min(max(int(query.get("limit", ["1000"])[0]), 1), 2000)
        except ValueError:
            limit = 1000
        clauses = ["1=1"]
        params: list[object] = []
        if status == "pending":
            clauses.append("r.pair_id IS NULL")
        elif status in DECISIONS:
            clauses.append("r.decision=?")
            params.append(status)
        if audit_status == "unaudited":
            clauses.append("a.pair_id IS NULL")
        elif audit_status in {"pass", "review", "fail", "api_failure"}:
            clauses.append("a.status=?")
            params.append(audit_status)
        if case_type:
            clauses.append("c.case_type=?")
            params.append(case_type)
        if search:
            clauses.append(
                "(c.case_type LIKE ? OR c.court_name LIKE ? OR c.court_case_no LIKE ? "
                "OR c.pair_id LIKE ? OR c.indictment_doc_id LIKE ?)"
            )
            params.extend([f"%{search}%"] * 5)
        params.append(limit)
        with connect(self.db_path) as connection:
            rows = connection.execute(
                f"""
                SELECT c.pair_id,c.case_type,c.court_name,c.court_case_no,c.document_type,
                       c.evidence_count,c.fact_count,c.source_updated_at,
                       r.decision,r.severity,r.updated_at,
                       a.status audit_status,a.highest_severity audit_severity,
                       a.checkpointed_at audit_updated_at
                FROM pair_cases c
                LEFT JOIN pair_reviews r USING(pair_id)
                LEFT JOIN pair_audits a USING(pair_id)
                WHERE {' AND '.join(clauses)}
                ORDER BY c.source_updated_at,c.pair_id LIMIT ?
                """, params
            ).fetchall()
            types = connection.execute(
                "SELECT case_type,COUNT(*) count FROM pair_cases GROUP BY case_type "
                "ORDER BY count DESC,case_type"
            ).fetchall()
        self.send_json({"cases": [dict(row) for row in rows], "case_types": [dict(row) for row in types]})

    def get_case(self, pair_id: str) -> None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT c.*,r.decision,r.severity,r.issues_json,r.notes,r.reviewer,
                       r.created_at,r.updated_at,
                       a.status audit_status,a.highest_severity audit_severity,
                       a.model audit_model,a.audit_pass,a.requires_manual_review,
                       a.result_json audit_result_json,a.checks_json audit_checks_json,
                       a.error audit_error,a.error_detail audit_error_detail,
                       a.checkpointed_at audit_updated_at
                FROM pair_cases c
                LEFT JOIN pair_reviews r USING(pair_id)
                LEFT JOIN pair_audits a USING(pair_id)
                WHERE c.pair_id=?
                """, (pair_id,)
            ).fetchone()
            if not row:
                return self.send_error_json("Pair not found", HTTPStatus.NOT_FOUND)
            history = connection.execute(
                "SELECT decision,severity,issues_json,notes,reviewer,recorded_at "
                "FROM pair_review_history WHERE pair_id=? ORDER BY id DESC LIMIT 20",
                (pair_id,),
            ).fetchall()
        data = dict(row)
        for source, target in (
            ("pair_entities_json", "pair_entities"),
            ("crime_facts_json", "crime_facts"),
            ("evidence_json", "evidence"),
            ("judgment_sections_json", "judgment_sections"),
        ):
            data[target] = json.loads(data.pop(source))
        issues_json = data.pop("issues_json", None)
        data["issues"] = json.loads(issues_json) if issues_json else []
        result_json = data.pop("audit_result_json", None)
        checks_json = data.pop("audit_checks_json", None)
        data["audit_result"] = json.loads(result_json) if result_json else None
        data["audit_checks"] = json.loads(checks_json) if checks_json else None
        data["history"] = []
        for item in history:
            record = dict(item)
            record["issues"] = json.loads(record.pop("issues_json"))
            data["history"].append(record)
        self.send_json(data)

    def save_review(self, pair_id: str, body: dict) -> None:
        decision = str(body.get("decision", ""))
        severity = str(body.get("severity", "none"))
        issues = body.get("issues", [])
        notes = str(body.get("notes", "")).strip()
        reviewer = str(body.get("reviewer", "")).strip()[:100]
        if decision not in DECISIONS:
            return self.send_error_json("Invalid decision")
        if severity not in SEVERITIES:
            return self.send_error_json("Invalid severity")
        if not isinstance(issues, list) or any(issue not in ISSUES for issue in issues):
            return self.send_error_json("Invalid issue type")
        if len(notes) > 10000:
            return self.send_error_json("Notes are too long")
        stamp = now_iso()
        issues_json = json.dumps(sorted(set(issues)), ensure_ascii=False)
        with connect(self.db_path) as connection:
            if not connection.execute(
                "SELECT 1 FROM pair_cases WHERE pair_id=?", (pair_id,)
            ).fetchone():
                return self.send_error_json("Pair not found", HTTPStatus.NOT_FOUND)
            prior = connection.execute(
                "SELECT created_at FROM pair_reviews WHERE pair_id=?", (pair_id,)
            ).fetchone()
            created_at = prior["created_at"] if prior else stamp
            connection.execute(
                """
                INSERT INTO pair_reviews(
                    pair_id,decision,severity,issues_json,notes,reviewer,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(pair_id) DO UPDATE SET
                    decision=excluded.decision,severity=excluded.severity,
                    issues_json=excluded.issues_json,notes=excluded.notes,
                    reviewer=excluded.reviewer,updated_at=excluded.updated_at
                """, (
                    pair_id, decision, severity, issues_json, notes, reviewer, created_at, stamp,
                )
            )
            connection.execute(
                "INSERT INTO pair_review_history(pair_id,decision,severity,issues_json,notes,reviewer,recorded_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (pair_id, decision, severity, issues_json, notes, reviewer, stamp),
            )
        self.send_json({"ok": True, "updated_at": stamp})

    def export_reviews(self, query: dict[str, list[str]]) -> None:
        fmt = query.get("format", ["csv"])[0]
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT c.pair_id,c.indictment_doc_id,c.judgment_doc_id,c.case_type,
                       c.court_name,c.court_case_no,
                       COALESCE(r.decision,'pending') decision,
                       COALESCE(r.severity,'none') severity,
                       COALESCE(r.issues_json,'[]') issues_json,
                       COALESCE(r.notes,'') notes,COALESCE(r.reviewer,'') reviewer,
                       r.created_at,r.updated_at
                FROM pair_cases c LEFT JOIN pair_reviews r USING(pair_id)
                ORDER BY c.source_updated_at,c.pair_id
                """
            ).fetchall()
        records = []
        for row in rows:
            item = dict(row)
            item["issues"] = json.loads(item.pop("issues_json"))
            records.append(item)
        if fmt == "jsonl":
            payload = "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in records
            ).encode("utf-8")
            content_type, suffix = "application/x-ndjson", "jsonl"
        else:
            output = io.StringIO(newline="")
            fields = [
                "pair_id", "indictment_doc_id", "judgment_doc_id", "case_type",
                "court_name", "court_case_no", "decision", "severity", "issues",
                "notes", "reviewer", "created_at", "updated_at",
            ]
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for item in records:
                item["issues"] = "|".join(item["issues"])
                writer.writerow(item)
            payload = ("\ufeff" + output.getvalue()).encode("utf-8")
            content_type, suffix = "text/csv", "csv"
        filename = f"pair_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{suffix}"
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Disposition", f"attachment; filename={quote(filename)}")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local paired-case review UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--db", type=Path, default=ROOT / "data/review/pair_reviews.sqlite3"
    )
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = args.db.resolve()
    init_database(db_path)
    sync_result = sync_pair_cases(db_path)
    server = ThreadingHTTPServer((args.host, args.port), PairReviewHandler)
    server.db_path = db_path  # type: ignore[attr-defined]
    server.sync_lock = threading.Lock()  # type: ignore[attr-defined]
    url = f"http://{args.host}:{args.port}"
    print(f"Pair review UI: {url}")
    print(f"Review database: {db_path}")
    print(f"Initial sync: {sync_result}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
