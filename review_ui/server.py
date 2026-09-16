#!/usr/bin/env python3
"""Local-only human review UI for the de-identified indictment corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import mimetypes
import sqlite3
import threading
import webbrowser
from datetime import datetime, timezone, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
TAIPEI = timezone(timedelta(hours=8))
DECISIONS = {"pass", "fail", "follow_up"}
SEVERITIES = {"none", "minor", "major", "critical"}
ISSUES = {
    "missed_person", "missed_organization", "missed_contact", "missed_address",
    "missed_datetime", "inconsistent_alias", "over_redaction", "semantic_loss",
    "formatting", "other",
}


def now_iso() -> str:
    return datetime.now(TAIPEI).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_database(db_path: Path, sample_size: int, seed: str) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {
        row["internal_doc_id"]: row
        for row in read_jsonl(ROOT / "data/intermediate/normalized/indictments_114_normalized.jsonl")
    }
    normalized_by_source = {row["source_id"]: internal_id for internal_id, row in normalized.items()}
    entities = {
        row["case_internal_id"]: row
        for row in read_jsonl(ROOT / "data/intermediate/entity_registry/indictments_114_entities.jsonl")
    }
    audits = {
        row["case_internal_id"]: row
        for row in read_jsonl(ROOT / "data/intermediate/audit/indictments_114_audit.jsonl")
    }
    mapping = {
        row["internal_clean_doc_id"]: row
        for row in read_jsonl(ROOT / "data/raw/metadata/indictments_114_source_mapping.jsonl")
    }
    clean = read_jsonl(ROOT / "data/clean/indictments/indictments_114_clean.jsonl")
    ai_enriched_path = ROOT / "data/clean/indictments/indictments_114_ai_enriched_pilot.jsonl"
    ai_enriched = {
        row["doc_id"]: row for row in read_jsonl(ai_enriched_path)
    } if ai_enriched_path.exists() else {}
    direct_experiment_path = ROOT / "data/intermediate/google_ai/direct_llm_bank_experiment.json"
    direct_experiments = {}
    if direct_experiment_path.exists():
        direct_experiment = json.loads(direct_experiment_path.read_text(encoding="utf-8"))
        direct_experiments[direct_experiment["doc_id"]] = direct_experiment
    two_pass_path = ROOT / "data/intermediate/google_ai/two_pass_gemma4_experiment.jsonl"
    two_pass_experiments = {
        row["doc_id"]: row for row in read_jsonl(two_pass_path)
    } if two_pass_path.exists() else {}
    hybrid_path = ROOT / "data/intermediate/google_ai/hybrid_gemma4_experiment.jsonl"
    hybrid_experiments = {
        row["doc_id"]: row for row in read_jsonl(hybrid_path)
    } if hybrid_path.exists() else {}
    model_audit_path = ROOT / "data/intermediate/model_audit/indictments_114_gemma3_4b_audit.jsonl"
    model_audits = {
        row["doc_id"]: row for row in read_jsonl(model_audit_path)
    } if model_audit_path.exists() else {}

    if not clean:
        raise RuntimeError("Clean corpus is empty")

    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cases (
                doc_id TEXT PRIMARY KEY,
                case_internal_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                case_type TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                clean_text TEXT NOT NULL,
                entities_json TEXT NOT NULL,
                audit_json TEXT NOT NULL,
                model_audit_json TEXT NOT NULL DEFAULT '{"status":"not_run","issues":[]}',
                evidence_json TEXT NOT NULL DEFAULT '[]',
                crime_facts TEXT NOT NULL DEFAULT '',
                crime_facts_summary_json TEXT NOT NULL DEFAULT '[]',
                physical_evidence_json TEXT NOT NULL DEFAULT '[]',
                google_ai_review_json TEXT NOT NULL DEFAULT '{"status":"not_run"}',
                direct_llm_json TEXT NOT NULL DEFAULT '{"status":"not_run"}',
                two_pass_llm_json TEXT NOT NULL DEFAULT '{"status":"not_run"}',
                hybrid_llm_json TEXT NOT NULL DEFAULT '{"status":"not_run"}',
                in_sample INTEGER NOT NULL DEFAULT 0,
                sample_rank INTEGER
            );
            CREATE TABLE IF NOT EXISTS reviews (
                doc_id TEXT PRIMARY KEY REFERENCES cases(doc_id),
                decision TEXT NOT NULL,
                severity TEXT NOT NULL,
                issues_json TEXT NOT NULL,
                notes TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS review_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL REFERENCES cases(doc_id),
                decision TEXT NOT NULL,
                severity TEXT NOT NULL,
                issues_json TEXT NOT NULL,
                notes TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cases_sample ON cases(in_sample, sample_rank);
            CREATE INDEX IF NOT EXISTS idx_cases_type ON cases(case_type);
            CREATE INDEX IF NOT EXISTS idx_reviews_decision ON reviews(decision);
            """
        )
        case_columns = {row["name"] for row in conn.execute("PRAGMA table_info(cases)")}
        if "evidence_json" not in case_columns:
            conn.execute("ALTER TABLE cases ADD COLUMN evidence_json TEXT NOT NULL DEFAULT '[]'")
        if "model_audit_json" not in case_columns:
            conn.execute(
                "ALTER TABLE cases ADD COLUMN model_audit_json TEXT NOT NULL "
                "DEFAULT '{\"status\":\"not_run\",\"issues\":[]}'"
            )
        for column, declaration in (
            ("crime_facts", "TEXT NOT NULL DEFAULT ''"),
            ("crime_facts_summary_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("physical_evidence_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("google_ai_review_json", "TEXT NOT NULL DEFAULT '{\"status\":\"not_run\"}'"),
            ("direct_llm_json", "TEXT NOT NULL DEFAULT '{\"status\":\"not_run\"}'"),
            ("two_pass_llm_json", "TEXT NOT NULL DEFAULT '{\"status\":\"not_run\"}'"),
            ("hybrid_llm_json", "TEXT NOT NULL DEFAULT '{\"status\":\"not_run\"}'"),
        ):
            if column not in case_columns:
                conn.execute(f"ALTER TABLE cases ADD COLUMN {column} {declaration}")
        for item in clean:
            doc_id = item["doc_id"]
            map_row = mapping.get(doc_id)
            if not map_row:
                raise RuntimeError(f"Missing source mapping for {doc_id}")
            source_id = map_row["source_id"]
            internal_id = normalized_by_source.get(source_id)
            if not internal_id or internal_id not in entities or internal_id not in audits:
                raise RuntimeError(f"Incomplete intermediate data for {source_id}")
            ai_item = ai_enriched.get(doc_id, {})
            direct_item = direct_experiments.get(doc_id)
            two_pass_item = two_pass_experiments.get(doc_id)
            hybrid_item = hybrid_experiments.get(doc_id)
            # A completed hybrid run is the preferred human-review candidate.  It
            # combines LLM analysis with deterministic replacement, so expose it
            # through the UI's primary text/summary/evidence fields instead of
            # leaving the improved result only in the experiment panel.
            display_text = hybrid_item.get("text", "") if hybrid_item else ai_item.get("text", item["text"])
            display_crime_facts = (
                hybrid_item.get("crime_facts", "") if hybrid_item else ai_item.get("crime_facts", "")
            )
            display_crime_facts_summary = (
                hybrid_item.get("crime_facts_summary", [])
                if hybrid_item else ai_item.get("crime_facts_summary", [])
            )
            display_evidence = (
                hybrid_item.get("evidence", []) if hybrid_item else ai_item.get("physical_evidence", [])
            )
            conn.execute(
                """
                INSERT INTO cases (
                    doc_id, case_internal_id, source_id, case_type, normalized_text,
                    clean_text, entities_json, audit_json, model_audit_json, evidence_json,
                    crime_facts, crime_facts_summary_json, physical_evidence_json, google_ai_review_json,
                    direct_llm_json, two_pass_llm_json, hybrid_llm_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    case_internal_id=excluded.case_internal_id,
                    source_id=excluded.source_id,
                    case_type=excluded.case_type,
                    normalized_text=excluded.normalized_text,
                    clean_text=excluded.clean_text,
                    entities_json=excluded.entities_json,
                    audit_json=excluded.audit_json,
                    model_audit_json=excluded.model_audit_json,
                    evidence_json=excluded.evidence_json,
                    crime_facts=excluded.crime_facts,
                    crime_facts_summary_json=excluded.crime_facts_summary_json,
                    physical_evidence_json=excluded.physical_evidence_json,
                    google_ai_review_json=excluded.google_ai_review_json,
                    direct_llm_json=excluded.direct_llm_json,
                    two_pass_llm_json=excluded.two_pass_llm_json,
                    hybrid_llm_json=excluded.hybrid_llm_json
                """,
                (
                    doc_id, internal_id, source_id, item.get("case_type", ""),
                    normalized[internal_id]["normalized_text"], display_text,
                    json.dumps(entities[internal_id], ensure_ascii=False),
                    json.dumps(audits[internal_id], ensure_ascii=False),
                    json.dumps(model_audits.get(doc_id, {"status": "not_run", "issues": []}), ensure_ascii=False),
                    json.dumps(item.get("evidence", []), ensure_ascii=False),
                    display_crime_facts,
                    json.dumps(display_crime_facts_summary, ensure_ascii=False),
                    json.dumps(display_evidence, ensure_ascii=False),
                    json.dumps(ai_item.get("google_ai_review", {"status": "not_run"}), ensure_ascii=False),
                    json.dumps({
                        "status": "complete",
                        "model": direct_item["model"],
                        "usage_metadata": direct_item.get("usage_metadata", {}),
                        "restricted": True,
                        "result": direct_item["result"],
                    }, ensure_ascii=False) if direct_item else json.dumps({"status": "not_run"}),
                    json.dumps({
                        "status": "complete",
                        "model": two_pass_item["model"],
                        "restricted": True,
                        "final_stage": "repair" if two_pass_item.get("repair") else "reviewed",
                        "repair_rounds": len(two_pass_item.get("repair_history", [])) + (1 if two_pass_item.get("repair") else 0),
                        "result": two_pass_item.get("repair") or two_pass_item["reviewed"],
                    }, ensure_ascii=False) if two_pass_item else json.dumps({"status": "not_run"}),
                    json.dumps({
                        "status": "complete",
                        "model": hybrid_item["model"],
                        "restricted": True,
                        "text": hybrid_item["text"],
                        "crime_facts": hybrid_item["crime_facts"],
                        "crime_facts_summary": hybrid_item["crime_facts_summary"],
                        "evidence": hybrid_item["evidence"],
                        "checks": hybrid_item["checks"],
                    }, ensure_ascii=False) if hybrid_item else json.dumps({"status": "not_run"}),
                ),
            )
        conn.execute(
            "DELETE FROM cases WHERE doc_id NOT IN (%s)"
            % ",".join("?" for _ in clean),
            [item["doc_id"] for item in clean],
        )
        configured = conn.execute("SELECT value FROM settings WHERE key='sample_seed'").fetchone()
        if configured is None:
            set_sample(conn, sample_size, seed)


def set_sample(conn: sqlite3.Connection, size: int, seed: str) -> int:
    rows = conn.execute(
        "SELECT doc_id,case_type,LENGTH(normalized_text) text_length FROM cases"
    ).fetchall()
    size = max(1, min(size, len(rows)))
    type_counts = dict(
        conn.execute("SELECT case_type,COUNT(*) FROM cases GROUP BY case_type").fetchall()
    )
    lengths = sorted(row["text_length"] for row in rows)
    long_threshold = lengths[int(0.95 * (len(lengths) - 1))]

    def priority(row: sqlite3.Row) -> int:
        # Long-tail charges are most likely to contain unusual prose and
        # identifying fact patterns; very long documents stress the rules.
        if type_counts[row["case_type"]] <= 2:
            return 0
        if row["text_length"] >= long_threshold:
            return 1
        return 2

    ranked = sorted(
        (
            priority(row),
            hashlib.sha256(f"{seed}:{row['doc_id']}".encode()).hexdigest(),
            row["doc_id"],
        )
        for row in rows
    )
    conn.execute("UPDATE cases SET in_sample=0, sample_rank=NULL")
    conn.executemany(
        "UPDATE cases SET in_sample=1, sample_rank=? WHERE doc_id=?",
        [(rank, doc_id) for rank, (_, _, doc_id) in enumerate(ranked[:size], start=1)],
    )
    conn.execute(
        "INSERT INTO settings(key,value) VALUES('sample_seed',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (seed,),
    )
    conn.execute(
        "INSERT INTO settings(key,value) VALUES('sample_size',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(size),),
    )
    conn.execute(
        "INSERT INTO settings(key,value) VALUES('sample_strategy','risk_priority_v1') "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
    )
    return size


def review_join_sql(scope: str) -> tuple[str, list]:
    where = "WHERE c.in_sample=1" if scope == "sample" else "WHERE 1=1"
    return where, []


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "JudgmentReview/1.0"

    @property
    def db_path(self) -> Path:
        return self.server.db_path  # type: ignore[attr-defined]

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
            return self.get_stats(parse_qs(parsed.query))
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
        if parsed.path.startswith("/api/reviews/"):
            return self.save_review(unquote(parsed.path.removeprefix("/api/reviews/")), body)
        if parsed.path == "/api/sample":
            return self.reset_sample(body)
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

    def get_stats(self, query: dict[str, list[str]]) -> None:
        scope = query.get("scope", ["sample"])[0]
        where, params = review_join_sql(scope)
        with connect(self.db_path) as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) total,
                       SUM(CASE WHEN r.doc_id IS NULL THEN 1 ELSE 0 END) pending,
                       SUM(CASE WHEN r.decision='pass' THEN 1 ELSE 0 END) passed,
                       SUM(CASE WHEN r.decision='fail' THEN 1 ELSE 0 END) failed,
                       SUM(CASE WHEN r.decision='follow_up' THEN 1 ELSE 0 END) follow_up
                FROM cases c LEFT JOIN reviews r ON r.doc_id=c.doc_id {where}
                """, params
            ).fetchone()
            settings = dict(conn.execute("SELECT key,value FROM settings").fetchall())
        self.send_json({**dict(row), "scope": scope, "sample_seed": settings.get("sample_seed", ""),
                        "sample_size": int(settings.get("sample_size", 0))})

    def get_cases(self, query: dict[str, list[str]]) -> None:
        scope = query.get("scope", ["sample"])[0]
        status = query.get("status", ["all"])[0]
        case_type = query.get("case_type", [""])[0]
        search = query.get("q", [""])[0].strip()
        try:
            limit = min(max(int(query.get("limit", ["200"])[0]), 1), 1000)
        except ValueError:
            limit = 200
        clauses = ["c.in_sample=1"] if scope == "sample" else ["1=1"]
        params: list[object] = []
        if status == "pending":
            clauses.append("r.doc_id IS NULL")
        elif status in DECISIONS:
            clauses.append("r.decision=?")
            params.append(status)
        if case_type:
            clauses.append("c.case_type=?")
            params.append(case_type)
        if search:
            clauses.append("(c.source_id LIKE ? OR c.case_type LIKE ? OR c.doc_id LIKE ?)")
            params.extend([f"%{search}%"] * 3)
        params.append(limit)
        with connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT c.doc_id,c.source_id,c.case_type,c.sample_rank,r.decision,r.severity,r.updated_at
                FROM cases c LEFT JOIN reviews r ON r.doc_id=c.doc_id
                WHERE {' AND '.join(clauses)}
                ORDER BY CASE WHEN c.sample_rank IS NULL THEN 1 ELSE 0 END,
                         c.sample_rank, c.source_id LIMIT ?
                """, params
            ).fetchall()
            types = conn.execute(
                "SELECT case_type,COUNT(*) count FROM cases GROUP BY case_type ORDER BY count DESC,case_type"
            ).fetchall()
        self.send_json({"cases": [dict(row) for row in rows], "case_types": [dict(row) for row in types]})

    def get_case(self, doc_id: str) -> None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT c.*,r.decision,r.severity,r.issues_json,r.notes,r.reviewer,
                       r.created_at,r.updated_at
                FROM cases c LEFT JOIN reviews r ON r.doc_id=c.doc_id WHERE c.doc_id=?
                """, (doc_id,)
            ).fetchone()
            if not row:
                return self.send_error_json("Case not found", HTTPStatus.NOT_FOUND)
            history = conn.execute(
                "SELECT decision,severity,issues_json,notes,reviewer,recorded_at FROM review_history WHERE doc_id=? ORDER BY id DESC LIMIT 20",
                (doc_id,),
            ).fetchall()
        data = dict(row)
        data["entities"] = json.loads(data.pop("entities_json"))
        data["audit"] = json.loads(data.pop("audit_json"))
        data["model_audit"] = json.loads(data.pop("model_audit_json"))
        data["evidence"] = json.loads(data.pop("evidence_json"))
        data["crime_facts_summary"] = json.loads(data.pop("crime_facts_summary_json"))
        data["physical_evidence"] = json.loads(data.pop("physical_evidence_json"))
        data["google_ai_review"] = json.loads(data.pop("google_ai_review_json"))
        data["direct_llm"] = json.loads(data.pop("direct_llm_json"))
        data["two_pass_llm"] = json.loads(data.pop("two_pass_llm_json"))
        data["hybrid_llm"] = json.loads(data.pop("hybrid_llm_json"))
        data["issues"] = json.loads(data.pop("issues_json")) if data.get("issues_json") else []
        data["history"] = [{**dict(item), "issues": json.loads(item["issues_json"])} for item in history]
        for item in data["history"]:
            item.pop("issues_json", None)
        self.send_json(data)

    def save_review(self, doc_id: str, body: dict) -> None:
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
        with connect(self.db_path) as conn:
            exists = conn.execute("SELECT 1 FROM cases WHERE doc_id=?", (doc_id,)).fetchone()
            if not exists:
                return self.send_error_json("Case not found", HTTPStatus.NOT_FOUND)
            prior = conn.execute("SELECT created_at FROM reviews WHERE doc_id=?", (doc_id,)).fetchone()
            created = prior["created_at"] if prior else stamp
            conn.execute(
                """
                INSERT INTO reviews(doc_id,decision,severity,issues_json,notes,reviewer,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(doc_id) DO UPDATE SET
                    decision=excluded.decision,severity=excluded.severity,
                    issues_json=excluded.issues_json,notes=excluded.notes,
                    reviewer=excluded.reviewer,updated_at=excluded.updated_at
                """, (doc_id, decision, severity, issues_json, notes, reviewer, created, stamp)
            )
            conn.execute(
                "INSERT INTO review_history(doc_id,decision,severity,issues_json,notes,reviewer,recorded_at) VALUES(?,?,?,?,?,?,?)",
                (doc_id, decision, severity, issues_json, notes, reviewer, stamp),
            )
        self.send_json({"ok": True, "updated_at": stamp})

    def reset_sample(self, body: dict) -> None:
        try:
            size = int(body.get("size", 100))
        except (TypeError, ValueError):
            return self.send_error_json("Invalid sample size")
        seed = str(body.get("seed", "20260906")).strip()[:100]
        if not seed:
            return self.send_error_json("Seed is required")
        with connect(self.db_path) as conn:
            actual = set_sample(conn, size, seed)
        self.send_json({"ok": True, "sample_size": actual, "sample_seed": seed})

    def export_reviews(self, query: dict[str, list[str]]) -> None:
        scope = query.get("scope", ["sample"])[0]
        fmt = query.get("format", ["csv"])[0]
        where, params = review_join_sql(scope)
        with connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT c.doc_id,c.source_id,c.case_type,c.in_sample,c.sample_rank,
                       COALESCE(r.decision,'pending') decision,COALESCE(r.severity,'none') severity,
                       COALESCE(r.issues_json,'[]') issues_json,COALESCE(r.notes,'') notes,
                       COALESCE(r.reviewer,'') reviewer,r.created_at,r.updated_at
                FROM cases c LEFT JOIN reviews r ON r.doc_id=c.doc_id {where}
                ORDER BY CASE WHEN c.sample_rank IS NULL THEN 1 ELSE 0 END,c.sample_rank,c.source_id
                """, params
            ).fetchall()
        records = []
        for row in rows:
            item = dict(row)
            item["issues"] = json.loads(item.pop("issues_json"))
            records.append(item)
        if fmt == "jsonl":
            payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode("utf-8")
            content_type, suffix = "application/x-ndjson", "jsonl"
        else:
            output = io.StringIO(newline="")
            fields = ["doc_id", "source_id", "case_type", "in_sample", "sample_rank", "decision",
                      "severity", "issues", "notes", "reviewer", "created_at", "updated_at"]
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for item in records:
                item["issues"] = "|".join(item["issues"])
                writer.writerow(item)
            payload = ("\ufeff" + output.getvalue()).encode("utf-8")
            content_type, suffix = "text/csv", "csv"
        filename = f"indictment_reviews_{scope}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{suffix}"
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Disposition", f"attachment; filename={quote(filename)}")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local indictment review UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", type=Path, default=ROOT / "data/review/indictment_reviews.sqlite3")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", default="20260906")
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_database(args.db.resolve(), args.sample_size, args.seed)
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    server.db_path = args.db.resolve()  # type: ignore[attr-defined]
    url = f"http://{args.host}:{args.port}"
    print(f"Review UI: {url}")
    print(f"Review database: {server.db_path}")
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
