from __future__ import annotations

import sys
import unittest
import urllib.error
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_case_pairs_with_google_ai as audit  # noqa: E402


def valid_result() -> dict:
    return {
        "crime_facts": {"status": "pass", "accuracy_issues": [], "privacy_leaks": []},
        "evidence": {
            "status": "pass", "unsupported_items": [], "missing_items": [],
            "privacy_leaks": [],
        },
        "aliases": {
            "status": "pass", "confusing_aliases": [],
            "cross_document_inconsistencies": [], "privacy_leaks": [],
            "safe_nicknames_observed": ["小黑"],
        },
        "cross_component_consistency": {"status": "pass", "issues": []},
        "overall": {"decision": "pass", "highest_severity": "none", "reasons": []},
    }


def sample_case() -> dict:
    return {
        "pair_id": "pair-1",
        "judgment_internal_id": "judgment-1",
        "linked_indictment_doc_id": "indictment-1",
        "raw_indictment": "被告王小明綽號小黑持手機犯案。",
        "raw_judgment": "被告王小明即小黑持手機犯案，有監視器照片可證。",
        "indictment": {
            "text": "被告甲綽號小黑持手機犯案。",
            "crime_facts_summary": ["被告甲持手機犯案。"],
            "evidence": [],
            "analysis_plan": {"persons": [{"mentions": ["王小明"]}]},
        },
        "judgment": {
            "text": "被告甲即小黑持手機犯案，有監視器照片可證。",
            "crime_facts_summary": ["被告甲持手機犯案。"],
            "evidence": [{"name": "監視器照片"}],
            "paired_crime_facts_summary": [{"text": "被告甲持手機犯案。"}],
            "paired_evidence": [{"name": "監視器照片", "provenance": ["judgment"]}],
            "analysis_plan": {"persons": [{"mentions": ["王小明"]}]},
            "pair_alias_registry": {"persons": [{
                "pair_person_id": "P01", "alias": "被告甲",
                "indictment_mentions": ["王小明"],
            }]},
        },
    }


class PairAuditTests(unittest.TestCase):
    def test_retry_after_prefers_google_hint_without_exponential_growth(self) -> None:
        error = urllib.error.HTTPError(
            "https://example.test", 429, "quota", {"Retry-After": "31"}, None
        )
        self.assertEqual(audit.retry_after_seconds(error, "", 300), 34.0)
        without_header = urllib.error.HTTPError(
            "https://example.test", 429, "quota", {}, None
        )
        self.assertAlmostEqual(
            audit.retry_after_seconds(without_header, "Please retry in 28.5s.", 300),
            31.5,
        )

    def test_prompt_preserves_safe_nickname_policy_and_evidence_scope(self) -> None:
        prompt = audit.audit_prompt(sample_case())
        self.assertIn("純暱稱、綽號", prompt)
        self.assertIn("不得假稱已看到未提供的卷宗", prompt)
        self.assertIn("小黑", prompt)

    def test_deterministic_check_does_not_treat_nickname_as_name_leak(self) -> None:
        checks = audit.deterministic_checks(sample_case())
        self.assertTrue(checks["pass"])
        self.assertEqual(checks["exact_person_mention_leaks"], [])

    def test_deterministic_check_catches_original_person_mention(self) -> None:
        case = sample_case()
        case["judgment"]["text"] = "被告王小明即小黑犯案。"
        checks = audit.deterministic_checks(case)
        self.assertFalse(checks["pass"])
        self.assertEqual(checks["exact_person_mention_leaks"][0]["span"], "王小明")

    def test_existing_court_codes_and_masked_plates_are_not_leaks(self) -> None:
        case = sample_case()
        case["indictment"]["analysis_plan"]["persons"].append({"mentions": ["A01", "甲男"]})
        case["indictment"]["text"] += " A01與甲男，車牌OO-OO。"
        checks = audit.deterministic_checks(case)
        self.assertTrue(checks["pass"])
        self.assertEqual(checks["identifier_pattern_hits"], [])

    def test_roc_date_is_not_misclassified_as_plate(self) -> None:
        case = sample_case()
        case["judgment"]["text"] += " 裁判日期113-05-20。"
        hits = audit.deterministic_checks(case)["identifier_pattern_hits"]
        self.assertFalse(any(item["kind"] == "PLATE" for item in hits))

    def test_repeated_professional_mask_is_not_alias_collision(self) -> None:
        case = sample_case()
        case["judgment"]["pair_alias_registry"]["persons"] = [
            {"pair_person_id": "P01", "alias": "〇〇〇", "role": "PROSECUTOR"},
            {"pair_person_id": "P02", "alias": "〇〇〇", "role": "JUDGE"},
        ]
        self.assertEqual(audit.deterministic_checks(case)["alias_collisions"], [])

    def test_alias_collision_forces_manual_review(self) -> None:
        case = sample_case()
        case["judgment"]["pair_alias_registry"]["persons"].append({
            "pair_person_id": "P02", "alias": "被告甲"
        })
        checks = audit.deterministic_checks(case)
        outcome = audit.derive_outcome(valid_result(), checks)
        self.assertFalse(outcome["audit_pass"])
        self.assertTrue(outcome["requires_manual_review"])

    def test_validate_requires_every_section(self) -> None:
        result = valid_result()
        del result["evidence"]
        with self.assertRaises(ValueError):
            audit.validate_audit_result(result)

    def test_warning_cannot_become_automatic_pass(self) -> None:
        result = valid_result()
        result["aliases"]["status"] = "warning"
        outcome = audit.derive_outcome(result, {"pass": True})
        self.assertEqual(outcome["effective_decision"], "review")

    def test_select_pending_skips_audited_and_failed_pairs(self) -> None:
        cases = [{"pair_id": "a"}, {"pair_id": "b"}, {"pair_id": "c"}]
        selected = audit.select_cases(cases, {"a"}, {"b"}, "pending", 10)
        self.assertEqual([row["pair_id"] for row in selected], ["c"])


if __name__ == "__main__":
    unittest.main()
