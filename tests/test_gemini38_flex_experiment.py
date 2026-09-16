from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import experiment_gemini38_flex_judgments as experiment  # noqa: E402


class Gemini38FlexExperimentTests(unittest.TestCase):
    def test_api_key_without_inline_comment_accepts_aq_auth_key(self) -> None:
        self.assertEqual(
            experiment.api_key_without_inline_comment("AQ.example-token # note"),
            "AQ.example-token",
        )

    def test_api_key_without_inline_comment_handles_quoted_dotenv_value(self) -> None:
        # The shared dotenv loader has already stripped the opening quote.
        self.assertEqual(
            experiment.api_key_without_inline_comment('AQ.example-token" # note'),
            "AQ.example-token",
        )

    def test_api_key_without_inline_comment_rejects_non_comment_suffix(self) -> None:
        with self.assertRaises(ValueError):
            experiment.api_key_without_inline_comment("AQ.example-token accidental")

    def test_only_audit_fail_cases_are_selected(self) -> None:
        audits = [
            {"pair_id": "a", "effective_decision": "fail"},
            {"pair_id": "b", "effective_decision": "review"},
        ]
        judgments = [{"pair_id": "a"}, {"pair_id": "b"}]
        selected = experiment.select_failed_cases(audits, judgments, set(), set())
        self.assertEqual([row[0]["pair_id"] for row in selected], ["a"])

    def test_exact_prompt_digest_can_match_baseline(self) -> None:
        baseline = {"pair_alias_registry": {"persons": []}}
        manifest = {"text": "主文\n被告甲無罪。"}
        _, prompt, digest = experiment.exact_baseline_prompt(baseline, manifest)
        self.assertEqual(digest, hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        self.assertIn("起訴書人物錨點", prompt)


if __name__ == "__main__":
    unittest.main()
