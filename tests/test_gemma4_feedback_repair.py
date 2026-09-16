import unittest

from scripts import experiment_gemma4_feedback_repair as experiment


class Gemma4FeedbackRepairTests(unittest.TestCase):
    def test_selects_only_audit_failures(self) -> None:
        audits = [
            {"pair_id": "a", "effective_decision": "fail"},
            {"pair_id": "b", "effective_decision": "review"},
        ]
        judgments = [{"pair_id": "a"}, {"pair_id": "b"}]
        selected = experiment.select_failed_cases(audits, judgments, set(), set())
        self.assertEqual([row[0]["pair_id"] for row in selected], ["a"])

    def test_repair_prompt_contains_feedback_and_pair_context(self) -> None:
        baseline = {"pair_alias_registry": {"persons": []}, "analysis_plan": {}}
        indictment = {"text": "起訴書", "evidence": [{"canonical_key": "酒測單"}]}
        audit = {"audit_result": {"overall": {"decision": "fail"}}}
        prompt = experiment.repair_prompt("判決書", baseline, indictment, audit)
        self.assertIn("Gemini 3.8 audit", prompt)
        self.assertIn("canonical_key", prompt)
        self.assertIn('"decision":"fail"', prompt)

    def test_augment_registry_records_explicit_judgment_mentions(self) -> None:
        registry = {
            "persons": [{"pair_person_id": "P01", "alias": "被告甲"}],
            "unique_mentions": {},
        }
        plan = {"persons": [{
            "linked_indictment_group_id": "P01", "mentions": ["王小明"]
        }]}
        repaired = experiment.augment_registry(registry, plan)
        self.assertEqual(repaired["persons"][0]["judgment_mentions"], ["王小明"])
        self.assertEqual(
            repaired["unique_mentions"]["王小明"]["pair_person_id"], "P01"
        )
        self.assertNotIn("judgment_mentions", registry["persons"][0])


if __name__ == "__main__":
    unittest.main()
