import copy
import unittest

from scripts import apply_gemini38_patch_plans as executor
from tests.test_gemini38_flex_patch_planner import sample_case


class ApplyGemini38PatchPlanTests(unittest.TestCase):
    def test_replaces_only_selected_json_pointer(self) -> None:
        case = sample_case()
        case["indictment"]["crime_facts_summary"] = ["被告甲未到庭"]
        operation = {
            "op": "RESTORE_FACT", "document": "indictment",
            "field_path": "/crime_facts_summary/0", "old_text": "未到庭",
            "replacement": "確未到庭", "source_quote": "王小明",
        }
        executor.apply_content_operation(case, operation)
        self.assertEqual(case["indictment"]["crime_facts_summary"], ["被告甲確未到庭"])
        self.assertEqual(case["indictment"]["text"], "起訴書記載被告甲")

    def test_replaces_evidence_instead_of_leaving_duplicate(self) -> None:
        case = sample_case()
        operation = {
            "op": "REPLACE_EVIDENCE", "document": "indictment",
            "field_path": "/evidence/0",
            "evidence": {"name": "新證據", "category": "書證", "quantity": None,
                         "proves": "待證事實", "canonical_key": "document"},
        }
        executor.apply_content_operation(case, operation)
        self.assertEqual(len(case["indictment"]["evidence"]), 1)
        self.assertEqual(case["indictment"]["evidence"][0]["name"], "新證據")

    def test_removes_registry_mention_and_reverse_index(self) -> None:
        case = sample_case()
        registry = case["judgment"]["pair_alias_registry"]
        registry["unique_mentions"] = {
            "王小明": {"pair_person_id": "P01", "alias": "被告甲"}
        }
        operation = {
            "op": "REMOVE_REGISTRY_MENTION", "document": "indictment",
            "pair_person_id": "P01", "mention": "王小明",
        }
        executor.apply_structural_operation(case, operation)
        self.assertEqual(registry["persons"][0]["indictment_mentions"], [])
        self.assertNotIn("王小明", registry["unique_mentions"])

    def test_updates_person_role_by_pointer(self) -> None:
        case = copy.deepcopy(sample_case())
        operation = {
            "op": "UPDATE_PERSON_ROLE", "document": "indictment",
            "field_path": "/analysis_plan/persons/0/role", "role": "WITNESS",
        }
        changed_i, changed_j = executor.apply_structural_operation(case, operation)
        self.assertTrue(changed_i)
        self.assertFalse(changed_j)
        self.assertEqual(case["indictment"]["analysis_plan"]["persons"][0]["role"], "WITNESS")


if __name__ == "__main__":
    unittest.main()
