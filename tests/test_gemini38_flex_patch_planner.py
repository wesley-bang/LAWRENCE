import unittest

from scripts import experiment_gemini38_flex_patch_planner as planner


def sample_case():
    return {
        "raw_indictment": "起訴書記載王小明",
        "raw_judgment": "判決書被告王小明",
        "indictment": {
            "text": "起訴書記載被告甲",
            "evidence": [{"name": "舊證據", "category": "書證"}],
            "analysis_plan": {"persons": [{"role": "DEFENDANT"}]},
        },
        "judgment": {"pair_alias_registry": {"persons": [{
            "pair_person_id": "P01", "alias": "被告甲",
            "indictment_mentions": ["王小明"],
        }]}, "text": "判決書被告甲", "analysis_plan": {"persons": []}},
    }


class Gemini38FlexPatchPlannerTests(unittest.TestCase):
    def test_valid_link_person_is_grounded(self) -> None:
        operation = {
            "op": "LINK_PERSON", "document": "judgment",
            "mentions": ["王小明"], "pair_person_id": "P01",
            "source_quote": "被告王小明", "rationale": "同一被告",
        }
        self.assertEqual(planner.validate_operation(operation, sample_case()), [])

    def test_rejects_unseen_quote_and_unknown_person(self) -> None:
        operation = {
            "op": "LINK_PERSON", "document": "judgment",
            "mentions": ["不存在"], "pair_person_id": "P99",
            "source_quote": "不存在", "rationale": "猜測",
        }
        errors = planner.validate_operation(operation, sample_case())
        self.assertIn("source_quote_not_found", errors)
        self.assertIn("unknown_pair_person_id", errors)
        self.assertIn("mention_not_found", errors)

    def test_rejects_unsafe_alias(self) -> None:
        operation = {
            "op": "REGISTER_PERSON", "document": "judgment",
            "mentions": ["王小明"], "role": "DEFENDANT",
            "proposed_pair_person_id": "P02", "alias": "王小明",
            "source_quote": "被告王小明", "rationale": "new",
        }
        self.assertIn(
            "unsafe_or_invalid_alias",
            planner.validate_operation(operation, sample_case()),
        )

    def test_rejects_unknown_field_path_and_missing_clean_target(self) -> None:
        operation = {
            "op": "MASK_SPAN", "document": "indictment",
            "field_path": "body", "old_text": "王小明",
            "replacement": "被告甲", "source_quote": "王小明",
        }
        errors = planner.validate_operation(operation, sample_case())
        self.assertIn("field_path_not_allowed", errors)

    def test_accepts_exact_json_pointer_and_rejects_invented_pointer(self) -> None:
        valid = {
            "op": "MASK_SPAN", "document": "indictment",
            "field_path": "/text", "old_text": "被告甲", "replacement": "被告乙",
            "source_quote": "王小明",
        }
        self.assertEqual(planner.validate_operation(valid, sample_case()), [])
        invented = dict(valid, field_path="/text/made-up")
        self.assertIn(
            "field_path_not_allowed", planner.validate_operation(invented, sample_case())
        )

    def test_validates_new_surgical_operations(self) -> None:
        replacement = {
            "op": "REPLACE_EVIDENCE", "document": "indictment",
            "field_path": "/evidence/0",
            "evidence": {"name": "正確證據", "category": "書證", "quantity": None,
                         "proves": "待證事實", "canonical_key": "document"},
            "source_quote": "王小明",
        }
        self.assertEqual(planner.validate_operation(replacement, sample_case()), [])
        remove = {
            "op": "REMOVE_REGISTRY_MENTION", "document": "indictment",
            "pair_person_id": "P01", "mention": "王小明", "source_quote": "王小明",
        }
        self.assertEqual(planner.validate_operation(remove, sample_case()), [])
        update = {
            "op": "UPDATE_PERSON_ROLE", "document": "indictment",
            "field_path": "/analysis_plan/persons/0/role", "role": "WITNESS",
            "source_quote": "王小明",
        }
        self.assertEqual(planner.validate_operation(update, sample_case()), [])

    def test_rejects_unsupported_person_role(self) -> None:
        operation = {
            "op": "REGISTER_PERSON", "document": "judgment",
            "mentions": ["王小明"], "role": "THIRD_PARTY",
            "proposed_pair_person_id": "P02", "alias": "人物甲",
            "source_quote": "被告王小明",
        }
        self.assertIn(
            "unsupported_person_role",
            planner.validate_operation(operation, sample_case()),
        )


if __name__ == "__main__":
    unittest.main()
