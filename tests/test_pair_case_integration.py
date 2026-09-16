from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import deidentify_linked_judgments as judgment  # noqa: E402
import pair_case_integration as pairing  # noqa: E402


class PairCaseIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.indictment_text = "被告王小明、被告許大華共同犯案。"
        self.indictment_plan = {"persons": [
            {"group_id": "P01", "role": "DEFENDANT", "mentions": ["王小明"]},
            {"group_id": "P02", "role": "DEFENDANT", "mentions": ["許大華"]},
        ]}
        self.pair_registry = pairing.build_pair_alias_registry(
            self.indictment_text, self.indictment_plan
        )
        self.metadata = {
            "court_code": "KLD", "court_name": "基隆地院", "sys": "M",
            "year": 114, "case_word": "訴", "number": 93, "date": "2025-06-30",
        }

    def test_judgment_reuses_indictment_alias_by_explicit_group(self) -> None:
        aliases = {x["pair_person_id"]: x["alias"] for x in self.pair_registry["persons"]}
        self.assertEqual(aliases["P02"], "被告乙")
        source = "上訴人即被告許大華坦承犯行。"
        plan = {"persons": [{
            "group_id": "J01", "linked_indictment_group_id": "P02",
            "role": "DEFENDANT", "mentions": ["許大華"],
        }]}
        replacements = judgment.build_judgment_replacements(
            source, plan, pair_alias_registry=self.pair_registry
        )
        rendered = judgment.render_judgment_text(source, replacements, self.metadata)
        self.assertEqual(rendered, "上訴人即被告乙坦承犯行。")

    def test_unique_masked_name_variant_can_resolve_automatically(self) -> None:
        registry = pairing.build_pair_alias_registry(
            "被告許OO涉案。",
            {"persons": [{
                "group_id": "P01", "role": "DEFENDANT", "mentions": ["許OO"]
            }]},
        )
        resolved = pairing.resolve_pair_alias(
            {"role": "DEFENDANT", "mentions": ["許○○"]}, registry
        )
        self.assertEqual(resolved["alias"], "被告甲")
        self.assertEqual(resolved["method"], "unique_normalized_mention")

    def test_explicit_group_link_replaces_existing_court_person_code(self) -> None:
        source = "被告許大華即A03坦承犯行。"
        plan = {"persons": [{
            "group_id": "J01", "linked_indictment_group_id": "P02",
            "role": "DEFENDANT", "mentions": ["許大華", "A03"],
        }]}
        replacements = judgment.build_judgment_replacements(
            source, plan, pair_alias_registry=self.pair_registry
        )
        rendered = judgment.render_judgment_text(source, replacements, self.metadata)
        self.assertEqual(rendered, "被告乙即被告乙坦承犯行。")

    def test_ambiguous_same_name_is_not_guessed(self) -> None:
        registry = pairing.build_pair_alias_registry(
            "被告林○○與證人林○○均到庭。",
            {"persons": [
                {"group_id": "P01", "role": "DEFENDANT", "mentions": ["林○○"]},
                {"group_id": "P02", "role": "WITNESS", "mentions": ["林○○"]},
            ]},
        )
        self.assertTrue(registry["manual_review_required"])
        self.assertIsNone(pairing.resolve_pair_alias(
            {"role": "OTHER", "mentions": ["林OO"]}, registry
        ))

    def test_evidence_union_merges_duplicates_and_keeps_provenance(self) -> None:
        indictment = [
            {"name": "被告乙警詢供述1份", "category": "供述", "proves": "承認取款"},
            {"name": "監視器照片", "category": "照片", "proves": "出現在現場"},
        ]
        decision = [
            {"name": "被告乙警詢供述", "category": "供述", "proves": "犯行經過"},
            {"name": "扣案手機", "category": "實體物證", "proves": "聯絡共犯"},
        ]
        merged = pairing.merge_pair_evidence(indictment, decision)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["provenance"], ["indictment", "judgment"])
        self.assertEqual(merged[0]["proves"], ["承認取款", "犯行經過"])
        self.assertEqual(len(merged[0]["source_records"]), 2)

    def test_public_pair_output_does_not_contain_raw_person_mentions(self) -> None:
        output = pairing.integrate_pair_records(
            {"doc_id": "clean-indictment", "text": "被告甲涉案。", "evidence": []},
            {"doc_id": "clean-judgment", "text": "被告甲有罪。", "evidence": []},
            self.pair_registry,
            pair_id="stable-clean-pair-id",
        )
        self.assertNotIn("pair_alias_registry", output)
        self.assertNotIn("王小明", str(output))
        self.assertNotIn("許大華", str(output))
        self.assertEqual(output["pair_id"], "stable-clean-pair-id")
        self.assertEqual(output["pair_entities"][1]["alias"], "被告乙")


if __name__ == "__main__":
    unittest.main()
