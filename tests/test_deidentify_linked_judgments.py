from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import deidentify_linked_judgments as judgment  # noqa: E402


class JudgmentDeterministicRenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            "court_code": "KLD", "court_name": "基隆地院", "sys": "M",
            "year": 114, "case_word": "訴", "number": 93, "date": "2025-06-30",
        }

    def test_masks_own_case_number_but_preserves_cited_precedent(self) -> None:
        source = (
            "臺灣基隆地方法院 114 年度訴字第 93 號刑事判決\n"
            "參照最高法院93年度台上字第5164號判決意旨。"
        )
        rendered = judgment.render_judgment_text(source, [], self.metadata)
        self.assertIn("[本案案號]刑事判決", rendered)
        self.assertNotIn("114 年度訴字第 93 號", rendered)
        self.assertIn("最高法院93年度台上字第5164號判決意旨", rendered)

    def test_generic_llm_case_replacement_cannot_override_own_case_marker(self) -> None:
        source = "臺灣基隆地方法院114年度訴字第93號刑事判決"
        replacements = [("114年度訴字第93號", "[案號]", "案號")]
        rendered = judgment.render_judgment_text(source, replacements, self.metadata)
        self.assertIn("[本案案號]", rendered)
        self.assertNotIn("[案號]", rendered)

    def test_masks_all_consolidated_dockets_in_document_header(self) -> None:
        source = (
            "臺灣雲林地方法院刑事判決114年度易字第484號 第663號 第821號"
            "公訴人 臺灣雲林地方檢察署檢察官被告王小明"
        )
        metadata = dict(self.metadata, court_code="ULD", court_name="雲林地院", number=821)
        rendered = judgment.render_judgment_text(source, [], metadata)
        self.assertIn("刑事判決[本案案號]公訴人", rendered)
        self.assertNotIn("第484號", rendered)
        self.assertNotIn("第663號", rendered)
        self.assertNotIn("第821號", rendered)

    def test_supports_judgment_specific_roles_without_duplicate_labels(self) -> None:
        source = "聲請人王小明與受刑人李大華均到庭。"
        plan = {"persons": [
            {"role": "PETITIONER", "mentions": ["王小明"]},
            {"role": "SENTENCED_PERSON", "mentions": ["李大華"]},
        ]}
        replacements = judgment.build_judgment_replacements(source, plan)
        rendered = judgment.render_judgment_text(source, replacements, self.metadata)
        self.assertEqual(rendered, "聲請人甲與受刑人甲均到庭。")
        self.assertNotIn("王小明", rendered)
        self.assertNotIn("李大華", rendered)

    def test_adapts_manifest_without_exposing_detail_url(self) -> None:
        record = {
            "detail_url": "https://judgment.judicial.gov.tw/FJUD/data.aspx?id=secret",
            "title": "臺灣基隆地方法院刑事判決",
            "text": "主文\n被告有罪。",
            "judgment_metadata": self.metadata,
        }
        adapted = judgment.adapt_linked_judgment(record)
        self.assertEqual(adapted["document_type"], "刑事判決")
        self.assertEqual(adapted["issuing_level"], "court")
        self.assertTrue(adapted["source_id"].startswith("JUDICIAL:"))
        self.assertNotIn("detail_url", adapted)
        self.assertNotIn("secret", adapted["source_id"])

    def test_extracts_judgment_sections(self) -> None:
        source = "主文\n被告有罪。\n犯罪事實\n一、被告犯案。\n理由\n證據充分。"
        sections = judgment.extract_judgment_sections(source)
        self.assertEqual(sections["主文"], "被告有罪。")
        self.assertEqual(sections["犯罪事實"], "一、被告犯案。")
        self.assertEqual(sections["理由"], "證據充分。")

    def test_removes_page_chrome_and_repairs_flattened_headings(self) -> None:
        source = (
            "去格式引用\n分享網址\n裁判字號:\n臺灣基隆地方法院 114 年度訴字第 93 號刑事判決\n"
            "臺灣基隆地方法院刑事判決114年度訴字第93號公訴人 檢察官\n"
            "主 文被告有罪。\n犯 罪\n事 實 及 理 由一、被告犯案。"
        )
        normalized = judgment.normalize_judgment_text(source)
        self.assertNotIn("分享網址", normalized)
        self.assertIn("\n主文\n被告有罪。", normalized)
        self.assertIn("\n犯罪事實及理由\n一、被告犯案。", normalized)
        sections = judgment.extract_judgment_sections(normalized)
        self.assertEqual(sections["主文"], "被告有罪。")
        self.assertEqual(sections["犯罪事實及理由"], "一、被告犯案。")


if __name__ == "__main__":
    unittest.main()
