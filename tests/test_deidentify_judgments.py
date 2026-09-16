from __future__ import annotations

import sys
import re
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import deidentify_judgments as deid  # noqa: E402


class DeidentificationRegressionTests(unittest.TestCase):
    def test_ministry_indictment_schema_adapter(self) -> None:
        source = {
            "barcode": "12345",
            "text": "被告王小明涉犯詐欺罪嫌。",
            "doc_type": "起訴書",
            "agency": {"short": "基隆地檢"},
            "charge": {"normalized": "詐欺"},
            "investigation": {"year_roc": 114, "year_ad": 2025},
        }
        adapted = deid.adapt_source_record(source)
        self.assertEqual(adapted["source_id"], "MOJ-PROSECUTION:12345")
        self.assertEqual(adapted["case_type"], "詐欺")
        self.assertEqual(adapted["source_year_roc"], 114)
        self.assertNotIn("detail_url", adapted)

    def test_leakage_audit_accepts_existing_aliases_and_procedural_phrase(self) -> None:
        organizations = [{"canonical_name": "高元有限公司", "alias": "機構甲", "type": "ORGANIZATION"}]
        text = "被告機構甲上列被告涉案；檢察官以證人身分傳喚甲到庭。"
        audit = deid.leakage_scan(text, [], organizations)
        self.assertTrue(audit["pass"])

    def test_reviewed_indictment_people_and_signature_are_handled(self) -> None:
        source = (
            "被告 杜志順 徐榮澤 鎖必信上列被告涉案，與少年林○浚共同犯案；"
            "證人鄭秉豪之供述及告訴人黃彥銘警詢之指訴。"
            "中華民國114年3月10日 書記官 劉芝麟所犯法條"
        )
        people = deid.extract_people(source)
        names = {person["canonical_name"] for person in people}
        self.assertTrue({"鎖必信", "林○浚", "鄭秉豪", "黃彥銘", "劉芝麟"} <= names)
        clean = deid.final_normalize(deid.replace_entities(source, people, []))
        for name in names:
            self.assertNotIn(name, clean)

    def test_source_masked_birth_bare_date_and_org_short_form(self) -> None:
        source = (
            "幼童江○錫(民國000年00月00日生,下稱A童)，"
            "於113年11月25日18時30分送至佛教慈濟財團法人花蓮慈濟醫院"
            "(下稱花蓮慈濟醫院)。"
        )
        people = deid.extract_people(source)
        organizations = deid.extract_organizations(source)
        clean = deid.replace_entities(source, people, organizations)
        clean, counts = deid.mask_deterministic_pii(clean)
        self.assertNotIn("江○錫", clean)
        self.assertIn("民國000年00月00日生", clean)
        self.assertIn("113年11月某日約18時", clean)
        match = re.search(r"(機構[甲乙丙丁])\(下稱(機構[甲乙丙丁])\)", clean)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), match.group(2))
        self.assertEqual(counts["SOURCE_MASKED_BIRTH_PRESERVED"], 1)

    def test_legal_terms_are_not_people(self) -> None:
        people = deid.extract_people("被害人A童頭部受傷，所謂之『凌虐』係指非人道待遇。")
        names = {person["canonical_name"] for person in people}
        self.assertNotIn("童頭部", names)
        self.assertNotIn("凌虐", names)

    def test_volume_reference_keeps_sentencing_factor_only(self) -> None:
        source = "身體狀況及罹患之疾病(見本院卷第41、[車牌]頁所示)等一切情狀"
        actual, count = deid.strip_volume_references(source)
        self.assertEqual(actual, "身體狀況及罹患之疾病")
        self.assertEqual(count, 1)

    def test_volume_reference_keeps_evidence_name(self) -> None:
        source = (
            "復有告訴人甲提供之對話紀錄文字檔各1份在卷可佐"
            "(見花蓮縣警察局吉安分局吉警偵字第1140025328號刑案偵查卷"
            "第21、23、31至47、59至65、93至107頁)"
        )
        actual, count = deid.strip_volume_references(source)
        self.assertEqual(actual, "復有告訴人甲提供之對話紀錄文字檔可佐")
        self.assertEqual(count, 1)
        evidence = deid.extract_evidence(actual)
        self.assertEqual(evidence[0]["name"], "告訴人甲提供之對話紀錄文字檔")
        self.assertIsNone(evidence[0]["content"])

    def test_inline_volume_reference_is_removed(self) -> None:
        actual, count = deid.strip_volume_references("道路交通事故調查報告表一見警卷第33頁可佐")
        self.assertEqual(actual, "道路交通事故調查報告表一可佐")
        self.assertEqual(count, 1)

    def test_bare_and_bracketed_court_conventions_are_removed(self) -> None:
        source = (
            "有法院前案紀錄表可佐(本院卷第14至20頁)；"
            "該筆錄可佐【見某地方檢察署[案號]卷,第61至69頁】；"
            "電話紀錄可佐【見本院[案號]卷內檢附電話紀錄表】。"
        )
        actual, count = deid.strip_volume_references(source)
        self.assertEqual(actual, "有法院前案紀錄表可佐；該筆錄可佐；電話紀錄可佐。")
        self.assertEqual(count, 3)

    def test_attachment_cross_reference_markers_are_removed(self) -> None:
        actual, _ = deid.strip_volume_references("其餘引用起訴書之記載(如附件)。《附件》起訴書內容")
        self.assertEqual(actual, "其餘引用起訴書之記載。起訴書內容")

    def test_all_zero_plate_and_account_are_preserved(self) -> None:
        source = "車牌號碼000-0000號，銀行帳號000000000000號；車牌AB-1234號，銀行帳號123456789號"
        actual, counts = deid.mask_deterministic_pii(source)
        self.assertIn("000-0000", actual)
        self.assertIn("000000000000", actual)
        self.assertIn("[車牌]", actual)
        self.assertIn("[銀行帳號]", actual)
        self.assertEqual(counts["PLATE"], 1)
        self.assertEqual(counts["BANK_ACCOUNT"], 1)
        self.assertEqual(counts["SOURCE_MASKED_PLATE_PRESERVED"], 1)
        self.assertEqual(counts["SOURCE_MASKED_ACCOUNT_PRESERVED"], 1)

    def test_source_masked_address_is_preserved_but_precise_address_is_generalized(self) -> None:
        source = "地址為高雄市○○區○○○路00○0號；另址為高雄市苓雅區中正一路100號。"
        actual, _ = deid.mask_deterministic_pii(source)
        self.assertIn("高雄市○○區○○○路00○0號", actual)
        self.assertNotIn("高雄市苓雅區中正一路100號", actual)
        self.assertIn("高雄市某處", actual)

    def test_url_mask_does_not_consume_adjacent_chinese_or_zero_account(self) -> None:
        source = "至假投資網站https://example.test/path銀行帳號000-000000000000號帳戶"
        actual, counts = deid.mask_deterministic_pii(source)
        self.assertEqual(actual, "至假投資網站[URL]銀行帳號000-000000000000號帳戶")
        self.assertEqual(counts["URL"], 1)
        self.assertEqual(counts["SOURCE_MASKED_ACCOUNT_PRESERVED"], 1)

    def test_court_code_is_not_consumed_by_org_or_identifier_rules(self) -> None:
        source = "嗣經A02經星展銀行通知；A05以網路銀行轉帳；尿液代號A00000000；三星SM-A205手機"
        organizations = deid.extract_organizations(source)
        self.assertFalse(any("A02" in item["canonical_name"] for item in organizations))
        self.assertFalse(any("A05" in item["canonical_name"] for item in organizations))
        replaced = deid.replace_entities(source, [], organizations)
        actual, counts = deid.mask_deterministic_pii(replaced)
        for code in ("A02", "A05", "A00000000"):
            self.assertIn(code, actual)
        self.assertIn("[車牌]手機", actual)
        self.assertEqual(counts["COURT_CODE_PRESERVED"], 3)

    def test_court_assigned_codes_are_kept_but_surnamed_oo_names_are_aliased(self) -> None:
        source = "被告A01與被告A02共同犯案；告訴人朱OO及檢察官莊OO均到庭。"
        people = deid.extract_people(source)
        clean = deid.replace_entities(source, people, [])
        self.assertIn("A01", clean)
        self.assertIn("A02", clean)
        self.assertNotIn("朱OO", clean)
        self.assertNotIn("莊OO", clean)
        self.assertTrue(any(person["canonical_name"] == "朱○○" for person in people))

    def test_masked_parties_receive_case_aliases_without_table_header_false_positive(self) -> None:
        text = "告訴人鍾○○、林○○證述綦詳。附表一：編號 被害人 詐騙方式 匯款時間"
        people = deid.extract_people(text)
        names = {person["canonical_name"]: person["alias"] for person in people}
        self.assertIn("鍾○○", names)
        self.assertIn("林○○", names)
        self.assertNotIn("詐騙方式", names)

    def test_reviewed_location_types_are_detected(self) -> None:
        text = (
            "在○○路000號萬家福斗六店內。位於○○路000號之「統一超商鑫花蓮門市」。"
            "花蓮縣警察局玉里分局交通分隊道路交通事故紀錄表。玉里簡易庭 法官 鍾 晴"
        )
        organizations = deid.extract_organizations(text)
        names = {item["canonical_name"] for item in organizations}
        self.assertTrue({"萬家福斗六店", "統一超商鑫花蓮門市", "花蓮縣警察局玉里分局交通分隊", "玉里簡易庭"} <= names)

    def test_spaced_judge_signature_keeps_role_with_circle_mask(self) -> None:
        text = "中華民國 115 年 8 月 25 日 玉里簡易庭 法官 鍾 晴\n以上正本證明與原本無異。"
        people = deid.extract_people(text)
        organizations = deid.extract_organizations(text)
        clean = deid.final_normalize(deid.replace_entities(text, people, organizations))
        self.assertNotIn("鍾 晴", clean)
        self.assertNotIn("鍾晴", clean)
        self.assertNotIn("玉里簡易庭", clean)
        self.assertIn("法官〇〇〇", clean)

    def test_relationship_people_malformed_header_and_foreign_names(self) -> None:
        text = (
            "被告 陳元龍上被告因案件遭訴；其前妻沈維妮所有之車輛；"
            "證人洪曼秀請其處理，友人林琮瑋、林東政於警詢證述。"
            "被告 VO TRAN DUY KHA搭載NGUYEN HUU LOC，案經NGUYEN THI LE HUYEN訴由警方偵辦。"
        )
        people = deid.extract_people(text)
        names = {person["canonical_name"] for person in people}
        expected = {
            "陳元龍", "沈維妮", "洪曼秀", "林琮瑋", "林東政",
            "VO TRAN DUY KHA", "NGUYEN HUU LOC", "NGUYEN THI LE HUYEN",
        }
        self.assertTrue(expected <= names)
        clean = deid.replace_entities(text, people, [])
        for name in expected:
            self.assertNotIn(name, clean)

    def test_spaced_prosecution_signatures_keep_roles_with_circle_masks(self) -> None:
        text = (
            "中華民國114年4月28日 檢察官 陳 筱 蓉"
            "本件正本證明與原本無異 中華民國114年5月某日 書記官 顏 偉 軒"
            "附錄本案所犯法條全文"
        )
        clean = deid.final_normalize(text)
        self.assertNotIn("陳 筱 蓉", clean)
        self.assertNotIn("顏 偉 軒", clean)
        self.assertIn("檢察官〇〇〇", clean)
        self.assertIn("書記官〇〇〇", clean)

    def test_model_discovered_people_and_organizations_are_covered(self) -> None:
        text = (
            "呂啟章則為公司協理；臉書貼文稱曾蜜小姐及徐先生；"
            "與真實年籍不詳之「哥仔」共同犯案。"
            "委託不知情之昇輝航空貨運承攬有限公司(下稱昇輝公司)，"
            "證據五1.國立臺灣大學醫學院附設醫院出具證明，另有敏盛綜合醫院診斷書。"
        )
        people = deid.extract_people(text)
        organizations = deid.extract_organizations(text)
        clean = deid.replace_entities(text, people, organizations)
        for value in ("呂啟章", "曾蜜小姐", "徐先生"):
            self.assertNotIn(value, clean)
        self.assertIn("哥仔", clean)
        for value in ("昇輝航空貨運承攬有限公司", "國立臺灣大學醫學院附設醫院", "敏盛綜合醫院"):
            self.assertNotIn(value, clean)

    def test_flattened_threads_item_number_does_not_corrupt_roc_year(self) -> None:
        normalized = deid.normalize_text("Threads14113年9月16日老師貼文")
        self.assertIn("Threads\n14. 113年9月16日", normalized)
        masked, _ = deid.mask_deterministic_pii(normalized)
        self.assertIn("113年9月某日", masked)

    def test_nickname_is_retained_but_landowner_name_is_removed(self) -> None:
        text = "被告梁景宇（暱稱金魚）向楊芙媄承租土地，未經楊芙媄同意開挖。"
        people = deid.extract_people(text)
        clean = deid.replace_entities(text, people, [])
        self.assertIn("金魚", clean)
        self.assertNotIn("楊芙媄", clean)


if __name__ == "__main__":
    unittest.main()
