from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hybrid_deidentify_with_google_ai as hybrid  # noqa: E402


class HybridDeterministicRenderTests(unittest.TestCase):
    def test_load_api_keys_accepts_primary_and_numbered_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                '"GOOGLE_STUDIO_API_KEY_2"="second"\n'
                'GOOGLE_STUDIO_API_KEY="primary"\n'
                'UNRELATED="ignored"\n',
                encoding="utf-8",
            )
            self.assertEqual(
                hybrid.load_api_keys(env_path),
                [
                    ("GOOGLE_STUDIO_API_KEY", "primary"),
                    ("GOOGLE_STUDIO_API_KEY_2", "second"),
                ],
            )

    def test_load_api_keys_strips_comments_after_quoted_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                'GOOGLE_STUDIO_API_KEY="primary" # first account\n'
                'GOOGLE_STUDIO_API_KEY_2="AQ.auth-token" # 第二個帳戶\n',
                encoding="utf-8",
            )
            self.assertEqual(
                hybrid.load_api_keys(env_path),
                [
                    ("GOOGLE_STUDIO_API_KEY", "primary"),
                    ("GOOGLE_STUDIO_API_KEY_2", "AQ.auth-token"),
                ],
            )

    @mock.patch.object(hybrid.time, "sleep")
    @mock.patch.object(hybrid, "request_json")
    def test_network_timeout_is_retried(self, request_json, sleep) -> None:
        request_json.side_effect = [TimeoutError("timed out"), ({"persons": []}, {})]
        case = {
            "doc_id": "doc-1", "source_id": "source-1", "case_type": "test",
            "normalized_text": "短文", "entities_json": '{"persons":[]}', "notes": "",
        }
        args = SimpleNamespace(
            scope="pending", force=False, max_source_chars=24000,
            retry_429_seconds=60.0, delay_seconds=0.0, model="gemma-4-31b-it",
            max_transient_attempts=2,
        )
        result = hybrid.request_case(
            case, None, args, "GOOGLE_STUDIO_API_KEY", "secret", {"last_request": None}
        )
        self.assertEqual(result["kind"], "success")
        self.assertEqual(request_json.call_count, 2)
        self.assertIn(mock.call(60.0), sleep.call_args_list)

    @mock.patch.object(hybrid.time, "sleep")
    @mock.patch.object(hybrid, "request_json")
    def test_repeated_network_errors_defer_case_and_release_worker(self, request_json, sleep) -> None:
        request_json.side_effect = [TimeoutError("first"), TimeoutError("second")]
        case = {
            "doc_id": "doc-2", "source_id": "source-2", "case_type": "test",
            "normalized_text": "短文", "entities_json": '{"persons":[]}', "notes": "",
        }
        args = SimpleNamespace(
            scope="pending", force=False, max_source_chars=24000,
            retry_429_seconds=60.0, delay_seconds=0.0, model="gemma-4-31b-it",
            max_transient_attempts=2,
        )
        result = hybrid.request_case(
            case, None, args, "GOOGLE_STUDIO_API_KEY_2", "secret", {"last_request": None}
        )
        self.assertEqual(result["kind"], "failure")
        self.assertEqual(result["reason"], "transient")
        self.assertEqual(result["record"]["error"], "transient_api_error")
        self.assertEqual(result["record"]["attempts"], 2)
        self.assertEqual(result["record"]["key_name"], "GOOGLE_STUDIO_API_KEY_2")
        self.assertEqual(request_json.call_count, 2)
        self.assertIn(mock.call(60.0), sleep.call_args_list)

    def test_registry_supplements_partially_masked_name_missed_by_llm(self) -> None:
        source = "被告王小明與少年林○浚共同犯案，少年林○浚之供述可佐。"
        plan = {
            "persons": [{"role": "DEFENDANT", "mentions": ["王小明"]}],
        }
        registry = {
            "persons": [{
                "canonical_name": "林○浚",
                "mentions": ["林○浚"],
                "roles": ["CO_OFFENDER"],
                "role": "CO_OFFENDER",
            }],
        }

        replacements = hybrid.build_replacements(source, plan, registry=registry)
        rendered = hybrid.render_text(source, replacements)

        self.assertEqual(rendered, "被告甲與少年甲共同犯案，少年甲之供述可佐。")
        self.assertNotIn("林○浚", rendered)

    def test_a01_style_alias_is_left_unchanged(self) -> None:
        source = "被告A01與被告A02共同犯案。"
        replacements = hybrid.build_replacements(source, {"persons": []}, registry={"persons": []})
        self.assertEqual(hybrid.render_text(source, replacements), source)

    def test_legal_professionals_keep_title_and_use_circle_mask(self) -> None:
        source = "檢察官王大明、書記官李小美、法官陳公平及選任辯護人林律師均到庭。"
        plan = {"persons": [
            {"role": "PROSECUTOR", "mentions": ["王大明"]},
            {"role": "CLERK", "mentions": ["李小美"]},
            {"role": "JUDGE", "mentions": ["陳公平"]},
            {"role": "DEFENSE_COUNSEL", "mentions": ["林律師"]},
        ]}
        rendered = hybrid.render_text(source, hybrid.build_replacements(source, plan))
        self.assertEqual(
            rendered,
            "檢察官〇〇〇、書記官〇〇〇、法官〇〇〇及選任辯護人〇〇〇均到庭。",
        )

    def test_llm_defendant_role_is_corrected_for_separately_investigated_cooffender(self) -> None:
        source = "被告王小明與共犯張博凱犯案，證人張博凱於警詢證述。"
        plan = {"persons": [
            {"role": "DEFENDANT", "mentions": ["王小明"]},
            {"role": "DEFENDANT", "mentions": ["張博凱"]},
        ]}
        registry = {"persons": [
            {"canonical_name": "王小明", "mentions": ["王小明"], "roles": ["DEFENDANT"]},
            {"canonical_name": "張博凱", "mentions": ["張博凱"], "roles": ["WITNESS"]},
        ]}
        replacements = hybrid.build_replacements(source, plan, registry=registry)
        self.assertEqual(
            hybrid.render_text(source, replacements),
            "被告甲與共犯甲犯案，證人即共犯甲於警詢證述。",
        )

    def test_public_institutions_are_not_masked(self) -> None:
        source = "衛生福利部旗山醫院、交通部公路局及國立屏東科技大學出具資料。"
        plan = {"organizations": [
            {"mentions": ["衛生福利部旗山醫院"], "action": "KEEP"},
            {"mentions": ["交通部公路局"], "action": "KEEP"},
            {"mentions": ["國立屏東科技大學"], "action": "KEEP"},
        ]}
        self.assertEqual(hybrid.render_text(source, hybrid.build_replacements(source, plan)), source)

    def test_related_investigation_number_is_masked_but_cited_precedent_is_kept(self) -> None:
        source = (
            "臺灣雲林地方檢察署114相字第102號檢驗報告書，"
            "並參照最高法院93年度台上字第5164號判決意旨。"
        )
        actual = hybrid.mask_uncited_case_numbers(source)
        self.assertIn("臺灣雲林地方檢察署[案號]檢驗報告書", actual)
        self.assertIn("最高法院93年度台上字第5164號判決意旨", actual)

        evidence = [{"name": "臺灣雲林地方檢察署114相字第102號檢驗報告書"}]
        masked = hybrid.mask_case_numbers_value(evidence)
        self.assertEqual(masked[0]["name"], "臺灣雲林地方檢察署[案號]檢驗報告書")


if __name__ == "__main__":
    unittest.main()
