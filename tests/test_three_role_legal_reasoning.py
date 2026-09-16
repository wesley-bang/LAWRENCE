import json
from pathlib import Path

import pytest

from scripts import generate_three_role_legal_reasoning as rolegen
from scripts import generate_gpt56_luna_flex_three_role_bulk as luna_bulk


def gemma_record(doc_id: str, **extra):
    record = {
        "doc_id": doc_id,
        "model": "gemma-4-31b-it",
        "text": f"TEXT-{doc_id}",
        "crime_facts_summary": [f"FACT-{doc_id}"],
        "evidence": [{"name": f"EVIDENCE-{doc_id}", "proves": "fact"}],
    }
    record.update(extra)
    return record


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sample_materials():
    pair = {
        "pair_id": "pair-1",
        "indictment": gemma_record("ind-1"),
        "judgment": gemma_record("jud-1"),
        "target_holding": "被告甲無罪。",
        "outcome_class": "acquittal",
    }
    return rolegen.case_materials(pair)


def valid_output(role: str) -> dict:
    common = {
        "role": role,
        "issues": ["issue"],
        "reasoning_trace": [{
            "evidence_ids": ["I-E001"],
            "support_snippets": ["fact"],
            "proposition": "fact",
        }],
        "new_evidence_claimed": [],
    }
    if role == "prosecutor":
        common["prosecution_brief"] = {
            "theory_of_case": {"statement": "theory", "evidence_ids": ["I-E001"]},
            "facts_asserted": [{"evidence_ids": ["I-E001"]}],
            "evidence_arguments": [{"evidence_ids": ["I-E001"]}],
        }
    elif role == "defense":
        common["defense_brief"] = {
            "admissions": [],
            "disputes": [{"evidence_ids": ["I-E001"]}],
            "evidence_arguments": [{"evidence_ids": ["I-E001"]}],
        }
        common["hypotheticals"] = [{"possibility": "possible", "is_asserted_fact": False}]
    else:
        common["decision"] = {
            "target_holding": "被告甲無罪。",
            "issue_findings": [{"evidence_ids": ["I-E001"]}],
            "response_to_prosecution": [{"evidence_ids": ["I-E001"]}],
            "response_to_defense": [{"evidence_ids": ["I-E001"]}],
            "rationale": [{"evidence_ids": ["I-E001"]}],
        }
    return common


def test_load_pairs_accepts_only_paired_gemma(tmp_path):
    indictments = tmp_path / "indictments.jsonl"
    judgments = tmp_path / "judgments.jsonl"
    write_jsonl(indictments, [
        gemma_record("good"),
        {**gemma_record("other"), "model": "some-other-model"},
    ])
    write_jsonl(judgments, [
        gemma_record(
            "jud-good", pair_id="pair-good", linked_indictment_doc_id="good",
            sections={"主文": "被告甲無罪。"},
        ),
        gemma_record(
            "jud-unpaired", pair_id="pair-bad", linked_indictment_doc_id="other",
            sections={"主文": "被告甲無罪。"},
        ),
    ])

    pairs = rolegen.load_pairs(indictments, judgments)

    assert [pair["pair_id"] for pair in pairs] == ["pair-good"]
    assert pairs[0]["outcome_class"] == "acquittal"


def test_evidence_ids_are_source_specific_and_stable():
    materials = sample_materials()

    assert [row["evidence_id"] for row in materials["all_evidence"]] == [
        "I-E001", "J-E001"
    ]
    assert materials["all_evidence"][0]["source_document"] == "indictment"
    assert materials["all_evidence"][1]["source_document"] == "judgment"


def test_role_prompts_enforce_visibility_boundaries():
    materials = sample_materials()
    prosecutor = valid_output("prosecutor")
    defense = valid_output("defense")

    prosecutor_prompt = rolegen.prompt_for_prosecutor(materials)
    defense_prompt = rolegen.prompt_for_defense(materials, prosecutor)
    judge_prompt = rolegen.prompt_for_judge(materials, prosecutor, defense)

    assert "TEXT-ind-1" in prosecutor_prompt
    assert "TEXT-jud-1" not in prosecutor_prompt
    assert "J-E001" not in prosecutor_prompt
    assert "被告甲無罪。" not in defense_prompt
    assert "J-E001" in defense_prompt
    assert "被告甲無罪。" in judge_prompt
    assert "defense_brief" in judge_prompt


def test_validator_rejects_unknown_or_new_evidence():
    output = valid_output("defense")
    output["reasoning_trace"][0]["evidence_ids"] = ["J-E999"]
    with pytest.raises(ValueError, match="unknown evidence"):
        rolegen.validate_role_output(
            output, "defense", {"I-E001", "J-E001"}, ["fact"]
        )

    output = valid_output("defense")
    output["new_evidence_claimed"] = ["invented witness"]
    with pytest.raises(ValueError, match="claimed new evidence"):
        rolegen.validate_role_output(
            output, "defense", {"I-E001", "J-E001"}, ["fact"]
        )


def test_validator_requires_exact_judge_holding_and_safe_hypotheticals():
    judge = valid_output("judge")
    rolegen.validate_role_output(
        judge, "judge", {"I-E001"}, ["fact"], "被告甲無罪。"
    )
    judge["decision"]["target_holding"] = "有罪"
    with pytest.raises(ValueError, match="exact target holding"):
        rolegen.validate_role_output(
            judge, "judge", {"I-E001"}, ["fact"], "被告甲無罪。"
        )

    defense = valid_output("defense")
    defense["hypotheticals"][0]["is_asserted_fact"] = True
    with pytest.raises(ValueError, match="is_asserted_fact=false"):
        rolegen.validate_role_output(defense, "defense", {"I-E001"}, ["fact"])


def test_validator_rejects_a_made_up_support_quote():
    output = valid_output("prosecutor")
    output["reasoning_trace"][0]["support_snippets"] = ["not in source"]
    with pytest.raises(ValueError, match="absent from visible materials"):
        rolegen.validate_role_output(output, "prosecutor", {"I-E001"}, ["fact"])


def test_support_quote_allows_only_unicode_width_and_whitespace_changes():
    output = valid_output("prosecutor")
    output["reasoning_trace"][0]["support_snippets"] = ["（被告 否認）"]
    rolegen.validate_role_output(
        output, "prosecutor", {"I-E001"}, ["source (被告否認) text"]
    )


@pytest.mark.parametrize("role", ["prosecutor", "defense", "judge"])
def test_luna_bulk_strict_schema_closes_every_object(role):
    schema = luna_bulk.role_schema(role)

    def visit(value):
        if isinstance(value, dict):
            if value.get("type") == "object":
                assert value.get("additionalProperties") is False
                assert set(value.get("required", [])) == set(value.get("properties", {}))
            if value.get("type") == "array":
                assert "items" in value
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)


def test_luna_bulk_repairs_only_a_high_similarity_support_quote():
    output = {
        "reasoning_trace": [{
            "step_id": "P01",
            "support_snippets": ["共計700萬元,充作股款經股東繳納之證明"],
        }]
    }
    source = "前文合計700萬元,充作股款經股東繳納之證明後文"

    repairs = luna_bulk.repair_support_snippets(output, [source])

    assert len(repairs) == 1
    assert output["reasoning_trace"][0]["support_snippets"][0] in source


def test_luna_bulk_does_not_repair_a_dissimilar_quote():
    output = {
        "reasoning_trace": [{
            "step_id": "P01",
            "support_snippets": ["監視器未顯示任何其他人在場"],
        }]
    }

    repairs = luna_bulk.repair_support_snippets(output, ["僅有監視器影像光碟一片"])

    assert repairs == []


def test_luna_bulk_drops_only_an_unverifiable_extra_quote():
    output = {
        "reasoning_trace": [{
            "step_id": "D01",
            "support_snippets": ["原文有效證據", "模型自行改寫而不存在的長句"],
        }]
    }

    repairs = luna_bulk.repair_support_snippets(output, ["案件記載原文有效證據。"])

    assert output["reasoning_trace"][0]["support_snippets"] == ["原文有效證據"]
    assert any(item["action"] == "drop_unverifiable_extra_quote" for item in repairs)
