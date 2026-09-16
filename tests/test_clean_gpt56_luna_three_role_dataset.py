from scripts import clean_gpt56_luna_three_role_dataset as cleaner


def test_clean_holding_removes_fact_section_but_not_fact_reference():
    value = "被告甲犯犯罪事實欄所示之罪。 事實被告甲於某日行為。"
    cleaned, changed = cleaner.clean_holding(value)
    assert changed is True
    assert cleaned == "被告甲犯犯罪事實欄所示之罪。"


def test_clean_holding_leaves_plain_holding_unchanged():
    value = "本件公訴不受理。扣案手機一支沒收。"
    assert cleaner.clean_holding(value) == (value, False)


def test_classify_outcome_handles_procedural_and_mixed_results():
    assert cleaner.classify_outcome("本件公訴不受理。") == "procedural"
    assert cleaner.classify_outcome("被告甲無罪。") == "acquittal"
    assert cleaner.classify_outcome("被告甲犯竊盜罪。") == "conviction"
    assert cleaner.classify_outcome("被告甲犯竊盜罪；被告乙公訴不受理。") == "mixed"
    assert cleaner.classify_outcome("本件再開辯論。") == "interlocutory"


def test_private_literal_detection_ignores_codes_aliases_and_nicknames():
    source = {
        "analysis_plan": {
            "persons": [
                {"mentions": ["王小明"], "same_person_reason": "被告"},
                {"mentions": ["A男", "BS000-A114023"], "same_person_reason": "被害人"},
                {"mentions": ["金魚"], "same_person_reason": "LINE暱稱"},
            ],
            "identifiers": [
                {"mention": "000-0000", "category": "PLATE"},
                {"mention": "臺北市○○路00號", "category": "ADDRESS"},
                {"mention": "ABC-1234", "category": "PLATE"},
                {"mention": "112年度台上字第1900號", "category": "CASE_NO", "reason": "引用裁判 KEEP"},
            ],
        }
    }
    assert cleaner.original_private_literals(source) == {"王小明", "ABC-1234"}


def test_private_replacement_map_preserves_nickname_and_aligns_documents():
    indictment = {
        "analysis_plan": {"persons": [
            {"role": "DEFENDANT", "mentions": ["王小明"], "same_person_reason": "被告"},
            {"role": "OTHER", "mentions": ["阿信"], "same_person_reason": "共犯"},
            {"role": "POLICE", "mentions": ["陳○○"], "same_person_reason": "員警"},
        ]}
    }
    judgment = {
        "analysis_plan": {"persons": [
            {"role": "DEFENDANT", "mentions": ["王小明"], "same_person_reason": "被告"},
        ]}
    }
    replacements = cleaner.private_replacement_map(indictment, judgment)
    assert replacements["王小明"] == "被告甲"
    assert replacements["陳○○"] == "員警甲"
    assert "阿信" not in replacements


def test_replace_private_literals_handles_catalog_and_roles():
    value, changed = cleaner.replace_private_literals(
        {"name": "王小明供述", "nested": ["王小明否認"]}, {"王小明": "被告甲"}
    )
    assert value == {"name": "被告甲供述", "nested": ["被告甲否認"]}
    assert changed == 2


def test_transform_strings_only_changes_string_values():
    value = {"a": "actual_target_holding", "b": [1, "x actual_target_holding"]}
    result = cleaner.transform_strings(
        value, lambda text: text.replace("actual_target_holding", "本案主文")
    )
    assert result == {"a": "本案主文", "b": [1, "x 本案主文"]}


def test_migrate_legal_rules_removes_evidence_ids_and_adds_sources():
    roles = {
        "judge": {
            "reasoning_trace": [{
                "step_id": "J01",
                "claim_type": "legal_rule",
                "evidence_ids": ["I-E001"],
                "support_snippets": ["本罪須告訴乃論"],
            }]
        }
    }
    repairs = cleaner.migrate_legal_rules(
        roles, {"indictment": "法律規定本罪須告訴乃論。", "judgment": "其他內容"}
    )
    step = roles["judge"]["reasoning_trace"][0]
    assert step["evidence_ids"] == []
    assert step["legal_sources"] == ["indictment"]
    assert repairs[0]["removed_evidence_ids"] == ["I-E001"]


def test_normalize_prefixed_aliases():
    value, repairs = cleaner.normalize_aliases(
        {"text": "被告甲○○否認，證人乙○○作證。"}, "被告甲○○、證人乙○○"
    )
    assert value["text"] == "被告甲否認，證人乙作證。"
    assert repairs
