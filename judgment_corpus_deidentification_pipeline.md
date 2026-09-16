# 台灣刑事法律書類資料蒐集與去識別化 Pipeline

> 目前實作資料源已改為法務部公開起訴書：`data/raw/indictments/indictments_114_source.jsonl` 與 `data/raw/indictments/indictments_115_source.jsonl`。第一輪嚴格選取前者民國 114 年的 940 筆；本文原有判決書原則仍作為通用去識別化依據，實際欄位與輸出以專案根目錄 `README.md` 為準。

> 目標：建立可用於法律 LLM 蒸餾／訓練的**刑事判決書 corpus**。本階段只處理：
>
> **司法院判決書取得 → 原始資料保存 → 文字解析 → 案件級實體抽取 → 去識別化 → 隱私與語義品質檢查 → 輸出 clean corpus**
>
> 暫不包含起訴書、檢／辯／審三方 rationale / CoT 生成。

---

## 0. 核心原則

這條 pipeline 建議遵守以下原則：

1. **不要爬裁判書查詢 HTML 當主要來源**：優先使用司法院官方開放資料／API。
2. **raw data 與 training data 嚴格分離**：原始判決可能仍含可識別資訊，不能直接進訓練集。
3. **去識別化以案件（case）為單位，而非以單一字串為單位**。
4. **同一案件內同一人必須使用一致代號**，例如王小明在全文都映射為「甲」。
5. **不同案件之間不要維持相同人物代號或可連結 ID**，避免跨案件 linkage。
6. **不是只有姓名要處理**：案號、地址、生日、電話、帳號、車牌、公司／學校／職稱、精確時間與地點等，都可能重新識別當事人。
7. **保留法律推理需要的語義**：例如「案發時 17 歲」可能影響少年法制，不應把年齡整個刪除。
8. **先去識別化，再拿去做 teacher-model 蒸餾**；之後若生成 rationale，建議再跑一次 privacy audit。

---

# 1. 判決書資料來源

## 1.1 首選：司法院資料開放平臺

司法院提供裁判書開放 API 與下載資料，可直接取得裁判全文，不需要逐頁爬裁判書查詢網頁。

官方來源：

- 司法院資料開放平臺：<https://opendata.judicial.gov.tw/>
- 裁判書開放 API 規格說明：<https://opendata.judicial.gov.tw/news/detail?newsId=3041>
- 裁判書查詢系統：<https://judgment.judicial.gov.tw/>
- 裁判書遮隱規則：<https://judgment.judicial.gov.tw/cover_rule/cover_rule.html>

API 裁判書內容可取得的核心欄位包含：

```text
JID        裁判書 ID
JYEAR      年度
JCASE      字別
JNO        號次
JDATE      裁判日期
JTITLE     案由
JFULLX     裁判全文資訊
  ├─ JFULLTYPE
  ├─ JFULLCONTENT   # 文字型裁判全文
  └─ JFULLPDF       # 若全文為檔案形式，提供 PDF URL
ATTACHMENTS          # 附件
```

### 建議下載策略

若建立 historical corpus：

```text
官方批次／月資料
        ↓
建立歷史 corpus
        ↓
之後每天／定期使用 API 取得異動
```

比逐筆查詢 HTML 穩定許多。

### 注意裁判異動與撤下

司法院 API 規格指出：裁判書公開後仍可能修改或移除。

因此資料庫必須支援：

```text
同 JID 再次出現
→ 覆蓋舊版本

官方回傳該裁判已移除／不公開
→ raw DB 與 downstream corpus 中同步移除
```

不要把下載後的裁判視為永久 immutable。

---

# 2. 建議資料架構

建議至少分為三層：

```text
data/
├── raw/                 # 原始官方資料；限制存取
│   ├── judgments/
│   └── metadata/
│
├── intermediate/        # entity extraction / anonymization 工作資料
│   ├── parsed/
│   ├── entity_registry/
│   └── audit/
│
└── clean/               # 可進後續 teacher / SFT pipeline 的版本
    └── judgments/
```

不要覆寫 raw data。

---

# 3. Raw Judgment Schema

推薦原始資料使用 JSONL。

例如：

```json
{
  "internal_doc_id": "sha256-generated-id",
  "source": "judicial_opendata",
  "jid": "ORIGINAL_JID",
  "jyear": "113",
  "jcase": "訴",
  "jno": "123",
  "jdate": "2024-05-20",
  "jtitle": "詐欺",
  "raw_text": "...原始裁判全文...",
  "source_first_seen": "2026-09-06T00:00:00+08:00",
  "source_last_seen": "2026-09-06T00:00:00+08:00",
  "source_deleted": false,
  "source_hash": "sha256(raw_text)"
}
```

其中：

- `jid` 只放在 restricted/raw DB。
- clean training corpus 不應直接保留原始 `jid`／完整案號。
- `source_hash` 可用來偵測來源是否變更。

---

# 4. 為什麼不能只把姓名改成甲乙丙丁

例如原文：

```text
被告王小明，民國78年3月4日生，住臺中市西屯區○○路100號，
任職於ABC科技股份有限公司，駕駛ABC-1234號自用小客車……
```

如果只改姓名：

```text
被告甲，民國78年3月4日生，住臺中市西屯區○○路100號，
任職於ABC科技股份有限公司，駕駛ABC-1234號自用小客車……
```

仍可能透過生日、公司、地址、車牌等資訊重新識別。

司法院自己的裁判書遮隱說明，也特別列出可能足以識別個人的資訊，包括：

- 地址
- 公司名稱
- 工作場所
- 職業／職稱
- 學校／年級
- 地標
- 證人姓名
- 事件年月日時
- 事件地點
- 親屬關係
- 其他可交叉識別資訊

因此此專案需要做的是**案件級 de-identification / pseudonymization**，不是單純的姓名 regex replacement。

---

# 5. 去識別化整體流程

推薦：

```text
Raw Judgment
      │
      ▼
[1] Parsing / normalization
      │
      ▼
[2] Deterministic PII detection
      │
      ▼
[3] Person / Organization entity extraction
      │
      ▼
[4] Case-level entity resolution
      │
      ▼
[5] 建立 Entity Registry
      │
      ▼
[6] Semantic-preserving replacement
      │
      ▼
[7] LLM / NER privacy audit
      │
      ▼
[8] Deterministic final sanitizer
      │
      ▼
[9] Leakage + semantic QA
      │
      ▼
Clean Judgment
```

---

# 6. Stage 1：文字正規化

在辨識 entity 前先做 normalization，但不要改變法律內容。

可處理：

```text
Unicode normalization (NFKC)
全形／半形統一
特殊空白統一
重複空格壓縮
換行格式統一
HTML entity decode
```

不要做：

```text
全文摘要
LLM paraphrase
同義詞改寫
刪除看似重複的句子
```

原因是這些操作可能改變判決語義或證據關係。

Python：

```python
import re
import unicodedata


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
```

---

# 7. Stage 2：Deterministic PII Detection

第一層先抓格式明確的 identifier。

建議至少涵蓋：

```text
身分證字號
護照／居留證號（能辨識者）
電話
Email
IP address
URL / 社群帳號
銀行帳號
信用卡號
車牌
出生年月日
完整地址
原始案號 / JID
```

這層優先使用 regex / parser，而不是 LLM。

## 7.1 範例 regex

注意：以下為起始範例，正式上線前需用大量真實判決做 false positive / false negative 測試。

```python
PATTERNS = {
    "TW_ID": r"\b[A-Z][12]\d{8}\b",

    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",

    "IPV4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",

    "MOBILE": r"(?<!\d)09\d{2}[- ]?\d{3}[- ]?\d{3}(?!\d)",

    # 台灣一般車牌格式很多，實作時應再依資料擴充
    "PLATE": r"\b[A-Z]{2,3}[-－]?\d{3,4}\b",
}
```

替換時建議保留 type：

```text
A123456789
→ [身分證字號]

0912-345-678
→ [電話]

ABC-1234
→ [車牌]
```

不要統一全部變成 `[REDACTED]`，因為保留 entity type 對法律文本理解較友善。

---

# 8. Stage 3：人物與機構實體抽取

Regex 不適合處理：

```text
王小明
王○明
王男
小明
綽號「阿明」
被告王小明
證人王小明
```

需要另外做 NER / rule-based extraction。

建議使用混合方法：

```text
司法文書結構規則
    +
中文 NER
    +
LLM entity extraction
```

其中**裁判書開頭的當事人欄是很好的 seed**。

例如：

```text
被      告  王小明
選任辯護人  李大華律師
告 訴 人    陳美華
```

可以先建立：

```text
王小明 → DEFENDANT
陳美華 → VICTIM / COMPLAINANT
李大華 → DEFENSE_COUNSEL
```

再掃全文 aliases。

---

# 9. Stage 4：Case-level Entity Resolution

最重要的一步之一。

假設全文出現：

```text
王小明
王男
被告王小明
被告
小明
綽號阿明
```

系統要判斷哪些 mention 是同一個 entity。

推薦資料結構：

```json
{
  "case_internal_id": "case_000001",
  "persons": [
    {
      "entity_id": "P1",
      "role": "DEFENDANT",
      "canonical_name": "王小明",
      "mentions": [
        "王小明",
        "被告王小明",
        "王男",
        "小明",
        "阿明"
      ],
      "alias": "甲"
    },
    {
      "entity_id": "P2",
      "role": "VICTIM",
      "canonical_name": "陳美華",
      "mentions": [
        "陳美華",
        "陳女"
      ],
      "alias": "乙"
    }
  ]
}
```

此表稱為 **Entity Registry**。

---

# 10. 人物代號策略

## 10.1 同案件內一致

```text
王小明 → 甲
陳美華 → 乙
林大華 → 丙
```

在同一案件全文都必須一致。

## 10.2 不要跨案件維持 identity

不要：

```text
case A：王小明 → PERSON_1371
case B：王小明 → PERSON_1371
```

建議：

```text
case A：王小明 → 甲
case B：王小明 → 乙
```

不同案件不應透過 training corpus 被 linkage。

## 10.3 角色稱謂最好保留

推薦：

```text
被告甲
被害人乙
證人丙
共同被告丁
告訴人乙
```

比單純：

```text
甲
乙
丙
```

更容易讓模型學到法律角色。

---

# 11. 法官、檢察官、律師如何處理

這些姓名通常不是案件法律推理的核心。

建議直接角色化：

```text
法官王小明
→ 法官

檢察官陳大華
→ 檢察官

辯護人林○○律師
→ 辯護人
```

若同案件需要區分多人，可使用：

```text
法官A / 法官B
檢察官A / 檢察官B
辯護人A / 辯護人B
```

不要與案件人物共用「甲乙丙丁」。

---

# 12. 法人、公司、醫院、學校等

不要把所有 entity 都映射成甲乙丙丁。

建議依 type 使用不同 namespace：

```text
自然人：甲、乙、丙、丁
公司：甲公司、乙公司
銀行：A銀行、B銀行
醫院：甲醫院、乙醫院
學校：甲學校、乙學校
商店：甲商店
機關：甲機關
```

例如：

```text
王小明任職於台灣ABC科技股份有限公司
```

可改：

```text
甲任職於甲公司
```

是否需要 anonymize 法人名稱，可依 re-identification risk 決定；對自然人案件而言，特定公司 + 特定職稱 + 日期可能形成強識別訊號，因此一般建議泛化。

---

# 13. 案號一定要從 training text 移除

這一點非常重要。

即使全文姓名已經全部匿名，只要保留：

```text
臺灣臺北地方法院113年度訴字第123號
```

就可能透過公開裁判書查詢系統重新找到原始案件。

因此 clean corpus 中：

```text
完整法院 + 年度 + 字別 + 號次
完整 JID
裁判書 URL
原始 PDF URL
```

都應移除。

若 downstream task 需要法院層級，可只保留泛化 metadata：

```json
{
  "court_level": "district_court",
  "case_domain": "criminal",
  "year_bucket": "2020s"
}
```

若地區本身與法律推理無關，可連法院所在地也不提供給模型。

---

# 14. Semantic-preserving Anonymization

去識別化不能破壞法律上重要資訊。

## 14.1 出生日期

原文：

```text
被告甲民國95年3月17日出生，案發日為民國112年2月10日。
```

不要單純改成：

```text
被告甲出生日期不詳。
```

如果年齡與法律責任有關，應轉成：

```text
被告甲案發時17歲。
```

或：

```text
被告甲案發時未滿18歲。
```

推薦先計算 legally-relevant derived feature，再刪除精確生日。

---

## 14.2 地址

原：

```text
臺中市西屯區臺灣大道三段99號8樓
```

若城市具有管轄／情境價值：

```text
臺中市某處
```

若沒有：

```text
某處
```

---

## 14.3 精確事件時間

原：

```text
民國112年5月17日上午9時13分27秒
```

依任務需求可以轉成：

```text
某日上午約9時
```

或：

```text
案發日上午
```

如果時間差本身涉及不在場證明、因果順序、酒測時間等，則必須保留足以推理的相對時間資訊。

---

## 14.4 學校／工作

原：

```text
就讀○○高中二年級
```

可改：

```text
就讀某高中二年級
```

保留「高中二年級」可能具有年齡／少年案件推理價值。

原：

```text
任○○科技股份有限公司執行長
```

若公司與職位不是案件要件：

```text
任某公司管理職
```

---

# 15. 建議 Replacement Policy

| 原始資訊 | clean corpus 建議 |
|---|---|
| 被告姓名 | 被告甲 |
| 被害人姓名 | 被害人乙 |
| 證人姓名 | 證人丙 |
| 共犯姓名 | 共同被告丁／共犯丁 |
| 法官姓名 | 法官 |
| 檢察官姓名 | 檢察官 |
| 律師姓名 | 辯護人 |
| 身分證字號 | `[身分證字號]` |
| 護照／居留證號 | `[證件號碼]` |
| 電話 | `[電話]` |
| Email | `[Email]` |
| 社群帳號 | `[社群帳號]` |
| IP | `[IP]` |
| 銀行帳號 | `[銀行帳號]` |
| 信用卡號 | `[信用卡號]` |
| 車牌 | `[車牌]` |
| 精確出生日期 | 年齡／年齡區間（若法律相關） |
| 精確地址 | 城市層級或「某處」 |
| 公司名稱 | 甲公司／某公司 |
| 學校名稱 | 甲學校／某學校 |
| 醫院名稱 | 甲醫院／某醫院 |
| 完整案號 | 移除 |
| JID | 移除 |
| 判決 URL | 移除 |
| 精確事件時間 | 視推理需求泛化 |
| 精確地標 | 視推理需求泛化 |

---

# 16. Stage 5：Replacement 的實作順序

replacement 順序很重要。

推薦：

```text
1. 建 Entity Registry
2. 對較長 mention 優先 replacement
3. 再替換較短 alias
4. 再跑 deterministic PII masker
5. 最後 privacy auditor
```

例如：

```text
被告王小明
王小明
小明
```

應按照字串長度由長到短替換，以免先把 `王小明` 的一部分替換造成後續 match 失敗。

示意：

```python

def replace_mentions(text, entity_registry):
    replacements = []

    for ent in entity_registry:
        alias = ent["alias"]
        for mention in ent["mentions"]:
            replacements.append((mention, alias))

    replacements.sort(key=lambda x: len(x[0]), reverse=True)

    for src, dst in replacements:
        text = text.replace(src, dst)

    return text
```

實際版需要額外處理 overlapping spans 與 contextual title，最好使用 span-based replacement 而非單純 `str.replace()`。

---

# 17. Stage 6：LLM Privacy Auditor

LLM 不建議直接負責改寫全文，但適合當**漏網 PII detector**。

輸入 anonymized text，要求只標 span，不准改文。

Prompt 可設計：

```text
你是一個法律文本隱私檢查器。

請找出下列裁判文字中仍可能直接或間接識別自然人的資訊。
不要改寫全文，只輸出 JSON。

需要檢查：
- 人名或別名
- 身分證件
- 電話、Email、帳號
- 詳細地址
- 公司、學校、醫院、特定工作場所
- 特殊職稱
- 精確事件時間與地點
- 親屬關係搭配姓名
- 車牌、銀行帳號、社群 ID
- 案號、JID、可回查案件的 URL
- 其他可與公開資訊交叉識別個人的描述

輸出：
[
  {
    "span": "...",
    "type": "...",
    "risk": "low|medium|high",
    "reason": "..."
  }
]
```

例：

```json
[
  {
    "span": "○○科技股份有限公司總經理",
    "type": "WORKPLACE_ROLE",
    "risk": "high",
    "reason": "特定公司與高階職稱可能唯一識別自然人"
  }
]
```

**重要：** auditor 只回傳 span + type，由 deterministic code 決定如何替換。

不要讓 auditor 直接 rewrite 全文，避免法律語義被偷偷修改。

---

# 18. Stage 7：Final Sanitizer

LLM audit 完之後，再跑 deterministic sanitizer。

流程：

```text
LLM 找出疑似 leakage span
        ↓
規則判斷 replacement policy
        ↓
span-based replace
        ↓
再次掃描 identifier regex
```

對 high-risk span 應預設 fail-closed：

```text
無法安全泛化
→ 直接刪除該 span
```

若整份判決因特殊案件而高度可識別，建議直接排除，不必強行進 training set。

---

# 19. Stage 8：Leakage QA

每份 clean judgment 至少跑以下檢查。

## 19.1 Exact leakage

確保 clean text 不含：

```text
raw JID
完整案號
raw URL
原始人名
原始電話
原始 Email
原始身分證號
原始車牌
```

可以直接從 Entity Registry 反查：

```python

def assert_no_original_mentions(clean_text, entity_registry):
    leaks = []

    for ent in entity_registry:
        for mention in ent["mentions"]:
            if mention and mention in clean_text:
                leaks.append(mention)

    return leaks
```

任何 leakage → 該 document 不輸出。

---

## 19.2 Regex scan

clean text 再跑一次：

```text
TW ID regex
電話 regex
Email regex
IP regex
車牌 regex
案號 regex
URL regex
```

命中高風險 pattern → reject / review。

---

## 19.3 Name NER scan

對 clean text 再跑一次 person NER。

若還抓到大量疑似真實中文姓名：

```text
→ 人工抽查
或
→ 第二次 entity resolution
```

---

# 20. Semantic QA

除了 privacy，還要確定 anonymization 沒有破壞案件。

建議抽樣檢查：

```text
人物關係是否仍一致？
甲是否在不同段落突然變成不同人？
共同被告是否被錯誤 merge？
被害人與證人是否被錯誤 merge？
時間順序是否保留？
犯罪金額是否保留？
犯罪次數是否保留？
年齡／未成年資訊是否保留？
證據與人物 binding 是否保留？
罪名／法條／法院理由是否完全未改？
```

推薦讓另一個 model 比較 raw / clean，但只做「semantic consistency score」，不要讓它重新改寫。

例如：

```json
{
  "identity_consistency": true,
  "temporal_consistency": true,
  "evidence_binding_consistency": true,
  "legal_fact_loss": false,
  "possible_privacy_leak": false,
  "pass": true
}
```

---

# 21. Clean Dataset Schema

推薦 JSONL：

```json
{
  "doc_id": "random-or-hash-not-derived-from-jid",
  "court_level": "district_court",
  "case_domain": "criminal",
  "year_bucket": "2020s",
  "case_type": "fraud",
  "text": "...去識別化後判決全文...",
  "privacy": {
    "version": "deid-v1",
    "entity_count": 6,
    "audit_pass": true,
    "regex_pass": true,
    "semantic_pass": true
  }
}
```

clean corpus 不建議放：

```text
jid
完整案號
原始人名
raw URL
raw PDF URL
可直接回查案件的 source identifier
```

如果研究需要追蹤來源，請放在**另一份 restricted mapping DB**：

```text
internal_clean_doc_id ↔ source_jid
```

而不是跟 training data 放在一起。

---

# 22. Entity Registry 儲存政策

Entity Registry 含有：

```text
真實姓名 ↔ 甲乙丙丁
```

因此本身屬於高敏感的 linkage data。

建議：

```text
raw DB
entity registry
source mapping
```

都放 restricted storage。

而：

```text
clean corpus
```

與上述資料實體分離。

如果未來要公開 dataset，公開版本原則上不應提供 reversible mapping。

---

# 23. 建議技術選型

最小可行版本：

```text
Python
├─ requests/httpx          官方 API / data download
├─ orjson                  JSONL
├─ regex                   deterministic PII
├─ hashlib                 source/content hash
├─ pydantic                schema validation
├─ pandas / polars         inspection/statistics
└─ LLM API / local model   entity extraction + privacy audit
```

進階版可加：

```text
Chinese NER model
spaCy/custom matcher
Presidio-style PII pipeline
SQLite / PostgreSQL
Ray / multiprocessing
```

中文司法文書的格式很特殊，因此不建議直接相信通用英文 PII library 的預設規則。

---

# 24. Pipeline Pseudocode

```python
for raw_doc in judicial_source:

    # 1. raw persistence
    save_raw(raw_doc)

    # 2. normalize
    text = normalize_text(raw_doc["raw_text"])

    # 3. extract structured header entities
    header_entities = extract_header_entities(text)

    # 4. NER / LLM entity extraction
    candidate_entities = extract_entities(text)

    # 5. entity resolution
    registry = resolve_case_entities(
        header_entities,
        candidate_entities,
        text,
    )

    # 6. assign aliases
    registry = assign_case_local_aliases(registry)

    # 7. span-based entity replacement
    clean = replace_entities_by_span(text, registry)

    # 8. deterministic PII masking / generalization
    clean = mask_identifiers(clean)
    clean = generalize_addresses(clean)
    clean = generalize_dates_if_safe(clean)
    clean = remove_case_number(clean)
    clean = remove_source_urls(clean)

    # 9. privacy audit
    findings = privacy_auditor(clean)
    clean = sanitize_audit_findings(clean, findings)

    # 10. final checks
    if contains_original_mentions(clean, registry):
        reject(raw_doc, "ENTITY_LEAK")
        continue

    if pii_regex_hit(clean):
        reject(raw_doc, "PII_REGEX_LEAK")
        continue

    if not semantic_consistency_pass(raw_doc["raw_text"], clean):
        reject(raw_doc, "SEMANTIC_DAMAGE")
        continue

    # 11. output
    save_clean(clean)
```

---

# 25. 推薦先做 MVP，再逐步提高 recall

第一版不要一次解決所有 edge case。

## MVP v0

先挑約 1,000–5,000 份刑事一審判決：

```text
下載
↓
header person extraction
↓
姓名甲乙丙丁化
↓
identifier regex
↓
地址／生日泛化
↓
案號 removal
↓
LLM privacy audit
↓
人工抽查 200–500 份
```

重點量測：

```text
PII Recall
Entity Resolution Accuracy
Alias Consistency
False Redaction Rate
Legal Semantic Preservation
Document Reject Rate
```

尤其要量 **Recall**：隱私資料漏掉一個通常比多遮一個更危險。

---

# 26. 建議建立 De-identification Benchmark

從 corpus 隨機抽一批判決，人工標註：

```text
PERSON
PERSON_ALIAS
ID_NUMBER
PHONE
EMAIL
ADDRESS
ORGANIZATION
SCHOOL
WORKPLACE
JOB_TITLE
VEHICLE_PLATE
BANK_ACCOUNT
SOCIAL_ID
EXACT_TIME
EXACT_LOCATION
CASE_NUMBER
SOURCE_URL
INDIRECT_IDENTIFIER
```

再用它來評估 de-id pipeline。

推薦主要指標：

```text
PII span recall
PII span precision
High-risk PII recall
Entity clustering accuracy
Alias consistency rate
Legal fact preservation rate
```

對 privacy pipeline，**high-risk recall 應優先於 precision**。

---

# 27. 不建議的做法

## 27.1 直接 regex 所有中文姓名

問題：

```text
一般詞彙、地名、公司名可能被誤判
王○○ / 王男 / 綽號等抓不到
同一人 aliases 無法 cluster
```

---

## 27.2 直接把全文丟給 LLM 說「幫我匿名」

問題：

```text
LLM 可能 paraphrase
可能改變數字
可能改變否認／承認語氣
可能改變證據與人物 binding
可能漏 PII
```

LLM 最適合做：

```text
entity detector
alias resolver
privacy auditor
```

而不是 unrestricted rewriter。

---

## 27.3 只遮姓名，不移除案號

這是極大的 re-identification leakage。

完整案號本身就是回查原始裁判的索引。

---

## 27.4 全部日期、地點、年齡都刪掉

會造成法律 reasoning 資訊嚴重損失。

應該採取：

```text
exact identifier
→ legally sufficient abstraction
```

而不是無條件 deletion。

---

# 28. 最終建議架構

```text
                 司法院 Open Data
                        │
                        ▼
                 Raw Judgment Store
                        │
                        ▼
                   Normalization
                        │
                        ▼
             Header / Entity Extraction
                        │
                        ▼
              Case Entity Resolution
                        │
                        ▼
                 Entity Registry
                        │
                        ▼
          Semantic-preserving Replacement
                        │
                        ▼
             Deterministic PII Masking
                        │
                        ▼
               LLM Privacy Auditor
                        │
                        ▼
                 Final Sanitizer
                        │
               ┌────────┴────────┐
               ▼                 ▼
          Privacy QA         Semantic QA
               └────────┬────────┘
                        ▼
                 Clean Judgment
                        │
                        ▼
             後續 Teacher / SFT Pipeline
```

---

# 29. 本專案建議的硬性規則

建議直接寫進 pipeline specification：

### MUST

- 每份判決建立案件級 entity registry。
- 同案件同人物 alias 必須一致。
- clean text 不得含完整案號／JID／source URL。
- 身分證、電話、Email、帳號、車牌等直接 identifier 必須完全遮蔽；但來源裁判已用全 `0` 表示的車牌或銀行帳號應原樣保留，不得再次替換成另一種標記。
- 人物姓名必須 pseudonymize。
- 法院原先配置的匿名代碼（例如 `A01`、`A02`）應原樣保留，不得納入人物 entity registry 或再次改碼。
- 帶有真實姓氏的遮蔽姓名（例如 `朱OO`、`朱ＯＯ`、`朱○○`）仍須正規化並改為案件級人物代號。
- 去識別化後必須做第二次 leakage scan。
- 原始文本與 clean text 必須分離儲存。
- source mapping / entity registry 不得與 training corpus 一起發布。

### SHOULD

- 地址泛化。
- 公司／學校／醫院等高識別性機構泛化。
- 精確生日轉換成年齡或年齡區間。
- 精確事件時間／地點依法律推理需求泛化。
- 使用 LLM / NER 做第二層 privacy audit。
- 人工建立小型 de-identification gold set。

### MUST NOT

- 不得只做姓名 regex replacement 就宣稱匿名完成。
- 不得讓 LLM 自由重寫整份判決來完成匿名。
- 不得在 clean training data 中留下可直接回查原裁判的 identifier。

---

# 30. 卷頁引用與證物欄位

判決正文中的「見偵卷第○頁」、「見本院卷第○頁」是來源定位資訊，不是法律事實或證據內容。clean text 應刪除卷名、偵查案號與頁碼，只保留其所支持的證據或量刑因素。括號、方頭括號、行內引用，以及不同法院使用的「警卷／偵卷／院卷／審理卷」寫法均適用；純交叉引用標記如「（如附件）」與「《附件》」亦應移除，但附件的實質文字不得因此刪除。

例如：

```text
身體狀況及罹患之疾病（見本院卷第41、43頁）等一切情狀
→ 身體狀況及罹患之疾病

告訴人林○○提供之對話紀錄文字檔各1份在卷可佐
（見○○警察局○○分局○○字第114...號刑案偵查卷第21至47頁）
→ 告訴人甲提供之對話紀錄文字檔可佐
```

同時在 clean record 建立結構化 `evidence`：

```json
{
  "evidence": [
    {
      "evidence_id": "E1",
      "name": "告訴人甲提供之對話紀錄文字檔",
      "content": null,
      "content_status": "not_publicly_available",
      "source": "judgment_text_reference"
    }
  ]
}
```

資料來源必須嚴格區分：

- 司法院裁判書 API 的 `ATTACHMENTS` 是裁判書附檔，例如附表 PDF，不等於偵查卷或本院卷。
- 公開裁判查詢頁若只有卷頁引用，不得推定已取得原始證物。
- 僅在合法、經授權且實際取得卷證內容時，才可填入 `content`，並記錄來源與內容 hash。
- 無法取得時，以去識別化後的證物名稱作為 fallback，`content` 必須維持 `null`。
- 偵查卷、本院卷與人工審查資料均視為 restricted data，不得與 clean corpus 一起發布。

---

# 31. 建議下一步

完成本文件中的 pipeline 後，再進入下一階段：

```text
Clean Judgment
       ↓
判決結構解析
       ↓
事實 / 爭點 / 證據 / 辯方主張 / 法院認定
       ↓
Judge rationale generation
       ↓
rationale verification
       ↓
SFT / distillation data
```

這樣可以確保 teacher model 從一開始看到的就是去識別化資料，降低後續 rationale 再次洩漏個資的風險。

---

# 參考官方資料

1. 司法院資料開放平臺  
   <https://opendata.judicial.gov.tw/>

2. 司法院裁判書開放 API 規格說明（114.08.22 版）  
   <https://opendata.judicial.gov.tw/news/detail?newsId=3041>

3. 司法院裁判書系統－系統說明  
   <https://judgment.judicial.gov.tw/readme.aspx>

4. 司法院裁判書遮隱規則  
   <https://judgment.judicial.gov.tw/cover_rule/cover_rule.html>

> 法律與隱私提醒：公開裁判書不代表其中資訊可不受限制地重新處理、散布或重新識別。若 corpus 將對外發布，或將使用於公開模型訓練，應另外確認個資保護、資料授權、研究倫理與機構規範；去識別化也應以實際 re-identification risk 測試，而非僅以「姓名已遮蔽」作為完成標準。
