# 起訴書語料去識別化

主資料每行一個 JSON 物件，代表一份起訴書。民國 114、115 年來源分別存放在 `data/raw/indictments/indictments_114_source.jsonl` 與 `data/raw/indictments/indictments_115_source.jsonl`；實際納入時仍以 `investigation.year_roc` 嚴格過濾，不只依檔名推定年份。

## 目前第一輪資料集

目前先使用 `indictments_114_source.jsonl` 中 `investigation.year_roc == 114` 的資料，共 940 筆、19 個地檢署。原始抽樣已按地檢署及案由多樣性配置，因此第一輪保留全部 940 筆，不再對大型地檢署重複降採樣。其他年份及民國 115 年來源暫不納入。

執行 normalization、去識別化與稽核：

```powershell
python -X utf8 scripts/deidentify_judgments.py
```

日後處理民國 115 年時可使用 `python -X utf8 scripts/deidentify_judgments.py --input data/raw/indictments/indictments_115_source.jsonl --year-roc 115`，輸出檔名會依年份自動切換，不會覆蓋 114 年資料。

主要輸出：

```text
data/intermediate/normalized/indictments_114_normalized.jsonl
data/intermediate/entity_registry/indictments_114_entities.jsonl
data/intermediate/audit/indictments_114_audit.jsonl
data/clean/indictments/indictments_114_clean.jsonl
data/clean/indictments/indictments_114_clean_manifest.json
```

第二輪篩選應以人工檢閱結果、稀有案由是否過度可識別、文字長度及去識別化稽核結果為依據，而不是再次按地檢署平均抽樣。檢閱 UI 的預設 100 筆會先選案由總數不超過 2 筆的長尾文件，再選全文長度最高的 5%，最後用固定 seed 補足名額。

## 下載起訴書對應判決

114、115 年來源資料的 `judgment.url` 是司法院案號查詢頁，不一定直接指向單一裁判；同一案號可能同時列出判決與後續裁定。下載程式會先開啟查詢結果清單，再依來源的 `judgment.date` 精確選取對應裁判。只連線司法院官方網站，請保留請求間隔：

```powershell
python -X utf8 scripts/download_linked_judgments.py --years 114 115 --delay-seconds 3
```

每抓完一份即原子寫入 checkpoint，不必等整批結束；重新執行同一命令會跳過完成項目。原始 HTML、逐筆 metadata、錯誤紀錄及年度 JSONL manifest 分別位於：

```text
data/raw/linked_judgments/{114,115}/html/
data/raw/linked_judgments/{114,115}/metadata/
data/raw/linked_judgments/{114,115}/errors/
data/raw/linked_judgments/linked_judgments_{114,115}.jsonl
```

年度是指起訴書來源年度；實際裁判日期可能落在次年。輸出保留其所連結的起訴書條碼與偵查案號，方便後續一對多或多對一合併。

## 判決書去識別化研究

判決書沿用「LLM 只產生分析計畫、程式做確定性替換」的架構，但使用判決書專用 prompt 與章節解析。`scripts/hybrid_deidentify_judgments_with_google_ai.py` 依 barcode 尋找已完成的配對起訴書，要求模型以 `linked_indictment_group_id` 回連人物，並由 `scripts/deidentify_linked_judgments.py` 確定性渲染。流程會移除司法院頁面介面、支援上訴人／聲請人／相對人／受刑人等角色、統一判決原有 A01/A02 人物代號、解析判決與裁定章節，並在保留其他裁判引用案號前強制遮蔽本案及合併審理案號。每筆成功後立即原子寫入 checkpoint，另輸出保留 indictment/judgment provenance 的 `paired_crime_facts_summary` 與 `paired_evidence`；中介檔含原始人物對照，必須維持 restricted。

兩把 key 分工執行範例：

```powershell
python -X utf8 scripts/hybrid_deidentify_with_google_ai.py --scope errors --limit 1000 --key-name GOOGLE_STUDIO_API_KEY
python -X utf8 scripts/hybrid_deidentify_judgments_with_google_ai.py --scope pending --limit 1000 --key-name GOOGLE_STUDIO_API_KEY_2
```

判決書命令省略 `--key-name` 時會載入 `.env` 內所有編號 key，以固定 worker 平行處理並由主執行緒統一寫入 checkpoint；也可重複指定 `--key-name` 只啟用選定的多把 key。

同一案件的跨文件人物代號與事證整合由 `scripts/pair_case_integration.py` 負責。起訴書分析先建立受限的 pair alias registry，保存人物 `group_id`、姓名變體與已渲染 alias；判決書分析日後應以 `linked_indictment_group_id` 明確連回同一人物，確定性渲染會強制沿用起訴書 alias。完全相同或 `OO/○○` 差異且沒有歧義的姓名可自動連結，同名歧義不得猜測。公開 pair 輸出只保留 `pair_person_id`、角色與 alias，不包含原始姓名。兩份文件的犯罪事實與證據採聯集、保留 indictment/judgment provenance；只有 canonical key 或正規化內容相同時才自動去重，避免把不同筆錄或不同物證錯誤合併。公開 `pair_id` 與判決書 `doc_id` 使用不可由來源反推的 clean ID；首次建立後必須在受限對照表保存並於重跑時傳回，才能保持穩定。

## 目前正式方法：

人工試驗後採用單次分析的混合流程，不讓 LLM 自由改寫起訴書全文：

1. `deidentify_judgments.py` 先正規化原始資料、建立案件內人物／機構 registry，並產生初步 clean corpus 與規則稽核。
2. 每份文件只呼叫一次 Google AI Studio 的 `gemma-4-31b-it`。模型只輸出 JSON 分析計畫：人物同一性與角色、私人機構、識別資訊、犯罪事實摘要，以及包含供述、證詞、書證、照片、數位證據、扣押文件、鑑定報告與實體物的完整證據清單。
3. Python 依分析計畫在原文上做確定性替換；模型沒有權限增刪或改寫原文。被告依案件內出現順序統一為 `被告甲`、`被告乙`，其他案件關係人依角色使用穩定代號，並避免產生「被告被告甲」。檢察官、書記官、法官、辯護人／律師、訴訟代理人與員警等專業人員則保留原職稱，姓名一律改成 `〇〇〇`，例如 `檢察官〇〇〇`、`書記官〇〇〇`，不使用甲乙編號。
4. 規則 registry 會補捉模型偶爾漏掉的部分遮罩姓名；`朱OO`、`林○○`、`林○浚` 等仍帶姓氏的名稱必須換成代號。原文既有的 `A01`、`A02`、甲男、乙女及純暱稱維持不變。
5. 政府機關、法院、地檢署、銀行及郵政名稱保留；涉案私人公司、商號、醫院、實驗室與其他私人組織換成機構代號。
6. 來源已用 `○／〇／Ｏ` 或全零形式遮罩的地址、公文識別、帳號、電話及車牌原樣保留。真正未遮罩的私人電話、證號、非全零帳號／車牌、本案案號及精確私人地址才處理；引用其他裁判的案號保留。
7. `crime_facts` 保存去識別化後的完整犯罪事實原段；`crime_facts_summary` 只整理犯罪構成事實；`evidence` 每一種證據各列一項，盡量保留原文明示數量與證明事項。卷頁引用不當成證據內容，找不到卷宗附件時只保存證據名稱。

目前 114 年 940 筆資料的剩餘案件可用下列命令處理：

```powershell
python -X utf8 scripts/hybrid_deidentify_with_google_ai.py --scope pending --limit 940 --delay-seconds 8
```

若 `.env` 同時設定 `GOOGLE_STUDIO_API_KEY`、`GOOGLE_STUDIO_API_KEY_2`（以及後續編號的 key），程式會為每把 key 建立一個固定 worker 平行處理。這些 key 應分屬不同 Google Cloud 專案，才會有各自的配額；主執行緒仍逐筆集中寫入 checkpoint，因此不要另外啟動第二份批次程序。`--delay-seconds` 是每個 worker 各自的最小請求間隔。

`--scope pending` 只選尚未有相同輸入 checkpoint 的文件。每完成一筆，程式便先寫入暫存檔，再以原子替換更新 `data/intermediate/google_ai/hybrid_gemma4_experiment.jsonl`；每筆包含輸入 SHA-256 與 `checkpointed_at`。程序、終端或對話中斷後執行同一命令即可續跑，已完成且輸入未變的文件不會再次送出。API 成功但沒有候選內容的單筆回應會原子記錄在 `data/intermediate/google_ai/hybrid_gemma4_failures.jsonl` 並跳過，不會卡死整批；之後可用 `--scope errors` 單獨重試。`--scope rendered` 只用既有分析重新套用最新確定性規則，不送 API。`--scope failed` 用於重新渲染所有人工退回案件；沒有輸入變更時會沿用既有 LLM 分析，不產生 API 請求。不要在批次續跑時使用 `--force`，因為它會重新送出既有文件。

每個 worker 逐筆同步呼叫 API，同一把 key 的兩筆開始時間至少間隔 8 秒。遇到 429 時，只有該 key 的 worker 依序等待 5、10、20、40、60 分鐘，之後每小時重試同一筆；因為 429 通常是該專案配額限制，直接換下一筆沒有幫助。單筆遇到暫時性 5xx、網路中斷或讀取逾時時，預設最多嘗試 4 次（可用 `--max-transient-attempts` 調整），之後原子記錄為 `transient_api_error` 並釋放 worker 處理下一筆；可稍後用 `--scope errors` 重試。模型 allowlist 僅允許本專案採用的 Gemma 免費型號，批次上限 1,000 筆，不會自動切換付費模型。API key 只從 `.env` 的 `GOOGLE_STUDIO_API_KEY` 與其編號變體讀取，不寫入輸出或日誌。執行前仍應確認各 Google AI Studio 專案維持 Free Tier，因本地程式無法替使用者變更或保證帳戶計費設定。

混合中介檔包含模型看到的身分關聯線索，屬 restricted。已完成的混合結果會在檢閱 UI 中優先顯示其全文、犯罪事實摘要與證據，但人工通過前不會覆蓋 canonical clean corpus。

啟動本地人工檢閱 UI：

```powershell
python -X utf8 review_ui/server.py
```

起訴書與判決書整合案件配對審核台使用獨立 port 與資料庫：

```powershell
python -X utf8 review_ui/pair_server.py
```

開啟 <http://127.0.0.1:8766>。介面會同時顯示兩份文書的正規化原文、去識別化文本、跨文書人物代號、整合犯罪事實與保留來源的事證物證；判決批次執行期間可在頁面同步最新 checkpoint。側欄可依 Gemini audit 的 Pass、Review、Fail、API failure 或尚未審核篩選，案件頁會分區呈現模型對犯罪事實、事證物證、人物代號與跨元件一致性的 finding，以及獨立的確定性規則命中。Gemini 結論僅供比較，不會覆蓋人工判定。審核紀錄保存於 `data/review/pair_reviews.sqlite3`，內容屬 restricted。

## 起訴書／判決書配對 Audit LLM

`scripts/audit_case_pairs_with_google_ai.py` 使用獨立的 Gemini 模型，同一次請求比較成對原文、去識別化全文、犯罪事實、各文書證據清單、整合證據與人物代號。它檢查事實正確性與隱私、證據是否受原文支持及完整收錄、會影響裁判理解的代號混淆、跨元件代號一致性；不把未洩漏本名的暱稱當成個資，也不會直接改寫任何既有結果。程式另以確定性規則掃描已知原名、直接識別碼與重複人物代號；規則或模型任一警示都不會自動通過。Gemini 3.8 Flash 不支援 `minimal` thinking level，因此 audit 預設使用 `medium`；需要降低延遲時可指定 `--thinking-level low`。

判決批次完成後執行：

```powershell
python -X utf8 scripts/audit_case_pairs_with_google_ai.py --scope pending --limit 1000
```

正式送出前可先用 `--dry-run` 檢查可配對數、prompt 長度與確定性警示數；此模式不讀取 key，也不呼叫 API：

```powershell
python -X utf8 scripts/audit_case_pairs_with_google_ai.py --dry-run --limit 1000
```

省略 `--key-name` 會以 `.env` 中不同專案的 key 建立固定平行 worker。每筆 audit 都原子 checkpoint 到 `data/intermediate/google_ai/pair_audit_gemini38.jsonl`；空回應、無效 JSON、過長 prompt 或連續暫時錯誤另存 failures，之後可用 `--scope errors` 重試。429 配額回應依 Google 的 `Retry-After`／訊息提示做固定短等待，不會與 500/503 共用指數退避；500/503 與網路錯誤則有限次退避，達上限便釋放 worker 處理下一組。Audit 輸入與 finding 可能含原始身分片段，因此輸出標為 restricted，不可直接放進公開訓練 corpus。免費層的可用模型、配額與資料使用條款可能變動，正式執行前仍須在 Google AI Studio 確認。

## 靜態檢閱網站

臺大個人網頁空間只提供靜態檔案。依資料擁有者確認，公開版可將來源網站本已公開的正規化原文置於左側，並在右側顯示 hybrid 去識別化候選文本，供逐筆對照；仍不包含人工舊備註、實體 registry、SQLite 或任何 API key。它只匯出 hybrid 已完成且本機尚未標記 `pass` 的案件、摘要與證據。網站上的審查紀錄保存在瀏覽器 `localStorage`，必須使用頁面上的按鈕匯出 JSON 才能帶回本機或換瀏覽器。

```powershell
python -X utf8 review_ui/export_public_site.py
python -X utf8 scripts/deploy_ntu_homepage.py --check
python -X utf8 scripts/deploy_ntu_homepage.py --replace
```

部署使用 FTP over TLS，只允許遠端 `public_html` 作為刪除與上傳根目錄。`--check` 僅驗證登入並列出檔名；`--replace` 才會刪除原網站內容並上傳 `review_ui/public_site`。密碼只從 FileZilla 已儲存站台或本機 `.env` 的 `NTU_FTPS_PASSWORD` 讀取，不可提交或寫入網站檔案。臺大目前只允許從校內網路以 FTP 更新個人網頁。

## 頂層

| 欄位 | 型別 | 說明 |
|---|---|---|
| `barcode` | str | 書類唯一條碼（法務部系統內的識別碼，最適合當主鍵去重） |
| `detail_id` | str | `detail.jsp?d=` 的 hex 值 |
| `detail_url` | str | 該書類全文頁完整網址 |
| `doc_type` | str | 書類類別，例：`起訴書`（理論上此次查詢都會是起訴書） |
| `title` | str | 全文頁標題，例：`法務部…-臺灣基隆地方檢察署 起訴書 114年度毒偵字第130號` |
| `text` | str | **起訴書全文**，已去除法律名詞的 hover 註解彈窗、正規化空白 |
| `text_len` | int | `text` 字數 |
| `law_articles` | list[str] | 全文中出現的法條引用（白名單比對），例：`["毒品危害防制條例第10條","刑法第47條"]` |
| `fetched_at` | str | 抓取時間 UTC ISO8601 |

## `agency`（承辦地檢）

| 欄位 | 型別 | 說明 |
|---|---|---|
| `name` | str | 全名，例：`臺灣基隆地方檢察署` |
| `short` | str | 簡稱，例：`基隆地檢` |
| `code` | str \| null | 查詢系統的機關代碼，例：`11` |
| `g_index` | int | 該次查詢結果側邊欄的分組索引 |

## `charge`（案由）

| 欄位 | 型別 | 說明 |
|---|---|---|
| `raw` | str | 結果頁原始案由，例：`違反毒品危害防制條例` |
| `normalized` | str | 標準化後（去「違反」「罪」、正規化），例：`毒品危害防制條例`。**抽樣分層依據** |
| `detail_page` | str \| null | 全文頁 metadata 區塊的案由（通常同 `raw`，偶有出入） |

## `investigation`（偵查案件）

| 欄位 | 型別 | 說明 |
|---|---|---|
| `seq_in_agency` | int | 在該地檢查詢結果中的序號（1 起） |
| `case_no` | str | 偵查案號（結果頁），格式 `年,字,號`，例：`114,毒偵,130` |
| `case_no_detail` | str \| null | 全文頁 metadata 的偵查案號（互相補洞用） |
| `header_case_nos` | list[str] | 起訴書抬頭列出的**所有**案號（含併案），例：`["114,毒偵,130","114,偵,2463"]` |
| `year_roc` | int | 民國年（由 `case_no` 拆出），例：`114` |
| `year_ad` | int | 西元年，例：`2025` |
| `prefix` | str | 案號字別，例：`毒偵`、`偵` |
| `number` | int | 案號號次，例：`130` |
| `close_date_roc` | str | 偵結日期（民國），例：`114-04-08` |
| `close_date` | str \| null | 偵結日期（西元），例：`2025-04-08` |

> 註：`year_roc / year_ad / prefix / number` 只有在 `case_no` 能成功拆解時才會出現。

## `judgment`（第一審裁判）— 整個物件可能是 `null`

當該筆還沒有第一審裁判時，`judgment` = `null`。

| 欄位 | 型別 | 說明 |
|---|---|---|
| `case_no` | str \| null | 裁判案號，例：`114,基簡,718` |
| `date_roc` | str \| null | 裁判日期（民國），例：`114-12-30` |
| `date` | str \| null | 裁判日期（西元） |
| `url` | str | 司法院裁判書查詢系統對應連結 |
| `court_code` | str \| null | 法院代碼，例：`KLD` |
| `court_name` | str \| null | 法院中文名（由代碼對照），例：`基隆地院` |
| `sys` | str \| null | 裁判系統別參數（`jud_sys`），例：`M`（刑事） |
| `year` | int \| null | 裁判案號的年 |
| `case_word` | str \| null | 裁判案號的字別，例：`基簡` |
| `number` | int \| null | 裁判案號的號次 |

## `sampling`（抽樣資訊，供追溯）

| 欄位 | 型別 | 說明 |
|---|---|---|
| `agency_quota` | int | 該地檢的目標抽樣數（通常 50，資料不足時較小） |
| `charge_alloc` | int | 該筆所屬案由在此地檢分配到的名額數 |

資料欄位的說明，然後這邊的取法是先選年份，然後把有100筆以上資料的地檢署的前500筆(系統限制只能看前500筆)選出盡量不一樣的案由總共50筆
