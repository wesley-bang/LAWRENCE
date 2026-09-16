const ISSUE_LABELS = {
  missed_person: "遺漏人名", missed_organization: "遺漏機構／場所", missed_contact: "遺漏聯絡資訊",
  missed_address: "遺漏地址", missed_datetime: "日期／時間過精確", inconsistent_alias: "代號不一致",
  over_redaction: "過度遮罩", semantic_loss: "法律語意受損", formatting: "格式／斷行問題", other: "其他"
};

const state = { scope: "sample", status: "pending", type: "", q: "", cases: [], current: -1, stats: null };
const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

function toast(message, error = false) {
  const el = $("toast"); el.textContent = message; el.className = error ? "show error" : "show";
  clearTimeout(toast.timer); toast.timer = setTimeout(() => el.className = "", 2600);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "操作失敗");
  return data;
}

function issueOptions() {
  $("issueOptions").innerHTML = Object.entries(ISSUE_LABELS).map(([value, label]) =>
    `<label><input type="checkbox" value="${value}"><span>${label}</span></label>`).join("");
}

async function refreshStats() {
  const stats = await api(`/api/stats?scope=${state.scope}`); state.stats = stats;
  const done = stats.total - stats.pending;
  $("doneCount").textContent = done; $("pendingCount").textContent = stats.pending;
  $("passCount").textContent = stats.passed; $("failCount").textContent = stats.failed;
  $("progressFill").style.width = `${stats.total ? done / stats.total * 100 : 0}%`;
  $("sampleSize").value = stats.sample_size; $("sampleSeed").value = stats.sample_seed;
}

async function refreshCases(preferredId = null) {
  const query = new URLSearchParams({scope: state.scope, status: state.status, limit: "1000"});
  if (state.type) query.set("case_type", state.type);
  if (state.q) query.set("q", state.q);
  const result = await api(`/api/cases?${query}`); state.cases = result.cases;
  if ($("typeFilter").options.length <= 1) {
    result.case_types.forEach(item => $("typeFilter").add(new Option(`${item.case_type} (${item.count})`, item.case_type)));
  }
  $("queueCount").textContent = `${state.cases.length} 筆`;
  renderList();
  if (!state.cases.length) { state.current = -1; $("reviewView").hidden = true; $("emptyState").hidden = false; return; }
  const wanted = preferredId ? state.cases.findIndex(x => x.doc_id === preferredId) : -1;
  state.current = wanted >= 0 ? wanted : Math.min(Math.max(state.current, 0), state.cases.length - 1);
  await openCase(state.current);
}

function renderList() {
  $("caseList").innerHTML = state.cases.map((item, index) => {
    const status = item.decision || "pending";
    const label = {pending:"待審", pass:"通過", fail:"退回", follow_up:"待確認"}[status];
    return `<button class="case-item ${index === state.current ? "active" : ""}" data-index="${index}">
      <span class="case-number">${item.sample_rank ? `#${item.sample_rank}` : "—"}</span>
      <span><strong>${escapeHtml(item.case_type)}</strong><small>${escapeHtml(item.source_id)}</small></span>
      <em class="review-status ${status}">${label}</em></button>`;
  }).join("");
  document.querySelectorAll(".case-item").forEach(el => el.addEventListener("click", () => openCase(Number(el.dataset.index))));
}

function highlighted(text, terms, cssClass) {
  if (!terms.length) return escapeHtml(text);
  const sorted = [...new Set(terms.filter(Boolean))].sort((a,b) => b.length-a.length);
  const re = new RegExp(sorted.map(x => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|"), "g");
  let out = "", last = 0;
  for (const match of text.matchAll(re)) {
    out += escapeHtml(text.slice(last, match.index));
    out += `<mark class="${cssClass}">${escapeHtml(match[0])}</mark>`; last = match.index + match[0].length;
  }
  return out + escapeHtml(text.slice(last));
}

async function openCase(index) {
  if (index < 0 || index >= state.cases.length) return;
  state.current = index; renderList();
  const item = await api(`/api/cases/${encodeURIComponent(state.cases[index].doc_id)}`);
  $("emptyState").hidden = true; $("reviewView").hidden = false;
  $("caseType").textContent = item.case_type; $("sampleRank").textContent = item.sample_rank ? `抽樣 #${item.sample_rank}` : "非抽樣";
  $("sourceJid").textContent = item.source_id; $("docId").textContent = item.doc_id;
  const persons = item.entities.persons.flatMap(x => x.mentions || [x.canonical_name]);
  const orgs = item.entities.organizations.map(x => x.canonical_name);
  $("normalizedText").innerHTML = highlighted(item.normalized_text, [...persons, ...orgs], "pii");
  const aliases = [...item.entities.persons.map(x => x.alias), ...item.entities.organizations.map(x => x.alias)];
  $("cleanText").innerHTML = highlighted(item.clean_text, aliases, "alias");
  const audit = item.audit;
  $("auditSummary").innerHTML = `<span>規則檢查 <b>${audit.leakage.pass ? "通過" : "失敗"}</b></span>
    <span>語意檢查 <b>${audit.semantic.pass ? "通過" : "失敗"}</b></span>
    <span>長度比 <b>${audit.semantic.length_ratio}</b></span><span>實體 <b>${persons.length + orgs.length}</b></span>`;
  const modelAudit = item.model_audit || {status:"not_run", issues:[]};
  const modelIssues = modelAudit.issues || [];
  $("modelIssueCount").textContent = modelAudit.status === "complete"
    ? `${modelIssues.length} 項可行動警示（原始 ${modelAudit.raw_issue_count ?? modelIssues.length} 項）`
    : "尚未執行";
  $("modelIssueList").innerHTML = modelAudit.status === "complete"
    ? (modelIssues.length ? modelIssues.map(issue =>
      `<div><b>${escapeHtml(issue.category)}</b><span>「${escapeHtml(issue.span)}」— ${escapeHtml(issue.reason)}</span><em>信心 ${escapeHtml(issue.confidence)}</em></div>`
    ).join("") : `<p>本地模型未發現額外疑點。</p>`)
    : `<p>${modelAudit.error ? escapeHtml(modelAudit.error) : "此文件尚未經本地模型檢查。"}</p>`;
  $("evidenceCount").textContent = `${item.evidence.length} 項`;
  $("evidenceList").innerHTML = item.evidence.length ? item.evidence.map(e =>
    `<div><b>${escapeHtml(e.evidence_id)}</b><span>${escapeHtml(e.name)}</span><em>${e.content === null ? "起訴書提及；原始卷證未公開" : "已取得內容"}</em></div>`
  ).join("") : `<p>此起訴書未抽取到明確證物名稱。</p>`;
  const googleReview = item.google_ai_review || {status:"not_run"};
  $("googleAiStatus").textContent = googleReview.status === "complete"
    ? `完成 · 修正 ${googleReview.redaction_count || 0} 處`
    : "尚未執行";
  $("googleAiSummary").innerHTML = googleReview.status === "complete"
    ? `<p>模型：${escapeHtml(googleReview.model || "")}；身分關聯提示 ${googleReview.identity_resolution_count || 0} 項。${googleReview.requires_priority_review ? "此案修正量較大，請優先複核。" : "結果仍須人工確認。"}</p>`
    : `<p>此文件尚未送交 Google AI 精查。</p>`;
  const directLlm = item.direct_llm || {status:"not_run"};
  $("directExperimentPanel").hidden = directLlm.status !== "complete";
  if (directLlm.status === "complete") {
    const directResult = directLlm.result || {};
    const directEvidence = directResult.evidence || [];
    $("directLlmStatus").textContent = `${directLlm.model} · Restricted · ${directEvidence.length} 項證據`;
    $("directLlmResult").innerHTML = `<p class="experiment-warning">模型原始輸出，未經任何規則修補；可能仍含識別資訊。</p>
      <div class="direct-document">${escapeHtml(directResult.deidentified_text || "")}</div>
      ${directEvidence.map((e, i) => `<div><b>證據 ${i + 1}</b><span>${escapeHtml(e.name || "")} ${e.quantity ? `（${escapeHtml(e.quantity)}）` : ""}<br>${escapeHtml(e.proves || "")}</span><em>${escapeHtml(e.category || "")}</em></div>`).join("")}`;
  }
  const twoPass = item.two_pass_llm || {status:"not_run"};
  $("twoPassPanel").hidden = twoPass.status !== "complete";
  if (twoPass.status === "complete") {
    const finalResult = twoPass.result || {};
    const finalEvidence = finalResult.evidence || [];
    $("twoPassStatus").textContent = `${twoPass.model} · ${twoPass.final_stage} · repair ${twoPass.repair_rounds} 輪 · ${finalEvidence.length} 項證據`;
    $("twoPassResult").innerHTML = `<p class="experiment-warning">LLM 最終原始輸出；已通過機械閘門，但仍屬 Restricted 實驗資料。</p>
      <div class="direct-document">${escapeHtml(finalResult.deidentified_text || "")}</div>
      ${finalEvidence.map((e, i) => `<div><b>證據 ${i + 1}</b><span>${escapeHtml(e.name || "")} ${e.quantity ? `（${escapeHtml(e.quantity)}）` : ""}<br>${escapeHtml(e.proves || "")}</span><em>${escapeHtml(e.category || "")}</em></div>`).join("")}`;
  }
  const hybrid = item.hybrid_llm || {status:"not_run"};
  $("hybridPanel").hidden = hybrid.status !== "complete";
  if (hybrid.status === "complete") {
    const hybridEvidence = hybrid.evidence || [];
    $("hybridStatus").textContent = `${hybrid.model} · 內容比 ${hybrid.checks?.length_ratio ?? "?"} · ${hybridEvidence.length} 項證據`;
    $("hybridResult").innerHTML = `<p class="experiment-warning">此版本已直接套用到右側去識別化結果，以及下方的犯罪事實摘要與證據。Gemma 僅負責分析；全文仍由確定性程式替換。</p>`;
  }
  const crimeFacts = item.crime_facts || "";
  $("crimeFactsCount").textContent = crimeFacts ? `${crimeFacts.length} 字` : "未抽取";
  $("crimeFactsText").textContent = crimeFacts || "此文件尚未抽取犯罪事實原段。";
  const factSummaries = item.crime_facts_summary || [];
  const physicalEvidence = item.physical_evidence || [];
  $("physicalEvidenceCount").textContent = `${factSummaries.length} 段摘要 · ${physicalEvidence.length} 項證據`;
  const factsHtml = factSummaries.length
    ? factSummaries.map((fact, i) => `<div><b>事實 ${i + 1}</b><span>${escapeHtml(fact)}</span></div>`).join("")
    : `<p>尚無 AI 犯罪事實摘要。</p>`;
  const physicalHtml = physicalEvidence.length
    ? physicalEvidence.map((e, i) => `<div><b>證據 ${i + 1}</b><span>${escapeHtml(e.name || "")}${e.quantity ? `（${escapeHtml(e.quantity)}）` : ""}${e.proves ? `<br>${escapeHtml(e.proves)}` : ""}</span><em>${escapeHtml(e.category || e.source_span || "")}</em></div>`).join("")
    : `<p>未抽取到結構化證據。</p>`;
  $("structuredExtraction").innerHTML = factsHtml + physicalHtml;
  document.querySelectorAll('input[name="decision"]').forEach(x => x.checked = x.value === item.decision);
  $("severity").value = item.severity || "none"; $("notes").value = item.notes || "";
  document.querySelectorAll('#issueOptions input').forEach(x => x.checked = item.issues.includes(x.value));
  $("saveStatus").textContent = item.updated_at ? `上次儲存：${item.updated_at}` : "尚未審查";
  renderHistory(item.history); $("historyPanel").hidden = true;
  $("prevCase").disabled = index === 0; $("nextCase").disabled = index === state.cases.length - 1;
  document.querySelector('.case-item.active')?.scrollIntoView({block:"nearest"});
  window.scrollTo({top: 0, behavior: "smooth"});
}

function renderHistory(history) {
  $("historyPanel").innerHTML = history.length ? history.map(x => `<div><b>${({pass:"通過",fail:"退回",follow_up:"待確認"})[x.decision]}</b>
    <span>${escapeHtml(x.recorded_at)} · ${escapeHtml(x.reviewer || "未署名")}</span><p>${escapeHtml(x.notes || "（無備註）")}</p></div>`).join("") : "尚無修改歷史。";
}

async function saveReview() {
  const current = state.cases[state.current]; if (!current) return;
  const decision = document.querySelector('input[name="decision"]:checked')?.value;
  if (!decision) return toast("請先選擇通過、退回或待確認", true);
  const reviewer = $("reviewer").value.trim(); localStorage.setItem("judgmentReviewer", reviewer);
  const body = {decision, severity: $("severity").value, notes: $("notes").value, reviewer,
    issues: [...document.querySelectorAll('#issueOptions input:checked')].map(x => x.value)};
  try {
    await api(`/api/reviews/${encodeURIComponent(current.doc_id)}`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    toast("審查紀錄已儲存"); await refreshStats(); await refreshCases();
  } catch (error) { toast(error.message, true); }
}

function debounce(fn, delay=250) { let timer; return (...args) => { clearTimeout(timer); timer=setTimeout(() => fn(...args), delay); }; }

function bindEvents() {
  document.querySelectorAll("[data-scope]").forEach(button => button.addEventListener("click", async () => {
    state.scope = button.dataset.scope; document.querySelectorAll("[data-scope]").forEach(x => x.classList.toggle("active", x===button));
    await refreshStats(); await refreshCases();
  }));
  $("statusFilter").addEventListener("change", e => { state.status=e.target.value; refreshCases(); });
  $("typeFilter").addEventListener("change", e => { state.type=e.target.value; refreshCases(); });
  $("search").addEventListener("input", debounce(e => { state.q=e.target.value.trim(); refreshCases(); }));
  $("prevCase").addEventListener("click", () => openCase(state.current-1)); $("nextCase").addEventListener("click", () => openCase(state.current+1));
  $("saveReview").addEventListener("click", saveReview);
  $("historyToggle").addEventListener("click", () => $("historyPanel").hidden = !$("historyPanel").hidden);
  document.querySelectorAll("[data-copy]").forEach(button => button.addEventListener("click", async () => {
    await navigator.clipboard.writeText($(button.dataset.copy).innerText); toast("已複製文字");
  }));
  $("exportCsv").addEventListener("click", () => location.href=`/api/export?scope=${state.scope}&format=csv`);
  $("exportJsonl").addEventListener("click", () => location.href=`/api/export?scope=${state.scope}&format=jsonl`);
  $("sampleSettings").addEventListener("click", () => $("sampleDialog").showModal());
  $("sampleForm").addEventListener("submit", async event => {
    if (event.submitter?.value === "cancel") return;
    event.preventDefault();
    try { await api("/api/sample", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({size:Number($("sampleSize").value),seed:$("sampleSeed").value})});
      $("sampleDialog").close(); toast("抽樣清單已更新，既有審查紀錄仍保留"); await refreshStats(); await refreshCases();
    } catch(error) { toast(error.message,true); }
  });
  document.addEventListener("keydown", event => {
    if (event.target.matches("input,textarea,select")) { if ((event.ctrlKey||event.metaKey)&&event.key==="Enter") saveReview(); return; }
    if (event.key==="ArrowLeft") openCase(state.current-1); if (event.key==="ArrowRight") openCase(state.current+1);
    const values={"1":"pass","2":"fail","3":"follow_up"}; if(values[event.key]) document.querySelector(`input[name="decision"][value="${values[event.key]}"]`).checked=true;
  });
}

async function start() {
  issueOptions(); bindEvents(); $("reviewer").value=localStorage.getItem("judgmentReviewer")||"";
  try { await refreshStats(); await refreshCases(); } catch(error) { toast(error.message,true); }
}
start();
