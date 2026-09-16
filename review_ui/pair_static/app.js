const ISSUE_LABELS = {
  indictment_privacy: "起訴書仍有識別資訊",
  judgment_privacy: "判決書仍有識別資訊",
  inconsistent_alias: "跨文書代號不一致",
  facts_mismatch: "犯罪事實整合錯誤",
  evidence_missing: "事證／物證遺漏",
  evidence_bad_merge: "不同證據被錯誤合併",
  over_redaction: "過度遮罩",
  semantic_loss: "法律語意受損",
  formatting: "格式／斷行問題",
  other: "其他"
};

const ROLE_LABELS = {
  DEFENDANT:"被告", WITNESS:"證人", COMPLAINANT:"告訴人", VICTIM:"被害人",
  INVESTOR:"投資人", PROSECUTOR:"檢察官", CLERK:"書記官", JUDGE:"法官",
  DEFENSE_COUNSEL:"辯護人", POLICE:"員警", APPELLANT:"上訴人",
  PETITIONER:"聲請人", RESPONDENT:"相對人", SENTENCED_PERSON:"受刑人",
  LEGAL_REPRESENTATIVE:"法定代理人", PRIVATE_PROSECUTOR:"自訴人", OTHER:"人物"
};

const AUDIT_LABELS = {
  pass:"PASS", review:"REVIEW", fail:"FAIL", api_failure:"API FAILURE", unaudited:"尚未審核"
};
const AUDIT_SECTION_LABELS = {
  crime_facts:"犯罪事實", evidence:"事證與物證", aliases:"人物代號",
  cross_component_consistency:"跨元件一致性"
};
const FIELD_LABELS = {
  accuracy_issues:"正確性問題", privacy_leaks:"隱私洩漏",
  unsupported_items:"缺乏原文支持", missing_items:"遺漏項目",
  confusing_aliases:"混淆代號", cross_document_inconsistencies:"跨文書不一致",
  safe_nicknames_observed:"可保留暱稱", issues:"一致性問題",
  document:"文件", source_document:"原始來源", missing_from:"遺漏位置",
  kind:"類型", severity:"嚴重度", quote:"原文片段", source_quote:"來源片段",
  explanation:"說明", item:"項目", alias_or_pair:"代號",
  affected_legal_binding:"影響", pair_person_id:"人物 ID", components:"涉及元件"
};

const state = {status:"pending", audit:"all", type:"", q:"", cases:[], current:-1, item:null, evidenceFilter:"all"};
const $ = id => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

function toast(message, error=false) {
  const element = $("toast");
  element.textContent = message;
  element.className = error ? "show error" : "show";
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.className = "", 2800);
}

async function api(url, options={}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "操作失敗");
  return data;
}

function highlighted(text, terms, cssClass) {
  const values = [...new Set((terms || []).filter(value => String(value).length >= 2))]
    .sort((a,b) => b.length-a.length);
  if (!values.length) return escapeHtml(text);
  const pattern = values.map(value => String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
  const expression = new RegExp(pattern, "g");
  let output = "", cursor = 0;
  for (const match of String(text || "").matchAll(expression)) {
    output += escapeHtml(text.slice(cursor, match.index));
    output += `<mark class="${cssClass}">${escapeHtml(match[0])}</mark>`;
    cursor = match.index + match[0].length;
  }
  return output + escapeHtml(String(text || "").slice(cursor));
}

function sourceTags(provenance=[]) {
  return `<div class="source-tags">${provenance.map(source =>
    `<span class="source-tag ${source}">${source === "indictment" ? "起訴書" : "判決書"}</span>`
  ).join("")}</div>`;
}

function setupIssueOptions() {
  $("issueOptions").innerHTML = Object.entries(ISSUE_LABELS).map(([value,label]) =>
    `<label><input type="checkbox" value="${value}"><span>${label}</span></label>`
  ).join("");
}

async function refreshStats() {
  const stats = await api("/api/stats");
  const done = stats.total - stats.pending;
  $("totalCount").textContent = stats.total;
  $("pendingCount").textContent = stats.pending;
  $("passCount").textContent = stats.passed;
  $("failCount").textContent = stats.failed;
  $("auditPassCount").textContent = stats.audit_pass;
  $("auditReviewCount").textContent = stats.audit_review;
  $("auditFailCount").textContent = stats.audit_fail;
  $("evidenceTotal").textContent = stats.evidence_count;
  $("progressFill").style.width = `${stats.total ? done / stats.total * 100 : 0}%`;
}

async function refreshCases(preferredId=null) {
  const query = new URLSearchParams({status:state.status, audit:state.audit, limit:"2000"});
  if (state.type) query.set("case_type", state.type);
  if (state.q) query.set("q", state.q);
  const result = await api(`/api/cases?${query}`);
  state.cases = result.cases;
  const currentType = $("typeFilter").value;
  $("typeFilter").innerHTML = `<option value="">全部案由</option>` + result.case_types.map(item =>
    `<option value="${escapeHtml(item.case_type)}">${escapeHtml(item.case_type)} (${item.count})</option>`
  ).join("");
  $("typeFilter").value = currentType;
  $("queueCount").textContent = `${state.cases.length} 個案件 pair`;
  if (!state.cases.length) {
    state.current = -1; state.item = null;
    $("reviewView").hidden = true; $("emptyState").hidden = false;
    renderList(); return;
  }
  const wanted = preferredId ? state.cases.findIndex(item => item.pair_id === preferredId) : -1;
  state.current = wanted >= 0 ? wanted : Math.min(Math.max(state.current, 0), state.cases.length - 1);
  renderList();
  await openCase(state.current);
}

function renderList() {
  const labels = {pending:"待審",pass:"通過",fail:"退回",follow_up:"待確認"};
  $("caseList").innerHTML = state.cases.map((item,index) => {
    const status = item.decision || "pending";
    const auditStatus = item.audit_status || "unaudited";
    return `<button class="case-item ${index === state.current ? "active" : ""}" data-index="${index}">
      <span><strong>${escapeHtml(item.case_type || "未分類案件")}</strong><small>${escapeHtml(item.court_name)} · ${escapeHtml(item.court_case_no)}</small></span>
      <span class="status-stack"><em class="review-status ${status}">${labels[status]}</em><em class="audit-status ${auditStatus}">${AUDIT_LABELS[auditStatus]}</em></span>
      <span class="case-meta"><b>${item.fact_count} 事實</b><span>${item.evidence_count} 事證</span><span>${escapeHtml(item.document_type)}</span></span>
    </button>`;
  }).join("");
  document.querySelectorAll(".case-item").forEach(element =>
    element.addEventListener("click", () => openCase(Number(element.dataset.index)))
  );
}

function findingHtml(item) {
  if (typeof item !== "object" || item === null) return `<li>${escapeHtml(item)}</li>`;
  const parts = Object.entries(item).filter(([,value]) => value !== null && value !== "" && (!Array.isArray(value) || value.length));
  return `<li>${parts.map(([key,value]) => {
    const shown = Array.isArray(value) ? value.join("、") : value;
    const css = key === "severity" ? ` finding-severity ${escapeHtml(String(value))}` : "";
    return `<span class="finding-field${css}"><b>${escapeHtml(FIELD_LABELS[key] || key)}</b>${escapeHtml(shown)}</span>`;
  }).join("")}</li>`;
}

function renderAuditSection(name, section) {
  const lists = Object.entries(section || {}).filter(([key,value]) => key !== "status" && Array.isArray(value));
  const content = lists.map(([key,items]) => items.length
    ? `<div class="audit-finding-group"><h5>${escapeHtml(FIELD_LABELS[key] || key)} · ${items.length}</h5><ul>${items.map(findingHtml).join("")}</ul></div>`
    : ""
  ).join("");
  const status = section?.status || "pass";
  return `<article class="audit-section-card"><header><h4>${escapeHtml(AUDIT_SECTION_LABELS[name] || name)}</h4><span class="audit-status ${escapeHtml(status)}">${escapeHtml(String(status).toUpperCase())}</span></header>${content || '<p class="audit-clear">未回報問題</p>'}</article>`;
}

function renderAudit(item) {
  const status = item.audit_status || "unaudited";
  $("auditBadge").className = `audit-status ${status}`;
  $("auditBadge").textContent = AUDIT_LABELS[status] || status;
  $("auditMeta").textContent = [item.audit_model, item.audit_severity && `最高 ${item.audit_severity}`, item.audit_updated_at].filter(Boolean).join(" · ");
  if (status === "unaudited") {
    $("auditOverall").innerHTML = `<p>這組案件尚未經 Gemini 3.8 audit。</p>`;
    $("auditSections").innerHTML = "";
    $("auditRules").hidden = true;
    return;
  }
  if (status === "api_failure") {
    $("auditOverall").innerHTML = `<p><b>${escapeHtml(item.audit_error || "API failure")}</b>：${escapeHtml(item.audit_error_detail || "模型沒有完成本案審核")}</p>`;
    $("auditSections").innerHTML = "";
    $("auditRules").hidden = true;
    return;
  }
  const result = item.audit_result || {};
  const overall = result.overall || {};
  $("auditOverall").innerHTML = `<strong>Gemini 結論：${escapeHtml(AUDIT_LABELS[overall.decision] || overall.decision || status)}</strong>${(overall.reasons || []).length ? `<ul>${overall.reasons.map(reason => `<li>${escapeHtml(reason)}</li>`).join("")}</ul>` : ""}`;
  $("auditSections").innerHTML = Object.keys(AUDIT_SECTION_LABELS).map(name => renderAuditSection(name, result[name] || {})).join("");
  const checks = item.audit_checks || {};
  const ruleGroups = [
    ["原始人物名稱殘留", checks.exact_person_mention_leaks || []],
    ["識別碼格式命中", checks.identifier_pattern_hits || []],
    ["人物代號碰撞", checks.alias_collisions || []]
  ];
  $("auditRuleContent").innerHTML = `<p class="audit-rule-result ${checks.pass ? "pass" : "fail"}">${checks.pass ? "規則檢查通過" : "規則檢查有警示"}</p>` + ruleGroups.map(([label,items]) => items.length ? `<div class="audit-finding-group"><h5>${label} · ${items.length}</h5><ul>${items.map(findingHtml).join("")}</ul></div>` : "").join("");
  $("auditRules").hidden = false;
}

function renderFacts(facts) {
  $("factsList").innerHTML = facts.length ? facts.map((fact,index) => {
    const item = typeof fact === "string" ? {text:fact,provenance:[]} : fact;
    return `<div class="fact-card"><b>${String(index+1).padStart(2,"0")}</b> ${escapeHtml(item.text || "")}${sourceTags(item.provenance || [])}</div>`;
  }).join("") : `<p class="empty-copy">尚未抽取到犯罪事實。</p>`;
}

function renderEvidence() {
  const evidence = state.item?.evidence || [];
  const filtered = evidence.filter(item => {
    const sources = item.provenance || [];
    if (state.evidenceFilter === "all") return true;
    if (state.evidenceFilter === "both") return sources.includes("indictment") && sources.includes("judgment");
    return sources.includes(state.evidenceFilter);
  });
  $("evidenceList").innerHTML = filtered.length ? filtered.map(item => {
    const proves = Array.isArray(item.proves) ? item.proves.join("；") : (item.proves || "");
    return `<div class="evidence-item"><span class="eid">${escapeHtml(item.evidence_id || "—")}</span>
      <div><strong>${escapeHtml(item.name || "未命名證據")}</strong>${proves ? `<p>${escapeHtml(proves)}</p>` : ""}${sourceTags(item.provenance || [])}</div>
      <span class="category">${escapeHtml(item.category || "其他")}</span></div>`;
  }).join("") : `<p class="empty-copy">此篩選條件下沒有事證。</p>`;
}

function renderHistory(history) {
  const labels = {pass:"通過",fail:"退回",follow_up:"待確認"};
  $("historyPanel").innerHTML = history.length ? history.map(item =>
    `<div><b>${labels[item.decision]}</b><span>${escapeHtml(item.recorded_at)} · ${escapeHtml(item.reviewer || "未署名")}</span><p>${escapeHtml(item.notes || "（無備註）")}</p></div>`
  ).join("") : "尚無審查歷史。";
}

async function openCase(index) {
  if (index < 0 || index >= state.cases.length) return;
  state.current = index; renderList();
  const item = await api(`/api/cases/${encodeURIComponent(state.cases[index].pair_id)}`);
  state.item = item;
  $("emptyState").hidden = true; $("reviewView").hidden = false;
  $("caseType").textContent = item.case_type || "未分類";
  $("courtName").textContent = item.court_name || "法院未標示";
  $("documentType").textContent = item.document_type || "裁判書";
  $("courtCaseNo").textContent = item.court_case_no || "案件配對";
  $("pairId").textContent = item.pair_id;
  $("sourceUpdated").textContent = item.source_updated_at || "—";
  $("position").textContent = `${index+1} / ${state.cases.length}`;
  $("factCount").textContent = item.fact_count;
  $("evidenceCount").textContent = item.evidence_count;
  renderAudit(item);

  const entityData = item.pair_entities || {persons:[],indictment_mentions:[],judgment_mentions:[]};
  const persons = entityData.persons || [];
  $("entityList").innerHTML = persons.length ? persons.map(person =>
    `<span class="entity-chip"><b>${escapeHtml(person.alias || "人物")}</b>${escapeHtml(ROLE_LABELS[person.role] || person.role || "")}</span>`
  ).join("") : `<span class="hint">沒有配對人物</span>`;

  const aliases = persons.map(person => person.alias).filter(Boolean);
  $("indictmentRaw").innerHTML = highlighted(item.indictment_raw, entityData.indictment_mentions, "pii");
  $("judgmentRaw").innerHTML = highlighted(item.judgment_raw, entityData.judgment_mentions, "pii");
  $("indictmentClean").innerHTML = highlighted(item.indictment_clean, aliases, "alias");
  $("judgmentClean").innerHTML = highlighted(item.judgment_clean, aliases, "alias");
  ["indictmentRaw","indictmentClean","judgmentRaw","judgmentClean"].forEach(id => $(id).scrollTop = 0);

  renderFacts(item.crime_facts || []);
  state.evidenceFilter = "all"; $("evidenceFilter").value = "all"; renderEvidence();
  document.querySelectorAll('input[name="decision"]').forEach(input => input.checked = input.value === item.decision);
  $("severity").value = item.severity || "none";
  $("notes").value = item.notes || "";
  document.querySelectorAll("#issueOptions input").forEach(input => input.checked = (item.issues || []).includes(input.value));
  $("saveStatus").textContent = item.updated_at ? `上次儲存：${item.updated_at}` : "尚未審查";
  renderHistory(item.history || []); $("historyPanel").hidden = true;
  $("prevCase").disabled = index === 0; $("nextCase").disabled = index === state.cases.length-1;
  document.querySelector(".case-item.active")?.scrollIntoView({block:"nearest"});
  document.querySelector(".workspace")?.scrollTo({top:0,behavior:"smooth"});
}

async function saveReview() {
  if (!state.item) return;
  const decision = document.querySelector('input[name="decision"]:checked')?.value;
  if (!decision) return toast("請先選擇審查結論", true);
  const reviewer = $("reviewer").value.trim();
  localStorage.setItem("pairReviewReviewer", reviewer);
  const body = {
    decision, reviewer, severity:$("severity").value, notes:$("notes").value,
    issues:[...document.querySelectorAll("#issueOptions input:checked")].map(input => input.value)
  };
  try {
    await api(`/api/reviews/${encodeURIComponent(state.item.pair_id)}`, {
      method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)
    });
    toast("案件 pair 審查已儲存");
    await refreshStats(); await refreshCases();
  } catch (error) { toast(error.message, true); }
}

function setupSynchronizedScroll(kind) {
  const left = $(`${kind}Raw`), right = $(`${kind}Clean`);
  let locked = false;
  const sync = (source,target) => {
    if (locked || !document.querySelector(`[data-sync="${kind}"]`).checked) return;
    locked = true;
    const maximum = source.scrollHeight - source.clientHeight;
    const ratio = maximum > 0 ? source.scrollTop / maximum : 0;
    target.scrollTop = ratio * (target.scrollHeight - target.clientHeight);
    requestAnimationFrame(() => locked = false);
  };
  left.addEventListener("scroll", () => sync(left,right));
  right.addEventListener("scroll", () => sync(right,left));
}

function debounce(callback, delay=250) {
  let timer; return (...args) => { clearTimeout(timer); timer=setTimeout(() => callback(...args),delay); };
}

function bindEvents() {
  $("statusFilter").addEventListener("change", event => {state.status=event.target.value;refreshCases();});
  $("auditFilter").addEventListener("change", event => {state.audit=event.target.value;refreshCases();});
  $("typeFilter").addEventListener("change", event => {state.type=event.target.value;refreshCases();});
  $("search").addEventListener("input", debounce(event => {state.q=event.target.value.trim();refreshCases();}));
  $("evidenceFilter").addEventListener("change", event => {state.evidenceFilter=event.target.value;renderEvidence();});
  $("prevCase").addEventListener("click", () => openCase(state.current-1));
  $("nextCase").addEventListener("click", () => openCase(state.current+1));
  $("saveReview").addEventListener("click", saveReview);
  $("historyToggle").addEventListener("click", () => $("historyPanel").hidden = !$("historyPanel").hidden);
  $("syncData").addEventListener("click", async () => {
    const button = $("syncData"); button.disabled = true; button.textContent = "同步中…";
    try {
      const result = await api("/api/sync", {method:"POST",headers:{"Content-Type":"application/json"},body:"{}"});
      toast(`已同步 ${result.synced} 筆判決結果`); await refreshStats(); await refreshCases(state.item?.pair_id);
    } catch(error) { toast(error.message,true); }
    finally { button.disabled=false; button.textContent="同步最新結果"; }
  });
  document.querySelectorAll("[data-copy]").forEach(button => button.addEventListener("click", async () => {
    await navigator.clipboard.writeText($(button.dataset.copy).innerText); toast("已複製文本");
  }));
  $("exportCsv").addEventListener("click", () => location.href="/api/export?format=csv");
  $("exportJsonl").addEventListener("click", () => location.href="/api/export?format=jsonl");
  document.addEventListener("keydown", event => {
    if (event.target.matches("input,textarea,select")) {
      if ((event.ctrlKey||event.metaKey) && event.key === "Enter") saveReview();
      return;
    }
    if (event.key === "ArrowLeft") openCase(state.current-1);
    if (event.key === "ArrowRight") openCase(state.current+1);
    const choices={"1":"pass","2":"fail","3":"follow_up"};
    if (choices[event.key]) document.querySelector(`input[name="decision"][value="${choices[event.key]}"]`).checked=true;
  });
  setupSynchronizedScroll("indictment"); setupSynchronizedScroll("judgment");
}

async function start() {
  setupIssueOptions(); bindEvents();
  $("reviewer").value = localStorage.getItem("pairReviewReviewer") || "";
  try { await refreshStats(); await refreshCases(); }
  catch(error) { toast(error.message,true); }
}

start();
