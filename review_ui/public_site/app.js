const $ = (id) => document.getElementById(id);
const STORAGE_KEY = "lawrence_public_reviews_v1";
const state = {
  all: [], filtered: [], current: 0,
  reviews: JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"),
};
const escapeHtml = (value) => String(value ?? "").replace(
  /[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]
);
const decisionOf = (id) => state.reviews[id]?.decision || "pending";

function applyFilter() {
  const query = $("search").value.trim().toLowerCase();
  const decision = $("decisionFilter").value;
  state.filtered = state.all.filter((item) =>
    (!query || item.case_type.toLowerCase().includes(query) || item.doc_id.includes(query)) &&
    (decision === "all" || decisionOf(item.doc_id) === decision)
  );
  state.current = Math.min(state.current, Math.max(0, state.filtered.length - 1));
  renderList(); renderCase();
}

function renderList() {
  const counts = {pending:0, pass:0, fail:0, follow_up:0};
  state.all.forEach((item) => counts[decisionOf(item.doc_id)]++);
  $("stats").textContent = `共 ${state.all.length} 筆 · 未審 ${counts.pending} · 通過 ${counts.pass} · 退回 ${counts.fail} · 待確認 ${counts.follow_up}`;
  $("caseList").innerHTML = state.filtered.map((item, index) =>
    `<div class="case-item ${index === state.current ? "active" : ""}" data-index="${index}">` +
    `<b>${escapeHtml(item.case_type || "未分類")}</b>` +
    `<span>${escapeHtml(item.doc_id.slice(0, 12))} · ${{pending:"未審",pass:"通過",fail:"退回",follow_up:"待確認"}[decisionOf(item.doc_id)]}</span></div>`
  ).join("");
  document.querySelectorAll(".case-item").forEach((element) => {
    element.onclick = () => { state.current = Number(element.dataset.index); renderList(); renderCase(); };
  });
}

function renderCase() {
  const item = state.filtered[state.current];
  $("workspace").hidden = !item; $("empty").hidden = Boolean(item);
  if (!item) { $("empty").innerHTML = "<h2>沒有符合條件的案件</h2>"; return; }
  $("position").textContent = `第 ${state.current + 1} / ${state.filtered.length} 筆`;
  $("caseType").textContent = item.case_type || "未分類";
  $("docId").textContent = item.doc_id;
  $("summaryCount").textContent = `${item.crime_facts_summary.length} 段`;
  $("summaries").innerHTML = item.crime_facts_summary.map((value, index) =>
    `<div><b>事實 ${index + 1}</b><span>${escapeHtml(value)}</span></div>`
  ).join("") || "<p>未抽取摘要</p>";
  $("evidenceCount").textContent = `${item.evidence.length} 項`;
  $("evidence").innerHTML = item.evidence.map((value, index) =>
    `<div><b>證據 ${index + 1}</b><span>${escapeHtml(value.name)}${value.quantity ? `（${escapeHtml(value.quantity)}）` : ""}<br>${escapeHtml(value.proves || "")}</span><em>${escapeHtml(value.category || "")}</em></div>`
  ).join("") || "<p>未抽取證據</p>";
  $("originalText").textContent = item.original_text || "（無原文）";
  $("cleanText").textContent = item.text;
  const review = state.reviews[item.doc_id] || {};
  document.querySelectorAll('input[name="decision"]').forEach((element) => element.checked = element.value === review.decision);
  $("severity").value = review.severity || "none"; $("notes").value = review.notes || "";
  $("saveStatus").textContent = review.updated_at ? `已儲存 ${review.updated_at}` : "尚未審查";
  $("previous").disabled = state.current === 0; $("next").disabled = state.current === state.filtered.length - 1;
  location.hash = item.doc_id;
}

function saveReview() {
  const item = state.filtered[state.current];
  const decision = document.querySelector('input[name="decision"]:checked')?.value;
  if (!decision) { alert("請先選擇審查結果"); return; }
  state.reviews[item.doc_id] = {decision, severity:$("severity").value, notes:$("notes").value, updated_at:new Date().toISOString()};
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.reviews));
  renderList();
  if (state.current < state.filtered.length - 1) { state.current++; renderList(); renderCase(); }
}

function exportReviews() {
  const blob = new Blob([JSON.stringify({exported_at:new Date().toISOString(), reviews:state.reviews}, null, 2)], {type:"application/json"});
  const link = document.createElement("a"); link.href = URL.createObjectURL(blob);
  link.download = `judgment_reviews_${new Date().toISOString().slice(0, 10)}.json`; link.click(); URL.revokeObjectURL(link.href);
}
async function importReviews(file) {
  const data = JSON.parse(await file.text()); state.reviews = {...state.reviews, ...(data.reviews || {})};
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.reviews)); applyFilter();
}

$("search").oninput = applyFilter; $("decisionFilter").onchange = applyFilter;
$("previous").onclick = () => { state.current--; renderList(); renderCase(); };
$("next").onclick = () => { state.current++; renderList(); renderCase(); };
$("save").onclick = saveReview; $("exportButton").onclick = exportReviews;
$("importButton").onclick = () => $("importFile").click();
$("importFile").onchange = (event) => event.target.files[0] && importReviews(event.target.files[0]);
fetch("cases.json", {cache:"no-store"}).then((response) => {
  if (!response.ok) throw Error(response.status); return response.json();
}).then((data) => {
  state.all = data.cases || []; const requested = location.hash.slice(1);
  state.current = Math.max(0, state.all.findIndex((item) => item.doc_id === requested)); applyFilter();
}).catch((error) => { $("empty").innerHTML = `<h2>資料載入失敗</h2><p>${escapeHtml(error.message)}</p>`; });
