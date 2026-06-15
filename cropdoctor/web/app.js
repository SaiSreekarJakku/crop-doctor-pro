// Crop Doctor Pro — frontend logic (no framework, no build step)
const $ = (id) => document.getElementById(id);
let currentFile = null;

const els = {
  dropzone: $("dropzone"), fileInput: $("fileInput"),
  previewWrap: $("previewWrap"), previewImg: $("previewImg"), changeBtn: $("changeBtn"),
  crop: $("crop"), region: $("region"), provider: $("provider"),
  minConf: $("minConf"), confVal: $("confVal"), temp: $("temp"), tempVal: $("tempVal"),
  btn: $("diagnoseBtn"), btnLabel: $("btnLabel"),
  placeholder: $("placeholder"), skeleton: $("skeleton"), result: $("result"),
  provBadge: $("provBadge"), modelBadge: $("modelBadge"), footModel: $("footModel"),
  toast: $("toast"),
};

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (p) => (p * 100).toFixed(1) + "%";

function toast(msg) {
  els.toast.textContent = msg;
  els.toast.classList.add("show");
  setTimeout(() => els.toast.classList.remove("show"), 3200);
}

// ---------- info / badges ----------
fetch("/api/info").then((r) => r.json()).then((info) => {
  els.provBadge.textContent = info.active_provider + " provider";
  const short = (info.model || "").split("/").pop();
  els.modelBadge.textContent = "🧠 " + short;
  els.footModel.textContent = info.model;
  // preselect ollama in dropdown if it's the active provider
  if (info.active_provider && [...els.provider.options].some(o => o.value === info.active_provider)) {
    // keep "auto" but reflect detection in badge only
  }
}).catch(() => {});

// ---------- file handling ----------
function setFile(file) {
  if (!file || !file.type.startsWith("image/")) { toast("Please choose an image file."); return; }
  currentFile = file;
  const url = URL.createObjectURL(file);
  els.previewImg.src = url;
  els.previewWrap.classList.add("show");
  els.dropzone.style.display = "none";
  els.btn.disabled = false;
}
els.dropzone.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", (e) => setFile(e.target.files[0]));
els.changeBtn.addEventListener("click", () => {
  els.previewWrap.classList.remove("show");
  els.dropzone.style.display = "";
  els.fileInput.value = "";
  currentFile = null; els.btn.disabled = true;
});
["dragenter", "dragover"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.remove("drag"); }));
els.dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
// paste from clipboard
window.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) setFile(item.getAsFile());
});

// ---------- sliders ----------
els.minConf.addEventListener("input", () => els.confVal.textContent = (+els.minConf.value).toFixed(2));
els.temp.addEventListener("input", () => els.tempVal.textContent = (+els.temp.value).toFixed(1));

// ---------- diagnose ----------
els.btn.addEventListener("click", async () => {
  if (!currentFile) return;
  els.btn.disabled = true;
  els.btnLabel.innerHTML = '<span class="spinner"></span> Analyzing…';
  els.placeholder.style.display = "none";
  els.result.style.display = "none";
  els.skeleton.classList.add("show");

  const fd = new FormData();
  fd.append("image", currentFile);
  fd.append("crop", els.crop.value.trim());
  fd.append("region", els.region.value.trim());
  fd.append("provider", els.provider.value);
  fd.append("min_confidence", els.minConf.value);
  fd.append("temperature", els.temp.value);

  try {
    const res = await fetch("/api/diagnose", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || "Request failed");
    render(data);
  } catch (err) {
    toast("Error: " + err.message);
    els.placeholder.style.display = "";
  } finally {
    els.skeleton.classList.remove("show");
    els.btn.disabled = false;
    els.btnLabel.textContent = "Diagnose";
  }
});

// ---------- demo preview (for visual checks: /?demo=ok or /?demo=abstain) ----------
const _demoOk = {"crop":"tomato","prediction":{"disease":"Early Blight","confidence":0.8394,"crop":"tomato","top_3":[{"disease":"Early Blight","crop":"tomato","prob":0.8394},{"disease":"Septoria Leaf Spot","crop":"tomato","prob":0.056},{"disease":"Late Blight","crop":"tomato","prob":0.0401}],"backend":"hf","entropy":0.202},"abstained":false,"gate":{"reasons":[]},"guidance":{"summary":"Early blight, caused by the fungus Alternaria solani, appears first as dark, concentric-ring (bullseye) lesions on the older lower leaves of tomato and then moves upward. Warm, humid conditions and rain splash of infected debris favor its spread.","immediate_steps":["Cut off and discard the infected lower leaves; do not add them to compost.","Stop watering from overhead and water at the soil line early in the day.","Increase airflow by staking plants and trimming the lower foliage."],"treatment_options":["Apply a copper-based fungicide following the label directions.","Apply chlorothalonil according to the product label.","For organic growers, use a Bacillus subtilis bio-fungicide as directed."],"prevention":["Lay mulch around the plant base to keep soil splashes off the leaves.","Rotate crops and avoid planting tomatoes or potatoes in the same bed for two years.","Space plants for good air movement and select resistant varieties when possible."],"sources":["KB-tomato-earlyblight"],"provider":"ollama","faithfulness":{"faithful":true}},"disclaimer":"Demonstration tool; confirm with a local agricultural extension service or a qualified agronomist before applying any treatment."};
const _demoAbstain = {"crop":"tomato","prediction":{"disease":"Powdery Mildew","confidence":0.5,"crop":"squash","top_3":[{"disease":"Powdery Mildew","crop":"squash","prob":0.50},{"disease":"Healthy","crop":"corn","prob":0.31},{"disease":"Healthy","crop":"strawberry","prob":0.19}],"backend":"hf","entropy":0.62},"abstained":true,"gate":{"reasons":["near_tie: top-1/top-2 margin 0.19 < 0.15","not_a_leaf: image does not appear to contain leaf tissue"]},"guidance":null,"disclaimer":"Demonstration tool; confirm with a local agricultural extension service or a qualified agronomist before applying any treatment."};
(function demoHook() {
  const m = new URLSearchParams(location.search).get("demo");
  if (!m) return;
  els.placeholder.style.display = "none";
  render(m === "abstain" ? _demoAbstain : _demoOk);
})();

// ---------- render ----------
function render(d) {
  const pred = d.prediction || {};
  const conf = pred.confidence || 0;
  const abstained = d.abstained;

  const top3 = (pred.top_3 || []).map((c) => `
    <div class="item">
      <div class="lbl">${esc(c.disease)} <small>${esc(c.crop)}</small></div>
      <div class="pct">${pct(c.prob)}</div>
      <div class="track"><i style="width:${(c.prob * 100).toFixed(1)}%"></i></div>
    </div>`).join("");

  let body;
  if (abstained) {
    const reasons = (d.gate?.reasons || []).map((r) => `<li>${esc(r)}</li>`).join("");
    body = `
      <div class="abstain-box">
        <div class="title">⚠️ Held back — not confident enough to advise</div>
        <ul>${reasons}</ul>
        <div class="rec">Recommendation: consult a local agricultural extension service or a qualified agronomist.</div>
      </div>`;
  } else {
    const g = d.guidance || {};
    const section = (icon, title, items) => `
      <div class="gsection">
        <h3>${icon} ${title}</h3>
        <ul>${(items || []).map((x) => `<li><span class="tick">✓</span><span>${esc(x)}</span></li>`).join("")}</ul>
      </div>`;
    const sources = (g.sources || []).map((s) => `<span class="chip src">${esc(s)}</span>`).join("");
    const faithful = g.faithfulness?.faithful !== false;
    body = `
      <div class="guidance">
        <div class="meta-row">
          <span class="chip prov">assembled by ${esc(g.provider)}</span>
          ${sources}
          <span class="chip">${faithful ? "✓ grounded" : "fallback"}</span>
        </div>
        <p class="gsummary">${esc(g.summary)}</p>
        ${section("⚡", "Immediate steps", g.immediate_steps)}
        ${section("💊", "Treatment options", g.treatment_options)}
        ${section("🛡️", "Prevention", g.prevention)}
      </div>`;
  }

  els.result.innerHTML = `
    <div class="result-head">
      <div>
        <div class="dx-name">${esc(pred.disease)}</div>
        <div class="dx-crop">${esc(d.crop || pred.crop || "")}</div>
      </div>
      <span class="verdict ${abstained ? "abstain" : "ok"}">
        ${abstained ? "⚠️ Abstained" : "✅ Diagnosis"}
      </span>
    </div>

    <div class="confbar-wrap">
      <div class="confbar-top"><span>Top-1 confidence</span><span>${pct(conf)} · entropy ${(pred.entropy ?? 0).toFixed(2)}</span></div>
      <div class="confbar ${abstained ? "warn" : ""}"><div class="fill" id="confFill"></div></div>
    </div>

    <div class="top3">${top3}</div>
    ${body}

    <details class="raw"><summary>Raw JSON response</summary><pre>${esc(JSON.stringify(d, null, 2))}</pre></details>
    <div class="disclaimer">${esc(d.disclaimer)}</div>
  `;
  els.result.style.display = "";
  // animate bars after paint
  requestAnimationFrame(() => {
    const fill = $("confFill");
    if (fill) fill.style.width = (conf * 100).toFixed(1) + "%";
    document.querySelectorAll(".top3 .track i").forEach((bar) => {
      const w = bar.style.width; bar.style.width = "0";
      requestAnimationFrame(() => (bar.style.width = w));
    });
  });
}
