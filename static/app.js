"use strict";

const $ = (sel) => document.querySelector(sel);
const form = $("#upload-form");
const statusLine = $("#status");
const player = $("#player");
let job = null;
let stopAt = null;

// ---------- helpers ----------
function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c != null) node.append(c);
  return node;
}

function clock(seconds) {
  const m = Math.floor(seconds / 60);
  const s = (seconds - m * 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

function setStatus(text, isError = false) {
  statusLine.textContent = text;
  statusLine.classList.toggle("error", isError);
}

function countTerms(text) {
  return new Set(text.split(/[\n,;]+/).map((t) => t.trim().toLowerCase()).filter(Boolean)).size;
}

// ---------- replay ----------
function replay(start, end, node) {
  document.querySelectorAll(".playing").forEach((n) => n.classList.remove("playing"));
  if (node) node.classList.add("playing");
  player.currentTime = start;
  stopAt = end;
  player.play();
}

player.addEventListener("timeupdate", () => {
  if (stopAt !== null && player.currentTime >= stopAt) {
    player.pause();
    stopAt = null;
    document.querySelectorAll(".playing").forEach((n) => n.classList.remove("playing"));
  }
});

// ---------- upload ----------
form.glossary.addEventListener("input", () => {
  $("#term-count").textContent = countTerms(form.glossary.value);
});
form.boost.addEventListener("input", () => { $("#boost-value").textContent = form.boost.value; });

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("#submit");
  button.disabled = true;
  setStatus("Transcribing… the first run also loads the speech model, which can take a minute.");
  try {
    const response = await fetch("/api/transcribe", { method: "POST", body: new FormData(form) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    job = data;
    player.src = `/api/jobs/${job.id}/audio`;
    render();
    setStatus("Done.");
    $("#results").hidden = false;
    $("#results").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    button.disabled = false;
  }
});

async function review(sid, decision) {
  const response = await fetch(`/api/jobs/${job.id}/suggestions/${sid}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision }),
  });
  const data = await response.json();
  if (!response.ok) { setStatus(data.error || "Could not save the review.", true); return; }
  job = data;
  render();
}

// ---------- rendering ----------
function render() {
  $("#file-name").textContent = job.filename;
  const t = job.timing;
  $("#timing").textContent =
    `${clock(job.duration)} audio · processed in ${(t.scoring_seconds + t.decoding_seconds).toFixed(1)} s` +
    (t.real_time_factor != null ? ` (RTF ${t.real_time_factor.toFixed(2)})` : "");
  $("#export-txt").href = `/api/jobs/${job.id}/export?format=txt`;
  $("#export-json").href = `/api/jobs/${job.id}/export?format=json`;
  renderTerms();
  renderQueue();
  renderTranscript();
  $("#standard-text").textContent = job.standard.text || "(no speech detected)";
  $("#assisted-text").textContent = job.assisted.text || "(no speech detected)";
}

function renderTerms() {
  const box = $("#terms");
  box.replaceChildren();
  const label = { suggested: "suggested", in_standard: "already in transcript", not_heard: "not heard" };
  for (const row of job.terms) {
    box.append(el("span", { class: `chip ${row.status}`, text: `${row.term}: ${label[row.status]}` }));
  }
}

function renderQueue() {
  const list = $("#queue");
  list.replaceChildren();
  const pending = job.suggestions.filter((s) => s.status === "pending").length;
  const badge = $("#queue-count");
  badge.textContent = pending ? `${pending} to review` : "all reviewed";
  badge.className = `badge${pending === 0 ? " done" : ""}`;
  badge.hidden = job.suggestions.length === 0;

  if (job.suggestions.length === 0) {
    const why = job.glossary.length
      ? "The glossary did not change any wording in this recording."
      : "No glossary was entered for this recording.";
    list.append(el("li", { class: "empty-queue", text: why }));
    return;
  }

  for (const s of job.suggestions) {
    const item = el("li", { class: s.status });
    const replayButton = el("button", {
      class: "small secondary", type: "button", text: "▶ Replay",
      onclick: () => replay(s.replay_start, s.replay_end, item),
    });
    const actions = s.status === "pending"
      ? [
          replayButton,
          el("button", { class: "small accept", type: "button", text: "Accept", onclick: () => review(s.id, "accept") }),
          el("button", { class: "small reject", type: "button", text: "Reject", onclick: () => review(s.id, "reject") }),
        ]
      : [
          replayButton,
          el("span", { class: "muted", text: s.status === "accepted" ? "Accepted" : "Rejected" }),
          el("button", { class: "small secondary", type: "button", text: "Undo", onclick: () => review(s.id, "undo") }),
        ];
    item.append(
      el("div", { class: "time", text: `${clock(s.start)} – ${clock(s.end)} · glossary: ${s.terms.join(", ")}` }),
      el("div", { class: "change" }, [
        el("span", { class: `old${s.standard_text ? "" : " empty"}`, text: s.standard_text || "(nothing)" }),
        " → ",
        el("span", { class: "new", text: s.suggested_text }),
      ]),
      el("div", { class: "actions" }, actions),
    );
    list.append(item);
  }
}

function renderTranscript() {
  const box = $("#transcript");
  box.replaceChildren();
  const byId = Object.fromEntries(job.suggestions.map((s) => [s.id, s]));
  for (const seg of job.segments) {
    if (seg.kind === "word") {
      const span = el("span", {
        class: `w${seg.low_confidence ? " low" : ""}`,
        text: seg.text,
        title: `${clock(seg.start)} · confidence ${(seg.confidence * 100).toFixed(0)}%`,
      });
      span.addEventListener("click", () =>
        replay(Math.max(0, seg.start - 1), Math.min(job.duration, seg.end + 1), span));
      box.append(span, " ");
      continue;
    }
    const s = byId[seg.id];
    if (s.status === "rejected") {
      if (s.standard_text) box.append(el("span", { text: s.standard_text }), " ");
      continue;
    }
    const text = s.status === "accepted" ? s.suggested_text : s.standard_text;
    const span = el("span", {
      class: `seg ${s.status}${text ? "" : " empty"}`,
      text: text || "[?]",
      title: s.status === "pending" ? `Suggested: ${s.suggested_text}` : "Accepted glossary suggestion",
    });
    span.addEventListener("click", () => replay(s.replay_start, s.replay_end, span));
    box.append(span, " ");
  }
}
