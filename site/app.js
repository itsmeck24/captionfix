// Fills the page from data.json, which scripts/build_site.py writes from results/.

const pct = (x, d = 1) => `${(100 * x).toFixed(d)}%`;
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function table(el, head, rows, numeric, shipRow) {
  el.innerHTML = `<thead><tr>${head.map((h, i) => `<th class="${numeric[i] ? "num" : ""}">${h}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r, k) => `<tr class="${k === shipRow ? "ship" : ""}">${r.map((c, i) =>
      `<td class="${numeric[i] ? "num" : ""}">${c}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

function render(d) {
  document.getElementById("stats").innerHTML = d.stats.map(s =>
    `<div class="stat"><strong>${s.value}${s.unit ? `<small> ${s.unit}</small>` : ""}</strong><p>${s.label}</p></div>`).join("");
  document.getElementById("problem-rate").textContent = d.problem;
  document.getElementById("results-intro").textContent = d.intro;

  table(document.getElementById("outcome"),
    ["Lectures", "Hours", "Course terms said", "Zoom got wrong", "Found in the 6-minute review", "Found by random checks"],
    d.outcome.map(o => [o.name, o.hours.toFixed(1), o.terms.toLocaleString(), `${o.zoom} (1 in ${Math.round(o.terms / o.zoom)})`,
      pct(o.caught, 0), pct(o.random, 0)]), [0, 1, 1, 1, 1, 1], -1);

  document.getElementById("misheard").innerHTML = d.misheard.map(m =>
    `<div><span>Zoom: <s>${esc(m.wrong)}</s></span><span>said: <b>${esc(m.said)}</b></span></div>`).join("");
  player(d.player);
  quiz(d.misheard);
}

// hero caption: Zoom's words, then the doubt mark with what the second model heard
function heroSwap() {
  const el = document.getElementById("swap");
  if (!el) return;
  const marked = '<span class="doubt">risk prediction<sup>misprediction?</sup></span>';
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) { el.innerHTML = marked; return; }
  let on = false;
  setInterval(() => { on = !on; el.innerHTML = on ? marked : "risk prediction"; }, 2400);
}

const nav = document.getElementById("nav");
addEventListener("scroll", () => nav.classList.toggle("scrolled", scrollY > 10), { passive: true });
heroSwap();
fetch("./data.json").then(r => r.json()).then(render);

// student player: captions with fixes and doubt marks over the slide that was up, plus a glossary
function player(p) {
  const root = document.getElementById("player");
  if (!root || !p) return;
  const cues = [];
  let cur = [];
  for (const w of p.words) {
    const len = cur.reduce((n, x) => n + x.text.length + 1, 0);
    if (cur.length && (len + w.text.length > 74 || w.t - cur[cur.length - 1].e > 1.2)) { cues.push(cur); cur = []; }
    cur.push(w);
  }
  if (cur.length) cues.push(cur);

  root.innerHTML = `
    <div class="screen">
      <div class="stage"><img id="pl-slide" alt="lecture slide"><p class="caption-bar"><span id="pl-cap"></span></p></div>
      <div class="controls">
        <button class="play" id="pl-play" aria-label="play">${'<svg viewBox="0 0 14 14"><path d="M2 1l11 6-11 6z"/></svg>'}</button>
        <div class="bar" id="pl-bar"><i id="pl-fill"></i></div><span id="pl-time">0:00</span>
        <label><input type="checkbox" id="pl-doubt" checked> doubt marks</label>
      </div>
    </div>
    <div class="side">
      <p class="micro">${esc(p.course)}<br>${esc(p.title)}, from ${Math.floor(p.start / 60)}:${String(Math.floor(p.start % 60)).padStart(2, "0")}</p>
      <div class="legend">
        <p><span class="f">blue</span> words were wrong in Zoom's captions and fixed after checking the audio.</p>
        <p><span class="d">dotted</span> words are course terms the second model heard differently. Small text shows what it heard.</p>
      </div>
      <h3>Course terms in this clip</h3>
      <div class="glossary" id="pl-gloss">${p.glossary.map(g => `<button data-t="${g.t}">${esc(g.term)}</button>`).join("")}</div>
      <p class="micro">Click a term to jump to where it is said.</p>
    </div>`;

  const audio = new Audio(p.audio), cap = root.querySelector("#pl-cap"), img = root.querySelector("#pl-slide");
  const fill = root.querySelector("#pl-fill"), time = root.querySelector("#pl-time"), btn = root.querySelector("#pl-play");
  const doubtBox = root.querySelector("#pl-doubt");
  const word = w => w.kind === "fixed" ? `<span class="fixed" title="Zoom wrote: ${esc(w.was || "")}">${esc(w.text)}</span>`
    : w.kind === "doubt" && doubtBox.checked ? `<span class="doubt">${esc(w.text)}<sup>${esc(w.was)}?</sup></span>` : esc(w.text);
  const first = p.words.find(w => w.kind);
  let started = false;
  const draw = () => {
    // before the first play, show the moment with the first mark so the page opens on it
    const t = started ? audio.currentTime : (first ? first.t + 0.2 : 0);
    const c = cues.find(c => t >= c[0].t - 0.2 && t <= c[c.length - 1].e + 0.8) || [];
    cap.innerHTML = c.map(word).join(" ");
    cap.style.visibility = c.length ? "visible" : "hidden";
    const s = p.slides.filter(s => s.t <= t + 0.01).pop() || p.slides[0];
    if (s && !img.src.endsWith(s.src)) img.src = s.src;
    fill.style.width = `${100 * t / (audio.duration || 80)}%`;
    time.textContent = `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  };
  const icon = on => on ? '<svg viewBox="0 0 14 14"><rect x="2" y="1" width="3.5" height="12"/><rect x="8.5" y="1" width="3.5" height="12"/></svg>'
                        : '<svg viewBox="0 0 14 14"><path d="M2 1l11 6-11 6z"/></svg>';
  btn.onclick = () => {
    if (!started && first) audio.currentTime = Math.max(0, first.t - 4);
    started = true;
    audio.paused ? audio.play() : audio.pause();
  };
  audio.onplay = () => { btn.innerHTML = icon(true); btn.classList.add("on"); };
  audio.onpause = () => { btn.innerHTML = icon(false); btn.classList.remove("on"); };
  audio.ontimeupdate = draw;
  doubtBox.onchange = draw;
  root.querySelector("#pl-bar").onclick = e => {
    const r = e.currentTarget.getBoundingClientRect();
    started = true;
    audio.currentTime = (audio.duration || 80) * (e.clientX - r.left) / r.width; draw();
  };
  root.querySelectorAll("#pl-gloss button").forEach(b => b.onclick = () => {
    started = true;
    audio.currentTime = Math.max(0, +b.dataset.t - 1.5); audio.play();
  });
  draw();
}

// sound-off quiz: real Zoom captions with one wrong course term; pick, then hear the clip
function quiz(items) {
  const root = document.getElementById("quiz-cards");
  if (!root || !items || !items.length) return;
  const audio = new Audio();
  let answered = 0, right = 0;
  root.innerHTML = items.slice(0, 3).map((q, i) => {
    const options = (i % 2 ? [q.said, q.wrong] : [q.wrong, q.said]);
    return `<article class="card" data-i="${i}">
      <p class="micro">${esc(q.course)} · ${esc(q.at)}</p>
      <p class="cap">${esc(q.before)} <b>${esc(q.wrong)}</b> ${esc(q.after)}</p>
      <div class="choices">${options.map(o => `<button data-o="${esc(o)}">${esc(o)}</button>`).join("")}</div>
      <p class="verdict"></p>
    </article>`;
  }).join("");
  root.querySelectorAll(".card").forEach(card => {
    const q = items[+card.dataset.i];
    card.querySelectorAll("button").forEach(b => b.onclick = () => {
      const ok = b.dataset.o === q.said;
      answered++; right += ok;
      card.querySelectorAll("button").forEach(x => {
        x.disabled = true;
        if (x.dataset.o === q.said) x.classList.add("right");
      });
      b.classList.add("picked");
      const cap = card.querySelector(".cap b");
      cap.className = "wrong";
      cap.insertAdjacentHTML("afterend", ` <b class="right">${esc(q.said)}</b>`);
      card.querySelector(".verdict").textContent = (ok ? "Right. " : "That is what Zoom wrote. ") + "Playing the clip.";
      audio.src = q.audio; audio.play();
      if (answered === Math.min(3, items.length)) {
        document.getElementById("quiz-score").innerHTML = `<b>You got ${right} of ${answered}.</b> You had the hint that
          one word was wrong. A student reading the captions gets no hint, and Zoom gets about 1 in 7 course terms wrong.
          CaptionFix finds where to look.`;
      }
    });
  });
}
