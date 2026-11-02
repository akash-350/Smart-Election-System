// Shared helpers for Smart Election pages.

/** Show a message in an .alert element. type: info | success | error | warn */
function setStatus(el, text, type = "info") {
  if (!el) return;
  el.className = "alert" + (type === "info" ? "" : " " + type);
  el.textContent = text || "";
}

/** POST JSON and return {ok, status, data}. Never throws on HTTP errors. */
async function postJSON(url, body) {
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    let data = {};
    try { data = await response.json(); } catch (_) { /* non-JSON */ }
    if (response.status === 401) {
      data.error = data.error || "Your admin session has ended. Please log in again.";
    }
    return { ok: response.ok, status: response.status, data };
  } catch (_) {
    return { ok: false, status: 0, data: { error: "Couldn't reach the server. Check your internet connection and try again." } };
  }
}

/** Start the webcam into a <video>. Returns the MediaStream. */
async function startCamera(video) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  return stream;
}

function stopCamera(stream) {
  if (stream) stream.getTracks().forEach(track => track.stop());
}

function cameraErrorText(error) {
  if (error && error.name === "NotAllowedError") return "Camera permission was denied. Allow camera access in the browser and try again.";
  if (error && error.name === "NotFoundError") return "No camera found. Connect a webcam and try again.";
  return "Could not start the camera: " + (error && error.message ? error.message : error);
}

/** Grab `count` JPEG frames from a <video>, `gap` ms apart. */
async function captureFrames(video, count, gap, onFrame, width = 640, quality = 0.8) {
  const canvas = document.createElement("canvas");
  const ratio = video.videoHeight / video.videoWidth || 0.75;
  canvas.width = width;
  canvas.height = Math.round(width * ratio);
  const context = canvas.getContext("2d");
  const frames = [];
  for (let i = 0; i < count; i++) {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    frames.push(canvas.toDataURL("image/jpeg", quality));
    if (onFrame) onFrame(i + 1, count);
    if (i < count - 1) await new Promise(resolve => setTimeout(resolve, gap));
  }
  return frames;
}

// Neutral chart palette (no party colours, so no candidate looks favoured).
const CANDIDATE_COLOURS = ["#1E3A8A", "#0E7490", "#B45309", "#64748B", "#7C3AED", "#BE185D"];
function candidateColour(index, name) {
  if (name === "NOTA") return "#94A3B8";
  return CANDIDATE_COLOURS[index % CANDIDATE_COLOURS.length];
}

// ---------------------------------------------------------------------------
// Theme (light / dark), remembered per browser
// ---------------------------------------------------------------------------
function currentTheme() {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("se-theme", theme); } catch (_) { /* private mode */ }
  document.querySelectorAll("[data-theme-toggle]").forEach(button => {
    button.setAttribute("aria-pressed", String(theme === "dark"));
    const label = button.querySelector("span");
    if (label) label.textContent = theme === "dark" ? "Light mode" : "Dark mode";
  });
  document.dispatchEvent(new CustomEvent("themechange", { detail: theme }));
}
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-theme-toggle]").forEach(button => {
    button.addEventListener("click", () => setTheme(currentTheme() === "dark" ? "light" : "dark"));
  });
  setTheme(currentTheme());
});
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// ---------------------------------------------------------------------------
// Confirmation dialog. Resolves to true / the typed text, or false if cancelled.
//   confirmDialog({ title, message, confirmLabel, danger, requireText, input, placeholder, value })
// ---------------------------------------------------------------------------
function confirmDialog(options) {
  return new Promise(resolve => {
    const dialog = document.createElement("dialog");
    dialog.className = "modal";
    const needsInput = options.requireText || options.input;
    dialog.innerHTML = `
      <form method="dialog">
        <h2></h2>
        <p></p>
        ${needsInput ? '<div class="field"><input class="input" autocomplete="off"></div>' : ""}
        <div class="modal-actions">
          <button class="btn btn-secondary" value="cancel" type="submit">Cancel</button>
          <button class="btn ${options.danger ? "btn-danger" : "btn-primary"}" value="ok" type="submit"></button>
        </div>
      </form>`;
    dialog.querySelector("h2").textContent = options.title || "Are you sure?";
    dialog.querySelector("p").textContent = options.message || "";
    const okButton = dialog.querySelector('button[value="ok"]');
    okButton.textContent = options.confirmLabel || "Confirm";
    const input = dialog.querySelector("input");
    if (input) {
      input.placeholder = options.placeholder || options.requireText || "";
      input.value = options.value || "";
      const check = () => {
        okButton.disabled = options.requireText ? input.value !== options.requireText : !input.value.trim();
      };
      input.addEventListener("input", check);
      check();
    }
    dialog.addEventListener("close", () => {
      const ok = dialog.returnValue === "ok";
      dialog.remove();
      resolve(ok ? (input ? input.value.trim() : true) : false);
    });
    document.body.appendChild(dialog);
    dialog.showModal();
    (input || okButton).focus();
  });
}

// ---------------------------------------------------------------------------
// Voice prompts (browser speech). Falls back to English if no voice exists
// for the chosen language.
// ---------------------------------------------------------------------------
function speak(text, lang = "en-IN", fallbackText = null) {
  if (!("speechSynthesis" in window) || !text) return;
  const voices = speechSynthesis.getVoices();
  const base = lang.split("-")[0];
  let voice = voices.find(v => v.lang === lang) || voices.find(v => v.lang && v.lang.startsWith(base));
  let say = text;
  if (!voice && base !== "en") {
    voice = voices.find(v => v.lang === "en-IN") || voices.find(v => v.lang && v.lang.startsWith("en"));
    say = fallbackText || "";
    lang = "en-IN";
  }
  if (!say) return;
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(say);
  utterance.lang = lang;
  if (voice) utterance.voice = voice;
  utterance.rate = 0.95;
  speechSynthesis.speak(utterance);
}
if ("speechSynthesis" in window) speechSynthesis.getVoices();  // warm up the voice list

// ---------------------------------------------------------------------------
// Beeps — so people with their eyes closed know when to open them.
// unlockAudio() must be called directly inside a click handler.
// ---------------------------------------------------------------------------
let audioContext = null;
function unlockAudio() {
  try {
    audioContext = audioContext || new (window.AudioContext || window.webkitAudioContext)();
    if (audioContext.state === "suspended") audioContext.resume();
  } catch (_) { audioContext = null; }
}
function beep(times = 1, frequency = 880) {
  if (!audioContext) return;
  for (let i = 0; i < times; i++) {
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();
    const start = audioContext.currentTime + i * 0.3;
    oscillator.frequency.value = frequency;
    oscillator.connect(gain);
    gain.connect(audioContext.destination);
    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.exponentialRampToValueAtTime(0.35, start + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.22);
    oscillator.start(start);
    oscillator.stop(start + 0.24);
  }
}
