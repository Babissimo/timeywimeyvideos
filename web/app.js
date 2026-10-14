// The page: open a video, set the slice, watch it play live, see where the
// slice sits in the cuboid, and render it. The video stays on this computer:
// the live view and the renders decode and slice it here, on the GPU.

import { probe } from "./src/decode.ts";
import { Live } from "./src/live.ts";
import { MAX_SEED } from "./src/noise.ts";
import { fullSize, Problem, readOptions } from "./src/options.ts";
import { render as renderClip } from "./src/render.ts";
import {
  changed, entryFolder, hasChanged, pickedFolder, same, Sources, VIDEO_TYPES,
} from "./src/sources.ts";

/**
 * @import { ClipInfo } from "./src/decode.ts"
 * @import { FrameInfo } from "./src/live.ts"
 * @import { Rendered, Stage } from "./src/render.ts"
 * @import { Folder, Source } from "./src/sources.ts"
 */

/**
 * The element of index.html with this id, typed as the script uses it. The overloads list
 * every id the script uses by its element's type, so a misspelt id fails the type check.
 * @overload
 * @param {"angle" | "angle-range" | "duration" | "file" | "folder" | "fps" | "loop-fade"
 *   | "loop-sides" | "noise" | "noise-amount" | "noise-range" | "noise-seed" | "noise-size"
 *   | "noise-speed" | "scale" | "scrub" | "side-fade" | "start"} id
 * @returns {HTMLInputElement}
 */
/**
 * @overload
 * @param {"motion" | "noise-direction" | "source"} id
 * @returns {HTMLSelectElement}
 */
/**
 * @overload
 * @param {"cancel" | "live-action" | "open-folder" | "play" | "render-full" | "render-preview"} id
 * @returns {HTMLButtonElement}
 */
/**
 * @overload
 * @param {"cube" | "live"} id
 * @returns {HTMLCanvasElement}
 */
/**
 * @overload
 * @param {"player"} id
 * @returns {HTMLVideoElement}
 */
/**
 * @overload
 * @param {"cube-loop" | "cube-noise" | "drop" | "edges-note" | "estimate" | "folder-note"
 *   | "frame-label" | "job" | "job-bar" | "job-text" | "live-info" | "live-message"
 *   | "live-stage" | "live-text" | "loop-fields" | "motion-field" | "motion-note"
 *   | "noise-fields" | "noise-note" | "now-playing" | "renders" | "side-fade-field"
 *   | "slice-note" | "source-info" | "unserved"} id
 * @returns {HTMLElement}
 */
/** @param {string} id */
function $(id) {
  return /** @type {HTMLElement} */ (document.getElementById(id));
}

// Why this page can't decode or encode video here, if it can't. WebCodecs, which does both,
// runs only in a secure context.
const UNABLE = !window.isSecureContext ?
  `The page can only decode video in a secure context, which ${location.origin} isn't: ` +
  "open it at localhost or over HTTPS." :
  !("VideoDecoder" in window && "VideoEncoder" in window) ?
  "This browser can't decode or encode video in a page, as it has no WebCodecs. Use a " +
  "recent Chrome or Edge." : "";

/**
 * @type {{ source: File | null, id: string, chosen: number, clip: ClipInfo | null,
 *          info: FrameInfo | null, pos: number, playing: boolean, scrubbing: boolean }}
 */
const state = {
  source: null,    // the File chosen
  id: "",          // and its id in the list
  chosen: 0,       // how many times a source has been chosen
  clip: null,      // what the chosen file holds, once probed
  info: null,      // the live view's description of the frame on screen
  pos: 0.5,        // how far through the output video, 0 to 1
  playing: false,
  scrubbing: false,
};

/** @type {Record<string, string>} */
const DESCRIBE = {
  rotate: "Samples evenly along the tilted plane, as if the cuboid were rotated. " +
    "Steeper angles show more time and less of the scene.",
  shear: "Keeps every input column, each delayed by tan(angle) frames per pixel " +
    "across. Needs an angle between −90° and 90°.",
  black: "The whole plane sweeps through the video, black wherever it's outside.",
  inside: "Each frame is the input's width and stays inside the video, so no " +
    "black, but steep angles may not fit the clip.",
  loop: "Time runs round in a ring, so the video plays on repeat without a " +
    "jump, and each frame is the input's width at any angle.",
  longest: "The longest straight line that fits; usually mostly through time.",
  perpendicular: "Like the whole-plane sweep. Every frame is a fresh slice, " +
    "but there's little room at small angles.",
  "perpendicular-sides": "Every frame is a fresh slice. The loop goes across the " +
    "width and round time a whole number of times, so it can be much longer than the clip.",
  time: "Forward through time, like ordinary playback but tilted.",
  "noise-time": "Each pixel is read up to that many frames earlier or later. " +
    "On a rotated frame near 90° this slides along the frame rather than off it.",
  "noise-perpendicular": "Each point moves straight off the plane. On a sheared " +
    "frame that moves columns sideways too, so Stay inside only fits at 0°.",
};

function sliceKind() {
  return /** @type {HTMLInputElement} */ (document.querySelector('input[name="slice"]:checked'))
    .value;
}

// What the frame does at the video's edges: black, inside or loop.
function edges() {
  return /** @type {HTMLInputElement} */ (document.querySelector('input[name="edges"]:checked'))
    .value;
}

/** @param {string} value */
function setEdges(value) {
  const radio = document.querySelector(`input[name="edges"][value="${value}"]`);
  /** @type {HTMLInputElement} */ (radio).checked = true;
  slicesChanged();
}

// Which ways the frame can move: rotate frames inside the video, or on a
// loop round the sides.
function moves() {
  return sliceKind() === "rotate" &&
    (edges() === "inside" || edges() === "loop" && $("loop-sides").checked);
}

function options() {
  const slice = sliceKind();
  const loop = edges() === "loop";
  const sides = loop && $("loop-sides").checked;
  const noise = $("noise").checked;
  return {
    slice,
    angle: sliceAngle(),
    inside: edges() === "inside" ? "1" : "",
    motion: moves() ? $("motion").value : "",
    loop: loop ? "1" : "",
    loop_fade: loop ? $("loop-fade").value : "",
    loop_sides: sides ? "1" : "",
    loop_side_fade: sides ? $("side-fade").value : "",
    noise: noise ? noiseAmount() : "",
    noise_size: noise ? $("noise-size").value : "",
    noise_speed: noise ? $("noise-speed").value : "",
    noise_direction: noise ? $("noise-direction").value : "",
    noise_seed: noise ? $("noise-seed").value : "",
    start: $("start").value,
    duration: $("duration").value,
    scale: $("scale").value || "1",
    fps: $("fps").value.trim(),
  };
}

/** @param {number} s */
function seconds(s) {
  const tenths = s.toFixed(1);
  if (Number(tenths) < 60) return `${tenths} s`;
  const whole = Math.round(s);  // rounded first, or 119.6 s would read 1:60
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

/** @param {number} n */
function bytes(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  return `${Math.max(1, Math.round(n / 1e3))} KB`;
}

// ---------------------------------------------------------------------------
// Live preview. Only one frame is worked on at a time; if the options or
// position change meanwhile, the latest ones are shown next, so dragging a
// slider never queues up stale frames.

/** @type {Live | null} */
let live = null;  // made on first use, so a browser without WebGL2 gets a message
/** @type {unknown} */
let unable = null;  // why it couldn't be made, as it is tried only once
let busy = false;
let again = false;

function liveView() {
  if (!live && !unable) {
    try {
      live = new Live($("live"));
      // This runs after Live's own listener, which readies it to draw again.
      $("live").addEventListener("webglcontextrestored", refresh);
    } catch (err) {
      unable = err;
    }
  }
  if (!live) throw unable;
  return live;
}

function refresh() {
  again = true;
  live?.supersede(options());  // a load for other options would only hold up theirs
  pump();
  estimate();
}

async function pump() {
  if (busy || !again) return;
  again = false;
  const file = state.source;
  if (!file) return;
  if (UNABLE) return showMessage(UNABLE);
  busy = true;
  const opts = options();
  // Once another source is chosen, the message is that one's to give.
  const chosen = state.chosen;
  /** @param {string} text */
  const loading = (text, progress = false) => {
    if (state.chosen === chosen) showMessage(text, null, progress);
  };
  let slow = false;
  const timer = setTimeout(() => {
    slow = true;
    loading("Loading the clip…");
  }, 300);
  try {
    const info = await liveView().show(file, opts, state.pos, (done, total) => {
      if (slow) loading(`Loading the clip… ${Math.floor(100 * done / total)}%`, true);
    });
    const resized = !state.info || state.info.frames !== info.frames;
    state.info = info;
    if (resized) anchor();
    hideMessage();
    describe();
    drawCube();
    loadFace();
  } catch (err) {
    const error = /** @type {Error} */ (err);
    if (error.name === "AbortError") {  // another source or other options took over
      if (slow) loading("Loading the clip…", true);  // as theirs loads, without this load's figure
    } else {
      pause();
      const kind = error instanceof Problem ? error.kind : "error";
      showMessage(error.message,
        kind === "does_not_fit" ?
          ["Let black in instead", () => setEdges("black")] :
        kind === "sideways" &&
          ["Wrap round the sides too", () => { $("loop-sides").checked = true; slicesChanged(); }]);
    }
  } finally {
    clearTimeout(timer);
    busy = false;
    pump();
  }
}

/**
 * Show a message over the live view, with a button for a way out if there is one. A busy
 * message, such as how far a load has got, isn't read out at each change.
 * @param {string} text
 * @param {[string, () => void] | false | null} [action]
 */
function showMessage(text, action, busy = false) {
  $("live-stage").setAttribute("aria-busy", String(busy));  // set before the text it governs
  $("live-message").hidden = false;
  $("live-text").textContent = text;
  const button = $("live-action");
  button.hidden = !action;
  if (action) {
    button.textContent = action[0];
    button.onclick = action[1];
  }
}

function hideMessage() {
  $("live-message").hidden = true;
  $("live-stage").setAttribute("aria-busy", "false");
}

function describe() {
  const info = state.info;
  if (!info) return;
  $("live-info").textContent =
    `${info.width}×${info.height} · ${info.frames} frames · ${seconds(info.seconds)}`;
  $("frame-label").textContent = `${info.frame + 1} / ${info.frames}`;
  if (!state.scrubbing) $("scrub").value = String(Math.round(state.pos * 1000));
  scrubText();
}

// What the scrubber says it is at, to assistive technology: a frame rather than 0 to 1000.
function scrubText() {
  const info = state.info;
  if (!info) return $("scrub").removeAttribute("aria-valuetext");
  const frame = Math.round(state.pos * (info.frames - 1)) + 1;
  $("scrub").setAttribute("aria-valuetext", `Frame ${frame} of ${info.frames}`);
}

// What a full-quality render would make. It needs only the clip and the options, not the
// live view's half-size copy, so it still holds when that copy can't be loaded.
/** @type {{ clip: ClipInfo | null, values: string }} */
let estimated = { clip: null, values: "" };  // what the estimate shown was worked out for

function estimate() {
  const values = options(), key = JSON.stringify(values);
  if (state.clip === estimated.clip && key === estimated.values) return;
  estimated = { clip: state.clip, values: key };
  let full = null;
  try {
    if (state.clip && !UNABLE) full = fullSize(state.clip, readOptions(values));
  } catch (err) {
    if (!(err instanceof Problem)) throw err;  // else the live view says what's wrong
  }
  $("estimate").classList.toggle("error", Boolean(full && "error" in full));
  $("estimate").textContent = !full ? "" : "error" in full ? `At full size: ${full.error}` :
    `Full quality makes ${full.width}×${full.height}, ${full.frames} frames ` +
    `(${seconds(full.seconds)}), holding ${bytes(full.memory)} of video on the GPU.`;
}

// Playback runs on the clock: each animation frame works out which output
// frame is due and shows it, skipping any the live view can't keep up with.

const clock = { start: 0, frame: 0 };
let ticking = 0;  // the request for tick's next animation frame

function anchor() {
  clock.start = performance.now();
  clock.frame = state.info ? state.pos * (state.info.frames - 1) : 0;
}

/** @param {number} now */
function tick(now) {
  if (!state.playing) return;
  const info = state.info;
  if (info && info.frames > 1 && !state.scrubbing) {
    const f = Math.floor(clock.frame + (now - clock.start) / 1000 * info.fps) % info.frames;
    if (f !== info.frame) {
      state.pos = f / (info.frames - 1);
      refresh();
    }
  }
  ticking = requestAnimationFrame(tick);
}

function play() {
  if (!state.info || state.info.frames < 2) return;
  if (state.info.frame === state.info.frames - 1) state.pos = 0;
  state.playing = true;
  $("play").textContent = "❚❚";
  $("play").setAttribute("aria-label", "Pause");
  anchor();
  cancelAnimationFrame(ticking);  // still due if paused and played again within a frame
  ticking = requestAnimationFrame(tick);
}

function pause() {
  state.playing = false;
  $("play").textContent = "▶";
  $("play").setAttribute("aria-label", "Play");
}

/** @param {number} by */
function step(by) {
  const info = state.info;
  if (!info || info.frames < 2) return;
  pause();
  const f = info.loop ? (info.frame + by + info.frames) % info.frames :
    Math.min(Math.max(info.frame + by, 0), info.frames - 1);
  state.pos = f / (info.frames - 1);
  refresh();
}

$("play").addEventListener("click", () => (state.playing ? pause() : play()));
$("scrub").addEventListener("pointerdown", () => { state.scrubbing = true; });
$("scrub").addEventListener("input", () => {
  state.pos = Number($("scrub").value) / 1000;
  scrubText();
  refresh();
});
for (const done of ["pointerup", "pointercancel", "change"]) {
  $("scrub").addEventListener(done, () => {
    state.scrubbing = false;
    anchor();
  });
}

document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;  // the browser's shortcuts, such as Back
  const target = /** @type {Element} */ (e.target);
  const range = target instanceof HTMLInputElement && target.type === "range";
  // Fields take their own keys, and so does the player's video.
  if (target.closest("input, select, textarea, video") && !range) return;
  if (e.key === " " && target.tagName !== "BUTTON") {
    e.preventDefault();
    state.playing ? pause() : play();
  } else if ((e.key === "ArrowRight" || e.key === "ArrowLeft") && !range) {
    e.preventDefault();
    step(e.key === "ArrowRight" ? 1 : -1);
  }
});

// ---------------------------------------------------------------------------
// Slice controls.

function slicesChanged() {
  const shear = sliceKind() === "shear";
  const limit = shear ? 89.5 : 180;
  for (const input of [$("angle"), $("angle-range")]) {
    input.min = String(-limit);
    input.max = String(limit);
  }
  setAngle(sliceAngle());
  const loop = edges() === "loop", sides = loop && $("loop-sides").checked;
  $("edges-note").textContent = DESCRIBE[edges()];
  $("loop-fields").hidden = !loop;
  $("side-fade-field").hidden = !sides;
  // No line round the sides is the longest.
  const longest = $("motion").querySelector('[value="longest"]');
  /** @type {HTMLOptionElement} */ (longest).disabled = sides;
  if (sides && $("motion").value === "longest") $("motion").value = "perpendicular";
  $("motion-field").hidden = !moves();
  $("slice-note").textContent = DESCRIBE[sliceKind()];
  const motion = $("motion").value;
  $("motion-note").textContent = DESCRIBE[sides && motion === "perpendicular" ?
    "perpendicular-sides" : motion];
  $("cube-loop").hidden = !loop;
  $("cube-loop").textContent = "Wrapping round, the clip's ends " +
    (sides ? "and sides join up, so the frame wraps round both: pieces " +
             "that leave the back or one side come in again at the front or the other." :
             "join up, so the frame wraps round time: pieces that leave the back " +
             "come in again at the front.");
  refresh();
}

/** @param {number} value */
function setAngle(value) {
  $("angle").value = String(value);
  $("angle-range").value = String(value);
}

for (const input of document.querySelectorAll(
  'input[name="slice"], input[name="edges"], #motion, #loop-sides')) {
  input.addEventListener("change", slicesChanged);
}
/**
 * A slider and the number box beside it, which set one value: the box's number, held within its
 * bounds. The slider shows the value to its own step, and the box shows it once left. While the
 * box holds no number, as while a minus sign is typed, the value stays as it was. Returns a
 * function that gives the value.
 * @param {HTMLInputElement} slider
 * @param {HTMLInputElement} box
 * @param {() => void} changed
 */
function pair(slider, box, changed) {
  let last = 0;
  const value = () => {
    if (box.value !== "") {  // it is "" while the box holds no number
      const min = box.min === "" ? -Infinity : Number(box.min);
      const max = box.max === "" ? Infinity : Number(box.max);
      last = Math.min(Math.max(Number(box.value), min), max);
    }
    return last;
  };
  slider.addEventListener("input", () => {
    box.value = slider.value;
    changed();
  });
  box.addEventListener("input", () => {
    if (box.value === "") return;
    slider.value = String(value());
    changed();
  });
  box.addEventListener("change", () => { box.value = String(value()); });
  return value;
}

const sliceAngle = pair($("angle-range"), $("angle"), refresh);
for (const input of [$("start"), $("duration"), $("scale"), $("fps"), $("loop-fade"),
                     $("side-fade")]) {
  input.addEventListener("change", refresh);
}

function noiseChanged() {
  const on = $("noise").checked;
  const direction = $("noise-direction").value;
  $("noise-fields").hidden = !on;
  $("noise-note").textContent = DESCRIBE[`noise-${direction}`];
  $("cube-noise").hidden = !on;
  $("cube-noise").textContent = "With noise on, the real surface is bumpy: the frame " +
    `is drawn on the flat plane its points are pushed off, up to ${noiseAmount()} ` +
    `frames ${direction === "time" ? "through time" : "perpendicular to it"}.`;
  refresh();
}

for (const input of [$("noise"), $("noise-size"), $("noise-speed"), $("noise-direction"),
                     $("noise-seed")]) {
  input.addEventListener("change", noiseChanged);
}
const noiseAmount = pair($("noise-range"), $("noise-amount"), noiseChanged);

// ---------------------------------------------------------------------------
// Sources: videos opened on this computer, on their own or a folder at a time,
// which the page reads where they lie as it needs them.

const sources = new Sources();
const NO_SOURCE = "Open a video, or drop one here.";
/** @type {string | null} */
let original = null;  // the address "watch the original" plays the chosen file at

function showSources() {
  const list = $("source");
  list.replaceChildren(new Option("Choose a video…", ""));
  if (sources.folder !== null) list.append(group(`Folder: ${sources.folder}`, sources.inFolder));
  if (sources.opened.length) list.append(group("Opened", sources.opened));
  list.value = sources.chosen?.id ?? "";
}

/**
 * @param {string} label
 * @param {readonly Source[]} items
 */
function group(label, items) {
  const group = document.createElement("optgroup");
  group.label = label;
  const seen = new Map();  // files of the same name from different places are numbered
  for (const { id, file } of items) {
    const n = (seen.get(file.name) ?? 0) + 1;
    seen.set(file.name, n);
    group.append(new Option(n > 1 ? `${file.name} (${n})` : file.name, id));
  }
  if (!items.length) group.append(Object.assign(new Option("No videos"), { disabled: true }));
  return group;
}

/**
 * A note on the folder, apart from the chosen video's details.
 * @param {string} text
 */
function folderNote(text, error = false) {
  $("folder-note").textContent = text;
  $("folder-note").classList.toggle("error", error);
}

function describeFolder() {
  const n = sources.inFolder.length, unreadable = sources.unreadable;
  folderNote((n ? `${sources.folder} has ${n} video${n === 1 ? "" : "s"}.` :
    `${sources.folder} has no videos at its top level (${[...VIDEO_TYPES].join(" ")}).`) +
    (unreadable.length ? ` Couldn't read ${unreadable.join(", ")}.` : ""));
}

/** @param {string} id */
async function chooseSource(id) {
  const file = sources.choose(id)?.file ?? null;
  $("source").value = file ? id : "";
  if (file === state.source) return;
  // The chosen video opened again, as a new File, takes the place of the File held without
  // loading afresh while that still reads. Once the file changes or moves it doesn't, and the
  // video loads again from the new File.
  const held = state.source;
  if (file && held && id === state.id && same(held, file)) {
    const stale = await hasChanged(held);
    if (state.source !== held || sources.chosen?.file !== file) return;  // chosen again since
    if (!stale) {
      live?.adopt(held, file);
      state.source = file;
      return;
    }
  }
  pause();
  live?.unload();
  forgetOriginal();
  state.source = file;
  state.id = file ? id : "";
  state.chosen++;
  state.clip = null;
  state.info = null;
  state.pos = 0.5;
  face = null;
  faceImage = null;
  drawCube();
  $("source-info").textContent = "";
  $("live-info").textContent = "";
  $("frame-label").textContent = "";
  scrubText();
  estimate();
  enableRender();
  if (!file) {
    showMessage(UNABLE || NO_SOURCE);
    return;
  }
  refresh();
  const chosen = state.chosen;
  try {
    const info = await probe(file);
    if (state.chosen !== chosen) return;
    state.clip = info;
    estimate();
    $("source-info").replaceChildren(
      `${info.width}×${info.height} · ${Number((info.fps.num / info.fps.den).toFixed(3))} fps` +
      (info.duration ? ` · ${seconds(info.duration)}` : "") + ` · ${bytes(file.size)} · `,
      link("watch the original", watchOriginal));
  } catch (err) {
    const message = await hasChanged(file) ? changed(file.name) :
      /** @type {Error} */ (err).message;
    if (state.chosen === chosen) $("source-info").textContent = message;
  }
}

function watchOriginal() {
  const file = /** @type {File} */ (state.source);  // its link shows only once one is chosen
  original ??= URL.createObjectURL(file);
  watch(original, file.name, file);
}

// Let go of the chosen file's address, and stop the player if it plays it.
function forgetOriginal() {
  if (!original) return;
  const player = $("player");
  if (player.src === original) {
    player.removeAttribute("src");
    player.load();
    player.hidden = true;
    $("now-playing").textContent = "";
    markPlaying();
  }
  URL.revokeObjectURL(original);
  original = null;
}

/** @param {File[]} files */
function openFiles(files) {
  const id = sources.open(files);
  if (!id) return;
  showSources();
  chooseSource(id);
}

async function pickFolder() {
  // Firefox and Safari have no showDirectoryPicker, but take a folder in a file input.
  if (!window.showDirectoryPicker) return $("folder").click();
  let folder;
  try {
    folder = await window.showDirectoryPicker({ id: "videos", startIn: "videos" });
  } catch (err) {
    const error = /** @type {Error} */ (err);
    if (error.name !== "AbortError") folderNote(error.message, true);
    return;  // else they closed the picker
  }
  await openFolder(folder);
}

/** @param {Folder} folder */
async function openFolder(folder) {
  try {
    if (!await sources.openFolder(folder)) return;  // another folder took its place
  } catch (err) {
    folderNote(`Couldn't read ${folder.name}: ${/** @type {Error} */ (err).message}`, true);
    return;
  }
  showSources();
  describeFolder();
  // The video chosen stays, as the File just listed where the folder lists it again, and comes
  // afresh if that is a newer copy.
  const chosen = sources.chosen;
  if (chosen && chosen.file !== state.source) chooseSource(chosen.id);
}

$("source").addEventListener("change", () => chooseSource($("source").value));
$("file").accept = ["video/*", ...VIDEO_TYPES].join(",");
$("noise-seed").max = String(MAX_SEED);
$("file").addEventListener("change", () => {
  openFiles([...$("file").files ?? []]);
  $("file").value = "";
});
$("open-folder").hidden =
  !(window.showDirectoryPicker || "webkitdirectory" in HTMLInputElement.prototype);
$("open-folder").addEventListener("click", pickFolder);
$("folder").addEventListener("change", () => {
  const files = [...$("folder").files ?? []];
  $("folder").value = "";
  const folder = pickedFolder(files);
  if (folder) openFolder(folder);
  else if (files.length) openFiles(files);  // from a browser that picked files, not a folder
  else folderNote("The folder chosen holds no files.");
});
document.addEventListener("dragover", (e) => {
  e.preventDefault();
  $("drop").classList.add("over");
});
document.addEventListener("dragleave", (e) => {
  if (!e.relatedTarget) $("drop").classList.remove("over");
});
document.addEventListener("drop", async (e) => {
  e.preventDefault();
  $("drop").classList.remove("over");
  // Everything is taken from the items before the first await: they empty once the event ends.
  const items = [...e.dataTransfer?.items ?? []].filter((item) => item.kind === "file");
  const entries = items.map((item) => item.webkitGetAsEntry?.() ?? null);
  const files = items.filter((_, i) => !entries[i]?.isDirectory).map((item) => item.getAsFile());
  // The first folder opens as the folder: by its handle where the browser gives one.
  const i = entries.findIndex((entry) => entry?.isDirectory);
  const handle = i < 0 ? null : items[i].getAsFileSystemHandle?.().catch(() => null);
  openFiles(files.filter((file) => file !== null));
  if (i < 0) return;
  // A folder's entry and handle are a directory's.
  const entry = /** @type {FileSystemDirectoryEntry} */ (entries[i]);
  openFolder(/** @type {FileSystemDirectoryHandle | null} */ (await handle) ?? entryFolder(entry));
});

// ---------------------------------------------------------------------------
// The cuboid: a wireframe of the clip in volume units (one pixel or one frame
// per unit), with its first frame on the front face, the current output frame
// drawn where it slices through, and the region the sweep covers shaded on top.
// The view is orthographic, so every flat rectangle stays a parallelogram and
// images can be mapped onto them with a plain 2D transform.

const cube = $("cube");
/** @type {Record<string, [number, number]>} */
const VIEWS = {
  three: [-0.6, -0.45],
  front: [0, 0],
  top: [0, -Math.PI / 2],
  side: [-Math.PI / 2, 0],
};
const camera = { yaw: VIEWS.three[0], pitch: VIEWS.three[1], zoom: 1 };
/** @type {HTMLCanvasElement | null} */
let face = null;       // canvas holding the clip's first frame
/** @type {ImageData | null} */
let faceImage = null;  // the live view's first frame, which it keeps while it keeps the clip

function loadFace() {
  const image = live?.face() ?? null;
  if (image === faceImage) return;
  faceImage = image;
  face = null;
  if (image) {
    face = document.createElement("canvas");
    face.width = image.width;
    face.height = image.height;
    /** @type {CanvasRenderingContext2D} */ (face.getContext("2d")).putImageData(image, 0, 0);
  }
  drawCube();
}

/**
 * A function from volume coordinates (x, y, t) to [screen x, screen y, depth].
 * @param {[number, number, number]} volume
 */
function projector(volume) {
  const [T, H, W] = volume;
  const w = cube.clientWidth, h = cube.clientHeight;
  const k = camera.zoom * Math.min(w, h) / (Math.hypot(W, H, T) * 1.1);
  const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
  const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
  /** @param {number} x @param {number} y @param {number} t */
  const project = (x, y, t) => {
    // Centre the cuboid, with y up and t running away from the viewer.
    const X = x - (W - 1) / 2, Y = (H - 1) / 2 - y, Z = t - (T - 1) / 2;
    const x1 = X * cy - Z * sy, z1 = X * sy + Z * cy;
    return [w / 2 + k * x1, h / 2 - k * (Y * cp - z1 * sp), Y * sp + z1 * cp];
  };
  // Whether the face with this outward normal (in x, y, t) faces the viewer.
  /** @param {number} nx @param {number} ny @param {number} nt */
  project.facing = (nx, ny, nt) => -ny * sp + (nx * sy + nt * cy) * cp < 0;
  return project;
}

/**
 * Draw an image onto a flat quad. map(u, v) gives the screen position of
 * image coordinates (u, v); only columns u0 to u1 are drawn.
 * @param {CanvasRenderingContext2D} ctx
 * @param {HTMLCanvasElement} image
 * @param {(u: number, v: number) => number[]} map
 * @param {number} u0
 * @param {number} u1
 * @param {number} alpha
 */
function drawOnto(ctx, image, map, u0, u1, alpha) {
  const o = map(0, 0), a = map(1, 0), b = map(0, 1);
  const dpr = devicePixelRatio || 1;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.setTransform(dpr * (a[0] - o[0]), dpr * (a[1] - o[1]),
                   dpr * (b[0] - o[0]), dpr * (b[1] - o[1]), dpr * o[0], dpr * o[1]);
  ctx.drawImage(image, u0, 0, u1 - u0, image.height, u0, 0, u1 - u0, image.height);
  ctx.restore();
}

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {number[][]} points
 * @param {{ fill?: string, stroke?: string, dash?: number[], width?: number }} style
 */
function polygon(ctx, points, { fill, stroke, dash = [], width = 1 }) {
  ctx.beginPath();
  points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) {
    ctx.setLineDash(dash);
    ctx.lineWidth = width;
    ctx.strokeStyle = stroke;
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {number[]} p
 * @param {number[]} q
 * @param {string} style
 * @param {number[]} [dash]
 */
function line(ctx, p, q, style, dash = [], width = 1) {
  ctx.beginPath();
  ctx.moveTo(p[0], p[1]);
  ctx.lineTo(q[0], q[1]);
  ctx.setLineDash(dash);
  ctx.lineWidth = width;
  ctx.strokeStyle = style;
  ctx.stroke();
  ctx.setLineDash([]);
}

/**
 * The columns u (0 to width, in image coordinates) of the output frame that
 * lie inside the cuboid once moved back by (shiftT, shiftX), or null if none do.
 * @param {FrameInfo} info
 * @param {number} dx
 * @param {number} dt
 */
function insideSpan(info, dx, dt, shiftT = 0, shiftX = 0) {
  const [T, , W] = info.volume;
  const t0 = info.line[0] - shiftT, x0 = info.line[1] - shiftX;
  let a = 0, b = info.width;
  for (const [p, d, hi] of [[x0 - 0.5 * dx, dx, W - 0.5], [t0 - 0.5 * dt, dt, T - 0.5]]) {
    if (Math.abs(d) < 1e-9) {
      if (p < -0.5 || p > hi) return null;
      continue;
    }
    let u1 = (-0.5 - p) / d, u2 = (hi - p) / d;
    if (u1 > u2) [u1, u2] = [u2, u1];
    a = Math.max(a, u1);
    b = Math.min(b, u2);
  }
  return b > a ? [a, b] : null;
}

function drawCube() {
  const dpr = devicePixelRatio || 1;
  const w = cube.clientWidth, h = cube.clientHeight;
  if (cube.width !== Math.round(w * dpr) || cube.height !== Math.round(h * dpr)) {
    cube.width = Math.round(w * dpr);
    cube.height = Math.round(h * dpr);
  }
  const ctx = /** @type {CanvasRenderingContext2D} */ (cube.getContext("2d"));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const info = state.info;
  if (!info) return;

  const P = projector(info.volume);
  const [T, H, W] = info.volume;
  const ends = [[-0.5, W - 0.5], [-0.5, H - 0.5], [-0.5, T - 0.5]];
  /** @param {number[]} bits */
  const corner = (bits) => P(ends[0][bits[0]], ends[1][bits[1]], ends[2][bits[2]]);
  /** @param {number} axis @param {number} high */
  const facing = (axis, high) => {
    /** @type {[number, number, number]} */
    const n = [0, 0, 0];
    n[axis] = high ? 1 : -1;
    return P.facing(...n);
  };

  // Each edge, and whether either face it borders is turned towards us.
  /** @type {[number[], number[], boolean][]} */
  const edges = [];
  for (const axis of [0, 1, 2]) {
    const [b, c] = [0, 1, 2].filter((other) => other !== axis);
    for (const hb of [0, 1]) for (const hc of [0, 1]) {
      const from = [0, 0, 0];
      from[b] = hb;
      from[c] = hc;
      const to = [...from];
      to[axis] = 1;
      edges.push([corner(from), corner(to), facing(b, hb) || facing(c, hc)]);
    }
  }
  for (const [p, q, front] of edges) if (!front) line(ctx, p, q, "rgba(150,160,175,.35)", [3, 4]);

  // The output frame: column u runs from (t0, x0) to (t1, x1) across it.
  const [t0, x0, t1, x1] = info.line;
  const dx = info.width > 1 ? (x1 - x0) / (info.width - 1) : 1;
  const dt = info.width > 1 ? (t1 - t0) / (info.width - 1) : 0;
  /**
   * @param {(u: number, v: number) => number[]} map
   * @param {number} u0
   * @param {number} u1
   * @param {number} v1
   */
  const quad = (map, u0, u1, v1) => [map(u0, 0), map(u1, 0), map(u1, v1), map(u0, v1)];

  // A looping frame runs round time in a ring, and on a loop round the sides
  // round x too, so each part of it past an end of the cuboid is drawn where
  // it wraps round to: the frame moved back by whole clip lengths or widths.
  /** @param {number} from @param {number} to @param {number} n @param {boolean} wraps */
  const turns = (from, to, n, wraps) => {
    if (!wraps) return [0];
    const first = Math.floor((Math.min(from, to) + 0.5) / n);
    const last = Math.floor((Math.max(from, to) + 0.5) / n);
    return Array.from({ length: last - first + 1 }, (_, i) => (first + i) * n);
  };
  const layers = [];
  for (const shiftT of turns(t0 - 0.5 * dt, t1 + 0.5 * dt, T, info.loop)) {
    for (const shiftX of turns(x0 - 0.5 * dx, x1 + 0.5 * dx, W, info.sides)) {
      /** @param {number} u @param {number} v */
      const onSlice = (u, v) =>
        P(x0 - shiftX + (u - 0.5) * dx, v - 0.5, t0 - shiftT + (u - 0.5) * dt);
      const span = insideSpan(info, dx, dt, shiftT, shiftX);
      if (info.loop && !span) continue;
      const middle = span ? (span[0] + span[1]) / 2 : info.width / 2;
      layers.push({
        depth: onSlice(middle, info.height / 2)[2],
        draw() {
          if (!info.loop) {
            polygon(ctx, quad(onSlice, 0, info.width, info.height),
                    { stroke: "rgba(242,163,58,.45)", dash: [4, 4] });
          }
          if (!span) return;
          drawOnto(ctx, $("live"), onSlice, span[0], span[1], 1);
          polygon(ctx, quad(onSlice, span[0], span[1], info.height),
                  { stroke: "#f2a33a", width: 1.5 });
        },
      });
    }
  }
  if (face) {
    const image = face;
    /** @param {number} u @param {number} v */
    const onFace = (u, v) => P(u - 0.5, v - 0.5, -0.5);
    layers.push({
      depth: P((W - 1) / 2, (H - 1) / 2, -0.5)[2],
      draw: () => drawOnto(ctx, image, onFace, 0, image.width, facing(2, 0) ? 0.35 : 0.2),
    });
  }
  layers.sort((a, b) => b.depth - a.depth).forEach((layer) => layer.draw());

  for (const [p, q, front] of edges) if (front) line(ctx, p, q, "rgba(205,212,224,.8)");

  // The strip of x-t the sweep passes through, on the top face, and the way
  // the frame's centre moves.
  if (info.frames > 1) {
    /** @param {number[]} point */
    const top = ([t, x]) => P(x, -0.5, t);
    const [f0, f1, f2, f3] = info.first, [l0, l1, l2, l3] = info.last;
    let from = top([(f0 + f2) / 2, (f1 + f3) / 2]);
    let to = top([(l0 + l2) / 2, (l1 + l3) / 2]);
    if (info.loop) {
      // A loop goes all the way round time, between the frame's ends in x,
      // or over all of x if it moves sideways round the sides too. Its
      // centre's path wraps round, so only its heading is drawn, across the
      // middle.
      const ht = (l0 + l2) / 2 - (f0 + f2) / 2, hx = (l1 + l3) / 2 - (f1 + f3) / 2;
      const [left, right] = Math.abs(hx) > 1e-9 ? [-0.5, W - 0.5] :
        [Math.min(f1, f3), Math.max(f1, f3)];
      polygon(ctx, [top([-0.5, left]), top([-0.5, right]), top([T - 0.5, right]),
                    top([T - 0.5, left])],
              { fill: "rgba(242,163,58,.10)", stroke: "rgba(242,163,58,.35)" });
      const reach = 0.4 * Math.min(T, W) / (Math.hypot(ht, hx) || 1);
      from = top([(T - 1) / 2 - ht * reach, (W - 1) / 2 - hx * reach]);
      to = top([(T - 1) / 2 + ht * reach, (W - 1) / 2 + hx * reach]);
    } else {
      polygon(ctx, [top([f0, f1]), top([f2, f3]), top([l2, l3]), top([l0, l1])],
              { fill: "rgba(242,163,58,.10)", stroke: "rgba(242,163,58,.35)" });
    }
    line(ctx, from, to, "rgba(242,163,58,.8)", [5, 4], 1.5);
    const angle = Math.atan2(to[1] - from[1], to[0] - from[0]);
    if (Math.hypot(to[0] - from[0], to[1] - from[1]) > 12) {
      polygon(ctx, [to, [to[0] - 9 * Math.cos(angle - 0.4), to[1] - 9 * Math.sin(angle - 0.4)],
                        [to[0] - 9 * Math.cos(angle + 0.4), to[1] - 9 * Math.sin(angle + 0.4)]],
              { fill: "rgba(242,163,58,.9)" });
    }
  }

  // Axis labels at the ends of the three edges leaving the first frame's
  // top-left corner.
  const origin = P(-0.5, -0.5, -0.5);
  const middle = P((W - 1) / 2, (H - 1) / 2, (T - 1) / 2);
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillStyle = "#9aa2b1";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  /** @type {[string, number[]][]} */
  const labels = [["x", P(W - 0.5, -0.5, -0.5)], ["y", P(-0.5, H - 0.5, -0.5)],
                  ["time", P(-0.5, -0.5, T - 0.5)]];
  for (const [label, end] of labels) {
    if (Math.hypot(end[0] - origin[0], end[1] - origin[1]) < 4) continue;  // end-on
    let ox = end[0] - middle[0], oy = end[1] - middle[1];
    const length = Math.hypot(ox, oy) || 1;
    ox = (ox / length) * 14;
    oy = (oy / length) * 14;
    ctx.fillText(label, end[0] + ox, end[1] + oy);
  }
}

/** @type {{ x: number, y: number, yaw: number, pitch: number } | null} */
let drag = null;
cube.addEventListener("pointerdown", (e) => {
  drag = { x: e.clientX, y: e.clientY, yaw: camera.yaw, pitch: camera.pitch };
  cube.setPointerCapture(e.pointerId);
});
cube.addEventListener("pointermove", (e) => {
  if (!drag) return;
  camera.yaw = drag.yaw + (e.clientX - drag.x) * 0.01;
  camera.pitch = Math.min(Math.max(drag.pitch - (e.clientY - drag.y) * 0.01,
                                   -Math.PI / 2), Math.PI / 2);
  drawCube();
});
for (const end of ["pointerup", "pointercancel"]) cube.addEventListener(end, () => { drag = null; });
cube.addEventListener("wheel", (e) => {
  e.preventDefault();
  camera.zoom = Math.min(Math.max(camera.zoom * Math.exp(-e.deltaY * 0.0015), 0.3), 10);
  drawCube();
}, { passive: false });
cube.addEventListener("dblclick", () => setView("three"));
const viewButtons =
  /** @type {NodeListOf<HTMLElement>} */ (document.querySelectorAll("[data-view]"));
for (const button of viewButtons) {
  button.addEventListener("click", () => setView(/** @type {string} */ (button.dataset.view)));
}

/** @param {string} name */
function setView(name) {
  [camera.yaw, camera.pitch] = VIEWS[name];
  camera.zoom = 1;
  drawCube();
}

new ResizeObserver(() => drawCube()).observe(cube);

// ---------------------------------------------------------------------------
// Renders, made here one at a time, each in a worker of its own, and kept until
// the page is left.

/** @type {AbortController | null} */
let job = null;     // the AbortController of the render under way
/**
 * @typedef {{ name: string, url: string, size: number, made: Date, about: string,
 *             saved: boolean }} Render
 * @type {Render[]}
 */
const renders = [];  // newest first

// What each stage of a render says, given how many frames it has done.
/** @type {Record<Stage, (done: number, total: number) => string>} */
const STAGES = {
  loading: (done, total) => `Loading the clip · frame ${done} of ${total}`,
  slicing: (done, total) => `Slicing · frame ${done} of ${total}`,
  finishing: () => "Finishing",
};

/** @param {boolean} preview */
async function render(preview) {
  const file = state.source;
  if (!file || job) return;
  if (UNABLE) return showJob({ error: true, text: UNABLE });
  const values = options();
  const controller = new AbortController();
  job = controller;
  enableRender();
  const began = performance.now();
  const kind = preview ? "preview" : "full-quality render";
  showJob({ running: true, text: "Starting…" });
  try {
    const made = await renderClip(file, values, {
      preview, signal: controller.signal,
      progress: ({ stage, done, total }) => showJob({
        running: true, done: done / total,
        text: `${STAGES[stage](done, total)} · ${seconds((performance.now() - began) / 1000)}`,
      }),
    });
    const took = (performance.now() - began) / 1000;
    const item = keep(made);
    showJob({ done: 1, text: `Made ${made.seconds.toFixed(1)} s of video in ${seconds(took)} ` +
                             `(${kind}).` });
    watch(item.url, item.name);
  } catch (err) {
    const error = /** @type {Error} */ (err);
    if (error.name === "AbortError") showJob({ text: "Cancelled." });
    else if (await hasChanged(file)) showJob({ error: true, text: changed(file.name) });
    else showJob({ error: true, text: error.message });
  } finally {
    job = null;
    enableRender();
  }
}

/**
 * What the render panel shows: whether one is running, how far through (0 to 1),
 * and what it says.
 * @param {{ running?: boolean, done?: number, text?: string, error?: boolean }} job
 */
function showJob({ running = false, done = 0, text = "", error = false }) {
  $("job").hidden = false;
  $("cancel").hidden = !running;
  $("job-bar").style.width = `${100 * done}%`;
  $("job-text").setAttribute("aria-busy", String(running));  // so only how it ends is read
  $("job-text").classList.toggle("error", error);
  $("job-text").textContent = text;
}

// The render buttons work once a video is chosen, for one render at a time.
function enableRender() {
  const off = Boolean(job) || !state.source;
  $("render-preview").disabled = off;
  $("render-full").disabled = off;
}

$("render-preview").addEventListener("click", () => render(true));
$("render-full").addEventListener("click", () => render(false));
$("cancel").addEventListener("click", () => job?.abort());

// Leaving the page loses a render under way and every render not yet downloaded.
window.addEventListener("beforeunload", (event) => {
  if (job || renders.some((r) => !r.saved)) event.preventDefault();
});

/**
 * Keep a render in the list, in place of an earlier one of the same name.
 * @param {Rendered} made
 */
function keep(made) {
  const earlier = renders.findIndex((r) => r.name === made.name);
  if (earlier >= 0) URL.revokeObjectURL(renders.splice(earlier, 1)[0].url);
  const item = {
    name: made.name, url: URL.createObjectURL(made.video), size: made.video.size,
    made: new Date(), saved: false,
    about: `${made.width}×${made.height}, ${made.frames} frames, ${made.rateControl}`,
  };
  renders.unshift(item);
  showRenders();
  return item;
}

function showRenders() {
  const list = $("renders");
  list.replaceChildren();
  for (const r of renders) {
    const item = document.createElement("li");
    item.dataset.src = r.url;
    const name = link(r.name, () => watch(r.url, r.name));
    name.classList.add("name");
    const meta = document.createElement("span");
    meta.className = "note";
    meta.textContent = `${bytes(r.size)} · ${r.made.toLocaleTimeString()} · ${r.about}`;
    const download = document.createElement("a");
    download.href = r.url;
    download.download = r.name;
    download.textContent = "Download";
    download.addEventListener("click", () => { r.saved = true; });
    const remove = link("Remove", () => forget(r));
    remove.classList.add("remove");
    item.append(name, meta, download, remove);
    list.append(item);
  }
  if (!renders.length) {
    const item = document.createElement("li");
    item.className = "note";
    item.textContent = "Nothing rendered yet. Renders stay here until you leave or reload " +
      "the page, so download the ones to keep.";
    list.append(item);
  }
  markPlaying();
}

/**
 * Drop a render from the list and let its video go.
 * @param {Render} r
 */
function forget(r) {
  renders.splice(renders.indexOf(r), 1);
  if ($("player").src === r.url) {
    $("player").removeAttribute("src");
    $("player").load();
    $("player").hidden = true;
    $("now-playing").textContent = "";
  }
  URL.revokeObjectURL(r.url);
  showRenders();
}

/**
 * @param {string} text
 * @param {() => void} onclick
 */
function link(text, onclick) {
  const button = document.createElement("button");
  button.className = "link";
  button.textContent = text;
  button.onclick = onclick;
  return button;
}

/**
 * Play a video in the player: the chosen file, or a render.
 * @param {string} url
 * @param {string} label
 * @param {File | null} [file]
 */
function watch(url, label, file = null) {
  const player = $("player"), note = $("now-playing");
  player.onerror = async () => {  // as for an original in a format only the page itself reads
    const gone = file && await hasChanged(file);
    if (player.src !== url) return;  // another video since
    player.hidden = true;
    note.textContent = gone ? changed(label) : `This browser's player can't play ${label}.`;
    note.classList.add("error");
    markPlaying();
  };
  player.hidden = false;
  player.src = url;
  player.play().catch(() => { /* autoplay may be blocked; the controls still work */ });
  note.textContent = label;
  note.classList.remove("error");
  markPlaying();
}

function markPlaying() {
  const playing = $("player").hidden ? "" : $("player").src;
  for (const item of $("renders").children) {
    item.classList.toggle("playing", /** @type {HTMLElement} */ (item).dataset.src === playing);
  }
}

// ---------------------------------------------------------------------------

function start() {
  $("unserved").remove();  // the note for when this script can't run
  slicesChanged();
  noiseChanged();
  showRenders();
  showSources();
  enableRender();  // as Firefox may keep a button's state from before a reload
  showMessage(UNABLE || NO_SOURCE);
}

start();
