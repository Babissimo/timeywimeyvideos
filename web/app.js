// The page: open a video, set the slice, watch it play live, see where the
// slice sits in the cuboid, and render it. The video stays on this computer:
// the live view and the renders decode and slice it here, on the GPU.

import { probe } from "./src/decode.ts";
import { Live } from "./src/live.ts";
import { handOver, render as renderClip } from "./src/render.ts";
import { changed, hasChanged, same, Sources, VIDEO_TYPES } from "./src/sources.ts";

const $ = (id) => document.getElementById(id);

// Why this page can't decode or encode video here, if it can't. WebCodecs, which does both,
// runs only in a secure context.
const UNABLE = !window.isSecureContext ?
  `The page can only decode video in a secure context, which ${location.origin} isn't: ` +
  "open it at localhost or over HTTPS." :
  !("VideoDecoder" in window && "VideoEncoder" in window) ?
  "This browser can't decode or encode video in a page, as it has no WebCodecs. Use a " +
  "recent Chrome or Edge." : "";

const state = {
  source: null,    // the File chosen
  id: "",          // and its id in the list
  chosen: 0,       // how many times a source has been chosen
  info: null,      // the live view's description of the frame on screen
  pos: 0.5,        // how far through the output video, 0 to 1
  playing: false,
  scrubbing: false,
};

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
  return document.querySelector('input[name="slice"]:checked').value;
}

// What the frame does at the video's edges: black, inside or loop.
function edges() {
  return document.querySelector('input[name="edges"]:checked').value;
}

function setEdges(value) {
  document.querySelector(`input[name="edges"][value="${value}"]`).checked = true;
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
    angle: $("angle").value || "0",
    inside: edges() === "inside" ? "1" : "",
    motion: moves() ? $("motion").value : "",
    loop: loop ? "1" : "",
    loop_fade: loop ? $("loop-fade").value : "",
    loop_sides: sides ? "1" : "",
    loop_side_fade: sides ? $("side-fade").value : "",
    noise: noise ? $("noise-amount").value || "0" : "",
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

function seconds(s) {
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.round(s - m * 60)).padStart(2, "0")}`;
}

function bytes(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  return `${Math.max(1, Math.round(n / 1e3))} KB`;
}

// ---------------------------------------------------------------------------
// Live preview. Only one frame is worked on at a time; if the options or
// position change meanwhile, the latest ones are shown next, so dragging a
// slider never queues up stale frames.

let live = null;  // made on first use, so a browser without WebGL2 gets a message
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
  if (unable) throw unable;
  return live;
}

function refresh() {
  again = true;
  pump();
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
  const loading = (text) => {
    if (state.chosen === chosen) showMessage(text);
  };
  let slow = false;
  const timer = setTimeout(() => {
    slow = true;
    loading("Loading the clip…");
  }, 300);
  try {
    const info = await liveView().show(file, opts, state.pos, (done, total) => {
      if (slow) loading(`Loading the clip… ${Math.floor(100 * done / total)}%`);
    });
    const resized = !state.info || state.info.frames !== info.frames;
    state.info = info;
    if (resized) anchor();
    hideMessage();
    describe();
    drawCube();
    loadFace();
  } catch (err) {
    if (err.name !== "AbortError") {  // else another source took over
      pause();
      showMessage(err.message,
        err.kind === "does_not_fit" ?
          ["Let black in instead", () => setEdges("black")] :
        err.kind === "sideways" &&
          ["Wrap round the sides too", () => { $("loop-sides").checked = true; slicesChanged(); }]);
    }
  } finally {
    clearTimeout(timer);
    busy = false;
    pump();
  }
}

function showMessage(text, action) {
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
}

function describe() {
  const info = state.info;
  if (!info) return;
  $("live-info").textContent =
    `${info.width}×${info.height} · ${info.frames} frames · ${seconds(info.seconds)}`;
  $("frame-label").textContent = `${info.frame + 1} / ${info.frames}`;
  if (!state.scrubbing) $("scrub").value = Math.round(state.pos * 1000);

  const full = info.full;
  $("estimate").classList.toggle("error", Boolean(full && full.error));
  $("estimate").textContent = !full ? "" : full.error ? `At full size: ${full.error}` :
    `Full quality makes ${full.width}×${full.height}, ${full.frames} frames ` +
    `(${seconds(full.seconds)}), holding ${bytes(full.memory)} of video on the GPU.`;
}

// Playback runs on the clock: each animation frame works out which output
// frame is due and shows it, skipping any the live view can't keep up with.

const clock = { start: 0, frame: 0 };

function anchor() {
  clock.start = performance.now();
  clock.frame = state.info ? state.pos * (state.info.frames - 1) : 0;
}

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
  requestAnimationFrame(tick);
}

function play() {
  if (!state.info || state.info.frames < 2) return;
  if (state.info.frame === state.info.frames - 1) state.pos = 0;
  state.playing = true;
  $("play").textContent = "❚❚";
  $("play").setAttribute("aria-label", "Pause");
  anchor();
  requestAnimationFrame(tick);
}

function pause() {
  state.playing = false;
  $("play").textContent = "▶";
  $("play").setAttribute("aria-label", "Play");
}

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
  state.pos = $("scrub").value / 1000;
  refresh();
});
for (const done of ["pointerup", "pointercancel", "change"]) {
  $("scrub").addEventListener(done, () => {
    state.scrubbing = false;
    anchor();
  });
}

document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, select, textarea") && e.target.type !== "range") return;
  if (e.key === " " && e.target.tagName !== "BUTTON") {
    e.preventDefault();
    state.playing ? pause() : play();
  } else if ((e.key === "ArrowRight" || e.key === "ArrowLeft") && e.target.type !== "range") {
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
    input.min = -limit;
    input.max = limit;
  }
  const angle = Number($("angle").value) || 0;
  if (Math.abs(angle) > limit) setAngle(Math.sign(angle) * limit);
  const loop = edges() === "loop", sides = loop && $("loop-sides").checked;
  $("edges-note").textContent = DESCRIBE[edges()];
  $("loop-fields").hidden = !loop;
  $("side-fade-field").hidden = !sides;
  // No line round the sides is the longest.
  $("motion").querySelector('[value="longest"]').disabled = sides;
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

function setAngle(value) {
  $("angle").value = value;
  $("angle-range").value = value;
}

for (const input of document.querySelectorAll(
  'input[name="slice"], input[name="edges"], #motion, #loop-sides')) {
  input.addEventListener("change", slicesChanged);
}
$("angle-range").addEventListener("input", () => {
  $("angle").value = $("angle-range").value;
  refresh();
});
$("angle").addEventListener("input", () => {
  if ($("angle").value === "" || !$("angle").checkValidity()) return;
  $("angle-range").value = $("angle").value;
  refresh();
});
for (const id of ["start", "duration", "scale", "fps", "loop-fade", "side-fade"]) {
  $(id).addEventListener("change", refresh);
}

function noiseChanged() {
  const on = $("noise").checked;
  const direction = $("noise-direction").value;
  $("noise-fields").hidden = !on;
  $("noise-note").textContent = DESCRIBE[`noise-${direction}`];
  $("cube-noise").hidden = !on;
  $("cube-noise").textContent = "With noise on, the real surface is bumpy: the frame " +
    `is drawn on the flat plane its points are pushed off, up to ${$("noise-amount").value || 0} ` +
    `frames ${direction === "time" ? "through time" : "perpendicular to it"}.`;
  refresh();
}

for (const id of ["noise", "noise-size", "noise-speed", "noise-direction", "noise-seed"]) {
  $(id).addEventListener("change", noiseChanged);
}
$("noise-range").addEventListener("input", () => {
  $("noise-amount").value = $("noise-range").value;
  noiseChanged();
});
$("noise-amount").addEventListener("input", () => {
  if ($("noise-amount").value === "" || !$("noise-amount").checkValidity()) return;
  $("noise-range").value = $("noise-amount").value;
  noiseChanged();
});

// ---------------------------------------------------------------------------
// Sources: videos opened on this computer, on their own or a folder at a time,
// which the page reads where they lie as it needs them.

const sources = new Sources();
const NO_SOURCE = "Open a video, or drop one here.";
let original = null;  // the address "watch the original" plays the chosen file at

function showSources() {
  const list = $("source");
  list.replaceChildren(new Option("Choose a video…", ""));
  if (sources.folder !== null) list.append(group(`Folder: ${sources.folder}`, sources.inFolder));
  if (sources.opened.length) list.append(group("Opened", sources.opened));
  list.value = sources.chosen?.id ?? "";
}

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

// A note on the folder, apart from the chosen video's details.
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
  state.info = null;
  state.pos = 0.5;
  face = null;
  faceImage = null;
  drawCube();
  $("source-info").textContent = "";
  $("live-info").textContent = "";
  $("frame-label").textContent = "";
  $("estimate").textContent = "";
  if (!file) {
    showMessage(UNABLE || NO_SOURCE);
    return;
  }
  refresh();
  const chosen = state.chosen;
  try {
    const info = await probe(file);
    if (state.chosen !== chosen) return;
    $("source-info").replaceChildren(
      `${info.width}×${info.height} · ${Number((info.fps.num / info.fps.den).toFixed(3))} fps` +
      (info.duration ? ` · ${seconds(info.duration)}` : "") + ` · ${bytes(file.size)} · `,
      link("watch the original", watchOriginal));
  } catch (err) {
    const message = await hasChanged(file) ? changed(file.name) : err.message;
    if (state.chosen === chosen) $("source-info").textContent = message;
  }
}

function watchOriginal() {
  original ??= URL.createObjectURL(state.source);
  watch(original, state.source.name);
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

function openFiles(files) {
  const id = sources.open(files);
  if (!id) return;
  showSources();
  chooseSource(id);
}

async function pickFolder() {
  let folder;
  try {
    folder = await window.showDirectoryPicker({ id: "videos", startIn: "videos" });
  } catch (err) {
    if (err.name !== "AbortError") folderNote(err.message, true);
    return;  // else they closed the picker
  }
  await openFolder(folder);
}

async function openFolder(folder) {
  try {
    await sources.openFolder(folder);
  } catch (err) {
    folderNote(`Couldn't read ${folder.name}: ${err.message}`, true);
    return;
  }
  showSources();
  describeFolder();
  // The video chosen stays, but comes afresh if the folder opened again holds a newer copy.
  const chosen = sources.chosen;
  if (chosen && chosen.file !== state.source) chooseSource(chosen.id);
}

$("source").addEventListener("change", () => chooseSource($("source").value));
$("file").accept = ["video/*", ...VIDEO_TYPES].join(",");
$("file").addEventListener("change", () => {
  openFiles([...$("file").files]);
  $("file").value = "";
});
$("open-folder").hidden = !("showDirectoryPicker" in window);
$("open-folder").addEventListener("click", pickFolder);
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
  const items = [...e.dataTransfer.items].filter((item) => item.kind === "file");
  const folders = items.map((item) => item.webkitGetAsEntry?.()?.isDirectory ?? false);
  const files = items.filter((_, i) => !folders[i]).map((item) => item.getAsFile());
  // The first folder opens as the folder, where the browser gives its handle.
  const dropped = items.find((_, i) => folders[i]);
  const handle = dropped?.getAsFileSystemHandle?.();
  openFiles(files.filter(Boolean));
  if (!dropped) return;
  if (!handle) {
    folderNote("This browser can't open a dropped folder: " + ("showDirectoryPicker" in window ?
      "use Open a folder of videos instead." : "open or drop the videos in it instead."));
    return;
  }
  try {
    await openFolder(await handle);
  } catch (err) {
    folderNote(err.message, true);
  }
});

// ---------------------------------------------------------------------------
// The cuboid: a wireframe of the clip in volume units (one pixel or one frame
// per unit), with its first frame on the front face, the current output frame
// drawn where it slices through, and the region the sweep covers shaded on top.
// The view is orthographic, so every flat rectangle stays a parallelogram and
// images can be mapped onto them with a plain 2D transform.

const cube = $("cube");
const VIEWS = {
  three: [-0.6, -0.45],
  front: [0, 0],
  top: [0, -Math.PI / 2],
  side: [-Math.PI / 2, 0],
};
const camera = { yaw: VIEWS.three[0], pitch: VIEWS.three[1], zoom: 1 };
let face = null;       // canvas holding the clip's first frame
let faceImage = null;  // the live view's first frame, which it keeps while it keeps the clip

function loadFace() {
  const image = live.face();
  if (image === faceImage) return;
  faceImage = image;
  face = null;
  if (image) {
    face = document.createElement("canvas");
    face.width = image.width;
    face.height = image.height;
    face.getContext("2d").putImageData(image, 0, 0);
  }
  drawCube();
}

// A function from volume coordinates (x, y, t) to [screen x, screen y, depth].
function projector(volume) {
  const [T, H, W] = volume;
  const w = cube.clientWidth, h = cube.clientHeight;
  const k = camera.zoom * Math.min(w, h) / (Math.hypot(W, H, T) * 1.1);
  const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
  const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
  const project = (x, y, t) => {
    // Centre the cuboid, with y up and t running away from the viewer.
    const X = x - (W - 1) / 2, Y = (H - 1) / 2 - y, Z = t - (T - 1) / 2;
    const x1 = X * cy - Z * sy, z1 = X * sy + Z * cy;
    return [w / 2 + k * x1, h / 2 - k * (Y * cp - z1 * sp), Y * sp + z1 * cp];
  };
  // Whether the face with this outward normal (in x, y, t) faces the viewer.
  project.facing = (nx, ny, nt) => -ny * sp + (nx * sy + nt * cy) * cp < 0;
  return project;
}

// Draw an image onto a flat quad. map(u, v) gives the screen position of
// image coordinates (u, v); only columns u0 to u1 are drawn.
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

// The columns u (0 to width, in image coordinates) of the output frame that
// lie inside the cuboid once moved back by (shiftT, shiftX), or null if none do.
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
  const ctx = cube.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const info = state.info;
  if (!info) return;

  const P = projector(info.volume);
  const [T, H, W] = info.volume;
  const ends = [[-0.5, W - 0.5], [-0.5, H - 0.5], [-0.5, T - 0.5]];
  const corner = (bits) => P(ends[0][bits[0]], ends[1][bits[1]], ends[2][bits[2]]);
  const facing = (axis, high) => {
    const n = [0, 0, 0];
    n[axis] = high ? 1 : -1;
    return P.facing(...n);
  };

  // Each edge, and whether either face it borders is turned towards us.
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
  const quad = (map, u0, u1, v1) => [map(u0, 0), map(u1, 0), map(u1, v1), map(u0, v1)];

  // A looping frame runs round time in a ring, and on a loop round the sides
  // round x too, so each part of it past an end of the cuboid is drawn where
  // it wraps round to: the frame moved back by whole clip lengths or widths.
  const turns = (from, to, n, wraps) => {
    if (!wraps) return [0];
    const first = Math.floor((Math.min(from, to) + 0.5) / n);
    const last = Math.floor((Math.max(from, to) + 0.5) / n);
    return Array.from({ length: last - first + 1 }, (_, i) => (first + i) * n);
  };
  const layers = [];
  for (const shiftT of turns(t0 - 0.5 * dt, t1 + 0.5 * dt, T, info.loop)) {
    for (const shiftX of turns(x0 - 0.5 * dx, x1 + 0.5 * dx, W, info.sides)) {
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
    const onFace = (u, v) => P(u - 0.5, v - 0.5, -0.5);
    layers.push({
      depth: P((W - 1) / 2, (H - 1) / 2, -0.5)[2],
      draw: () => drawOnto(ctx, face, onFace, 0, face.width, facing(2, 0) ? 0.35 : 0.2),
    });
  }
  layers.sort((a, b) => b.depth - a.depth).forEach((layer) => layer.draw());

  for (const [p, q, front] of edges) if (front) line(ctx, p, q, "rgba(205,212,224,.8)");

  // The strip of x-t the sweep passes through, on the top face, and the way
  // the frame's centre moves.
  if (info.frames > 1) {
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
  for (const [label, end] of [["x", P(W - 0.5, -0.5, -0.5)], ["y", P(-0.5, H - 0.5, -0.5)],
                              ["time", P(-0.5, -0.5, T - 0.5)]]) {
    if (Math.hypot(end[0] - origin[0], end[1] - origin[1]) < 4) continue;  // end-on
    let ox = end[0] - middle[0], oy = end[1] - middle[1];
    const length = Math.hypot(ox, oy) || 1;
    ox = (ox / length) * 14;
    oy = (oy / length) * 14;
    ctx.fillText(label, end[0] + ox, end[1] + oy);
  }
}

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
for (const button of document.querySelectorAll("[data-view]")) {
  button.addEventListener("click", () => setView(button.dataset.view));
}

function setView(name) {
  [camera.yaw, camera.pitch] = VIEWS[name];
  camera.zoom = 1;
  drawCube();
}

new ResizeObserver(() => drawCube()).observe(cube);

// ---------------------------------------------------------------------------
// Renders, made here one at a time, each in a worker of its own, and kept until
// the page is left.

let job = null;     // the AbortController of the render under way
const renders = [];  // newest first: { name, url, size, made, about, saved }

// What each stage of a render says, given how many frames it has done.
const STAGES = {
  loading: (done, total) => `Loading the clip · frame ${done} of ${total}`,
  slicing: (done, total) => `Slicing · frame ${done} of ${total}`,
  finishing: () => "Finishing",
};

async function render(preview) {
  const file = state.source;
  if (!file || job) return;
  if (UNABLE) return showJob({ error: true, text: UNABLE });
  const values = options();
  const controller = new AbortController();
  job = controller;
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
    if (err.name === "AbortError") showJob({ text: "Cancelled." });
    else if (await hasChanged(file)) showJob({ error: true, text: changed(file.name) });
    else if (err.command) showJob({ error: true, text: handOver(err), command: err.command });
    else showJob({ error: true, text: err.message });
  } finally {
    job = null;
  }
}

// What the render panel shows: whether one is running, how far through (0 to 1),
// what it says, and for a clip the browser can't render, timeslice.py's command.
function showJob({ running = false, done = 0, text = "", error = false, command }) {
  $("render-preview").disabled = running;
  $("render-full").disabled = running;
  $("job").hidden = false;
  $("cancel").hidden = !running;
  $("job-bar").style.width = `${100 * done}%`;
  $("job-text").classList.toggle("error", error);
  $("job-text").textContent = text;
  $("job-command").hidden = !command;
  $("job-command").textContent = command || "";
}

$("render-preview").addEventListener("click", () => render(true));
$("render-full").addEventListener("click", () => render(false));
$("cancel").addEventListener("click", () => job?.abort());

// Leaving the page loses a render under way and every render not yet downloaded.
window.addEventListener("beforeunload", (event) => {
  if (job || renders.some((r) => !r.saved)) event.preventDefault();
});

// Keep a render in the list, in place of an earlier one of the same name.
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

// Drop a render from the list and let its video go.
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

function link(text, onclick) {
  const button = document.createElement("button");
  button.className = "link";
  button.textContent = text;
  button.onclick = onclick;
  return button;
}

// Play a video in the player: the chosen file, or a render.
function watch(url, label) {
  const player = $("player");
  player.hidden = false;
  player.src = url;
  player.play().catch(() => { /* autoplay may be blocked; the controls still work */ });
  $("now-playing").textContent = label;
  markPlaying();
}

function markPlaying() {
  const playing = $("player").hidden ? "" : $("player").src;
  for (const item of $("renders").children) {
    item.classList.toggle("playing", item.dataset.src === playing);
  }
}

// ---------------------------------------------------------------------------

function start() {
  $("unserved").remove();  // the note for when this script can't run
  slicesChanged();
  noiseChanged();
  showRenders();
  showSources();
  showMessage(UNABLE || NO_SOURCE);
}

start();
