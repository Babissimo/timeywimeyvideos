"use strict";
// The page for webapp.py: choose a source, set the slice, watch live frames
// from the server, see where the slice sits in the cuboid, and run renders.

const $ = (id) => document.getElementById(id);

const state = {
  source: "",
  info: null,      // the server's description of the frame on screen
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
    source: state.source,
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

async function problemFrom(res) {
  let message = `The server said ${res.status}.`, kind = "error";
  try {
    const body = await res.json();
    message = body.error || message;
    kind = body.kind || kind;
  } catch { /* not JSON */ }
  return Object.assign(new Error(message), { kind });
}

async function getJSON(url, init) {
  const res = await fetch(url, init);
  if (!res.ok) throw await problemFrom(res);
  return res.json();
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

// Paint raw RGB bytes from the server onto a canvas.
function paint(canvas, width, height, rgb) {
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(width, height);
  const px = image.data;
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    px[j] = rgb[i];
    px[j + 1] = rgb[i + 1];
    px[j + 2] = rgb[i + 2];
    px[j + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}

function frameURL(opts, pos) {
  return "/api/frame?" + new URLSearchParams({ ...opts, pos });
}

// ---------------------------------------------------------------------------
// Live preview. Only one frame request is in flight at a time; if the options
// or position change meanwhile, the latest ones are fetched next, so dragging
// a slider never queues up stale frames.

let busy = false;
let again = false;

function refresh() {
  again = true;
  pump();
}

async function pump() {
  if (busy || !again) return;
  again = false;
  if (!state.source) return;
  busy = true;
  const opts = options();
  const slow = setTimeout(() => showMessage("Loading the clip…"), 300);
  try {
    const res = await fetch(frameURL(opts, state.pos));
    if (!res.ok) throw await problemFrom(res);
    const info = JSON.parse(res.headers.get("X-Info"));
    paint($("live"), info.width, info.height, new Uint8Array(await res.arrayBuffer()));
    const resized = !state.info || state.info.frames !== info.frames;
    state.info = info;
    if (resized) anchor();
    hideMessage();
    describe();
    drawCube();
    loadFace(opts);
  } catch (err) {
    pause();
    showMessage(err.message,
      err.kind === "does_not_fit" ?
        ["Let black in instead", () => setEdges("black")] :
      err.kind === "sideways" &&
        ["Wrap round the sides too", () => { $("loop-sides").checked = true; slicesChanged(); }]);
  } finally {
    clearTimeout(slow);
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
    `(${seconds(full.seconds)}), holding ${bytes(full.memory)} of video in memory.`;
}

// Playback runs on the clock: each animation frame works out which output
// frame is due and asks for it, skipping any the server can't keep up with.

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
// Sources.

async function loadSources(select) {
  const data = await getJSON("/api/sources");
  const list = $("source");
  list.replaceChildren(new Option("Choose a video…", ""));
  const folder = data.folder.split(/[\\/]/).filter(Boolean).pop() || data.folder;
  for (const [label, kind, names] of [[`Folder: ${folder}`, "videos", data.videos],
                                       ["Uploads", "uploads", data.uploads]]) {
    if (!names.length) continue;
    const group = document.createElement("optgroup");
    group.label = label;
    for (const name of names) group.append(new Option(name, `${kind}/${name}`));
    list.append(group);
  }
  list.value = select || "";
  if (!data.videos.length && !data.uploads.length) {
    $("source-info").textContent = `No videos in ${data.folder} yet. Upload one, ` +
      "or restart with --videos pointing at a folder of videos.";
  }
}

async function chooseSource(source) {
  pause();
  state.source = source;
  state.info = null;
  state.pos = 0.5;
  face = null;
  faceKey = "";
  drawCube();
  $("source-info").textContent = "";
  $("live-info").textContent = "";
  $("frame-label").textContent = "";
  $("estimate").textContent = "";
  if (!source) {
    showMessage("Pick a video, or upload one.");
    return;
  }
  try {
    const info = await getJSON("/api/info?" + new URLSearchParams({ source }));
    const name = source.slice(source.indexOf("/") + 1);
    $("source-info").replaceChildren(
      `${info.width}×${info.height} · ${Number(info.fps.toFixed(3))} fps` +
      (info.duration ? ` · ${seconds(info.duration)}` : "") + ` · ${bytes(info.size)} · `,
      link("watch the original", () => watch(source, name)));
  } catch (err) {
    $("source-info").textContent = err.message;
  }
  refresh();
}

$("source").addEventListener("change", () => chooseSource($("source").value));

function upload(file) {
  const bar = $("upload-bar");
  bar.hidden = false;
  bar.firstElementChild.style.width = "0";
  $("source-info").textContent = `Uploading ${file.name}…`;
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload");
  xhr.setRequestHeader("X-Filename", encodeURIComponent(file.name));
  xhr.setRequestHeader("Content-Type", "application/octet-stream");
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) bar.firstElementChild.style.width = `${100 * e.loaded / e.total}%`;
  };
  xhr.onload = async () => {
    bar.hidden = true;
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
    if (xhr.status !== 200) {
      $("source-info").textContent = body.error || `Upload failed (${xhr.status}).`;
      return;
    }
    await loadSources(body.source);
    chooseSource(body.source);
  };
  xhr.onerror = () => {
    bar.hidden = true;
    $("source-info").textContent = "Upload failed: lost the connection to the server.";
  };
  xhr.send(file);
}

$("file").addEventListener("change", () => {
  if ($("file").files[0]) upload($("file").files[0]);
  $("file").value = "";
});
document.addEventListener("dragover", (e) => {
  e.preventDefault();
  $("drop").classList.add("over");
});
document.addEventListener("dragleave", (e) => {
  if (!e.relatedTarget) $("drop").classList.remove("over");
});
document.addEventListener("drop", (e) => {
  e.preventDefault();
  $("drop").classList.remove("over");
  if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
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
let face = null;      // canvas holding the clip's first frame
let faceKey = "";

async function loadFace(opts) {
  const key = [opts.source, opts.start, opts.duration, opts.scale, opts.loop_fade,
               opts.loop_side_fade].join("|");
  if (key === faceKey) return;
  faceKey = key;
  face = null;
  try {
    const res = await fetch(frameURL(
      { ...opts, slice: "rotate", angle: 0, inside: "", motion: "", noise: "" }, 0));
    if (!res.ok || key !== faceKey) return;
    const info = JSON.parse(res.headers.get("X-Info"));
    const canvas = document.createElement("canvas");
    paint(canvas, info.width, info.height, new Uint8Array(await res.arrayBuffer()));
    if (key !== faceKey) return;
    face = canvas;
    drawCube();
  } catch { /* the cuboid just goes without */ }
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
// Renders.

let polling = 0;

async function render(preview) {
  if (!state.source) return;
  try {
    showJob(await getJSON("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...options(), preview }),
    }));
    poll();
  } catch (err) {
    showJob({ state: "failed", message: err.message });
  }
}

function poll() {
  clearTimeout(polling);
  polling = setTimeout(async () => {
    let job;
    try {
      job = await getJSON("/api/render");
    } catch {
      return poll();  // try again
    }
    showJob(job);
    if (job.state === "running") return poll();
    if (job.state === "done") {
      await loadRenders();
      watch(`renders/${job.output}`, job.output);
    }
  }, 400);
}

function showJob(job) {
  const running = job.state === "running";
  $("render-preview").disabled = running;
  $("render-full").disabled = running;
  $("job").hidden = job.state === "idle";
  $("cancel").hidden = !running;
  const done = running && job.frames ? job.frame / job.frames : job.state === "done" ? 1 : 0;
  $("job-bar").style.width = `${100 * done}%`;
  $("job-text").classList.toggle("error", job.state === "failed");
  const kind = job.preview ? "preview" : "full-quality render";
  $("job-text").textContent =
    running ? `${job.stage}${job.frames ? ` · frame ${job.frame} of ${job.frames}` : "…"}` +
              ` · ${seconds(job.elapsed || 0)}` :
    job.state === "done" ? `${job.message || "Done"} (${kind}).` :
    job.state === "cancelled" ? "Cancelled." :
    job.message || "";
}

$("render-preview").addEventListener("click", () => render(true));
$("render-full").addEventListener("click", () => render(false));
$("cancel").addEventListener("click", async () => showJob(await getJSON("/api/render/cancel", { method: "POST" })));

async function loadRenders() {
  const list = $("renders");
  let renders = [];
  try {
    renders = await getJSON("/api/renders");
  } catch { /* leave the list empty */ }
  list.replaceChildren();
  for (const r of renders) {
    const item = document.createElement("li");
    item.dataset.src = `renders/${r.name}`;
    const name = link(r.name, () => watch(`renders/${r.name}`, r.name, r.modified));
    name.classList.add("name");
    const meta = document.createElement("span");
    meta.className = "note";
    meta.textContent = `${bytes(r.size)} · ${new Date(r.modified * 1000).toLocaleString()}`;
    const download = document.createElement("a");
    download.href = `/media/renders/${encodeURIComponent(r.name)}`;
    download.download = r.name;
    download.textContent = "Download";
    item.append(name, meta, download);
    list.append(item);
  }
  if (!renders.length) {
    const item = document.createElement("li");
    item.className = "note";
    item.textContent = "Nothing rendered yet. Renders are saved in the renders folder.";
    list.append(item);
  }
}

function link(text, onclick) {
  const button = document.createElement("button");
  button.className = "link";
  button.textContent = text;
  button.onclick = onclick;
  return button;
}

// Play a source ("videos/…", "uploads/…") or render ("renders/…") in the player.
function watch(src, label, version = Date.now()) {
  const player = $("player");
  const [kind, ...rest] = src.split("/");
  player.hidden = false;
  player.src = `/media/${kind}/${encodeURIComponent(rest.join("/"))}?v=${version}`;
  player.play().catch(() => { /* autoplay may be blocked; the controls still work */ });
  $("now-playing").textContent = label;
  for (const item of $("renders").children) {
    item.classList.toggle("playing", item.dataset.src === src);
  }
}

// ---------------------------------------------------------------------------

async function start() {
  slicesChanged();
  noiseChanged();
  await Promise.all([loadSources(), loadRenders()]);
  const job = await getJSON("/api/render");
  showJob(job);
  if (job.state === "running") poll();
}

start();
