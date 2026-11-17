// The cuboid: a wireframe of the clip in volume units (one pixel or one frame
// per unit), with its first frame on the front face, the current output frame
// drawn where it slices through, and the region the sweep covers shaded on top.
// The view is orthographic, so every flat rectangle stays a parallelogram and
// images can be mapped onto them with a plain 2D transform.

import type { FrameInfo } from "../live";
import { $ } from "./elements";
import { state } from "./state";
import { live } from "./view";

const cube = $("cube");
const VIEWS: Record<string, [number, number]> = {
  three: [-0.6, -0.45],
  front: [0, 0],
  top: [0, -Math.PI / 2],
  side: [-Math.PI / 2, 0],
};
const camera = { yaw: VIEWS.three[0], pitch: VIEWS.three[1], zoom: 1 };
let face: HTMLCanvasElement | null = null;  // canvas holding the clip's first frame
// The live view's first frame, which it keeps while it keeps the clip.
let faceImage: ImageData | null = null;

export function loadFace() {
  const image = live?.face() ?? null;
  if (image === faceImage) return;
  faceImage = image;
  face = null;
  if (image) {
    face = document.createElement("canvas");
    face.width = image.width;
    face.height = image.height;
    (face.getContext("2d") as CanvasRenderingContext2D).putImageData(image, 0, 0);
  }
  drawCube();
}

// Let go of the first frame, as the clip it came from is gone.
export function forgetFace() {
  face = null;
  faceImage = null;
}

/** A function from volume coordinates (x, y, t) to [screen x, screen y, depth]. */
function projector(volume: [number, number, number]) {
  const [T, H, W] = volume;
  const w = cube.clientWidth, h = cube.clientHeight;
  const k = camera.zoom * Math.min(w, h) / (Math.hypot(W, H, T) * 1.1);
  const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
  const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
  const project = (x: number, y: number, t: number) => {
    // Centre the cuboid, with y up and t running away from the viewer.
    const X = x - (W - 1) / 2, Y = (H - 1) / 2 - y, Z = t - (T - 1) / 2;
    const x1 = X * cy - Z * sy, z1 = X * sy + Z * cy;
    return [w / 2 + k * x1, h / 2 - k * (Y * cp - z1 * sp), Y * sp + z1 * cp];
  };
  // Whether the face with this outward normal (in x, y, t) faces the viewer.
  project.facing = (nx: number, ny: number, nt: number) => -ny * sp + (nx * sy + nt * cy) * cp < 0;
  return project;
}

/**
 * Draw an image onto a flat quad. map(u, v) gives the screen position of
 * image coordinates (u, v); only columns u0 to u1 are drawn.
 */
function drawOnto(ctx: CanvasRenderingContext2D, image: HTMLCanvasElement,
                  map: (u: number, v: number) => number[], u0: number, u1: number,
                  alpha: number) {
  const o = map(0, 0), a = map(1, 0), b = map(0, 1);
  const dpr = devicePixelRatio || 1;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.setTransform(dpr * (a[0] - o[0]), dpr * (a[1] - o[1]),
                   dpr * (b[0] - o[0]), dpr * (b[1] - o[1]), dpr * o[0], dpr * o[1]);
  ctx.drawImage(image, u0, 0, u1 - u0, image.height, u0, 0, u1 - u0, image.height);
  ctx.restore();
}

function polygon(ctx: CanvasRenderingContext2D, points: number[][],
                 { fill, stroke, dash = [], width = 1 }:
                   { fill?: string; stroke?: string; dash?: number[]; width?: number }) {
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

function line(ctx: CanvasRenderingContext2D, p: number[], q: number[], style: string,
              dash: number[] = [], width = 1) {
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
 */
function insideSpan(info: FrameInfo, dx: number, dt: number, shiftT = 0, shiftX = 0) {
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

export function drawCube() {
  const dpr = devicePixelRatio || 1;
  const w = cube.clientWidth, h = cube.clientHeight;
  if (cube.width !== Math.round(w * dpr) || cube.height !== Math.round(h * dpr)) {
    cube.width = Math.round(w * dpr);
    cube.height = Math.round(h * dpr);
  }
  const ctx = cube.getContext("2d") as CanvasRenderingContext2D;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const info = state.info;
  if (!info) return;

  const P = projector(info.volume);
  const [T, H, W] = info.volume;
  const ends = [[-0.5, W - 0.5], [-0.5, H - 0.5], [-0.5, T - 0.5]];
  const corner = (bits: number[]) => P(ends[0][bits[0]], ends[1][bits[1]], ends[2][bits[2]]);
  const facing = (axis: number, high: number) => {
    const n: [number, number, number] = [0, 0, 0];
    n[axis] = high ? 1 : -1;
    return P.facing(...n);
  };

  // Each edge, and whether either face it borders is turned towards us.
  const edges: [number[], number[], boolean][] = [];
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
  const quad = (map: (u: number, v: number) => number[], u0: number, u1: number, v1: number) =>
    [map(u0, 0), map(u1, 0), map(u1, v1), map(u0, v1)];

  // A looping frame runs round time in a ring, and on a loop round the sides
  // round x too, so each part of it past an end of the cuboid is drawn where
  // it wraps round to: the frame moved back by whole clip lengths or widths.
  const turns = (from: number, to: number, n: number, wraps: boolean) => {
    if (!wraps) return [0];
    const first = Math.floor((Math.min(from, to) + 0.5) / n);
    const last = Math.floor((Math.max(from, to) + 0.5) / n);
    return Array.from({ length: last - first + 1 }, (_, i) => (first + i) * n);
  };
  const layers = [];
  for (const shiftT of turns(t0 - 0.5 * dt, t1 + 0.5 * dt, T, info.loop)) {
    for (const shiftX of turns(x0 - 0.5 * dx, x1 + 0.5 * dx, W, info.sides)) {
      const onSlice = (u: number, v: number) =>
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
    const onFace = (u: number, v: number) => P(u - 0.5, v - 0.5, -0.5);
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
    const top = ([t, x]: number[]) => P(x, -0.5, t);
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
  const labels: [string, number[]][] = [["x", P(W - 0.5, -0.5, -0.5)],
    ["y", P(-0.5, H - 0.5, -0.5)], ["time", P(-0.5, -0.5, T - 0.5)]];
  for (const [label, end] of labels) {
    if (Math.hypot(end[0] - origin[0], end[1] - origin[1]) < 4) continue;  // end-on
    let ox = end[0] - middle[0], oy = end[1] - middle[1];
    const length = Math.hypot(ox, oy) || 1;
    ox = (ox / length) * 14;
    oy = (oy / length) * 14;
    ctx.fillText(label, end[0] + ox, end[1] + oy);
  }
}

let drag: { id: number; x: number; y: number; yaw: number; pitch: number } | null = null;
cube.addEventListener("pointerdown", (e) => {
  if (!e.isPrimary) return;  // a second finger leaves the turn to the first
  drag = { id: e.pointerId, x: e.clientX, y: e.clientY, yaw: camera.yaw, pitch: camera.pitch };
  cube.setPointerCapture(e.pointerId);
});
cube.addEventListener("pointermove", (e) => {
  if (drag?.id !== e.pointerId) return;
  camera.yaw = drag.yaw + (e.clientX - drag.x) * 0.01;
  camera.pitch = Math.min(Math.max(drag.pitch - (e.clientY - drag.y) * 0.01,
                                   -Math.PI / 2), Math.PI / 2);
  drawCube();
});
// A cancelled drag, as when the browser takes a touch to scroll the page, puts the cuboid back.
cube.addEventListener("pointercancel", (e) => {
  if (drag?.id !== e.pointerId) return;
  [camera.yaw, camera.pitch] = [drag.yaw, drag.pitch];
  drag = null;
  drawCube();
});
const release = (e: PointerEvent) => { if (drag?.id === e.pointerId) drag = null; };
cube.addEventListener("pointerup", release);
cube.addEventListener("lostpointercapture", release);
cube.addEventListener("wheel", (e) => {
  if (!e.ctrlKey && !e.metaKey) return;  // a plain wheel scrolls the page; a pinch sets ctrlKey
  e.preventDefault();
  // A pinch's deltas add up to -100 ln(its scale); a mouse wheel's notch, often 100, is cut to 25.
  const delta = Math.min(Math.max(e.deltaY, -25), 25);
  camera.zoom = Math.min(Math.max(camera.zoom * Math.exp(-delta / 100), 0.3), 10);
  drawCube();
}, { passive: false });
cube.addEventListener("dblclick", () => setView("three"));
const viewButtons = document.querySelectorAll<HTMLElement>("[data-view]");
for (const button of viewButtons) {
  button.addEventListener("click", () => setView(button.dataset.view as string));
}

function setView(name: string) {
  [camera.yaw, camera.pitch] = VIEWS[name];
  camera.zoom = 1;
  drawCube();
}

new ResizeObserver(() => drawCube()).observe(cube);
