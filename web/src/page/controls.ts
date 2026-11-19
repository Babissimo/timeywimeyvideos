// The slice, noise and clip controls, and the options they set.

import { MAX_SEED } from "../noise";
import { $ } from "./elements";
import { refresh } from "./view";

const DESCRIBE: Record<string, string> = {
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
  return (document.querySelector('input[name="slice"]:checked') as HTMLInputElement).value;
}

// What the frame does at the video's edges: black, inside or loop.
function edges() {
  return (document.querySelector('input[name="edges"]:checked') as HTMLInputElement).value;
}

export function setEdges(value: string) {
  const radio = document.querySelector(`input[name="edges"][value="${value}"]`);
  (radio as HTMLInputElement).checked = true;
  slicesChanged();
}

// Which ways the frame can move: rotate frames inside the video, or on a
// loop round the sides.
function moves() {
  return sliceKind() === "rotate" &&
    (edges() === "inside" || edges() === "loop" && $("loop-sides").checked);
}

export function options() {
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

export function slicesChanged() {
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
  (longest as HTMLOptionElement).disabled = sides;
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

function setAngle(value: number) {
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
 */
function pair(slider: HTMLInputElement, box: HTMLInputElement, changed: () => void) {
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

export function noiseChanged() {
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

$("noise-seed").max = String(MAX_SEED);
for (const input of [$("noise"), $("noise-size"), $("noise-speed"), $("noise-direction"),
                     $("noise-seed")]) {
  input.addEventListener("change", noiseChanged);
}
const noiseAmount = pair($("noise-range"), $("noise-amount"), noiseChanged);
