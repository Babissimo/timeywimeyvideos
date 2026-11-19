// Live preview. Only one frame is worked on at a time; if the options or
// position change meanwhile, the latest ones are shown next, so dragging a
// slider never queues up stale frames.

import type { ClipInfo } from "../decode";
import { Live } from "../live";
import { bytes, seconds } from "../numbers";
import { fullSize, Problem, readOptions } from "../options";
import { options, setEdges, slicesChanged } from "./controls";
import { drawCube, loadFace } from "./cuboid";
import { $ } from "./elements";
import { state, UNABLE } from "./state";

// Made on first use, so a browser without WebGL2 gets a message.
export let live: Live | null = null;
let unable: unknown = null;  // why it couldn't be made, as it is tried only once
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

export function refresh() {
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
  const loading = (text: string, progress = false) => {
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
    const error = err as Error;
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
 */
export function showMessage(text: string, action?: [string, () => void] | false | null,
                            busy = false) {
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
export function scrubText() {
  const info = state.info;
  if (!info) return $("scrub").removeAttribute("aria-valuetext");
  const frame = Math.round(state.pos * (info.frames - 1)) + 1;
  $("scrub").setAttribute("aria-valuetext", `Frame ${frame} of ${info.frames}`);
}

// What a full-quality render would make. It needs only the clip and the options, not the
// live view's half-size copy, so it still holds when that copy can't be loaded.
let estimated: { clip: ClipInfo | null; values: string } =
  { clip: null, values: "" };  // what the estimate shown was worked out for

export function estimate() {
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

function tick(now: number) {
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

export function pause() {
  state.playing = false;
  $("play").textContent = "▶";
  $("play").setAttribute("aria-label", "Play");
}

function step(by: number) {
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
  const target = e.target as Element;
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
