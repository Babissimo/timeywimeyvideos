// Renders, made here one at a time, each in a worker of its own. The browser keeps
// them for the page, or where it keeps no files for it, they last until it is left.

import { bytes, seconds } from "../numbers";
import { render as renderClip, type Rendered, type Stage } from "../render";
import { changed, hasChanged } from "../sources";
import { discard, stored } from "../store";
import { options } from "./controls";
import { $ } from "./elements";
import { state, UNABLE } from "./state";

let job: AbortController | null = null;  // the AbortController of the render under way
/** A render, saved once it is kept in the browser or downloaded. */
interface Render {
  name: string; url: string; size: number; made: Date; about: string; saved: boolean;
}
const renders: Render[] = [];  // newest first
let storing = false;  // whether the browser keeps files for the page

// What each stage of a render says, given how many frames it has done.
const STAGES: Record<Stage, (done: number, total: number) => string> = {
  loading: (done, total) => `Loading the clip · frame ${done} of ${total}`,
  slicing: (done, total) => `Slicing · frame ${done} of ${total}`,
  finishing: () => "Finishing",
};

async function render(preview: boolean) {
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
    const error = err as Error;
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
 */
function showJob({ running = false, done = 0, text = "", error = false }:
                   { running?: boolean; done?: number; text?: string; error?: boolean }) {
  $("job").hidden = false;
  $("cancel").hidden = !running;
  $("job-bar").style.width = `${100 * done}%`;
  $("job-text").setAttribute("aria-busy", String(running));  // so only how it ends is read
  $("job-text").classList.toggle("error", error);
  $("job-text").textContent = text;
}

// The render buttons work once a video is chosen, for one render at a time.
export function enableRender() {
  const off = Boolean(job) || !state.source;
  $("render-preview").disabled = off;
  $("render-full").disabled = off;
}

$("render-preview").addEventListener("click", () => render(true));
$("render-full").addEventListener("click", () => render(false));
$("cancel").addEventListener("click", () => job?.abort());

// Leaving the page loses a render under way and every render held in memory and
// not yet downloaded.
window.addEventListener("beforeunload", (event) => {
  if (job || renders.some((r) => !r.saved)) event.preventDefault();
});

/** Keep a render in the list, in place of an earlier one of the same name. */
function keep(made: Rendered) {
  const earlier = renders.findIndex((r) => r.name === made.name);
  if (earlier >= 0) URL.revokeObjectURL(renders.splice(earlier, 1)[0].url);
  const item = {
    name: made.name, url: URL.createObjectURL(made.video), size: made.video.size,
    made: new Date(), saved: made.kept,
    about: `${made.width}×${made.height}, ${made.frames} frames, ${made.rateControl}`,
  };
  renders.unshift(item);
  showRenders();
  return item;
}

/** List the renders the browser kept from before. */
export async function restore() {
  const files = await stored().catch(() => null);
  storing = files !== null;
  for (const file of files ?? []) {
    if (renders.some((r) => r.name === file.name)) continue;  // made since the page opened
    renders.push({ name: file.name, url: URL.createObjectURL(file), size: file.size,
                   made: new Date(file.lastModified), about: "", saved: true });
  }
  showRenders();
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
    const when = r.made.toDateString() === new Date().toDateString()
      ? r.made.toLocaleTimeString() : r.made.toLocaleString();
    meta.textContent = [bytes(r.size), when, r.about].filter(Boolean).join(" · ");
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
    item.textContent = storing
      ? "Nothing rendered yet. Renders are kept in this browser until you remove them."
      : "Nothing rendered yet. Renders stay here until you leave or reload the page, so " +
        "download the ones to keep.";
    list.append(item);
  }
  markPlaying();
}

/** Drop a render from the list and let its video go. */
function forget(r: Render) {
  renders.splice(renders.indexOf(r), 1);
  discard(r.name);
  if ($("player").src === r.url) {
    $("player").removeAttribute("src");
    $("player").load();
    $("player").hidden = true;
    $("now-playing").textContent = "";
  }
  URL.revokeObjectURL(r.url);
  showRenders();
}

export function link(text: string, onclick: () => void) {
  const button = document.createElement("button");
  button.className = "link";
  button.textContent = text;
  button.onclick = onclick;
  return button;
}

/** Play a video in the player: the chosen file, or a render. */
export function watch(url: string, label: string, file: File | null = null) {
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

export function markPlaying() {
  const playing = $("player").hidden ? "" : $("player").src;
  for (const item of $("renders").children) {
    item.classList.toggle("playing", (item as HTMLElement).dataset.src === playing);
  }
}
