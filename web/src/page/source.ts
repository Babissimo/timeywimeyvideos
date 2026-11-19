// Sources: videos opened on this computer, on their own or a folder at a time,
// which the page reads where they lie as it needs them.

import { probe } from "../decode";
import { bytes, seconds } from "../numbers";
import {
  changed, entryFolder, hasChanged, pickedFolder, same, Sources, VIDEO_TYPES,
  type Folder, type Source,
} from "../sources";
import { drawCube, forgetFace } from "./cuboid";
import { $ } from "./elements";
import { enableRender, link, markPlaying, watch } from "./renders";
import { state, UNABLE } from "./state";
import { estimate, live, pause, refresh, scrubText, showMessage } from "./view";

const sources = new Sources();
export const NO_SOURCE = "Open a video, or drop one here.";
let original: string | null = null;  // the address "watch the original" plays the chosen file at

export function showSources() {
  const list = $("source");
  list.replaceChildren(new Option("Choose a video…", ""));
  if (sources.folder !== null) list.append(group(`Folder: ${sources.folder}`, sources.inFolder));
  if (sources.opened.length) list.append(group("Opened", sources.opened));
  list.value = sources.chosen?.id ?? "";
}

function group(label: string, items: readonly Source[]) {
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

/** A note on the folder, apart from the chosen video's details. */
function folderNote(text: string, error = false) {
  $("folder-note").textContent = text;
  $("folder-note").classList.toggle("error", error);
}

function describeFolder() {
  const n = sources.inFolder.length, unreadable = sources.unreadable;
  folderNote((n ? `${sources.folder} has ${n} video${n === 1 ? "" : "s"}.` :
    `${sources.folder} has no videos at its top level (${[...VIDEO_TYPES].join(" ")}).`) +
    (unreadable.length ? ` Couldn't read ${unreadable.join(", ")}.` : ""));
}

async function chooseSource(id: string) {
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
  forgetFace();
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
    const message = await hasChanged(file) ? changed(file.name) : (err as Error).message;
    if (state.chosen === chosen) $("source-info").textContent = message;
  }
}

function watchOriginal() {
  const file = state.source as File;  // its link shows only once one is chosen
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

function openFiles(files: File[]) {
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
    const error = err as Error;
    if (error.name !== "AbortError") folderNote(error.message, true);
    return;  // else they closed the picker
  }
  await openFolder(folder);
}

async function openFolder(folder: Folder) {
  try {
    if (!await sources.openFolder(folder)) return;  // another folder took its place
  } catch (err) {
    folderNote(`Couldn't read ${folder.name}: ${(err as Error).message}`, true);
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
  const entry = entries[i] as FileSystemDirectoryEntry;
  openFolder(((await handle) as FileSystemDirectoryHandle | null) ?? entryFolder(entry));
});
