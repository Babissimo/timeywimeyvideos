/**
 * The videos the page offers: files opened on their own, picked or dropped, and the videos at
 * the top level of a folder opened whole. Each is a File the page reads where it lies, a piece
 * at a time as it needs it; nothing is uploaded or copied.
 */

/** The extensions of the videos a folder's listing offers. */
export const VIDEO_TYPES: ReadonlySet<string> =
  new Set([".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".gif"]);

/** Whether a file's name ends in one of VIDEO_TYPES, in any case. A hidden file's never does. */
export function isVideo(name: string): boolean {
  const dot = name.lastIndexOf(".");
  return !name.startsWith(".") && dot > 0 && VIDEO_TYPES.has(name.slice(dot).toLowerCase());
}

/** What a folder lists, as a FileSystemDirectoryHandle does. */
export type Entry =
  | { readonly kind: "directory"; readonly name: string }
  | { readonly kind: "file"; readonly name: string; getFile(): Promise<File> };

/** A folder opened whole: a FileSystemDirectoryHandle, from `showDirectoryPicker`. */
export interface Folder {
  readonly kind: "directory";
  readonly name: string;
  values(): AsyncIterable<Entry>;
  isSameEntry(other: Folder): Promise<boolean>;
}

/** A folder's videos, and the names of those it listed but couldn't give. */
export interface Listing { files: File[]; unreadable: string[] }

// Names in the order people count them, so clip2 comes before clip10.
const byName = new Intl.Collator(undefined, { numeric: true });

/** The videos at the top level of a folder, each list ordered by name. */
export async function listFolder(folder: Folder): Promise<Listing> {
  // Each read settles as it is asked for, so one that fails is never left unhandled.
  const reads: Promise<File | string>[] = [];
  for await (const entry of folder.values()) {
    if (entry.kind === "file" && isVideo(entry.name))
      reads.push(entry.getFile().catch(() => entry.name));
  }
  const files: File[] = [], unreadable: string[] = [];
  for (const read of await Promise.all(reads)) {
    if (typeof read === "string") unreadable.push(read);
    else files.push(read);
  }
  return { files: files.sort((a, b) => byName.compare(a.name, b.name)),
           unreadable: unreadable.sort(byName.compare) };
}

/**
 * Whether the browser now refuses to read a file. A File is the file as it was when opened, and
 * once that changes or moves, every read of it fails, which the decoder takes for a file with no
 * video in it.
 */
export async function hasChanged(file: Blob): Promise<boolean> {
  try {
    await file.slice(0, 1).arrayBuffer();
    return false;
  } catch (error) {
    return error instanceof DOMException
      && (error.name === "NotReadableError" || error.name === "NotFoundError");
  }
}

/** What the page says of a file that has changed or moved since it was opened. */
export function changed(name: string): string {
  return `${name} has changed or moved since it was opened: open it, or its folder, again.`;
}

/** A video the page offers, and the id, unique among those offered, that names it. */
export interface Source { readonly id: string; readonly file: File }

/** Whether two Files are the same file as it was when each was read, as far as they tell. */
export const same = (a: File, b: File) =>
  a.name === b.name && a.size === b.size && a.lastModified === b.lastModified;

const inOrder = (sources: Source[]) =>
  sources.sort((a, b) => byName.compare(a.file.name, b.file.name));

/**
 * The page's sources: the videos in the folder opened last and the files opened on their own,
 * each list in order of name, and which of them is chosen.
 */
export class Sources {
  /** The name of the folder open, or null before one is. */
  folder: string | null = null;
  /** The videos in the folder open. */
  inFolder: readonly Source[] = [];
  /** The names of the videos the folder open listed but couldn't give. */
  unreadable: readonly string[] = [];
  /** The files opened on their own. Files of the same name from elsewhere are each offered. */
  opened: readonly Source[] = [];
  private handle: Folder | null = null;
  private made = 0;
  private chosenId = "";

  private source(file: File): Source {
    return { id: String(++this.made), file };
  }

  /**
   * Offer files picked or dropped, whatever they hold. One offered already, here or in the
   * folder, keeps its id and takes the File just given, as the one it had may no longer read.
   * Returns the first one's id, or "".
   */
  open(files: Iterable<File>): string {
    let opened = [...this.opened];
    let first = "";
    for (const file of files) {
      const offered = (source: Source) => same(source.file, file);
      const inFolder = this.inFolder.find(offered);
      const alone = inFolder ? undefined : opened.find(offered);
      const source = { id: inFolder?.id ?? alone?.id ?? this.source(file).id, file };
      if (inFolder) this.inFolder = this.inFolder.map((s) => (s === inFolder ? source : s));
      else opened = alone ? opened.map((s) => (s === alone ? source : s)) : [...opened, source];
      first ||= source.id;
    }
    this.opened = inOrder(opened);
    return first;
  }

  /**
   * Offer a folder's videos in place of the last folder's, each as the File just listed.
   * Opened again, the folder's videos keep their ids, matched by name. Any other video it lists
   * that is offered on its own, or was chosen from the folder before, joins it under its id.
   * Otherwise the video chosen from another folder stays offered, as opened on its own.
   */
  async openFolder(folder: Folder): Promise<void> {
    const { files, unreadable } = await listFolder(folder);
    const again = this.handle !== null && await this.handle.isSameEntry(folder);
    const before = again ? this.inFolder : [];
    const chosen = this.inFolder.find((source) => source.id === this.chosenId);
    let opened = [...this.opened];
    const listed = files.map((file) => {
      const known = before.find((source) => source.file.name === file.name);
      if (known) return { id: known.id, file };
      const alone = opened.find((source) => same(source.file, file))
        ?? (chosen && same(chosen.file, file) ? chosen : undefined);
      opened = opened.filter((source) => source !== alone);
      return alone ? { id: alone.id, file } : this.source(file);
    });
    if (chosen && !listed.some((source) => source.id === chosen.id)) opened.push(chosen);
    this.opened = inOrder(opened);
    this.handle = folder;
    this.folder = folder.name;
    this.inFolder = listed;
    this.unreadable = unreadable;
  }

  /** The source with this id, or null if none is offered. */
  get(id: string): Source | null {
    return this.inFolder.find((source) => source.id === id)
      ?? this.opened.find((source) => source.id === id) ?? null;
  }

  /** Choose the source with this id, or none with "". Returns it, or null. */
  choose(id: string): Source | null {
    this.chosenId = id;
    return this.chosen;
  }

  /** The source chosen, while it is offered. */
  get chosen(): Source | null {
    return this.get(this.chosenId);
  }
}
