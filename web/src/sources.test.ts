import { describe, expect, test } from "vitest";
import { isVideo, listFolder, Sources, VIDEO_TYPES, type Folder, type Source } from "./sources";

/** An entry that lists but can't be read, as a file removed since the folder was opened. */
class Gone {
  constructor(readonly name: string) {}
}

/**
 * A folder as a FileSystemDirectoryHandle lists it: files and folders (named by a string), in
 * no set order. `place` stands for where it is on disk; folders at one place are the same.
 */
function folder(name: string, entries: (File | Gone | string)[], place = name):
    Folder & { place: string; reads: () => number } {
  let reads = 0;
  return {
    kind: "directory",
    name,
    place,
    reads: () => reads,
    async isSameEntry(other) {
      return (other as { place?: string }).place === place;
    },
    async *values() {
      for (const entry of entries) {
        if (typeof entry === "string") {
          yield { kind: "directory", name: entry };
        } else {
          yield { kind: "file", name: entry.name, getFile: async () => {
            reads++;
            if (entry instanceof Gone) throw new DOMException("gone", "NotFoundError");
            return entry;
          } };
        }
      }
    },
  };
}

const file = (name: string, text = name, lastModified = 0) =>
  new File([text], name, { lastModified });

/** Expect these very Files, in order: toEqual would take any File for any other. */
function expectFiles(got: readonly (Source | File)[], files: File[]): void {
  expect(got).toHaveLength(files.length);
  got.forEach((item, i) => expect(item instanceof File ? item : item.file).toBe(files[i]));
}

/** Expect a source with this id and this very File. */
function expectSource(got: Source | null | undefined, id: string, file: File): void {
  expect(got?.id).toBe(id);
  expect(got?.file).toBe(file);
}

describe("isVideo", () => {
  test("takes the seven video extensions, in any case", () => {
    expect([...VIDEO_TYPES].sort()).toEqual([".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4",
                                             ".webm"]);
    for (const name of ["clip.mp4", "CLIP.MOV", "a.b.m4v", "x.Mkv", "y.webm", "z.avi", "w.gif"])
      expect(isVideo(name), name).toBe(true);
  });

  test("refuses other files, and hidden ones", () => {
    for (const name of ["notes.txt", "clip.mp4.part", "clip", "mp4", "clip.", ".mp4",
                        "._clip.mp4", ".clip.mov"])
      expect(isVideo(name), name).toBe(false);
  });
});

describe("listFolder", () => {
  test("lists the videos at the top level, by name as people count", async () => {
    const files = [file("b.mov"), file("notes.txt"), file("clip10.mp4"), file("._clip2.mp4"),
                   file("clip2.mp4"), file("A.webm")];
    const listing = folder("Movies", [...files, "clips.mp4"]);
    const { files: listed, unreadable } = await listFolder(listing);
    expect(listed.map((f) => f.name)).toEqual(["A.webm", "b.mov", "clip2.mp4", "clip10.mp4"]);
    for (const f of listed) expect(files).toContain(f);  // the folder's own Files
    expect(unreadable).toEqual([]);
  });

  test("reads only the videos", async () => {
    const listing = folder("Movies", [file("a.mp4"), file("b.txt"), file("c.pdf"), "d"]);
    await listFolder(listing);
    expect(listing.reads()).toBe(1);
  });

  test("lists the rest when some videos can't be read", async () => {
    const clip = file("clip.mp4");
    const listing = await listFolder(folder("Movies", [new Gone("walk.mov"), clip,
                                                       new Gone("b.mp4"), new Gone("a.txt")]));
    expectFiles(listing.files, [clip]);
    expect(listing.unreadable).toEqual(["b.mp4", "walk.mov"]);
  });

  test("lists nothing from a folder without videos", async () => {
    expect(await listFolder(folder("Empty", []))).toEqual({ files: [], unreadable: [] });
    expect((await listFolder(folder("Notes", [file("a.txt"), "inner"]))).files).toEqual([]);
  });
});

describe("Sources", () => {
  test("keeps the Files opened, in order of name, and offers each by an id of its own", () => {
    const sources = new Sources();
    const one = file("one.mp4"), two = file("two.mov"), notes = file("notes.txt");
    const first = sources.open([two, one, notes]);
    expect(sources.get(first)?.file).toBe(two);
    expectFiles(sources.opened, [notes, one, two]);
    const ids = sources.opened.map((s) => s.id);
    expect(new Set(ids).size).toBe(3);
    expect(sources.get(ids[1])?.file).toBe(one);
    expect(sources.get(ids[0])?.file).toBe(notes);  // the decoder says what's wrong with it
    expect(sources.get("")).toBeNull();
    expect(sources.get("missing")).toBeNull();
    expect(sources.open([])).toBe("");
  });

  test("offers files of the same name from elsewhere each in turn", () => {
    const sources = new Sources();
    const here = file("clip.mp4", "here", 1), there = file("clip.mp4", "there, longer", 1);
    const a = sources.open([here]);
    const b = sources.open([there]);
    expect(a).not.toBe(b);
    expectFiles(sources.opened, [here, there]);
    expect(sources.get(a)?.file).toBe(here);
    expect(sources.get(b)?.file).toBe(there);
    // Opened again, a file keeps its place among those of its name, and so its label.
    const hereAgain = file("clip.mp4", "here", 1);
    expect(sources.open([hereAgain])).toBe(a);
    expectFiles(sources.opened, [hereAgain, there]);
  });

  test("offers a file opened again under its id, as the File just given", () => {
    const sources = new Sources();
    const id = sources.open([file("clip.mp4", "same", 5)]);
    // As the File opened first may no longer read: the file may have moved since.
    const again = file("clip.mp4", "same", 5);
    expect(sources.open([again])).toBe(id);
    expectSource(sources.get(id), id, again);
    expect(sources.opened).toHaveLength(1);
    // Changed since, it is another file.
    expect(sources.open([file("clip.mp4", "same", 6)])).not.toBe(id);
    expect(sources.opened).toHaveLength(2);
  });

  test("lists a folder's videos, the last folder opened in place of the one before", async () => {
    const sources = new Sources();
    expect(sources.folder).toBeNull();
    expect(sources.inFolder).toEqual([]);
    const clip = file("clip.mp4");
    await sources.openFolder(folder("Movies", [file("notes.txt"), clip, new Gone("lost.mp4")]));
    expect(sources.folder).toBe("Movies");
    expectFiles(sources.inFolder, [clip]);
    expect(sources.unreadable).toEqual(["lost.mp4"]);
    const id = sources.inFolder[0].id;
    expect(sources.get(id)?.file).toBe(clip);

    const alone = file("clip.mp4", "another");  // opened on its own, apart from the folder's
    const opened = sources.open([alone]);
    expect(opened).not.toBe(id);
    expect(sources.get(id)?.file).toBe(clip);

    const other = file("other.mov");
    await sources.openFolder(folder("Holiday", [other]));
    expect(sources.folder).toBe("Holiday");
    expectFiles(sources.inFolder, [other]);
    expect(sources.unreadable).toEqual([]);
    expect(sources.get(id)).toBeNull();
    expectFiles(sources.opened, [alone]);
  });

  test("an empty folder still counts as open", async () => {
    const sources = new Sources();
    await sources.openFolder(folder("Empty", []));
    expect(sources.folder).toBe("Empty");
    expect(sources.inFolder).toEqual([]);
  });

  test("a folder that can't be listed leaves the list as it was", async () => {
    const sources = new Sources();
    const clip = file("clip.mp4");
    await sources.openFolder(folder("Movies", [clip]));
    const broken: Folder = {
      kind: "directory",
      name: "Locked",
      isSameEntry: async () => false,
      async *values() { throw new DOMException("gone", "NotFoundError"); },
    };
    await expect(sources.openFolder(broken)).rejects.toThrow("gone");
    expect(sources.folder).toBe("Movies");
    expectFiles(sources.inFolder, [clip]);
  });

  test("chooses a source by id while it is offered", () => {
    const sources = new Sources();
    expect(sources.chosen).toBeNull();
    const clip = file("clip.mp4");
    const id = sources.open([clip]);
    expectSource(sources.choose(id), id, clip);
    expect(sources.chosen?.file).toBe(clip);
    expect(sources.choose("")).toBeNull();
    expect(sources.chosen).toBeNull();
    expect(sources.choose("missing")).toBeNull();
  });

  test("keeps a folder's video chosen when the folder is opened again", async () => {
    const sources = new Sources();
    const walk = file("walk.mp4", "walk", 1), run = file("run.mp4", "run", 1);
    await sources.openFolder(folder("Movies", [walk, run]));
    const id = sources.inFolder.find((s) => s.file === walk)!.id;
    sources.choose(id);

    // Changed or not, it comes as the File just listed, under the same id.
    const again = file("walk.mp4", "walk", 1);
    await sources.openFolder(folder("Movies", [again, run]));
    expectSource(sources.chosen, id, again);
    const edited = file("walk.mp4", "walk, edited", 2);
    await sources.openFolder(folder("Movies", [edited, run]));
    expectSource(sources.chosen, id, edited);
    expect(sources.opened).toEqual([]);
  });

  test("keeps the video chosen from a folder when another folder opens", async () => {
    const sources = new Sources();
    const walk = file("walk.mp4", "here", 1);
    await sources.openFolder(folder("Movies", [walk, file("run.mp4")]));
    const id = sources.inFolder.find((s) => s.file === walk)!.id;
    sources.choose(id);

    // Another folder of the same name, with a video of the same name, is still another.
    const elsewhere = file("walk.mp4", "there", 1);
    await sources.openFolder(folder("Movies", [elsewhere], "~/Desktop/Movies"));
    expectFiles(sources.inFolder, [elsewhere]);
    expect(sources.inFolder[0].id).not.toBe(id);
    expectSource(sources.chosen, id, walk);
    expectFiles(sources.opened, [walk]);

    // A video not chosen goes with its folder, and one gone from its folder stays if chosen.
    expect(sources.opened.some((s) => s.file.name === "run.mp4")).toBe(false);
    const inner = sources.inFolder[0].id;
    sources.choose(inner);
    await sources.openFolder(folder("Movies", [], "~/Desktop/Movies"));
    expectSource(sources.chosen, inner, elsewhere);

    // The first folder again takes its video back, as the File just listed.
    sources.choose(id);
    const back = file("walk.mp4", "here", 1);
    await sources.openFolder(folder("Movies", [back]));
    expect(sources.inFolder).toHaveLength(1);
    expectSource(sources.inFolder[0], id, back);
    expectFiles(sources.opened, [elsewhere]);
    expectSource(sources.chosen, id, back);
  });

  test("lists the video chosen from a folder once when another folder holds it too", async () => {
    const sources = new Sources();
    const run = file("run.mp4", "run", 1);
    await sources.openFolder(folder("Movies", [run]));
    const id = sources.inFolder[0].id;
    sources.choose(id);
    const backup = file("run.mp4", "run", 1);
    await sources.openFolder(folder("Backup", [backup]));
    expect(sources.inFolder).toHaveLength(1);
    expectSource(sources.inFolder[0], id, backup);
    expect(sources.opened).toEqual([]);
    expectSource(sources.chosen, id, backup);
  });

  test("offers a file opened that the folder lists as the folder's", async () => {
    const sources = new Sources();
    await sources.openFolder(folder("Movies", [file("walk.mp4", "walk", 1)]));
    const id = sources.inFolder[0].id;
    const again = file("walk.mp4", "walk", 1);
    expect(sources.open([again])).toBe(id);
    expectSource(sources.inFolder[0], id, again);
    expect(sources.opened).toEqual([]);
  });
});
