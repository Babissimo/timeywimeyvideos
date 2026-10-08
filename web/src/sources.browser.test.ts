import { afterEach, expect, test } from "vitest";
import { changed, hasChanged, listFolder, Sources } from "./sources";

// The origin's private file system gives real FileSystemDirectoryHandles without asking.
const root = () => navigator.storage.getDirectory();

async function write(folder: FileSystemDirectoryHandle, name: string, text: string) {
  const writable = await (await folder.getFileHandle(name, { create: true })).createWritable();
  await writable.write(text);
  await writable.close();
}

afterEach(async () => {
  for (const name of ["videos", "more", "changing.mp4"])
    await (await root()).removeEntry(name, { recursive: true }).catch(() => {});
});

test("lists the videos in a real folder", async () => {
  const folder = await (await root()).getDirectoryHandle("videos", { create: true });
  for (const name of ["walk.MOV", "notes.txt", "run.mp4", ".hidden.mp4"])
    await write(folder, name, name);
  await folder.getDirectoryHandle("inner.mp4", { create: true });

  const { files, unreadable } = await listFolder(folder);
  expect(files.map((file) => file.name)).toEqual(["run.mp4", "walk.MOV"]);
  expect(await files[1].text()).toBe("walk.MOV");
  expect(unreadable).toEqual([]);

  const sources = new Sources();
  await sources.openFolder(folder);
  expect(sources.folder).toBe("videos");
  const run = sources.inFolder[0];
  expect(await sources.choose(run.id)?.file.text()).toBe("run.mp4");

  // The same folder, from a handle of its own, keeps the choice; another doesn't list it.
  await sources.openFolder(await (await root()).getDirectoryHandle("videos"));
  expect(sources.chosen).toEqual(run);
  const more = await (await root()).getDirectoryHandle("more", { create: true });
  await write(more, "run.mp4", "another run");
  await sources.openFolder(more);
  expect(sources.inFolder[0].id).not.toBe(run.id);
  expect(sources.chosen).toEqual(run);
});

test("tells when a file has changed since it was opened", async () => {
  const handle = await (await root()).getFileHandle("changing.mp4", { create: true });
  const writable = await handle.createWritable();
  await writable.write("first");
  await writable.close();
  const file = await handle.getFile();
  expect(await hasChanged(file)).toBe(false);
  const again = await handle.createWritable();
  await again.write("second, longer");
  await again.close();
  expect(await hasChanged(file)).toBe(true);
  expect(await hasChanged(await handle.getFile())).toBe(false);
  await (await root()).removeEntry("changing.mp4");
  expect(await hasChanged(file)).toBe(true);
  expect(changed("clip.mp4"))
    .toBe("clip.mp4 has changed or moved since it was opened: open it, or its folder, again.");
});
