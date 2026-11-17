/**
 * Renders kept in the origin private file system, where they outlast the page. A render is
 * written under a name of its own as it is made, then moved to its file name, so only finished
 * renders are listed, and two pages making the same render never write into one file.
 */
import { BufferTarget, StreamTarget, type Target } from "mediabunny";
import { Problem } from "./options";

const PART = ".part";  // the end of the name of a render being written

// What this uses of the file system that the DOM types leave out: moving a file, and writing it
// in place, which only a worker can.
declare global {
  interface FileSystemFileHandle {
    move(folder: FileSystemDirectoryHandle, name: string): Promise<void>;
    createSyncAccessHandle(): Promise<{
      write(data: AllowSharedBufferSource, options: { at: number }): number;
      close(): void;
    }>;
  }
}

/** Where a render's MP4 goes as it is muxed. */
export interface Destination {
  target: Target;
  /** Once the MP4 is written: the video, as the file `name`, and whether it is kept. */
  finish(name: string): Promise<{ video: Blob; kept: boolean }>;
}

/**
 * Somewhere to write a render: a file in the origin private file system, written in place as
 * the muxer goes, as only a worker can; else memory, where the browser keeps no files for the
 * page, as in Firefox's private windows.
 */
export async function destination(): Promise<Destination> {
  try {
    const folder = await navigator.storage.getDirectory();
    const temporary = `${crypto.randomUUID()}${PART}`;
    // The part's lock, held until the worker ends, keeps sweeps off it.
    await new Promise<void>((held, failed) => {
      navigator.locks.request(temporary, () => {
        held();
        return new Promise(() => {});
      }).catch(failed);
    });
    await sweep(folder);
    const part = await folder.getFileHandle(temporary, { create: true });
    const file = await part.createSyncAccessHandle();
    return {
      target: new StreamTarget(new WritableStream({
        write: ({ data, position }) => {
          try {
            file.write(data, { at: position });
          } catch (error) {
            if ((error as Error).name !== "QuotaExceededError") throw error;
            throw new Problem("The browser has no room left to keep this render. Remove some "
                              + "renders, or set a shorter duration or a smaller scale.");
          }
        },
      })),
      async finish(name) {
        file.close();
        // In place of any render of that name, one move at a time, as Chrome refuses to move a
        // file onto one being moved onto. A lock's name may not start with "-".
        const video = await navigator.locks.request(`move ${name}`, async () => {
          await part.move(folder, name);
          return part.getFile();
        });
        return { video, kept: true };
      },
    };
  } catch {
    const target = new BufferTarget();
    return {
      target,
      finish: async () => ({ video: new Blob([target.buffer!], { type: "video/mp4" }),
                             kept: false }),
    };
  }
}

/** The renders kept, newest first, or null where the browser keeps no files for the page. */
export async function stored(): Promise<File[] | null> {
  let folder: FileSystemDirectoryHandle;
  try {
    folder = await navigator.storage.getDirectory();
  } catch {
    return null;
  }
  await sweep(folder);
  const files: Promise<File | null>[] = [];
  for await (const handle of folder.values()) {
    if (handle.kind === "file" && handle.name.endsWith(".mp4"))
      files.push(handle.getFile().catch(() => null));  // unless removed meanwhile
  }
  return (await Promise.all(files)).filter((file) => file !== null)
    .sort((a, b) => b.lastModified - a.lastModified);
}

/** Remove a kept render, if there is one of this name. */
export async function discard(name: string): Promise<void> {
  try {
    await (await navigator.storage.getDirectory()).removeEntry(name);
  } catch {
    // none kept, or no files for the page
  }
}

/** Remove the parts renders stopped partway left: those whose lock no render holds. */
async function sweep(folder: FileSystemDirectoryHandle): Promise<void> {
  const parts: string[] = [];
  for await (const name of folder.keys()) if (name.endsWith(PART)) parts.push(name);
  await Promise.all(parts.map((name) => navigator.locks.request(name, { ifAvailable: true },
    async (lock) => { if (lock) await folder.removeEntry(name); }).catch(() => {})));
}
