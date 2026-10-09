import { describe, expect, test } from "vitest";
import { ContainerError, readMatroska, readMp4 } from "./container";

type Bytes = Uint8Array<ArrayBuffer>;
const concat = (...parts: Uint8Array[]): Bytes => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  parts.reduce((at, p) => (out.set(p, at), at + p.length), 0);
  return out;
};
const ascii = (text: string) => new Uint8Array([...text].map((c) => c.charCodeAt(0)));
const bigEndian = (value: number, length: number) =>
  new Uint8Array(Array.from({ length }, (_, i) => Math.floor(value / 256 ** (length - 1 - i)) % 256));

/** An EBML element: its ID's bytes, an 8-byte size (or `size` in its place), then the data. */
function el(id: number, data: Uint8Array, size?: Uint8Array): Bytes {
  const idBytes = bigEndian(id, Math.ceil(Math.log2(id + 1) / 8));
  return concat(idBytes, size ?? concat(new Uint8Array([0x01]), bigEndian(data.length, 7)), data);
}
const uint = (id: number, value: number) =>
  el(id, bigEndian(value, Math.max(1, Math.ceil(Math.log2(value + 1) / 8))));
const UNKNOWN = new Uint8Array([0x01, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);

const header = el(0x1a45dfa3, el(0x4282, ascii("matroska")));
const track = (number: number, type: number, codec: string, defaultDuration?: number) => el(0xae, concat(
  uint(0xd7, number), uint(0x83, type), el(0x86, ascii(codec)),
  ...defaultDuration === undefined ? [] : [uint(0x23e383, defaultDuration)],
));
const cluster = el(0x1f43b675, concat(uint(0xe7, 0), el(0xa3, new Uint8Array([0x81, 0, 0, 0x80, 1, 2, 3]))));
const segment = (...parts: Uint8Array[]) => concat(header, el(0x18538067, concat(...parts), UNKNOWN));

describe("readMatroska", () => {
  test("counts the streams ffmpeg makes and reads default durations, tracks past the first MiB", async () => {
    const tracks = el(0x1654ae6b, concat(
      track(1, 1, "V_VP9", 33_333_333),
      track(2, 0x11, "D_WEBVTT/SUBTITLES"),
      track(3, 2, "V_MISMATCHED"),
      track(4, 0x12, "B_BUTTONS"),
    ));
    const attachments = el(0x1941a469, concat(
      el(0x61a7, concat(el(0x466e, ascii("a.ttf")), el(0x4660, ascii("font/ttf")),
                        el(0x465c, ascii("glyphs")))),
      el(0x61a7, concat(el(0x466e, ascii("b.ttf")), el(0x465c, ascii("glyphs")))),
    ));
    const blob = new Blob([segment(el(0xec, new Uint8Array(2 << 20)), tracks, attachments, cluster)]);
    const read = await readMatroska(blob);
    expect(read.streams).toBe(3);
    expect([...read.defaultDurations]).toEqual([[1, 33_333_333]]);
  });

  test("finds tracks after the clusters through the seek head", async () => {
    const seekHead = (position: number) =>
      el(0x114d9b74, el(0x4dbb, concat(el(0x53ab, bigEndian(0x1654ae6b, 4)),
                                       el(0x53ac, bigEndian(position, 4)))));
    const before = seekHead(0).length + cluster.length;
    const tracks = el(0x1654ae6b, track(1, 1, "V_VP9", 40_000_000));
    const blob = new Blob([segment(seekHead(before), cluster, tracks)]);
    const read = await readMatroska(blob);
    expect(read.streams).toBe(1);
    expect(read.defaultDurations.get(1)).toBe(40_000_000);
  });

  test("reads a file cut short in its first cluster", async () => {
    const whole = concat(header, el(0x18538067, concat(el(0x1654ae6b, track(1, 1, "V_VP9")), cluster)));
    expect((await readMatroska(new Blob([whole.subarray(0, whole.length - 3)]))).streams).toBe(1);
  });

  const block = (track: number, time: number) =>
    el(0xa3, concat(new Uint8Array([0x80 | track]), bigEndian(time, 2), new Uint8Array([0x80, 1, 2])));
  const clusterAt = (time: number, ...blocks: Uint8Array[]) =>
    el(0x1f43b675, concat(uint(0xe7, time), ...blocks));
  const withSubtitles = el(0x1654ae6b, concat(track(1, 1, "V_VP9"), track(2, 0x11, "S_TEXT/UTF8")));

  test("starts subtitles at their first block", async () => {
    const clusters = clusterAt(0, block(2, 0), block(1, 400));
    const read = await readMatroska(new Blob([segment(withSubtitles, clusters)]));
    expect(read.textStart).toBe(0);
  });

  test("counts ticks of the file's timestamp scale", async () => {
    const info = el(0x1549a966, uint(0x2ad7b1, 500_000));
    const clusters = clusterAt(4, block(2, 6), block(1, 0));
    const read = await readMatroska(new Blob([segment(info, withSubtitles, clusters)]));
    expect(read.textStart).toBe(5_000);
  });

  test("stops reading clusters once the video has begun", async () => {
    const read = await readMatroska(new Blob([segment(withSubtitles, clusterAt(0, block(1, 0)),
                                                      clusterAt(2000, block(2, 0)))]));
    expect(read.textStart).toBeNull();
  });

  test("ends a cluster of unknown size at the next cluster", async () => {
    const open = el(0x1f43b675, concat(uint(0xe7, 0), block(1, 200)), UNKNOWN);
    const read = await readMatroska(new Blob([segment(withSubtitles, open, clusterAt(100, block(2, 0)))]));
    expect(read.textStart).toBe(100_000);
  });

  test.each([
    ["an element of unknown size inside the tracks",
     el(0x1654ae6b, el(0xae, concat(el(0xd7, new Uint8Array([1]), new Uint8Array([0xff])), uint(0x83, 1))))],
    ["an integer longer than 8 bytes", el(0x1654ae6b, el(0xae, el(0xd7, new Uint8Array(9))))],
    ["an element longer than its parent",
     el(0x1654ae6b, el(0xae, uint(0xd7, 1), new Uint8Array([0x80 | 72])))],
    ["a size far past the end of the file",
     el(0x1654ae6b, el(0xd7, new Uint8Array(1), new Uint8Array([1, 0x7f, 255, 255, 255, 255, 255, 254])))],
    ["no tracks", cluster],
  ])("refuses %s", async (_, part) => {
    await expect(readMatroska(new Blob([segment(part)]))).rejects.toThrow(ContainerError);
  });
});

/** An ISOBMFF box, with a 64-bit size when `large`. */
function box(type: string, data: Uint8Array, large = false): Bytes {
  return large
    ? concat(bigEndian(1, 4), ascii(type), bigEndian(16 + data.length, 8), data)
    : concat(bigEndian(8 + data.length, 4), ascii(type), data);
}
/** A box that starts with a version and flags. */
const full = (type: string, version: number, ...data: Uint8Array[]) =>
  box(type, concat(new Uint8Array([version, 0, 0, 0]), ...data));
const u32 = (value: number) => bigEndian(value >>> 0, 4);
const table = (type: string, runs: [number, number][]) =>
  full(type, 0, u32(runs.length), ...runs.flatMap(([count, value]) => [u32(count), u32(value)]));

/** A track with ID `id`, handler type `handler` and time scale `scale`: its edit list's
 * entries ([duration, media time]) and its stts and ctts runs ([count, value]). */
function trak(id: number, handler: string, scale: number, edits: [number, number][] | null,
              stts: [number, number][], ctts?: [number, number][]): Bytes {
  return box("trak", concat(
    full("tkhd", 1, new Uint8Array(16), u32(id), new Uint8Array(64)),
    ...edits ? [box("edts", full("elst", 0, u32(edits.length),
                                 ...edits.flatMap(([d, t]) => [u32(d), u32(t), u32(0x10000)])))] : [],
    box("mdia", concat(
      full("mdhd", 0, new Uint8Array(8), u32(scale), u32(0), new Uint8Array(4)),
      full("hdlr", 0, new Uint8Array(4), ascii(handler), new Uint8Array(13)),
      box("minf", box("stbl", concat(table("stts", stts), ...ctts ? [table("ctts", ctts)] : []))),
    )),
  ));
}
const mvhd = (scale: number) => full("mvhd", 0, new Uint8Array(8), u32(scale), new Uint8Array(84));

describe("readMp4", () => {
  test("reads every track's handler, edit list and start, and the timing tables of the one asked for", async () => {
    const moov = box("moov", concat(
      mvhd(1000),
      trak(1, "vide", 15360, [[1000, 1024]], [[3, 512], [1, 1024]], [[1, 1024], [2, 0], [1, 512]]),
      trak(2, "soun", 44100, null, [[10, 1024]]),
      // Starting after an empty edit of half a second, and at its first sample's offset.
      trak(3, "tmcd", 15360, [[500, -1], [1000, 0]], [[1, 15360]]),
      trak(4, "subt", 1000, null, [[1, 100]], [[1, 40]]),
      // Its samples are in fragments.
      trak(5, "text", 1000, null, []),
    ));
    const blob = new Blob([box("ftyp", ascii("isom")), box("mdat", new Uint8Array(1000), true), moov]);
    const read = await readMp4(blob, 1);
    expect(read.streams).toBe(5);
    expect(read.textStart).toBe(40_000);
    expect(read.tracks.get(1)).toEqual({
      handler: "vide", start: 0, edited: true, edit: { delay: 0, time: 1024, duration: 15360 },
      durations: [512, 512, 512, 1024], offsets: [1024, 0, 0, 512], reordered: true, dtsShift: 0,
      counted: { samples: 4, duration: 2560 },
    });
    expect(read.tracks.get(2)).toEqual({
      handler: "soun", start: 0, edited: false, edit: null, durations: [], offsets: [],
      reordered: false, dtsShift: 0, counted: { samples: 0, duration: 0 },
    });
    expect(read.tracks.get(3)).toMatchObject({ start: 500_000, edit: { delay: 7680, time: 0 } });
    expect(read.tracks.get(5)).toMatchObject({ start: null });
  });

  test("starts a track with an edit list at its empty edits", async () => {
    const moov = box("moov", concat(mvhd(1000), trak(3, "tmcd", 15360, [[500, -1], [1000, 0]], [[1, 1]])));
    expect((await readMp4(new Blob([moov]))).textStart).toBe(500_000);
  });

  test("shifts decode timestamps for negative offsets, but not those of the last two entries", async () => {
    const offsets: [number, number][] = [[1, -512], [1, -1024], [1, 0], [1, -2048], [1, -4096]];
    const moov = box("moov", trak(1, "vide", 15360, null, [[5, 512]], offsets));
    const read = await readMp4(new Blob([moov]), 1);
    expect(read.tracks.get(1)).toMatchObject({
      reordered: true, dtsShift: 1024, offsets: [-512, -1024, 0, -2048, -4096],
    });
  });

  test("leaves the tracks not asked for as runs, however many samples they hold", async () => {
    // Six minutes of 48 kHz PCM is one sample per audio sample.
    const moov = box("moov", concat(trak(1, "vide", 600, null, [[200_000, 20]]),
                                    trak(2, "soun", 48000, null, [[48000 * 360, 1]])));
    const read = await readMp4(new Blob([moov]), 1);
    expect(read.tracks.get(1)?.offsets).toHaveLength(200_000);
    expect(read.tracks.get(2)).toMatchObject({ start: 0, durations: [] });
  });

  test.each([
    ["fewer than 8 bytes left at the end of a box", new Uint8Array(4)],
    ["a box running past its parent", concat(u32(200), ascii("udta"), new Uint8Array(8))],
  ])("passes over %s, as ffmpeg does", async (_, tail) => {
    const moov = box("moov", concat(trak(1, "vide", 30, null, [[2, 1]]), tail));
    const read = await readMp4(new Blob([box("ftyp", ascii("isom")), moov]), 1);
    expect(read.streams).toBe(1);
    expect(read.tracks.get(1)?.durations).toEqual([1, 1]);
  });

  describe("counts the samples in movie fragments as far as ffmpeg reads them", () => {
    /** Default durations by track, as trex boxes give them. */
    const mvex = (...defaults: [number, number][]) => box("mvex", concat(
      ...defaults.map(([id, duration]) => full("trex", 0, u32(id), u32(1), u32(duration), u32(0), u32(0)))));
    /** A track fragment of track `id`: its default duration, if it gives one, and its runs,
     * each a sample count or the samples' own durations. */
    const traf = (id: number, duration: number | null, ...runs: (number | number[])[]) => box("traf", concat(
      box("tfhd", concat(new Uint8Array([0, 0, 0, duration === null ? 0 : 0x08]), u32(id),
                         ...duration === null ? [] : [u32(duration)])),
      ...runs.map((run) => typeof run === "number"
        ? box("trun", concat(new Uint8Array(4), u32(run)))
        : box("trun", concat(new Uint8Array([0, 0, 1, 0]), u32(run.length), ...run.map(u32)))),
    ));
    const moof = (...trafs: Bytes[]) => box("moof", concat(full("mfhd", 0, u32(1)), ...trafs));
    const mdat = box("mdat", new Uint8Array(16));
    const moov = (stts: [number, number][]) => box("moov", concat(
      mvhd(1000), trak(1, "vide", 600, null, stts), trak(2, "soun", 48000, null, []),
      mvex([1, 20], [2, 1024])));
    /** A segment index of track 1 whose references, starting `gap` bytes after it, sum to `size`. */
    const sidx = (size: number, gap = 0) => full("sidx", 0, u32(1), u32(600), u32(0), u32(gap),
                                                 new Uint8Array([0, 0, 0, 1]), u32(size), u32(600), u32(0));
    const fragments = [moof(traf(1, null, 3), traf(2, null, 5)), mdat,
                       moof(traf(1, 25, 2, [21, 22]), traf(2, 512, [1024])), mdat];
    const counted = async (...parts: Bytes[]) =>
      (await readMp4(new Blob([box("ftyp", ascii("isom")), ...parts]), 1)).tracks.get(1)?.counted;
    const after = (...parts: Bytes[]) => parts.reduce((sum, part) => sum + part.length, 0);

    test("in every fragment, with the movie header's samples", async () => {
      // The trex default, then the track fragment's, then the samples' own.
      expect(await counted(moov([]), ...fragments))
        .toEqual({ samples: 7, duration: 3 * 20 + 2 * 25 + 21 + 22 });
      expect(await counted(moov([[2, 30]]), ...fragments)).toEqual({ samples: 9, duration: 60 + 153 });
    });

    test("past the fields before each duration", async () => {
      // A base data offset and a sample description index before the default duration; a
      // data offset and first sample flags before the samples, which have sizes and
      // composition offsets besides, and in the second run no durations of their own.
      const tfhd = box("tfhd", concat(new Uint8Array([0, 0, 0, 0x0b]), u32(1), bigEndian(1000, 8), u32(1),
                                      u32(40)));
      const trun = (flags: number, ...samples: number[][]) => box("trun", concat(
        new Uint8Array([0, 0, flags >> 8, flags & 0xff]), u32(samples.length), u32(100), u32(0x2000000),
        ...samples.flat().map(u32)));
      const fragment = moof(box("traf", concat(tfhd, trun(0xb05, [31, 200, 0], [32, 200, 66]),
                                               trun(0xa05, [200, 0]))));
      expect(await counted(moov([]), fragment, mdat)).toEqual({ samples: 3, duration: 31 + 32 + 40 });
    });

    test("not for a track the movie header gives no defaults for", async () => {
      const bare = box("moov", concat(mvhd(1000), trak(1, "vide", 600, null, [])));
      expect(await counted(bare, ...fragments)).toEqual({ samples: 0, duration: 0 });
    });

    test("only before the first media data after a segment index that covers the file", async () => {
      const covers = sidx(after(...fragments));
      expect(await counted(moov([]), covers, ...fragments)).toEqual({ samples: 3, duration: 60 });
      // An index closed by a movie fragment random access box covers the file too.
      const mfra = box("mfra", full("mfro", 0, u32(24)));
      expect(await counted(moov([]), covers, ...fragments, mfra)).toEqual({ samples: 3, duration: 60 });
      // One that falls short, or starts after a gap, doesn't.
      expect((await counted(moov([]), sidx(after(...fragments) - 1), ...fragments))?.samples).toBe(7);
      expect((await counted(moov([]), sidx(after(...fragments) - 8, 8), ...fragments))?.samples).toBe(7);
      // Media data holding none doesn't count.
      const empty = box("mdat", new Uint8Array(0));
      expect(await counted(moov([]), sidx(after(empty, ...fragments)), empty, ...fragments))
        .toEqual({ samples: 3, duration: 60 });
    });
  });

  test.each([
    ["no movie header", box("ftyp", ascii("isom"))],
    ["a movie header after a box too small for its own header",
     concat(box("ftyp", ascii("isom")), bigEndian(4, 4), ascii("free"), box("moov", mvhd(1000)))],
    ["a sample table of more samples than any video holds",
     box("moov", trak(1, "vide", 30, null, [[2 ** 24, 1], [1, 1]]))],
  ])("refuses %s", async (_, bytes) => {
    await expect(readMp4(new Blob([bytes]), 1)).rejects.toThrow(ContainerError);
  });
});
