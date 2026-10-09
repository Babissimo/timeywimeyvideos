/**
 * What ffmpeg reads from a container that Mediabunny doesn't report: how many
 * streams it makes of the file, when its subtitle and data streams start, and
 * per-track details that frame timing depends on. Both readers follow the
 * file's structure, reading only the parts they need.
 */

/** A container header that can't be read. */
export class ContainerError extends Error {
  override name = "ContainerError";
}

export interface MatroskaHeader {
  /** How many streams ffmpeg makes: tracks of a type it knows, with a codec of that kind,
   * and complete attachments. */
  streams: number;
  /** When the earliest subtitle or metadata track starts (at its first block), in
   * microseconds, so far as it can come before the video and audio: null when none has a
   * block before the first cluster to start after a block of theirs. */
  textStart: number | null;
  /** Each track's default duration in nanoseconds, by track number. */
  defaultDurations: Map<number, number>;
}

export interface Mp4Header {
  /** How many tracks the movie holds, of any kind (timecode, subtitles and data included). */
  streams: number;
  /** When the earliest track other than video and audio starts, as ffmpeg starts it, in
   * microseconds; null if there is none with samples in the movie header. */
  textStart: number | null;
  /** The tracks by ID. */
  tracks: Map<number, Mp4Track>;
}

export interface Mp4Track {
  /** The handler type, such as "vide", "soun", "subt" or "tmcd". */
  handler: string;
  /** When ffmpeg starts the track, in microseconds; null if it has no samples in the movie header. */
  start: number | null;
  /** Whether the track has an edit list. */
  edited: boolean;
  /** The edit list's first edit that shows media, null if there is none. */
  edit: Edit | null;
  /** For the track `readMp4` was asked to expand, its samples' durations and composition
   * offsets in decode order, in the track's time scale. Empty for the others, and when the
   * samples are in fragments rather than the movie header. */
  durations: number[]; offsets: number[];
  /** Whether the track has composition offsets, and how far ffmpeg moves its decode
   * timestamps back so that negative ones still show after decoding. */
  reordered: boolean; dtsShift: number;
}

/** An edit that shows a track's media, in the track's time scale: it follows empty edits
 * lasting `delay`, and shows the media from `time` for `duration`. */
export interface Edit { delay: number; time: number; duration: number }

/** Where an element or box sits in the file: its header from `start`, its data from `data`
 * to `end`. One of unknown size is `open`, and runs to the end of its parent. */
interface Span { id: number; start: number; data: number; end: number; open?: boolean }

// The parts read whole are small; one claiming more than this is corrupt.
const MAX_READ = 64 << 20;

async function read(source: Blob, from: number, to: number): Promise<Uint8Array> {
  if (to - from > MAX_READ) throw new ContainerError(`a header of ${to - from} bytes at byte ${from}`);
  return new Uint8Array(await source.slice(from, to).arrayBuffer());
}

/** a × b / c (c positive), rounded to the nearest integer, halves away from zero, as
 * ffmpeg's av_rescale. */
function rescale(a: number, b: number, c: number): number {
  const n = BigInt(a) * BigInt(b);
  const half = BigInt(c) / 2n;
  return Number(n < 0n ? -((-n + half) / BigInt(c)) : (n + half) / BigInt(c));
}

/** The EBML elements from `from` to `to`, reading only their headers. Each must end by `to`,
 * unless `ragged` lets one run past it (as in a file cut short). */
async function* walk(source: Blob, from: number, to: number, ragged = false) {
  for (let at = from; at < to;) {
    const span = ebml(await read(source, at, Math.min(at + 16, to)), 0, at, to);
    if (span.end > to && !ragged) throw new ContainerError(`the header at byte ${at} runs past its parent`);
    yield span;
    at = span.end;
  }
}

/** The EBML elements inside `bytes`, the whole data of a parent at file position `base`. */
function* within(bytes: Uint8Array, base: number) {
  const to = base + bytes.length;
  for (let at = base; at < to;) {
    const span = ebml(bytes, at - base, at, to);
    if (span.open) throw new ContainerError(`an element of unknown size at byte ${at}`);
    if (span.end > to) throw new ContainerError(`the header at byte ${at} runs past its parent`);
    yield span;
    at = span.end;
  }
}

const EBML = {
  header: 0x1a45dfa3, segment: 0x18538067, seekHead: 0x114d9b74, seek: 0x4dbb, seekId: 0x53ab,
  seekPosition: 0x53ac, info: 0x1549a966, timestampScale: 0x2ad7b1, tracks: 0x1654ae6b,
  trackEntry: 0xae, trackNumber: 0xd7, trackType: 0x83, codecId: 0x86, defaultDuration: 0x23e383,
  attachments: 0x1941a469, attachedFile: 0x61a7, fileName: 0x466e, fileMimeType: 0x4660,
  fileData: 0x465c, cluster: 0x1f43b675, timestamp: 0xe7, simpleBlock: 0xa3, blockGroup: 0xa0,
  block: 0xa1, cues: 0x1c53bb6b, chapters: 0x1043a770, tags: 0x1254c367,
};
// The elements that can follow a cluster at the top level, and so end one of unknown size.
const TOP_LEVEL = new Set([EBML.seekHead, EBML.info, EBML.tracks, EBML.cluster, EBML.cues,
                           EBML.attachments, EBML.chapters, EBML.tags]);

/** An EBML variable-length integer: an ID keeps its length marker, a size of all ones is unknown (null). */
function vint(bytes: Uint8Array, offset: number, isId: boolean, position: number) {
  const head = bytes[offset] ?? 0;
  const length = Math.clz32(head) - 23;
  if (head === 0 || (isId && length > 4) || offset + length > bytes.length) {
    throw new ContainerError(`a malformed EBML header at byte ${position}`);
  }
  let value = isId ? head : head & (0xff >> length);
  let unknown = !isId && value === 0xff >> length;
  for (let i = 1; i < length; i++) {
    value = value * 256 + bytes[offset + i]!;
    unknown &&= bytes[offset + i] === 0xff;
  }
  return { value: unknown ? null : value, length };
}

/** The EBML element whose header is at `offset` in `bytes` (file position `position`),
 * inside a parent ending at `to`. */
function ebml(bytes: Uint8Array, offset: number, position: number, to: number): Span {
  const id = vint(bytes, offset, true, position);
  const size = vint(bytes, offset + id.length, false, position);
  const data = position + id.length + size.length;
  if (size.value === null) {
    return { id: id.value!, start: position, data, end: Math.max(to, data), open: true };
  }
  return { id: id.value!, start: position, data, end: data + size.value };
}

/** An unsigned integer element's value, from `bytes` holding the file from `base`. */
function uint(bytes: Uint8Array, span: Span, base: number): number {
  const length = span.end - span.data;
  if (length > 8) throw new ContainerError(`an integer of ${length} bytes at byte ${span.start}`);
  let value = 0;
  for (let i = span.data - base; i < span.end - base; i++) value = value * 256 + bytes[i]!;
  return value;
}

/** Read the parts of a Matroska file's header that ffmpeg makes streams of. */
export async function readMatroska(source: Blob): Promise<MatroskaHeader> {
  let segment: Span | undefined;
  for await (const top of walk(source, 0, source.size, true)) {
    if (top.start === 0 && top.id !== EBML.header) break;
    if (top.id === EBML.segment) {
      segment = { ...top, end: Math.min(top.end, source.size) };
      break;
    }
  }
  if (!segment) throw new ContainerError("no Matroska segment");

  // The top level as far as the first cluster, then through the seek head
  // whatever the muxer put after the clusters.
  const found = new Map<number, Span>();
  const seeks = new Map<number, number>();
  let clusters: number | undefined;
  for await (const part of walk(source, segment.data, segment.end, true)) {
    if (part.id === EBML.cluster) clusters = part.start;
    if (part.id === EBML.cluster || part.end > segment.end) break;
    if (!found.has(part.id)) found.set(part.id, part);
    if (part.id !== EBML.seekHead) continue;
    const bytes = await read(source, part.data, part.end);
    for (const seek of within(bytes, part.data)) {
      if (seek.id !== EBML.seek) continue;
      let id: number | undefined;
      let at: number | undefined;
      for (const field of within(bytes.subarray(seek.data - part.data, seek.end - part.data), seek.data)) {
        if (field.id === EBML.seekId) id = uint(bytes, field, part.data);
        if (field.id === EBML.seekPosition) at = segment.data + uint(bytes, field, part.data);
      }
      if (id !== undefined && at !== undefined && at < segment.end) seeks.set(id, at);
    }
  }
  for (const id of [EBML.info, EBML.tracks, EBML.attachments]) {
    const at = seeks.get(id);
    if (found.has(id) || at === undefined) continue;
    const part = ebml(await read(source, at, Math.min(at + 16, segment.end)), 0, at, segment.end);
    if (part.id === id && !part.open && part.end <= segment.end) found.set(id, part);
  }

  const tracks = found.get(EBML.tracks);
  if (!tracks || tracks.open) throw new ContainerError("no Matroska tracks");
  const bytes = await read(source, tracks.data, tracks.end);
  const part = (span: Span) => bytes.subarray(span.data - tracks.data, span.end - tracks.data);
  let streams = 0;
  const textTracks = new Set<number>();
  const defaultDurations = new Map<number, number>();
  for (const entry of within(bytes, tracks.data)) {
    if (entry.id !== EBML.trackEntry) continue;
    let number: number | undefined;
    let type = 0;
    let codec: number | undefined;
    let defaultDuration: number | undefined;
    for (const field of within(part(entry), entry.data)) {
      if (field.id === EBML.trackNumber) number = uint(bytes, field, tracks.data);
      if (field.id === EBML.trackType) type = uint(bytes, field, tracks.data);
      if (field.id === EBML.codecId) codec = part(field)[0];
      if (field.id === EBML.defaultDuration) defaultDuration = uint(bytes, field, tracks.data);
    }
    if (number !== undefined && defaultDuration !== undefined) defaultDurations.set(number, defaultDuration);
    // Video, audio, subtitle and metadata tracks, whose codec IDs start V, A, and S or D.
    const initials = ({ 1: "V", 2: "A", 0x11: "SD", 0x21: "SD" } as Record<number, string>)[type];
    if (codec === undefined || !initials?.includes(String.fromCharCode(codec))) continue;
    streams++;
    if (type > 2 && number !== undefined) textTracks.add(number);
  }

  const attachments = found.get(EBML.attachments);
  if (attachments && !attachments.open) {
    for await (const file of walk(source, attachments.data, attachments.end)) {
      if (file.id !== EBML.attachedFile) continue;
      const fields = new Map<number, Span>();
      for await (const field of walk(source, file.data, file.end)) fields.set(field.id, field);
      const data = fields.get(EBML.fileData);
      if (fields.has(EBML.fileName) && fields.has(EBML.fileMimeType) && data && data.end > data.data) {
        streams++;
      }
    }
  }

  let textStart: number | null = null;
  if (textTracks.size > 0 && clusters !== undefined) {
    const ticks = await firstTextBlock(source, clusters, segment.end, textTracks);
    if (ticks !== null) {
      let scale = 1_000_000;
      const info = found.get(EBML.info);
      if (info && !info.open) {
        const fields = await read(source, info.data, info.end);
        for (const field of within(fields, info.data)) {
          if (field.id === EBML.timestampScale) scale = uint(fields, field, info.data);
        }
      }
      textStart = rescale(ticks, scale, 1000);
    }
  }
  return { streams, textStart, defaultDurations };
}

/**
 * The earliest first block of these tracks (in TimestampScale ticks), reading the clusters
 * from `from` only until one starts after a block of another track, or until each of these
 * has had a block: a later first block can't come before the other tracks start.
 */
async function firstTextBlock(source: Blob, from: number, to: number,
                              text: Set<number>): Promise<number | null> {
  const seen = new Set<number>();
  let earliest: number | null = null;
  let others = Infinity;
  const block = async (at: number, end: number, time: number) => {
    const head = await read(source, at, Math.min(at + 11, end));
    const track = vint(head, 0, false, at);
    if (track.value === null || head.length < track.length + 2) return;
    const t = time + new DataView(head.buffer, head.byteOffset).getInt16(track.length);
    if (!text.has(track.value)) others = Math.min(others, t);
    else if (!seen.has(track.value)) {
      seen.add(track.value);
      if (earliest === null || t < earliest) earliest = t;
    }
  };
  for (let at = from; at < to && seen.size < text.size;) {
    const cluster = ebml(await read(source, at, Math.min(at + 16, to)), 0, at, to);
    let next = Math.min(cluster.end, to);
    if (cluster.id === EBML.cluster) {
      let time = 0;
      for await (const child of walk(source, cluster.data, next, true)) {
        // A cluster of unknown size ends where the next top-level element begins.
        if (TOP_LEVEL.has(child.id)) {
          next = child.start;
          break;
        }
        const end = Math.min(child.end, to);
        if (child.id === EBML.timestamp) {
          time = uint(await read(source, child.data, end), { ...child, end }, child.data);
          if (time > others) return earliest;
        } else if (child.id === EBML.simpleBlock) {
          await block(child.data, end, time);
        } else if (child.id === EBML.blockGroup) {
          for await (const inner of walk(source, child.data, end, true)) {
            if (inner.id === EBML.block) await block(inner.data, Math.min(inner.end, end), time);
          }
        }
      }
    } else if (cluster.open) {
      break;
    }
    at = next;
  }
  return earliest;
}

/** The ISOBMFF box whose header is at `offset` in `bytes` (file position `position`), inside
 * a parent ending at `to`, its type read as a number; null where ffmpeg would stop reading. */
function box(bytes: Uint8Array, offset: number, position: number, to: number): Span | null {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let size = view.getUint32(offset);
  let data = position + 8;
  if (size === 1 && position + 16 <= to) {
    size = Number(view.getBigUint64(offset + 8));
    data += 8;
  } else if (size === 0) {
    size = to - position;
  }
  if (position + size < data) return null;
  // ffmpeg cuts a box that runs past its parent short at the parent's end.
  return { id: view.getUint32(offset + 4), start: position, data, end: Math.min(position + size, to) };
}

/** The boxes from `from` to `to` as ffmpeg's mov_read_default walks them, reading only their
 * headers: it stops with fewer than 8 bytes left, or at a size too small for its header. */
async function* boxes(source: Blob, from: number, to: number) {
  for (let at = from; to - at >= 8;) {
    const span = box(await read(source, at, Math.min(at + 16, to)), 0, at, to);
    if (!span) return;
    yield span;
    at = span.end;
  }
}

/** The boxes inside `bytes`, the whole data of a parent at file position `base`. */
function* boxesWithin(bytes: Uint8Array, base: number) {
  for (let at = base; base + bytes.length - at >= 8;) {
    const span = box(bytes, at - base, at, base + bytes.length);
    if (!span) return;
    yield span;
    at = span.end;
  }
}

const fourcc = (name: string) => [...name].reduce((id, c) => id * 256 + c.charCodeAt(0), 0);
const BOX = Object.fromEntries(
  ["moov", "mvhd", "trak", "tkhd", "edts", "elst", "mdia", "mdhd", "hdlr", "minf", "stbl", "stts", "ctts"]
    .map((name) => [name, fourcc(name)]),
) as Record<string, number>;

/** One `trak` box as read: its edit list's entries ([duration in the movie's time scale,
 * media time]) and the contents of its stts and ctts boxes. */
interface Trak {
  id: number; handler: string; timescale: number;
  edits: [number, number][] | null; stts: DataView | null; ctts: DataView | null;
}

/** Read the parts of an MP4 or QuickTime movie header that ffmpeg makes streams of, with
 * the samples of the track whose ID is `expand`. */
export async function readMp4(source: Blob, expand?: number): Promise<Mp4Header> {
  for await (const moov of boxes(source, 0, source.size)) {
    if (moov.id !== BOX.moov) continue;
    let movieScale = 0;
    const traks: Trak[] = [];
    for await (const part of boxes(source, moov.data, moov.end)) {
      if (part.id === BOX.mvhd) movieScale = timescale(await read(source, part.data, part.end));
      if (part.id === BOX.trak) traks.push(await readTrak(source, part));
    }
    let textStart: number | null = null;
    const tracks = new Map<number, Mp4Track>();
    for (const trak of traks) {
      const track = mp4Track(trak, movieScale, trak.id === expand);
      tracks.set(trak.id, track);
      if (["vide", "soun"].includes(track.handler) || track.start === null) continue;
      if (textStart === null || track.start < textStart) textStart = track.start;
    }
    return { streams: traks.length, textStart, tracks };
  }
  throw new ContainerError("no movie header");
}

/** A movie or media header's time scale, after its version, flags and times (4 or 8 bytes each). */
function timescale(bytes: Uint8Array): number {
  const at = bytes[0] === 1 ? 20 : 12;
  return bytes.length >= at + 4 ? new DataView(bytes.buffer, bytes.byteOffset).getUint32(at) : 0;
}

/** The [count, value] runs of an stts or ctts box, after its version, flags and entry
 * count, as far as the box has room for them. */
function* runs(table: DataView, signed: boolean) {
  const room = Math.floor((table.byteLength - 8) / 8);
  const entries = Math.min(room >= 0 ? table.getUint32(4) : 0, room);
  for (let i = 0; i < entries; i++) {
    const value = signed ? table.getInt32(12 + 8 * i) : table.getUint32(12 + 8 * i);
    yield { count: table.getInt32(8 + 8 * i), value, last: i + 2 >= entries };
  }
}

// More samples than this in one track (some 75 hours at 60 fps) is a corrupt table.
const MAX_SAMPLES = 1 << 24;

/** A table's values sample by sample, at most `limit` of them. */
function expandRuns(table: DataView, signed: boolean, limit: number): number[] {
  let total = 0;
  for (const { count } of runs(table, signed)) total += Math.max(count, 0);
  if (Math.min(total, limit) > MAX_SAMPLES) {
    throw new ContainerError(`a sample table of over ${MAX_SAMPLES} samples`);
  }
  const out: number[] = [];
  for (const { count, value } of runs(table, signed)) {
    for (let j = 0; j < count && out.length < limit; j++) out.push(value);
  }
  return out;
}

function mp4Track(trak: Trak, movieScale: number, expand: boolean): Mp4Track {
  const { handler, timescale, edits, stts, ctts } = trak;
  // ffmpeg converts each edit's duration from the movie's time scale to the track's. Empty
  // edits before the first that shows media delay the track; later edits are not modelled.
  const inTrack = (d: number) => movieScale ? rescale(d, timescale, movieScale) : 0;
  let delay = 0;
  let edit: Edit | null = null;
  for (const [duration, time] of edits ?? []) {
    if (time === -1) {
      delay += inTrack(duration);
      continue;
    }
    if (movieScale) edit = { delay, time, duration: inTrack(duration) };
    break;
  }

  let samples = 0;
  for (const { count } of stts ? runs(stts, false) : []) samples += Math.max(count, 0);
  let first: number | undefined;
  let reordered = false;
  let dtsShift = 0;
  for (const { count, value, last } of ctts ? runs(ctts, true) : []) {
    if (count <= 0) continue;
    first ??= value;
    reordered = true;
    // ffmpeg leaves the last two entries out of the shift.
    if (!last && value < 0) dtsShift = Math.max(dtsShift, value === -(2 ** 31) ? 2 ** 31 - 1 : -value);
  }
  // ffmpeg starts a track at its empty edits, or without an edit list at its first
  // sample's presentation time.
  const edited = edits !== null;
  const start = samples && timescale ? rescale(edited ? delay : first ?? 0, 1_000_000, timescale) : null;

  let durations: number[] = [];
  let offsets: number[] = [];
  if (expand && stts) {
    durations = expandRuns(stts, false, Infinity);
    offsets = ctts ? expandRuns(ctts, true, durations.length) : [];
    // Samples without a composition offset show at their decode time.
    const given = offsets.length;
    offsets.length = durations.length;
    offsets.fill(0, given);
  }
  return { handler, start, edited, edit, durations, offsets, reordered, dtsShift };
}

async function readTrak(source: Blob, trak: Span): Promise<Trak> {
  const contents = async (span: Span) => {
    const bytes = await read(source, span.data, span.end);
    return { bytes, view: new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength) };
  };
  const result: Trak = { id: -1, handler: "", timescale: 0, edits: null, stts: null, ctts: null };
  for await (const child of boxes(source, trak.data, trak.end)) {
    if (child.id === BOX.tkhd) {
      const { bytes, view } = await contents(child);
      // version and flags, creation and modification times (4 or 8 bytes each), track ID
      const at = bytes[0] === 1 ? 20 : 12;
      if (bytes.length >= at + 4) result.id = view.getUint32(at);
    } else if (child.id === BOX.edts) {
      const { bytes } = await contents(child);
      for (const elst of boxesWithin(bytes, child.data)) {
        if (elst.id !== BOX.elst) continue;
        const entries = bytes.subarray(elst.data - child.data, elst.end - child.data);
        const view = new DataView(entries.buffer, entries.byteOffset, entries.byteLength);
        const version = entries[0];
        const size = version === 1 ? 20 : 12;
        // ffmpeg counts the entries the box has room for, and passes over a list of none.
        const count = Math.max(0, Math.floor((entries.length - 8) / size));
        if (!count) continue;
        result.edits = Array.from({ length: count }, (_, i) => {
          const at = 8 + size * i;
          const duration = version === 1 ? Number(view.getBigUint64(at)) : view.getUint32(at);
          const time = version === 1 ? Number(view.getBigInt64(at + 8)) : view.getInt32(at + 4);
          return [duration, time];
        });
      }
    } else if (child.id === BOX.mdia) {
      for await (const media of boxes(source, child.data, child.end)) {
        if (media.id === BOX.mdhd) result.timescale = timescale((await contents(media)).bytes);
        if (media.id === BOX.hdlr) {
          const { bytes } = await contents(media);
          // version and flags, a predefined field, then the handler type
          if (bytes.length >= 12) result.handler = String.fromCharCode(...bytes.subarray(8, 12));
        }
        if (media.id !== BOX.minf) continue;
        for await (const info of boxes(source, media.data, media.end)) {
          if (info.id !== BOX.stbl) continue;
          for await (const table of boxes(source, info.data, info.end)) {
            if (table.id === BOX.stts) result.stts = (await contents(table)).view;
            if (table.id === BOX.ctts) result.ctts = (await contents(table)).view;
          }
        }
      }
    }
  }
  return result;
}
