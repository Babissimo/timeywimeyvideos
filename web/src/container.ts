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
  /** The tracks by ID, the last of any that share one. */
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
  /** For the track `readMp4` was asked to expand, the samples ffmpeg takes its average frame
   * rate over: how many the movie header and the fragments it reads with it hold, and how
   * long they last together, in the track's time scale. Zero for the others. */
  counted: { samples: number; duration: number };
  /** How long ffmpeg takes the track to last, in microseconds, counted from the start of its
   * media rather than from the track's start (see `readMp4`). */
  duration: number;
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
  ["moov", "mvhd", "trak", "tkhd", "edts", "elst", "mdia", "mdhd", "hdlr", "minf", "stbl", "stts", "ctts",
   "mvex", "trex", "moof", "traf", "tfhd", "tfdt", "trun", "sidx", "mdat"]
    .map((name) => [name, fourcc(name)]),
) as Record<string, number>;

/** One `trak` box as read: its media header's time scale and duration, its edit list's
 * entries ([duration in the movie's time scale, media time]) and the contents of its stts
 * and ctts boxes. */
interface Trak {
  id: number; handler: string; timescale: number; duration: number;
  edits: [number, number][] | null; stts: DataView | null; ctts: DataView | null;
}

/** A track's timing as ffmpeg reads the movie, in the track's time scale: how long it takes
 * the track to last, where its media so far ends, how far its edit list sets its decode
 * times back from its media's, its place among the tracks, and whether a segment index has
 * indexed it. */
interface Extent {
  scale: number; duration: number; end: number; offset: number; rank: number; indexed: boolean;
}

/**
 * Read the parts of an MP4 or QuickTime movie header that ffmpeg makes streams of, with
 * the samples of the track whose ID is `expand`. The top level is read as far as ffmpeg
 * reads it before probing: past the movie header and the first media data, and through any
 * movie fragments, until a box ends the file or a segment index has covered it.
 *
 * A track lasts as long as the movie header makes it (`extentsOf`); each run of samples in
 * the fragments read takes it to where the run ends, if that is later; and a segment index
 * sets it to where its references end, one covering the file setting each track without an
 * index of its own to the track whose index starts first in the file. While probing, ffmpeg
 * reads on through a few seconds of fragments after such an index, whose runs can take a
 * track past the index's span where that falls short of them; that is not modelled.
 */
export async function readMp4(source: Blob, expand?: number): Promise<Mp4Header> {
  let header: Mp4Header | undefined;
  let defaults = new Map<number, number>();
  // Each track's timing beside the track, and by ID the first track's, which ffmpeg gives the
  // fragments and indexes of every track with that ID.
  const timings: [Mp4Track, Extent][] = [];
  const extents = new Map<number, Extent>();
  // Where segment indexes place each movie fragment they reference, by its position in the
  // file and then by track, in the track's time scale.
  const places = new Map<number, Map<number, number>>();
  // The index whose references start first in the file, the earlier track's on a tie.
  let lead: { at: number; extent: Extent } | undefined;
  // The size of a movie fragment random access box closing the file, as its last 4 bytes give it.
  let closing: Promise<number> | undefined;
  const closingSize = () => closing ??= read(source, source.size - 4, source.size)
    .then((tail) => new DataView(tail.buffer, tail.byteOffset).getUint32(0));
  let media = false;
  let covered = false;
  for await (const top of boxes(source, 0, source.size)) {
    if (top.id === BOX.moov && !header) {
      let movieScale = 0;
      const traks: Trak[] = [];
      for await (const part of boxes(source, top.data, top.end)) {
        if (part.id === BOX.mvhd) {
          movieScale = timeScale(timing(await read(source, part.data, part.end)).scale);
        }
        if (part.id === BOX.trak) traks.push(await readTrak(source, part));
        if (part.id === BOX.mvex) defaults = await fragmentDefaults(source, part);
      }
      let textStart: number | null = null;
      const tracks = new Map<number, Mp4Track>();
      const timed = extentsOf(traks, movieScale);
      for (const [i, trak] of traks.entries()) {
        const track = mp4Track(trak, movieScale, trak.id === expand);
        tracks.set(trak.id, track);
        timings.push([track, timed[i]!]);
        if (!extents.has(trak.id)) extents.set(trak.id, timed[i]!);
        if (["vide", "soun"].includes(track.handler) || track.start === null) continue;
        if (textStart === null || track.start < textStart) textStart = track.start;
      }
      header = { streams: traks.length, textStart, tracks };
    } else if (header && top.id === BOX.moof) {
      // A track fragment's first run starts at the last decode time the movie fragment gave
      // its track, else where a segment index places the movie fragment, else where the
      // track's media ends; each further run where the one before it ended.
      const decodeTimes = new Map<number, number>();
      // ffmpeg looks the movie fragment up 8 bytes before its data, past its start if its size
      // takes 64 bits.
      const placed = places.get(top.data - 8);
      for (const { id, parts } of trackFragments(await read(source, top.data, top.end), top.data, defaults)) {
        const extent = extents.get(id);
        if (!extent) continue;
        let next: number | undefined;
        for (const part of parts) {
          if ("time" in part) {
            decodeTimes.set(id, part.time);
            extent.end = part.time;
          } else if (part.samples) {  // ffmpeg passes over a run of no samples
            const at = next ?? decodeTimes.get(id) ?? placed?.get(id) ?? extent.end;
            next = extent.end = at + part.duration;
            extent.duration = Math.max(extent.duration, extent.end);
            if (id !== expand) continue;
            const { counted } = header.tracks.get(id)!;
            counted.samples += part.samples;
            counted.duration += part.duration;
          }
        }
      }
    } else if (header && top.id === BOX.sidx) {
      const index = await segmentIndex(source, top, extents);
      if (index) {
        let complete = index.reach === source.size;
        if (!complete && source.size >= 4) complete = index.reach === source.size - await closingSize();
        const { extent } = index;
        extent.duration = extent.end = index.end;
        extent.indexed = true;
        // ffmpeg takes a reference's time for its fragment's first decode time, which the edit
        // list's offset puts that much later in the track's media.
        for (const [at, time] of index.places) {
          const placed = places.get(at) ?? new Map<number, number>();
          places.set(at, placed.set(index.id, rescale(time, extent.scale, index.scale) + extent.offset));
        }
        if (!lead || index.first < lead.at || (index.first === lead.at && extent.rank < lead.extent.rank)) {
          lead = { at: index.first, extent };
        }
        const by = lead.extent;
        for (const [, other] of complete ? timings : []) {
          if (!other.indexed) other.duration = other.end = rescale(by.duration, other.scale, by.scale);
        }
        // ffmpeg stops at the next media data only after an index covering the file from its
        // own end.
        covered ||= complete && index.first === top.end;
      }
    }
    // ffmpeg takes no notice of media data that holds none.
    if (top.id === BOX.mdat && top.end > top.data) media = true;
    if (header && media && (covered || top.end === source.size)) break;
  }
  if (!header) throw new ContainerError("no movie header");
  for (const [track, extent] of timings) track.duration = rescale(extent.duration, 1_000_000, extent.scale);
  return header;
}

/** Each track's timing as ffmpeg reads it from the movie header, in its order. A track lasts
 * as long as its media header says, or its sample table if that is shorter and lasts at all,
 * and no longer than its edit list, if it has samples in the header and every track up to it
 * has a sample table with entries (ffmpeg's mov_fix_index). */
function extentsOf(traks: Trak[], movieScale: number): Extent[] {
  let fitted = true;
  return traks.map((trak, rank) => {
    const scale = timeScale(trak.timescale);
    let [entries, end, samples] = [0, 0, false];
    for (const { count, value } of trak.stts ? runs(trak.stts, false) : []) {
      entries++;
      end += Math.max(count, 0) * value;
      samples ||= count > 0;
    }
    let duration = end ? Math.min(trak.duration, end) : trak.duration;
    fitted &&= entries > 0;
    // Samples in the header, as the sample table counts them.
    if (fitted && trak.edits && movieScale && samples) {
      const edited = trak.edits.reduce((sum, [length]) => sum + rescale(length, scale, movieScale), 0);
      duration = Math.min(duration, edited);
    }
    const offset = timeOffset(trak, scale, movieScale);
    return { scale, duration, end, offset, rank, indexed: false };
  });
}

/** How far ffmpeg sets a track's decode times back from its media's (its time_offset), in
 * the track's time scale: the media time its edit list starts at, less a first empty edit. */
function timeOffset(trak: Trak, scale: number, movieScale: number): number {
  const [first, second] = trak.edits ?? [];
  if (!first || !movieScale) return 0;
  const empty = first[1] === -1 ? rescale(first[0], scale, movieScale) : 0;
  return Math.max((first[1] === -1 ? second?.[1] : first[1]) ?? 0, 0) - empty;
}

/** Each track's default sample duration in its fragments, by track ID, from an mvex box's
 * trex boxes: the first for a track counts. */
async function fragmentDefaults(source: Blob, mvex: Span): Promise<Map<number, number>> {
  const bytes = await read(source, mvex.data, mvex.end);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const defaults = new Map<number, number>();
  for (const trex of boxesWithin(bytes, mvex.data)) {
    const at = trex.data - mvex.data;
    // version and flags, track ID, sample description index, then the duration
    if (trex.id !== BOX.trex || trex.end - trex.data < 16) continue;
    const id = view.getUint32(at + 4);
    if (!defaults.has(id)) defaults.set(id, view.getUint32(at + 12));
  }
  return defaults;
}

/** The track fragments in a movie fragment's data `bytes` (from file position `base`) of
 * tracks the movie header gives `defaults` for, as ffmpeg reads them, each header starting
 * one: the track, then in order each decode time the fragment gives it and each run of
 * samples, how many the run holds and how long they last together. A sample without its own
 * duration takes its track fragment's default, else the movie header's. */
function* trackFragments(bytes: Uint8Array, base: number, defaults: Map<number, number>) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  type Part = { time: number } | { samples: number; duration: number };
  for (const traf of boxesWithin(bytes, base)) {
    if (traf.id !== BOX.traf) continue;
    let id: number | undefined;
    let length = 0;
    let parts: Part[] = [];
    for (const part of boxesWithin(bytes.subarray(traf.data - base, traf.end - base), traf.data)) {
      const at = part.data - base;
      const end = part.end - base;
      if (end - at < 8) continue;
      const flags = view.getUint32(at) & 0xffffff;
      if (part.id === BOX.tfhd) {
        // ffmpeg credits what comes before any header, or under the header of a track the
        // movie header gives no defaults for, to the movie fragment's track fragment before,
        // failing to read the file if there is none; here it is passed over.
        if (id !== undefined && defaults.has(id)) yield { id, parts };
        parts = [];
        id = view.getUint32(at + 4);
        // A base data offset and a sample description index come before the default duration.
        const field = at + 8 + (flags & 0x01 ? 8 : 0) + (flags & 0x02 ? 4 : 0);
        length = flags & 0x08 && field + 4 <= end ? view.getUint32(field) : defaults.get(id) ?? 0;
      } else if (part.id === BOX.tfdt) {
        // A version other than 0 gives the time in 8 bytes.
        if (!bytes[at]) parts.push({ time: view.getUint32(at + 4) });
        else if (end - at >= 12) parts.push({ time: Number(view.getBigUint64(at + 4)) });
      } else if (part.id === BOX.trun) {
        const entries = view.getUint32(at + 4);
        // Each sample's duration, size, flags and composition offset, as the flags say.
        const size = 4 * [0x100, 0x200, 0x400, 0x800].filter((field) => flags & field).length;
        const first = at + 8 + (flags & 0x01 ? 4 : 0) + (flags & 0x04 ? 4 : 0);
        const samples = size ? Math.min(entries, Math.max(0, Math.floor((end - first) / size))) : entries;
        let duration = 0;
        if (flags & 0x100) for (let i = 0; i < samples; i++) duration += view.getUint32(first + i * size);
        else duration = samples * length;
        parts.push({ samples, duration });
      }
    }
    if (id !== undefined && defaults.has(id)) yield { id, parts };
  }
}

/** A segment index of one of the movie's tracks, as ffmpeg reads it: the track, and the timing
 * of the first with its ID, to which ffmpeg gives the index; the index's time scale; where its
 * references end in time, in that scale, which ffmpeg takes for the track's; where in the file
 * they start and end; and where each starts, in the file and in time. Null for one ffmpeg
 * makes nothing of. */
async function segmentIndex(source: Blob, sidx: Span, extents: Map<number, Extent>) {
  const bytes = await read(source, sidx.data, sidx.end);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = bytes[0];
  // version and flags, reference ID, time scale, earliest presentation time, first offset
  const fields = version === 1 ? 32 : 24;
  if (bytes.length < fields || version > 1) return null;
  const [id, scale] = [view.getUint32(4), view.getUint32(8)];
  // ffmpeg reads the time scale as a signed integer and fails to read a file whose index has
  // one of 0 or less; here the index is passed over.
  const extent = extents.get(id);
  if (!extent || !scale || scale >= 2 ** 31) return null;
  let end = version === 1 ? Number(view.getBigUint64(12)) : view.getUint32(12);
  const first = sidx.end + (version === 1 ? Number(view.getBigUint64(20)) : view.getUint32(16));
  const count = view.getUint16(fields - 2);
  let reach = first;
  const places: [number, number][] = [];
  for (let i = 0; i < count; i++) {
    const at = fields + 12 * i;
    // ffmpeg doesn't follow a reference to another segment index.
    if (at + 12 > bytes.length || view.getUint32(at) & 0x80000000) return null;
    places.push([reach, end]);
    reach += view.getUint32(at);
    end += view.getUint32(at + 4);
  }
  return count ? { id, extent, scale, end, first, reach, places } : null;
}

/** A movie or media header's time scale as ffmpeg takes it: read as a signed integer, and 1
 * if that is 0 or less. */
function timeScale(scale: number): number {
  return scale > 0 && scale < 2 ** 31 ? scale : 1;
}

/** A movie or media header's time scale and duration, after its version, flags and times
 * (4 or 8 bytes each, as the duration is). ffmpeg takes a duration of all ones as 0. */
function timing(bytes: Uint8Array): { scale: number; duration: number } {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const long = bytes[0] === 1;
  const at = long ? 20 : 12;
  const scale = bytes.length >= at + 4 ? view.getUint32(at) : 0;
  const duration = bytes.length < at + (long ? 12 : 8) ? 0n
    : long ? view.getBigUint64(at + 4) : BigInt(view.getUint32(at + 4));
  return { scale, duration: duration === (long ? 2n ** 64n : 2n ** 32n) - 1n ? 0 : Number(duration) };
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
  const counted = { samples: 0, duration: 0 };
  if (expand && stts) {
    durations = expandRuns(stts, false, Infinity);
    offsets = ctts ? expandRuns(ctts, true, durations.length) : [];
    // Samples without a composition offset show at their decode time.
    const given = offsets.length;
    offsets.length = durations.length;
    offsets.fill(0, given);
    // ffmpeg counts the table's samples only if they last at all.
    const duration = durations.reduce((sum, d) => sum + d, 0);
    if (duration > 0) Object.assign(counted, { samples: durations.length, duration });
  }
  // readMp4 times the track once it has read as far as ffmpeg does.
  return { handler, start, edited, edit, durations, offsets, reordered, dtsShift, counted, duration: 0 };
}

async function readTrak(source: Blob, trak: Span): Promise<Trak> {
  const contents = async (span: Span) => {
    const bytes = await read(source, span.data, span.end);
    return { bytes, view: new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength) };
  };
  const result: Trak = {
    id: -1, handler: "", timescale: 0, duration: 0, edits: null, stts: null, ctts: null,
  };
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
        if (media.id === BOX.mdhd) {
          const { scale, duration } = timing((await contents(media)).bytes);
          [result.timescale, result.duration] = [timeScale(scale), duration];
        }
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
