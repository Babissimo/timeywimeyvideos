/**
 * Reading a video into frames in the browser: Mediabunny demuxes, WebCodecs
 * decodes. Which frames are kept follows timeslice.load_video, which asks
 * ffmpeg for them with `-ss start -t duration -i video -vf fps=rate`.
 */
import {
  ALL_FORMATS, BlobSource, EncodedPacketSink, Input, IsobmffInputFormat, MatroskaInputFormat,
  UnsupportedInputFormatError, VideoSample, VideoSampleSink, type InputTrack, type InputVideoTrack,
} from "mediabunny";
import { avcFrameRate } from "./avc";
import { readMatroska, readMp4, type Mp4Track } from "./container";
import { hevcFrameRate } from "./hevc";
import { gcd, limitDenominator, roundHalfEven } from "./pymath";
import type { Rate } from "./types";

/** The browser couldn't read or decode a video. */
export class VideoError extends Error {
  override name = "VideoError";
}

/** What `probe` finds: the upright size, the frame rate ffmpeg decodes at and the length in seconds. */
export interface ClipInfo { width: number; height: number; fps: Rate; duration: number | null }

/**
 * scale resizes each frame; timeScale keeps that fraction of the frames
 * (0.5 = every other one); start and duration pick a window in seconds; fast
 * resizes by nearest neighbour instead of smoothly.
 */
export interface LoadOptions {
  scale?: number; timeScale?: number; start?: number; duration?: number
  fast?: boolean; signal?: AbortSignal
}

/** What `load` delivers: `frames` frames of width x height, at `fps` (the source rate x timeScale). */
export interface LoadPlan { width: number; height: number; frames: number; fps: Rate }

/** Return the size (turned upright), the frame rate ffmpeg decodes at and the duration (seconds,
 * or null) of a video. */
export async function probe(source: Blob): Promise<ClipInfo> {
  const clip = await open(source);
  try {
    return (await readPackets(clip, false)).info;
  } finally {
    clip.input.dispose();
  }
}

/** How many seconds of a video `length` seconds long (null if unknown) are used when
 * starting at `start` and keeping at most `duration`. */
export function clipSeconds(length: number | null, start?: number, duration?: number): number | null {
  let seconds = length === null ? null : Math.max(length - (start || 0), 0);
  if (duration !== undefined) seconds = seconds === null ? duration : Math.min(seconds, duration);
  return seconds;
}

/** What `load` will deliver with these options, found from the packets' timestamps without
 * decoding, so a volume can be allocated first. */
export async function planLoad(source: Blob, options: LoadOptions = {}): Promise<LoadPlan> {
  options.signal?.throwIfAborted();
  const clip = await open(source);
  try {
    const { width, height, fps, sources } = await plan(clip, options);
    return { width, height, frames: sources.length, fps };
  } finally {
    clip.input.dispose();
  }
}

/**
 * Decode the frames `planLoad` describes and hand each to `put` in order, as an
 * upright ImageBitmap of the planned size that `put` then owns. Waits for `put`
 * when it returns a promise. Rejects with the signal's reason once aborted.
 */
export async function load(
  source: Blob, options: LoadOptions,
  put: (index: number, frame: ImageBitmap) => void | Promise<void>,
  progress?: (done: number, total: number) => void,
): Promise<LoadPlan> {
  const { signal } = options;
  signal?.throwIfAborted();
  const clip = await open(source);
  try {
    const { width, height, fps, sources, resolution } = await plan(clip, options);
    const total = sources.length;
    const resize: ImageBitmapOptions = {
      resizeWidth: width, resizeHeight: height, resizeQuality: options.fast ? "pixelated" : "high",
    };
    const upright = (await clip.track.getRotation()) === 0 && !(await clip.track.getFlip());
    let canvas: OffscreenCanvas | undefined;

    // A frame stored turned is drawn upright at full size first, so it is resized
    // after turning, as ffmpeg does.
    const draw = (sample: VideoSample): Promise<ImageBitmap> => {
      if (upright) return createImageBitmap(sample.toCanvasImageSource(), resize);
      canvas ??= new OffscreenCanvas(sample.displayWidth, sample.displayHeight);
      const context = canvas.getContext("2d")!;
      context.clearRect(0, 0, canvas.width, canvas.height);
      sample.draw(context, 0, 0, canvas.width, canvas.height);
      return createImageBitmap(canvas, resize);
    };
    const bitmap = async (sample: VideoSample): Promise<ImageBitmap> => {
      let opaque: VideoSample | undefined;
      try {
        // ffmpeg's rgb24 ignores alpha, keeping the colour beneath it.
        if (sample.hasAlpha) {
          const frame = sample.toVideoFrame();
          try {
            opaque = new VideoSample(new VideoFrame(frame, { alpha: "discard" }),
                                     { rotation: sample.rotation, flip: sample.flip });
          } finally {
            frame.close();
          }
        }
        return await draw(opaque ?? sample);
      } catch (error) {
        throw new VideoError(`failed to decode ${clip.name}`, { cause: error });
      } finally {
        opaque?.close();
      }
    };
    let done = 0;
    const deliver = async (sample: VideoSample) => {
      const frame = await bitmap(sample);
      if (signal?.aborted) {
        frame.close();
        signal.throwIfAborted();
      }
      await put(done, frame);
      done++;
      progress?.(done, total);
      signal?.throwIfAborted();
    };

    const samples = new VideoSampleSink(clip.track)
      .samples(sources[0]! / resolution, (sources[total - 1]! + 0.5) / resolution);
    // The latest sample decoded, kept open for any planned frame the decoder skipped.
    let held: VideoSample | undefined;
    try {
      while (done < total) {
        let next: IteratorResult<VideoSample, void>;
        try {
          next = await samples.next();
        } catch (error) {
          signal?.throwIfAborted();
          throw new VideoError(`failed to decode ${clip.name}`, { cause: error });
        }
        if (next.done) break;
        const sample = next.value;
        if (signal?.aborted) {
          sample.close();
          signal.throwIfAborted();
        }
        const tick = Math.round(sample.timestamp * resolution);
        try {
          while (done < total && sources[done]! < tick) await deliver(held ?? sample);
          while (done < total && sources[done] === tick) await deliver(sample);
        } finally {
          held?.close();
          held = sample;
        }
      }
      if (done < total && !held) throw new VideoError(`no frames decoded from ${clip.name}`);
      while (done < total) await deliver(held!);
    } finally {
      held?.close();
      await samples.return();
    }
    return { width, height, frames: total, fps };
  } finally {
    clip.input.dispose();
  }
}

interface Clip { input: Input; track: InputVideoTrack; name: string; source: Blob }

async function open(source: Blob): Promise<Clip> {
  const name = source instanceof File ? source.name : "this file";
  const input = new Input({ source: new BlobSource(source), formats: ALL_FORMATS });
  let track: InputVideoTrack | null = null;
  try {
    track = await input.getPrimaryVideoTrack();
  } catch (error) {
    input.dispose();
    const kind = error instanceof UnsupportedInputFormatError ? await container(source) : null;
    if (kind) {
      throw new VideoError(`can't read ${name}: the browser can't open ${kind} files`, { cause: error });
    }
    throw new VideoError(`can't read a video stream from ${name}`, { cause: error });
  }
  if (!track) {
    input.dispose();
    throw new VideoError(`can't read a video stream from ${name}`);
  }
  return { input, track, name, source };
}

// Containers ffmpeg reads and Mediabunny doesn't, by the bytes they start with.
const CONTAINERS: [string, number, string][] = [
  ["AVI", 8, "AVI "], ["FLV", 0, "FLV"], ["MPEG program stream", 0, "\x00\x00\x01\xba"],
  ["ASF", 0, "\x30\x26\xb2\x75\x8e\x66\xcf\x11"], ["RealMedia", 0, ".RMF"],
  ["MXF", 0, "\x06\x0e\x2b\x34"], ["IVF", 0, "DKIF"], ["Y4M", 0, "YUV4MPEG2"], ["GIF", 0, "GIF8"],
];

/** The name of the container `source` is in, if it is one ffmpeg reads and the browser doesn't. */
async function container(source: Blob): Promise<string | null> {
  const head = String.fromCharCode(...new Uint8Array(await source.slice(0, 16).arrayBuffer()));
  return CONTAINERS.find(([, at, magic]) => head.startsWith(magic, at))?.[0] ?? null;
}

/**
 * A clip's video packets as ffmpeg's demuxer sees them, in decode order, the
 * file around them, and what `probe` reports. Times are in ticks of
 * 1/resolution seconds.
 */
interface Packets {
  info: ClipInfo;
  resolution: number;
  isobmff: boolean;
  /** The timestamps Mediabunny gives, which its decoded samples carry. */
  samples: number[];
  /** ffmpeg's presentation and decode timestamps, durations (0 if the container gives none)
   * and key frames. */
  pts: number[]; dts: number[]; lengths: number[]; keys: boolean[];
  /** Decode indices in presentation order. */
  order: number[];
  /** The packets in ffmpeg's index, from `first` to `last`, and which of those it decodes
   * without showing. */
  first: number; last: number; hidden: boolean[];
  /** How far before its target an MP4 seek compares decode timestamps. */
  seekBack: number;
  /** ffprobe's r_frame_rate, or the rate that stands in for a time base taken as one, from
   * which ffmpeg's demuxer makes up a missing packet duration. */
  rFrameRate: Rate;
  /** When the file starts, in microseconds: its streams' earliest first timestamp. */
  fileStart: bigint;
  /** How many streams the file holds, of any kind. */
  streams: number;
}

// ffmpeg judges the frame rate from at most this many packets.
const RATE_PACKETS = 41;

/** Read a clip's packets: every one, or with `all` false only enough for `probe`. */
async function readPackets(clip: Clip, all: boolean, signal?: AbortSignal): Promise<Packets> {
  const { input, track, name, source } = clip;
  try {
    const format = await input.getFormat();
    const isobmff = format instanceof IsobmffInputFormat;
    const resolution = await track.getTimeResolution();
    // An MP4's frame rate depends on every sample's duration, a Matroska file's on its header.
    const limit = all || isobmff ? Infinity : RATE_PACKETS;
    const samples: number[] = [];
    const seconds: number[] = [];
    const keys: boolean[] = [];
    for await (const packet of new EncodedPacketSink(track).packets(undefined, undefined, { metadataOnly: true })) {
      if (samples.length === limit) break;
      samples.push(Math.round(packet.timestamp * resolution));
      seconds.push(packet.duration);
      keys.push(packet.type === "key");
      signal?.throwIfAborted();
    }

    let pts = samples;
    const order = presentationOrder(pts);
    let dts = decodeTimestamps(pts, order);
    let lengths: number[];
    const tracks = await input.getTracks();
    // Other containers count only the video and audio tracks Mediabunny lists, missing
    // for instance an MPEG-TS stream of subtitles or data, and its start.
    let streams = tracks.length;
    let textStart: number | null = null;
    let starts: Map<number, Mp4Track> | undefined;
    let index: MovIndex | undefined;
    let seekBack = 0;
    // The rate a Matroska track's default duration gives.
    let defaultRate: Rate | null = null;
    // ffmpeg's avg_frame_rate: an MP4's over the samples its header and the fragments read
    // with it hold, a Matroska track's its default duration's. Other containers' is not modelled.
    let average: Rate | null = null;
    if (isobmff) {
      const header = await readMp4(source, track.id);
      const video = header.tracks.get(track.id);
      ({ streams, textStart, tracks: starts } = header);
      const counted = video?.counted;
      if (counted && counted.samples > 0 && counted.duration > 0) {
        average = avReduce(BigInt(Math.round(resolution)) * BigInt(counted.samples), BigInt(counted.duration),
                           INT_MAX);
      }
      // Packets last as their samples do. ffmpeg parses VP8 and VP9 in full, which drops the
      // durations; that is not modelled.
      if (video && video.durations.length === pts.length) {
        index = movIndex(video, keys);
        lengths = video.durations;
      } else {
        // A fragmented file's tables are in its fragments, so the decode timestamps stay
        // rebuilt, and a sample lasts until the next one's.
        lengths = dts.map((t, i) => i + 1 < dts.length ? dts[i + 1]! - t
                                                       : Math.round(seconds[i]! * resolution));
        // Under an edit list, which starts the presentation at the frame shown first,
        // ffmpeg's seek looks back by the reorder delay; without one, not at all.
        if (video?.edited && pts.length > 0) seekBack = pts[order[0]!]! - dts[0]!;
      }
    } else if (format instanceof MatroskaInputFormat) {
      const header = await readMatroska(source);
      ({ streams, textStart } = header);
      const defaultDuration = header.defaultDurations.get(track.id) ?? null;
      if (defaultDuration) average = defaultRate = avReduce(1_000_000_000n, BigInt(defaultDuration), 30000n);
      // Blocks take the track's default duration, cut to whole ticks. In ffmpeg a block's
      // own BlockDuration, which a remux of variable-rate video writes, overrides it; Mediabunny
      // doesn't report that, so here the default stands.
      const length = defaultDuration === null ? 0 : Math.floor(defaultDuration * resolution / 1e9);
      lengths = pts.map(() => length);
    } else {
      lengths = seconds.map((s) => Math.round(s * resolution));
    }
    const first = index?.first ?? 0;
    const last = index?.last ?? pts.length - 1;
    const hidden = index?.hidden ?? pts.map(() => false);
    if (index) ({ pts, dts, seekBack } = index);
    // ffmpeg probes the rate from the packets its index holds.
    const rateDts = index ? dts.slice(first, last + 1)
      : isobmff || pts.length <= RATE_PACKETS ? dts : decodeTimestamps(pts.slice(0, RATE_PACKETS));
    const codec = await track.getCodec();
    const found = frameRate(rateDts, lengths.slice(first, last + 1), resolution, index?.table ?? null,
                            defaultRate, codec);
    // Finding none, ffmpeg takes the rate the codec declares (doubled for H.264, which counts
    // fields) where the time base can tell its frames apart, and failing that the time base.
    const declared = found ? null : await declaredRate(track, codec);
    const fields = declared && codec === "avc" ? multiply(declared, { num: 2, den: 1 }) : declared;
    const base = Math.round(resolution);
    const byTimeBase = !found && !(fields && fields.num <= base * fields.den);
    let rFrameRate = found ?? (byTimeBase ? { num: base, den: 1 } : fields!);
    // ffmpeg paces decoded frames to av_guess_frame_rate: r_frame_rate, or the average where
    // that is under 70 fps and r_frame_rate over 210.
    const toAverage = average !== null && rFrameRate.num > 210 * rFrameRate.den
      && average.num < 70 * average.den;
    // A time base over 210 Hz is no frame rate. Where ffmpeg, and so load_video, would pace to
    // one for want of anything else, the mean rate of the decode timestamps read stands in,
    // for the packet durations ffmpeg makes up too.
    if (byTimeBase && !toAverage && base > 210) {
      const span = (rateDts[rateDts.length - 1] ?? 0) - (rateDts[0] ?? 0);
      rFrameRate = span > 0 ? limitDenominator((rateDts.length - 1) * resolution / span, 1001)
                            : { num: 25, den: 1 };
    }
    const fps = toAverage ? average! : rFrameRate;
    const fileStart = await startOf(tracks, isobmff, textStart, starts);

    const rotation = await track.getRotation();
    const [codedWidth, codedHeight] = [await track.getCodedWidth(), await track.getCodedHeight()];
    const turned = rotation % 180 !== 0;
    let duration = await input.getDurationFromMetadata();
    const stated = format instanceof MatroskaInputFormat && duration !== null;
    duration ??= await input.computeDuration();
    // ffmpeg takes a Matroska header's duration as it stands, and otherwise measures
    // from the file's start to its end.
    if (!stated) duration -= Number(fileStart) / 1e6;
    const info: ClipInfo = {
      width: turned ? codedHeight : codedWidth, height: turned ? codedWidth : codedHeight, fps,
      duration: Number.isFinite(duration) && duration > 0 ? duration : null,
    };
    return {
      info, resolution, isobmff, samples, pts, dts, lengths, keys, order, first, last, hidden, seekBack,
      rFrameRate, fileStart, streams,
    };
  } catch (error) {
    if (error instanceof VideoError || signal?.aborted) throw error;
    throw new VideoError(`can't read a video stream from ${name}`, { cause: error });
  }
}

/**
 * The video's frames as ffmpeg's demuxer and decoder see them, and the file
 * around them. Times are in ticks of 1/resolution seconds.
 */
interface Timeline {
  info: ClipInfo;
  resolution: number;
  isobmff: boolean;
  /** How many streams the file holds, of any kind. */
  streams: number;
  /** When the file starts, in microseconds: its streams' earliest first timestamp. */
  fileStart: bigint;
  /** How far before its target an MP4 seek compares decode timestamps. */
  seekBack: number;
  /** ffprobe's r_frame_rate, or the rate that stands in for a time base taken as one, from
   * which ffmpeg's demuxer makes up a missing packet duration. */
  rFrameRate: Rate;
  /** In presentation order: when ffmpeg shows each frame, the timestamp its decoded sample
   * carries here, its packet's duration (0 if the container gives none) and its place in
   * decode order. */
  starts: number[]; samples: number[]; lengths: number[]; decodeIndex: number[];
  /** In decode order: decode timestamps, which packets are key frames, and which ffmpeg
   * decodes without showing. Its index holds those from `first` to `last`. */
  dts: number[]; keys: boolean[]; hidden: boolean[]; first: number; last: number;
}

async function readTimeline(clip: Clip, signal?: AbortSignal): Promise<Timeline> {
  const { samples, pts, lengths, order, ...rest } = await readPackets(clip, true, signal);
  return {
    ...rest,
    starts: order.map((i) => pts[i]!), samples: order.map((i) => samples[i]!),
    lengths: order.map((i) => lengths[i]!), decodeIndex: order,
  };
}

/** When a file starts, in microseconds: the earliest of its streams' first timestamps,
 * as ffmpeg takes them, given the earliest start of its subtitle and data streams and,
 * for an MP4, when its header starts each track. */
async function startOf(tracks: InputTrack[], isobmff: boolean, textStart: number | null,
                       header?: Map<number, Mp4Track>): Promise<bigint> {
  let start: bigint | null = null;
  for (const track of tracks) {
    const stated = header?.get(track.id)?.start;
    if (stated !== undefined && stated !== null) {
      if (start === null || BigInt(stated) < start) start = BigInt(stated);
      continue;
    }
    let first: number;
    let res: number;
    try {
      const packet = await new EncodedPacketSink(track).getFirstPacket({ metadataOnly: true });
      // ffmpeg ignores a stream without packets, having no start time for it.
      if (!packet) continue;
      res = await track.getTimeResolution();
      first = Math.round(packet.timestamp * res);
    } catch {
      // A stream Mediabunny can't read has no start time either.
      continue;
    }
    // ffmpeg starts an MP4 stream at its edit list, so AAC priming before 0 doesn't count.
    // A Matroska Opus track's CodecDelay moves its packets earlier in ffmpeg but not its
    // start, to which ffmpeg adds the samples it will skip.
    if (isobmff) first = Math.max(first, 0);
    const us = rescale(BigInt(first), MICRO, BigInt(Math.round(res)));
    if (start === null || us < start) start = us;
  }
  // ffmpeg takes a subtitle or data stream's start only when it is less than a second
  // before the others' (or they have none). Mediabunny lists no such streams.
  if (textStart !== null) {
    const text = BigInt(textStart);
    if (start === null || (start > text && start - text < MICRO)) start = text;
  }
  return start ?? 0n;
}

/** Decode indices sorted by presentation timestamp. */
function presentationOrder(pts: number[]): number[] {
  return pts.map((_, i) => i).sort((a, b) => pts[a]! - pts[b]! || a - b);
}

/** Decode timestamps rebuilt for packets with these presentation timestamps, where the
 * container gives none: in presentation order, delayed by the deepest reordering. An
 * encoder that declares a deeper delay than it uses starts the real ones earlier. */
function decodeTimestamps(pts: number[], order = presentationOrder(pts)): number[] {
  const sorted = order.map((i) => pts[i]!);
  let delay = 0;
  order.forEach((i, rank) => { delay = Math.max(delay, i - rank); });
  return pts.map((_, i) => i >= delay ? sorted[i - delay]! : sorted[i]! - (sorted[delay]! - sorted[0]!));
}

/** ffmpeg's index of an MP4 track's packets, in ticks of its time scale. */
interface MovIndex {
  /** In decode order: when each packet is shown, and its decode timestamp in the index. */
  pts: number[]; dts: number[];
  /** The packets the index holds, from `first` to `last`, and which of those are decoded
   * without being shown. */
  first: number; last: number; hidden: boolean[];
  /** How far before its target a seek compares decode timestamps. */
  seekBack: number;
  /** Every sample's decode timestamp as the header is read, before the edit list applies. */
  table: number[];
}

/**
 * The index ffmpeg's mov demuxer makes of a track's samples (mov_build_index, then
 * mov_fix_index for an edit list). Under an edit, decoding starts at the last key frame
 * shown at or before the edit's media time; frames shown before that time, or from the
 * edit's end on, are decoded but not shown; and the index ends at the first key frame
 * that lasts to the edit's end (the second, with composition offsets). The first frame
 * shown in decode order is decoded at the empty edits' end, the others keeping their
 * steps from it, and all then move earlier so that the earliest shown starts there.
 * Edits after the first that shows media are not modelled.
 */
function movIndex(track: Mp4Track, keys: boolean[]): MovIndex {
  const { durations, offsets, edit, reordered, dtsShift } = track;
  const n = durations.length;
  const table: number[] = [];
  for (let i = 0, t = -dtsShift; i < n; t += durations[i]!, i++) table.push(t);
  const hidden = table.map(() => false);
  const show = (dts: number[]) => dts.map((t, i) => t + dtsShift + offsets[i]!);
  if (!edit) return { pts: show(table), dts: table, first: 0, last: n - 1, hidden, seekBack: dtsShift, table };

  const { delay, time, duration } = edit;
  const end = time + duration;
  // The key frame to decode from: the last at or before the media time by decode
  // timestamp, then, with composition offsets, back to one shown by then. Failing a key
  // frame, ffmpeg tries any frame, and failing that the first.
  const target = time - dtsShift;
  const find = (any: boolean) => {
    let found = searchIndex(table, keys, null, target, any);
    for (let i = found; i > 0 && table[i] === table[i - 1]; i--) if (any || keys[i - 1]) found = i - 1;
    if (reordered) while (found >= 0 && !(keys[found] && table[found]! + offsets[found]! <= target)) found--;
    return found;
  };
  let first = find(false);
  if (first < 0) first = find(true);
  if (first < 0) first = 0;

  let kept: number | undefined;
  let earliest = -1;
  let last = n - 1;
  let pastEnd = false;
  for (let i = first; i < n; i++) {
    const shown = table[i]! + dtsShift + offsets[i]!;
    const length = i + 1 < n ? table[i + 1]! - table[i]! : duration;
    if (shown < time || shown >= end) hidden[i] = true;
    else {
      kept ??= i;
      // ffmpeg takes the earliest presentation time, except that a negative one so far
      // gives way to the next.
      const at = delay + table[i]! - table[kept]! + offsets[i]!;
      earliest = earliest < 0 ? at : Math.min(earliest, at);
    }
    if (shown + length >= end && keys[i]) {
      if (reordered && !pastEnd) {
        pastEnd = true;
        continue;
      }
      last = i;
      break;
    }
  }
  const corrected = earliest - delay;
  const shift = delay - dtsShift - table[kept ?? first]! - Math.max(corrected, 0);
  const dts = table.map((t) => t + shift);
  return { pts: show(dts), dts, first, last, hidden, seekBack: corrected + dtsShift, table };
}

/**
 * ffmpeg's ff_index_search_timestamp looking backward: the last packet whose timestamp is
 * at or before `wanted`, then back to a key frame unless `any`; -1 if there is none.
 * Packets decoded without being shown are stepped over as it narrows the search.
 */
function searchIndex(timestamps: number[], keys: boolean[], hidden: boolean[] | null, wanted: number,
                     any = false): number {
  const n = timestamps.length;
  let a = -1;
  let b = n;
  if (n && timestamps[n - 1]! < wanted) a = n - 1;
  while (b - a > 1) {
    let m = (a + b) >> 1;
    while (hidden?.[m] && m < b && m < n - 1) {
      m++;
      if (m === b && timestamps[m]! >= wanted) {
        m = b - 1;
        break;
      }
    }
    if (timestamps[m]! >= wanted) b = m;
    if (timestamps[m]! <= wanted) a = m;
  }
  let m = a;
  if (!any) while (m >= 0 && !keys[m]) m--;
  return m;
}

interface Plan {
  width: number; height: number; fps: Rate; resolution: number;
  /** For each frame delivered, the timestamp (in ticks) its source frame's decoded sample carries. */
  sources: number[];
}

async function plan(clip: Clip, options: LoadOptions): Promise<Plan> {
  const { scale = 1, timeScale = 1, start, duration, signal } = options;
  if (!Number.isFinite(scale)) throw new RangeError(`scale must be a number, not ${scale}`);
  const share = limitDenominator(timeScale, 1000);
  if (share.num <= 0) throw new RangeError(`a timeScale of ${timeScale} keeps no frames`);
  const codec = await clip.track.getCodec();
  if (!codec || !(await clip.track.canDecode())) {
    const kind = codec ?? await clip.track.getInternalCodecId() ?? "unknown";
    throw new VideoError(`can't decode ${clip.name}: the browser has no decoder for its ${kind} video`);
  }
  const timeline = await readTimeline(clip, signal);
  const kept = multiply(timeline.info.fps, share);
  const chosen = chooseFrames(timeline, kept, start, duration);
  if (chosen.length === 0) throw new VideoError(`no frames decoded from ${clip.name}`);
  const { width, height } = timeline.info;
  return {
    width: Math.max(1, roundHalfEven(width * scale)),
    height: Math.max(1, roundHalfEven(height * scale)),
    fps: kept, resolution: timeline.resolution,
    sources: chosen.map((i) => timeline.samples[i]!),
  };
}

/** A frame on its way to ffmpeg's output: which source frame, when (in output
 * frames, fractional) and for how long. */
interface Timed { frame: number; at: number; length: number }

/**
 * Which frames (indices into `timeline.starts`, in output order, some repeated)
 * ffmpeg returns for `-ss start -t duration -i video -vf fps=rate -f rawvideo`,
 * where the fps filter is only there when `rate` differs from the source's.
 *
 * Timestamps move so that `start` (whole microseconds, as ffmpeg parses it)
 * after the file's start is 0. Seeking an MP4 finds the last key frame in the
 * index whose decode timestamp is at or before `start` less `seekBack`; frames
 * decoded before it are lost, even ones shown after `start`. A trim keeps
 * frames from 0 on, for `duration` after the first one kept (a duration of 0 sets
 * no limit). The fps filter rounds each frame's time to the nearest tick of the
 * new rate, and fills every tick from the first frame's up to the end of the
 * last with the latest frame at or before it. Last, `pace` fits the frames to a
 * constant rate.
 */
function chooseFrames(timeline: Timeline, rate: Rate, start?: number, duration?: number): number[] {
  const { starts, decodeIndex, dts, keys, hidden, first: indexed, last: lastIndexed, resolution } = timeline;
  const res = BigInt(Math.round(resolution));
  const startUs = start === undefined ? null : BigInt(microseconds(start));
  const offset = rescale(-(timeline.fileStart + (startUs ?? 0n)), res, MICRO);

  let firstDecoded = indexed;
  if (startUs !== null && timeline.isobmff) {
    const target = rescale(timeline.fileStart + startUs, res, MICRO) - BigInt(timeline.seekBack);
    const to = lastIndexed + 1;
    // A seek that finds nothing leaves ffmpeg reading from the index's start.
    const found = searchIndex(dts.slice(indexed, to), keys.slice(indexed, to), hidden.slice(indexed, to),
                              Number(target));
    if (found >= 0) firstDecoded += found;
  }
  const decoded = starts.map((_, k) => k).filter((k) => {
    const i = decodeIndex[k]!;
    return i >= firstDecoded && i <= lastIndexed && !hidden[i];
  });

  // A packet without a duration lasts one frame at r_frame_rate, rounded down, as ffmpeg's
  // demuxer makes one up.
  const { info: { fps }, rFrameRate } = timeline;
  const made = Math.floor(resolution * rFrameRate.den / rFrameRate.num);
  const lengths = new Map<number, number>();
  for (const k of decoded) lengths.set(k, timeline.lengths[k]! > 0 ? timeline.lengths[k]! : Math.max(made, 1));

  const us = duration === undefined ? 0 : microseconds(duration);
  const limit = us === 0 ? null : rescale(BigInt(us), res, MICRO);
  const window: number[] = [];
  let first: bigint | undefined;
  let end: bigint | undefined;
  for (const k of decoded) {
    const pts = BigInt(starts[k]!) + offset;
    if (pts < 0n) continue;
    first ??= pts;
    if (limit !== null && pts - first >= limit) {
      end = pts;
      break;
    }
    window.push(k);
  }
  if (window.length === 0) return [];
  const last = window[window.length - 1]!;
  if (end === undefined) {
    // The last frame ends as ffmpeg's decoder estimates: after its duration, unless that is
    // missing, or a single tick after a step of more than two, which it takes as made up;
    // then after the step from the frame before, or one frame at the rate decoded at, whereas
    // ffmpeg takes the codec's declared rate, else the stream's average.
    const before = decoded[decoded.indexOf(last) - 1];
    const step = before === undefined ? 0 : starts[last]! - starts[before]!;
    const given = timeline.lengths[last]! > 0 ? timeline.lengths[last]! : made;
    const period = Math.max(1, Math.round(resolution * fps.den / fps.num));
    const length = given > 0 && !(given === 1 && step > 2) ? given : step > 0 ? step : period;
    end = BigInt(starts[last]! + length) + offset;
  }

  const timed: Timed[] = [];
  if (rate.num === fps.num && rate.den === fps.den) {
    // Without the fps filter ffmpeg paces to av_guess_frame_rate. Its switch to an H.264
    // stream's declared rate, which timeslice.probe can't see, is not modelled.
    // In output frames to 1/2^bits of a frame (2^bits × fps.num within 2^29, bits at
    // most 16), then 2^-17 further from 0 unless whole, as ffmpeg times them.
    const bits = Math.min(Math.max(29 - Math.floor(Math.log2(fps.num)), 0), 16);
    for (const k of window) {
      let at = Number(rescale(BigInt(starts[k]!) + offset, BigInt(fps.num) << BigInt(bits),
                              res * BigInt(fps.den))) / 2 ** bits;
      if (at !== Math.round(at)) at += Math.sign(at) / 2 ** 17;
      timed.push({ frame: k, at, length: lengths.get(k)! * (1 / resolution) / (fps.den / fps.num) });
    }
  } else {
    const tick = (pts: bigint) => rescale(pts, BigInt(rate.num), res * BigInt(rate.den));
    const ticks = window.map((k) => tick(BigInt(starts[k]!) + offset));
    const stop = tick(end);
    for (let n = ticks[0]!, q = 0; n < stop; n++) {
      while (q + 1 < ticks.length && ticks[q + 1]! <= n) q++;
      timed.push({ frame: window[q]!, at: Number(n), length: 1 });
    }
  }
  return pace(timed, timeline.streams === 1);
}

/**
 * Fit frames to a constant rate as ffmpeg does before writing raw video. A frame
 * that ends more than 1.1 frames past the next output slot fills the slots up to
 * its end, the first of them with the frame before it when it also starts more
 * than 1.1 frames late; one that ends more than 1.1 frames before the next slot
 * is dropped. With a single stream in the file the first frame may start late
 * without repeats.
 */
function pace(frames: Timed[], singleStream: boolean): number[] {
  const out: number[] = [];
  const history = [0, 0, 0];
  let next = 0;
  let previous: number | undefined;
  for (const { frame, at, length: given } of frames) {
    let length = given;
    let delta0 = at - next;
    let delta = delta0 + length;
    let copies = 1;
    let repeats = 0;
    if (delta0 < 0 && delta > 0) {
      length += delta0;
      delta0 = 0;
    }
    if (singleStream && out.length === 0 && delta0 >= 0.5) {
      delta = length;
      delta0 = 0;
      next = roundHalfEven(at);
    }
    if (delta < -1.1) copies = 0;
    else if (delta > 1.1) {
      copies = roundHalfEven(Math.fround(delta));
      if (delta0 > 1.1) repeats = roundHalfEven(Math.fround(delta0 - 0.6));
    }
    history.unshift(repeats);
    history.length = 3;
    for (let i = 0; i < copies; i++) {
      out.push(i < repeats && previous !== undefined ? previous : frame);
      next++;
    }
    previous = frame;
  }
  // At the end, the last frame is repeated as often as the median of the last three repeats.
  const [a, b, c] = history as [number, number, number];
  const tail = Math.max(Math.min(a, b), Math.min(Math.max(a, b), c));
  for (let i = 0; i < tail && previous !== undefined; i++) out.push(previous);
  return out;
}

/**
 * The frame rate ffprobe calls r_frame_rate, where ffmpeg finds one in the container or the
 * timestamps; null where it doesn't. An MP4 whose sample table (`table`, each sample's
 * decode timestamp) has every sample in the index but the last lasting as long as the first
 * gives it exactly; so does the rate a Matroska track's default duration gives (`defaultRate`),
 * between 5 and 1000 fps; otherwise, for H.264 and HEVC and for time bases finer than
 * 1/100 s or coarser than 1/5 s, it is estimated from decode timestamps: as ffmpeg reads an
 * MP4's header, from the first 99 samples in its table (taken only for such a time base),
 * then from the packets read while probing (`dts` and `lengths`), the first 20 frame steps of
 * them (40 for a time base coarser than 0.5 ms). The estimate is the greatest common step if
 * every step after the third shares one, else the standard rate whose ticks the timestamps
 * fall on most evenly.
 */
function frameRate(dts: number[], lengths: number[], resolution: number, table: number[] | null,
                   defaultRate: Rate | null, codec: string | null): Rate | null {
  if (table && lengths[0]! > 0 && lengths.slice(1, -1).every((d) => d === lengths[0])) {
    return avReduce(BigInt(Math.round(resolution)), BigInt(lengths[0]!), INT_MAX);
  }
  if (defaultRate && defaultRate.num < defaultRate.den * 1000 && defaultRate.num > defaultRate.den * 5) {
    return defaultRate;
  }
  const unreliableBase = resolution >= 101 || resolution < 5;
  if (!unreliableBase && codec !== "avc" && codec !== "hevc") return null;
  // Reading the header, ffmpeg knows no codec yet, so judges by the time base alone. It
  // carries the common step found there into the probe.
  const header = table ? standardRate(table.slice(0, 99), resolution) : { rate: null, common: 0 };
  return (unreliableBase ? header.rate : null) ?? standardRate(dts, resolution, lengths, header.common).rate;
}

/**
 * The frame rate an H.264 or HEVC stream declares in the parameter sets of its decoder
 * configuration, reduced as ffmpeg's decoder takes it; null if it declares none. Parameter
 * sets carried only in the packets, and other codecs' declared rates, are not read: VP8 and
 * VP9 have none, and AV1's timing info is seldom written.
 */
async function declaredRate(track: InputVideoTrack, codec: string | null): Promise<Rate | null> {
  if (codec !== "avc" && codec !== "hevc") return null;
  let rate: Rate | null;
  try {
    const description = (await track.getDecoderConfig())?.description;
    if (!description) return null;
    const record = ArrayBuffer.isView(description)
      ? new Uint8Array(description.buffer, description.byteOffset, description.byteLength)
      : new Uint8Array(description);
    rate = codec === "avc" ? avcFrameRate(record) : hevcFrameRate(record);
  } catch {
    return null;  // parameter sets that can't be read declare nothing
  }
  if (!rate) return null;
  // ffmpeg reduces the frame's duration to terms within 2^30.
  const period = avReduce(BigInt(rate.den), BigInt(rate.num), 1n << 30n);
  return { num: period.den, den: period.num };
}

// ffmpeg's standard frame rates, as multiples of 1/12012 fps: twelfths up to
// 30, whole rates up to 60, a few higher, and the NTSC family.
const STANDARD_RATES = [
  ...Array.from({ length: 30 * 12 }, (_, i) => (i + 1) * 1001),
  ...Array.from({ length: 30 }, (_, i) => (i + 31) * 1001 * 12),
  ...[80, 120, 240].map((r) => r * 1001 * 12),
  ...[24, 30, 60, 12, 15, 48].map((r) => r * 1000 * 12),
];

/**
 * ffmpeg's ff_rfps_calculate over these decode timestamps, and the common step it found,
 * starting from `common`. Given the packets' `lengths`, over as many as ffmpeg reads
 * while probing; without, over all of them, as the mov demuxer passes its header's.
 */
function standardRate(dts: number[], resolution: number, lengths?: number[],
                      common = 0): { rate: Rate | null; common: number } {
  const want = !lengths ? Infinity : 1 / resolution > 0.0005 ? 40 : 20;
  const n = STANDARD_RATES.length;
  // For each rate, and for ticks counted from 0 or from half a tick: the sums of
  // each timestamp's distance from its nearest tick, and of its square.
  const sums = [new Float64Array(n), new Float64Array(n)];
  const squares = [new Float64Array(n), new Float64Array(n)];
  let count = 0;
  let stepSum = 0;
  let probed = 0;
  let last: number | undefined;
  for (let p = 0; p < dts.length && count < want; p++) {
    if (lengths && p >= 2) {
      if (probed / resolution >= 5) break;  // ffmpeg probes at most 5 s
      probed += lengths[p] ?? 0;
    }
    const ts = dts[p]!;
    if (last !== undefined && ts > last) {
      const time = ts * (1 / resolution);
      for (let i = 0; i < n; i++) {
        if (squares[0]![i]! >= 1e10) continue;
        const scaled = time * STANDARD_RATES[i]! / 12012;
        for (let j = 0; j < 2; j++) {
          const error = scaled - roundHalfEven(scaled + j * 0.5) + j * 0.5;
          sums[j]![i]! += error;
          squares[j]![i]! += error * error;
        }
      }
      count++;
      stepSum += ts - last;
      if (count % 10 === 0) {
        for (let i = 0; i < n; i++) {
          if (squares[0]![i]! >= 1e10) continue;
          const spread = [0, 1].map((j) => squares[j]![i]! / count - (sums[j]![i]! / count) ** 2);
          if (spread[0]! > 0.04 && spread[1]! > 0.04) squares[0]![i] = squares[1]![i] = 2e10;
        }
      }
      if (count > 3) common = gcd(common, ts - last);
    }
    last = ts;
  }
  if (count > 15 && common > Math.max(1, Math.floor(resolution / 500))) {
    return { rate: avReduce(BigInt(Math.round(resolution)), BigInt(common), INT_MAX), common };
  }
  if (count < 2) return { rate: null, common };
  let best = 0.01;
  let found = 0;
  for (let i = 0; i < n; i++) {
    const rate = STANDARD_RATES[i]!;
    if (probed && probed * (1 / resolution) < (1001 * 11.5) / rate) continue;
    if (!probed && rate < 1001 * 12) continue;
    if ((1 / resolution) * stepSum / count < (1001 * 12.0 * 0.8) / rate) continue;
    for (let j = 0; j < 2; j++) {
      const mean = sums[j]![i]! / count;
      const spread = squares[j]![i]! / count - mean * mean;
      if (spread < best && best > 1e-9) {
        best = spread;
        found = rate;
      }
    }
  }
  const rate = found && found / 12012 < 1.01 * resolution ? avReduce(BigInt(found), 12012n, INT_MAX) : null;
  return { rate, common };
}

const MICRO = 1_000_000n;
const INT_MAX = 2_147_483_647n;

/** a × b / c rounded to the nearest integer, halves away from zero, as ffmpeg's
 * av_rescale_rnd with AV_ROUND_NEAR_INF. c is positive. */
function rescale(a: bigint, b: bigint, c: bigint): bigint {
  const n = a * b;
  return n < 0n ? -((-n + c / 2n) / c) : (n + c / 2n) / c;
}

/** num/den as ffmpeg's av_reduce makes it: the nearest fraction whose terms are at most max. */
function avReduce(num: bigint, den: bigint, max: bigint): Rate {
  const divisor = bigGcd(num, den) || 1n;
  num /= divisor;
  den /= divisor;
  let [n0, d0, n1, d1] = [0n, 1n, 1n, 0n];
  if (num <= max && den <= max) return { num: Number(num), den: Number(den) };
  while (den) {
    let x = num / den;
    const nextDen = num - den * x;
    const n2 = x * n1 + n0;
    const d2 = x * d1 + d0;
    if (n2 > max || d2 > max) {
      if (n1) x = (max - n0) / n1;
      if (d1) x = bigMin(x, (max - d0) / d1);
      if (den * (2n * x * d1 + d0) > num * d1) [n1, d1] = [x * n1 + n0, x * d1 + d0];
      break;
    }
    [n0, d0, n1, d1] = [n1, d1, n2, d2];
    [num, den] = [den, nextDen];
  }
  return { num: Number(n1), den: Number(d1) };
}

function bigGcd(a: bigint, b: bigint): bigint {
  while (b) [a, b] = [b, a % b];
  return a < 0n ? -a : a;
}

function bigMin(a: bigint, b: bigint): bigint {
  return a < b ? a : b;
}

/** Seconds as whole microseconds, as ffmpeg parses the decimal Python passes it:
 * digits past the sixth decimal place are dropped. */
function microseconds(seconds: number): number {
  const parts = /^(-?)(\d+)(?:\.(\d*))?$/.exec(String(seconds));
  if (!parts) return Math.trunc(seconds * 1e6);
  const us = Number(parts[2]) * 1e6 + Number((parts[3] ?? "").slice(0, 6).padEnd(6, "0"));
  return parts[1] ? -us : us;
}

function multiply(a: Rate, b: Rate): Rate {
  const num = a.num * b.num;
  const den = a.den * b.den;
  const divisor = gcd(num, den) || 1;
  return { num: num / divisor, den: den / divisor };
}
