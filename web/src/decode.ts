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
import { isText, readMatroska, readMp4, type Mp4Track } from "./container";
import { hevcFrameRate } from "./hevc";
import { gcd, limitDenominator, roundHalfEven } from "./pymath";
import type { Rate } from "./types";

/** The browser couldn't read or decode a video. */
export class VideoError extends Error {
  override name = "VideoError";
}

/** What `probe` finds: the upright size, the frame rate as load_video takes it and the length in
 * seconds. */
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

/** Return the size (turned upright), the frame rate as load_video takes it and the duration
 * (seconds, or null) of a video. */
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
  /** ffmpeg's presentation and decode timestamps, durations (0 where its demuxer reads none)
   * and key frames. */
  pts: number[]; dts: number[]; lengths: number[]; keys: boolean[];
  /** Decode indices in presentation order. */
  order: number[];
  /** The packets in ffmpeg's index, from `first` to `last`, and which of those it decodes
   * without showing. */
  first: number; last: number; hidden: boolean[];
  /** How far before its target an MP4 seek compares decode timestamps. */
  seekBack: number;
  /** The duration ffmpeg's demuxer makes up for a packet without one: read while probing,
   * before it knows r_frame_rate, and after. */
  made: { probing: number; after: number };
  /** How many packets, from `first`, ffmpeg reads while probing before it knows r_frame_rate. */
  probePackets: number;
  /** The rate ffmpeg paces decoded frames to without the fps filter. */
  paced: Rate;
  /** When the file starts, in microseconds: its streams' earliest first timestamp. */
  fileStart: bigint;
  /** How many streams the file holds, of any kind. */
  streams: number;
}

// ffmpeg judges the frame rate from at most this many packets.
const RATE_PACKETS = 41;
// The mean rate that can stand in for ffmpeg's is over the first RATE_PACKETS frames shown.
// They are among the first RATE_PACKETS + REORDER packets, as H.264 and HEVC hold back at most
// 16 frames to show in order.
const REORDER = 16;

/** Read a clip's packets: every one, or with `all` false only enough for `probe`. */
async function readPackets(clip: Clip, all: boolean, signal?: AbortSignal): Promise<Packets> {
  const { input, track, name, source } = clip;
  try {
    const format = await input.getFormat();
    const isobmff = format instanceof IsobmffInputFormat;
    const resolution = await track.getTimeResolution();
    // An MP4's frame rate depends on every sample's duration, a Matroska file's on its header
    // and first frames.
    const limit = all || isobmff ? Infinity : RATE_PACKETS + REORDER;
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
    let movie: Map<number, Mp4Track> | undefined;
    let index: MovIndex | undefined;
    let seekBack = 0;
    const codec = await track.getCodec();
    // The r_frame_rate a container's header gives, which ffmpeg knows before reading packets.
    let headerRate: Rate | null = null;
    // ffmpeg's avg_frame_rate as the header gives it: an MP4's over the samples its header and
    // the fragments read with it hold, a Matroska track's its default duration's.
    let average: Rate | null = null;
    if (isobmff) {
      const header = await readMp4(source, track.id);
      const video = header.tracks.get(track.id);
      ({ streams, textStart, tracks: movie } = header);
      const counted = video?.counted;
      if (counted && counted.samples > 0 && counted.duration > 0) {
        average = avReduce(BigInt(Math.round(resolution)) * BigInt(counted.samples), BigInt(counted.duration),
                           INT_MAX);
      }
      // Packets last as their samples do.
      if (video && video.durations.length === pts.length) {
        index = movIndex(video, keys);
        lengths = video.durations;
        // A sample table in which every sample in the index but the last lasts as long as the
        // first gives r_frame_rate exactly.
        const steps = lengths.slice(index.first, index.last + 1);
        if (steps[0]! > 0 && steps.slice(1, -1).every((d) => d === steps[0])) {
          headerRate = avReduce(BigInt(Math.round(resolution)), BigInt(steps[0]!), INT_MAX);
        }
      } else {
        // A fragmented file's tables are in its fragments, so the decode timestamps stay
        // rebuilt, and a sample lasts until the next one's.
        lengths = dts.map((t, i) => i + 1 < dts.length ? dts[i + 1]! - t
                                                       : Math.round(seconds[i]! * resolution));
        // Under an edit list, which starts the presentation at the frame shown first,
        // ffmpeg's seek looks back by the reorder delay; without one, not at all.
        if (video?.edited && pts.length > 0) seekBack = pts[order[0]!]! - dts[0]!;
      }
      // ffmpeg's mov demuxer has VP8 and VP9 parsed in full, which drops the packets' durations.
      if (codec === "vp8" || codec === "vp9") lengths = lengths.map(() => 0);
    } else if (format instanceof MatroskaInputFormat) {
      const header = await readMatroska(source);
      ({ streams, textStart } = header);
      const defaultDuration = header.defaultDurations.get(track.id) ?? null;
      if (defaultDuration) {
        average = avReduce(1_000_000_000n, BigInt(defaultDuration), 30000n);
        // ffmpeg takes it for r_frame_rate between 5 and 1000 fps.
        if (average.num > average.den * 5 && average.num < average.den * 1000) headerRate = average;
      }
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
    // The rate the codec declares, from which ffmpeg can make up durations, and whether it
    // parses the stream: of the codecs whose rates are read, its mov and Matroska demuxers parse
    // H.264 but not HEVC.
    const declared = await declaredRate(track, codec);
    const parsed = codec === "avc";
    const probeLength = madeUp(resolution, null, declared, parsed);
    // Matroska gives no decode timestamps, and ffmpeg infers H.264's only once it has decoded
    // several frames, so times probing by them from later; that is not modelled.
    const spans = codec !== "avc" || !(format instanceof MatroskaInputFormat);
    // The header's rate stands, known before any packet is read.
    const { rate: found, probePackets } = headerRate ? { rate: headerRate, probePackets: 0 }
      : frameRate(rateDts, resolution, index?.table ?? null, codec,
                  { lengths: lengths.slice(first, last + 1), made: probeLength, average, spans });
    // Finding none, ffmpeg takes the rate the codec declares (doubled for H.264, which counts
    // fields) where the time base can tell its frames apart, and failing that the time base.
    const fields = declared && codec === "avc" ? multiply(declared, { num: 2, den: 1 }) : declared;
    const base = Math.round(resolution);
    const rFrameRate = found ?? (fields && fields.num <= base * fields.den ? fields : { num: base, den: 1 });
    // Without an average in the header, ffmpeg averages the durations of the packets it read
    // while probing, from the third on: in a Matroska file, the ones it made up. Other
    // containers' is not modelled.
    const averaged = average ?? (format instanceof MatroskaInputFormat
      && probeLength > 0 && probePackets > 2 ? averageRate(resolution, probeLength) : null);
    const paced = guessFrameRate(rFrameRate, averaged, codec === "avc" ? declared : null);
    // For want of anything better ffmpeg paces to the time base, or a rate an encoder declared
    // from it, as with a browser's WebM recording. Over 210 fps that is no frame rate unless the
    // frames come about that often: where the first shown average half as often or less,
    // load_video sets their mean rate with the fps filter instead.
    let fps = paced;
    if (paced.num > 210 * paced.den) {
      const end = Math.min(last + 1, first + RATE_PACKETS + REORDER);
      const shown = pts.slice(first, end).filter((_, i) => !hidden[first + i])
        .sort((a, b) => a - b).slice(0, RATE_PACKETS);
      const span = (shown[shown.length - 1] ?? 0) - (shown[0] ?? 0);
      const mean = span > 0 ? limitDenominator((shown.length - 1) * resolution / span, 1001) : null;
      if (mean && mean.num > 0 && 2 * mean.num * paced.den <= paced.num * mean.den) fps = mean;
    }
    // Where an MP4 repeats a track ID, its header can't say which of those tracks Mediabunny
    // lists, and Mediabunny's measures stand.
    const distinct = movie?.size === streams ? movie : undefined;
    // One at a time: Mediabunny keeps only the last movie fragment it read.
    const starts: (bigint | null)[] = [];
    for (const each of tracks) starts.push(await trackStart(each, isobmff, distinct));
    const fileStart = startOf(starts, textStart);

    const rotation = await track.getRotation();
    const [codedWidth, codedHeight] = [await track.getCodedWidth(), await track.getCodedHeight()];
    const turned = rotation % 180 !== 0;
    // ffmpeg times an MP4 by its tracks' durations, takes a Matroska header's duration as it
    // stands, and otherwise measures from the file's start to its end.
    let duration: number;
    const timed = distinct ? timedLength(tracks, starts, fileStart, distinct) : null;
    if (timed !== null) {
      duration = Number(timed) / 1e6;
    } else {
      const stated = await input.getDurationFromMetadata();
      duration = stated !== null && format instanceof MatroskaInputFormat ? stated
        : (stated ?? await input.computeDuration()) - Number(fileStart) / 1e6;
    }
    const info: ClipInfo = {
      width: turned ? codedHeight : codedWidth, height: turned ? codedWidth : codedHeight, fps,
      duration: Number.isFinite(duration) && duration > 0 ? duration : null,
    };
    const made = { probing: probeLength, after: madeUp(resolution, rFrameRate, declared, parsed) };
    return {
      info, resolution, isobmff, samples, pts, dts, lengths, keys, order, first, last, hidden, seekBack,
      made, probePackets, paced, fileStart, streams,
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
  /** The duration ffmpeg's demuxer makes up for a packet without one: read while probing,
   * before it knows r_frame_rate, and after. */
  made: { probing: number; after: number };
  /** How many packets, from `first`, ffmpeg reads while probing before it knows r_frame_rate. */
  probePackets: number;
  /** The rate ffmpeg paces decoded frames to without the fps filter. */
  paced: Rate;
  /** In presentation order: when ffmpeg shows each frame, the timestamp its decoded sample
   * carries here, its packet's duration (0 where the demuxer reads none) and its place in
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

/** When ffmpeg starts a stream, in microseconds: at its first timestamp, or where the header
 * starts an MP4 track with samples there. Null for a stream without packets. */
async function trackStart(track: InputTrack, isobmff: boolean,
                          header?: Map<number, Mp4Track>): Promise<bigint | null> {
  const stated = header?.get(track.id)?.start;
  if (stated !== undefined && stated !== null) return BigInt(stated);
  let first: number;
  let res: number;
  try {
    const packet = await new EncodedPacketSink(track).getFirstPacket({ metadataOnly: true });
    // ffmpeg has no start time for a stream without packets.
    if (!packet) return null;
    res = await track.getTimeResolution();
    first = Math.round(packet.timestamp * res);
  } catch {
    // A stream Mediabunny can't read has no start time either.
    return null;
  }
  // ffmpeg starts an MP4 stream at its edit list, so AAC priming before 0 doesn't count.
  // A Matroska Opus track's CodecDelay moves its packets earlier in ffmpeg but not its
  // start, to which ffmpeg adds the samples it will skip.
  if (isobmff) first = Math.max(first, 0);
  return rescale(BigInt(first), MICRO, BigInt(Math.round(res)));
}

/** When a file starts, in microseconds: the earliest of its streams' starts, ignoring those
 * without one, given the earliest start of its subtitle and data streams. */
function startOf(starts: (bigint | null)[], textStart: number | null): bigint {
  let start: bigint | null = null;
  for (const us of starts) if (us !== null && (start === null || us < start)) start = us;
  // Mediabunny lists no subtitle or data streams.
  return withText(start, textStart === null ? null : BigInt(textStart), -1n) ?? 0n;
}

/** The other streams' start, end or duration, `main`, or a subtitle or data stream's, `text`,
 * where ffmpeg takes that instead: when the others have none, or when it is less than a second
 * past theirs, later for `past` 1 and earlier for -1. */
function withText(main: bigint | null, text: bigint | null, past: 1n | -1n): bigint | null {
  if (main === null || text === null) return main ?? text;
  const gap = (text - main) * past;
  return gap > 0n && gap < MICRO ? text : main;
}

/** How long ffmpeg takes an MP4 to last from its streams' durations, given its header's
 * tracks, in microseconds: from the file's start to where the last stream ends, its duration
 * after its own start, or the longest duration if that is more. Mediabunny lists the video
 * and audio, which start as given; a subtitle or data track starts only where the header
 * starts it, so not at all where its samples are all in fragments, though ffmpeg starts it at
 * the first it reads. Null if the header holds none of the streams. */
function timedLength(tracks: InputTrack[], starts: (bigint | null)[], fileStart: bigint,
                     header: Map<number, Mp4Track>): bigint | null {
  const later = (a: bigint | null, b: bigint) => a === null || b > a ? b : a;
  let [end, longest, textEnd, textLongest]: (bigint | null)[] = [null, null, null, null];
  for (const [i, track] of tracks.entries()) {
    const duration = header.get(track.id)?.duration;
    if (duration === undefined) continue;
    const start = starts[i] ?? null;
    if (start !== null) end = later(end, start + BigInt(duration));
    longest = later(longest, BigInt(duration));
  }
  for (const { handler, start, duration } of header.values()) {
    if (!isText(handler)) continue;
    if (start !== null) textEnd = later(textEnd, BigInt(start + duration));
    textLongest = later(textLongest, BigInt(duration));
  }
  end = withText(end, textEnd, 1n);
  longest = withText(longest, textLongest, 1n);
  if (longest === null) return null;
  return end !== null && end - fileStart > longest ? end - fileStart : longest;
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
 * where the fps filter is only there when `rate` differs from the one ffmpeg
 * paces to without it.
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

  // A packet without a duration has the one ffmpeg's demuxer makes up, which differs for those
  // read while probing, unless a seek drops them to be read again.
  const { made, paced } = timeline;
  const probing = startUs === null ? indexed + timeline.probePackets : 0;
  const lengths = new Map<number, number>();
  for (const k of decoded) {
    const given = timeline.lengths[k]!;
    lengths.set(k, given > 0 ? given : decodeIndex[k]! < probing ? made.probing : made.after);
  }
  // One frame at the rate ffmpeg paces to.
  const period = Math.max(1, Math.round(resolution * paced.den / paced.num));

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
    // then after the step from the frame before, or one frame at the rate paced to, whereas
    // ffmpeg takes the codec's declared rate, else the stream's average.
    const before = decoded[decoded.indexOf(last) - 1];
    const step = before === undefined ? 0 : starts[last]! - starts[before]!;
    const given = lengths.get(last)!;
    const length = given > 0 && !(given === 1 && step > 2) ? given : step > 0 ? step : period;
    end = BigInt(starts[last]! + length) + offset;
  }

  const timed: Timed[] = [];
  if (rate.num === paced.num && rate.den === paced.den) {
    // In output frames to 1/2^bits of a frame (2^bits × paced.num within 2^29, bits at
    // most 16), then 2^-17 further from 0 unless whole, as ffmpeg times them.
    const bits = Math.min(Math.max(29 - Math.floor(Math.log2(paced.num)), 0), 16);
    for (const k of window) {
      let at = Number(rescale(BigInt(starts[k]!) + offset, BigInt(paced.num) << BigInt(bits),
                              res * BigInt(paced.den))) / 2 ** bits;
      if (at !== Math.round(at)) at += Math.sign(at) / 2 ** 17;
      // ffmpeg's filters give a frame still without a duration one frame.
      const length = lengths.get(k)! || period;
      timed.push({ frame: k, at, length: length * (1 / resolution) / (paced.den / paced.num) });
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
 * The frame rate ffprobe calls r_frame_rate, where ffmpeg estimates one from decode timestamps
 * (null where it doesn't), and how many packets it reads while probing before it knows it. It
 * estimates one for H.264 and HEVC and for time bases finer than 1/100 s or coarser than 1/5 s:
 * as it reads an MP4's header, from the first 99 samples in its table (`table`, each sample's
 * decode timestamp; taken only for such a time base), then from the packets read while probing
 * (`dts` and `probe`, as `standardRate` takes them). The estimate is the greatest common step if
 * every step after the third shares one, else the standard rate whose ticks the timestamps fall
 * on most evenly. For other streams, probing reads a single packet.
 */
function frameRate(dts: number[], resolution: number, table: number[] | null, codec: string | null,
                   probe: Probe): { rate: Rate | null; probePackets: number } {
  const unreliableBase = resolution >= 101 || resolution < 5;
  if (!unreliableBase && codec !== "avc" && codec !== "hevc") {
    return { rate: null, probePackets: Math.min(dts.length, 1) };
  }
  // Reading the header, ffmpeg knows no codec yet, so judges by the time base alone. It
  // carries the common step found there into the probe.
  const header = table ? standardRate(table.slice(0, 99), resolution) : { rate: null, common: 0 };
  if (unreliableBase && header.rate) return { rate: header.rate, probePackets: 0 };
  const { rate, read } = standardRate(dts, resolution, probe, header.common);
  return { rate, probePackets: read };
}

/**
 * The duration ffmpeg's demuxer makes up for a packet without one, in ticks: a frame at
 * r_frame_rate (`rate`, once known), unless it parses the stream (`parsed`) and the codec
 * declares a rate (`declared`); failing that, a tick of a time base coarser than 1 ms; failing
 * that, a frame at a declared rate under 1000 fps; else none.
 */
function madeUp(resolution: number, rate: Rate | null, declared: Rate | null, parsed: boolean): number {
  if (rate && !(parsed && declared)) return Math.floor(resolution * rate.den / rate.num);
  if (resolution < 1000) return 1;
  if (declared && declared.num < declared.den * 1000) return Math.floor(resolution * declared.den / declared.num);
  return 0;
}

/** The avg_frame_rate ffmpeg works out from packets that each last `length` ticks: their rate,
 * or the standard rate nearest it if one is within 1%. */
function averageRate(resolution: number, length: number): Rate {
  const rate = avReduce(BigInt(Math.round(resolution)), BigInt(length), 60000n);
  let best = 0.01;
  let found = 0;
  for (const standard of STANDARD_RATES) {
    const error = Math.abs(rate.num / rate.den / (standard / 12012) - 1);
    if (error < best) {
      best = error;
      found = standard;
    }
  }
  return found ? avReduce(BigInt(found), 12012n, INT_MAX) : rate;
}

/**
 * ffmpeg's av_guess_frame_rate, the rate it paces decoded frames to without the fps filter:
 * r_frame_rate, or the average where that is under 70 fps and r_frame_rate over 210; then,
 * for H.264, which can count fields, the rate the stream declares (`declared`) where that is
 * under 0.7 of the rate so far and the average over a tenth away from it.
 */
function guessFrameRate(rFrameRate: Rate, average: Rate | null, declared: Rate | null): Rate {
  let rate = rFrameRate;
  if (average && average.num < 70 * average.den && rate.num > 210 * rate.den) rate = average;
  // Without an average the comparison fails, as ffmpeg's does with its 0/0.
  const off = average ? Math.abs(1 - (average.num / average.den) / (rate.num / rate.den)) : NaN;
  if (declared && declared.num / declared.den < rate.num / rate.den * 0.7 && off > 0.1) rate = declared;
  return rate;
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

/** What ffmpeg's probe goes by besides the decode timestamps: the packets' durations (0 for
 * none), the duration it makes up for a packet without one, the stream's average rate, and
 * whether it can time probing by the decode timestamps' span. */
interface Probe { lengths: number[]; made: number; average: Rate | null; spans: boolean }

/**
 * ffmpeg's ff_rfps_calculate over these decode timestamps, the common step it found, starting
 * from `common`, and how many packets it read. Given a `probe`, over as many as ffmpeg reads
 * while probing: 20 frame steps (40 for a time base coarser than 0.5 ms), or 5 s by the longer
 * of the packets' durations and the frames at the average rate, or where it `spans` and lacks
 * both, after 30 frames, by the span of the decode timestamps from the third. Without, over all
 * of them, as the mov demuxer passes its header's. Other streams' packets, which can end probing
 * sooner or later, and ffmpeg's 5 MB limit on what it reads are not modelled.
 */
function standardRate(dts: number[], resolution: number, probe?: Probe,
                      common = 0): { rate: Rate | null; common: number; read: number } {
  const want = !probe ? Infinity : 1 / resolution > 0.0005 ? 40 : 20;
  const n = STANDARD_RATES.length;
  // For each rate, and for ticks counted from 0 or from half a tick: the sums of
  // each timestamp's distance from its nearest tick, and of its square.
  const sums = [new Float64Array(n), new Float64Array(n)];
  const squares = [new Float64Array(n), new Float64Array(n)];
  let count = 0;
  let stepSum = 0;
  let probed = 0;
  let last: number | undefined;
  let read = 0;
  const res = BigInt(Math.round(resolution));
  for (let p = 0; p < dts.length && count < want; p++) {
    read = p + 1;  // the packet that reaches the time limit is read but not counted
    if (probe && p >= 2) {
      // ffmpeg probes at most 5 s, timed in whole microseconds.
      const { average } = probe;
      const byLengths = rescale(BigInt(probed), MICRO, res);
      const byFrames = average?.num ? rescale(BigInt(p) * BigInt(average.den), MICRO, BigInt(average.num)) : 0n;
      if (byLengths >= 5_000_000n || byFrames >= 5_000_000n) break;
      if (probe.spans && !byLengths && !byFrames && p > 30
          && rescale(BigInt(dts[p]! - dts[2]!), MICRO, res) >= 5_000_000n) break;
      const length = probe.lengths[p] ?? 0;
      probed += length > 0 ? length : probe.made;
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
    return { rate: avReduce(res, BigInt(common), INT_MAX), common, read };
  }
  if (count < 2) return { rate: null, common, read };
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
  return { rate, common, read };
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
