/**
 * Reading a video into frames in the browser: Mediabunny demuxes and judges the frame rate,
 * WebCodecs decodes. A clip is held at one constant rate, each slot holding the source frame on
 * screen about its middle, so a frame that lasts longer fills more slots.
 */
import {
  ALL_FORMATS, BlobSource, EncodedPacketSink, Input, UnsupportedInputFormatError, VideoSample,
  VideoSampleSink, type FrameRateMetrics, type InputVideoTrack,
} from "mediabunny";
import { gcd, limitDenominator, roundHalfEven } from "./numbers";
import type { Rate } from "./types";

/** The browser couldn't read or decode a video. */
export class VideoError extends Error {
  override name = "VideoError";
}

/** What `probe` finds: the upright size in square pixels, the rate the frames are held at and
 * the length in seconds, from the start of the first frame to the end of the last. */
export interface ClipInfo { width: number; height: number; fps: Rate; duration: number | null }

/**
 * scale resizes each frame; timeScale keeps that fraction of the frames
 * (0.5 = every other one); start and duration pick a window in seconds from the
 * first frame; fast resizes by nearest neighbour instead of smoothly.
 */
export interface LoadOptions {
  scale?: number; timeScale?: number; start?: number; duration?: number
  fast?: boolean; signal?: AbortSignal
}

/** What `load` delivers: `frames` frames of width x height, at `fps` (the clip's rate x
 * timeScale). */
export interface LoadPlan { width: number; height: number; frames: number; fps: Rate }

/** A video's size, frame rate and length. Rejects with a VideoError if the browser can't open
 * or decode it. */
export async function probe(source: Blob): Promise<ClipInfo> {
  const { width, height, fps, length } = await measure(source);
  return { width, height, fps, duration: length };
}

/** How many seconds of a video `length` seconds long (null if unknown) are used when
 * starting at `start` and keeping at most `duration`. */
export function clipSeconds(length: number | null, start?: number, duration?: number): number | null {
  let seconds = length === null ? null : Math.max(length - (start || 0), 0);
  if (duration !== undefined) seconds = seconds === null ? duration : Math.min(seconds, duration);
  return seconds;
}

/** What `load` will deliver with these options, found without decoding, so a volume can be
 * allocated first. */
export async function planLoad(source: Blob, options: LoadOptions = {}): Promise<LoadPlan> {
  const facts = await measure(source, options.signal);
  const { width, height, frames, fps } = plan(facts, options);
  planned.set(source, facts);
  return { width, height, frames, fps };
}

// What each file was last planned from, so that the load after a plan delivers what it said
// without measuring the file again. Each plan measures afresh: Firefox reads on through a file
// changed on disk.
const planned = new WeakMap<Blob, Facts>();

const TIE = 1e-6;  // seconds

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
  const facts = planned.get(source) ?? await measure(source, signal);
  const { width, height, frames: total, fps, times } = plan(facts, options);
  const clip = await open(source, signal);
  try {
    const resize: ImageBitmapOptions = {
      resizeWidth: width, resizeHeight: height, resizeQuality: options.fast ? "pixelated" : "high",
    };
    const upright = (await clip.track.getRotation()) === 0 && !(await clip.track.getFlip());
    let canvas: OffscreenCanvas | undefined;

    // A frame stored turned is drawn upright at full size first, then resized as the others are.
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
        // The volume holds no alpha: a frame keeps the colour beneath it.
        if (sample.hasAlpha) {
          const frame = sample.toVideoFrame();
          try {
            opaque = new VideoSample(await withoutAlpha(frame),
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

    // Each slot holds the latest frame to start before its time. One starting at that time, to
    // within the microseconds decoders count in, counts as later, so it doesn't hide the one
    // before it.
    const before = (k: number) => times[k]! - TIE;
    const samples = new VideoSampleSink(clip.track).samples(before(0), before(total - 1));
    // The latest sample decoded, kept open for slots until a later one starts.
    let held: VideoSample | undefined;
    try {
      while (done < total) {
        let next: IteratorResult<VideoSample, void>;
        try {
          next = await samples.next();
        } catch (error) {
          throw new VideoError(`failed to decode ${clip.name}`, { cause: error });
        }
        if (next.done) break;
        const sample = next.value;
        if (signal?.aborted) {
          sample.close();
          signal.throwIfAborted();
        }
        try {
          while (done < total && sample.timestamp >= before(done)) await deliver(held ?? sample);
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
  } catch (error) {
    signal?.throwIfAborted();  // whatever failed once the file was let go
    throw error;
  } finally {
    clip.close();
  }
}

/**
 * The frame without its alpha. Firefox draws an RGB frame's alpha even once discarded, and as if
 * premultiplied, so such a frame's pixels are copied out with the alpha made opaque.
 */
async function withoutAlpha(frame: VideoFrame): Promise<VideoFrame> {
  if (frame.format !== "RGBA" && frame.format !== "BGRA")
    return new VideoFrame(frame, { alpha: "discard" });
  const pixels = new Uint8Array(frame.allocationSize());
  await frame.copyTo(pixels);  // the visible rectangle, its rows packed tight
  for (let i = 3; i < pixels.length; i += 4) pixels[i] = 255;
  const { width, height } = frame.visibleRect!;
  return new VideoFrame(pixels, {
    format: frame.format, codedWidth: width, codedHeight: height,
    displayWidth: frame.displayWidth, displayHeight: frame.displayHeight,
    timestamp: frame.timestamp, duration: frame.duration ?? undefined,
    colorSpace: frame.colorSpace.toJSON(), transfer: [pixels.buffer],
  } as VideoFrameBufferInit);
}

/** A video open for reading, until `close` lets it go. */
interface Clip { track: InputVideoTrack; name: string; close: () => void }

/** Open a video's primary video track. Aborting the signal stops its reads, which can run through
 * the whole file, and rejects with the signal's reason. */
async function open(source: Blob, signal?: AbortSignal): Promise<Clip> {
  signal?.throwIfAborted();
  const name = source instanceof File ? source.name : "this file";
  const input = new Input({ source: new BlobSource(source), formats: ALL_FORMATS });
  const stop = () => input.dispose();
  signal?.addEventListener("abort", stop);
  const close = () => {
    signal?.removeEventListener("abort", stop);
    input.dispose();
  };
  let track: InputVideoTrack | null = null;
  try {
    track = await input.getPrimaryVideoTrack();
    signal?.throwIfAborted();
  } catch (error) {
    close();
    const kind = error instanceof UnsupportedInputFormatError
      ? await container(source).catch(() => null) : null;
    signal?.throwIfAborted();
    if (kind) {
      throw new VideoError(`The browser can't open ${kind} files such as ${name}. Convert it to `
                           + "an H.264 MP4 first.", { cause: error });
    }
    throw new VideoError(`can't read a video stream from ${name}`, { cause: error });
  }
  if (!track) {
    close();
    throw new VideoError(`can't read a video stream from ${name}`);
  }
  return { track, name, close };
}

// Containers Mediabunny doesn't read, by the bytes they start with, so the page can name them.
const CONTAINERS: [string, number, string][] = [
  ["AVI", 8, "AVI "], ["FLV", 0, "FLV"], ["MPEG program stream", 0, "\x00\x00\x01\xba"],
  ["ASF", 0, "\x30\x26\xb2\x75\x8e\x66\xcf\x11"], ["RealMedia", 0, ".RMF"],
  ["MXF", 0, "\x06\x0e\x2b\x34"], ["IVF", 0, "DKIF"], ["Y4M", 0, "YUV4MPEG2"], ["GIF", 0, "GIF8"],
];

/** The name of the container `source` is in, if it is one of CONTAINERS. */
async function container(source: Blob): Promise<string | null> {
  const head = String.fromCharCode(...new Uint8Array(await source.slice(0, 16).arrayBuffer()));
  return CONTAINERS.find(([, at, magic]) => head.startsWith(magic, at))?.[0] ?? null;
}

/** What loading a video needs to know of it. Times are in seconds. */
interface Facts {
  name: string;
  /** The upright size in square pixels. */
  width: number; height: number;
  /** The rate the frames are held at, when the first starts and how long they last. */
  fps: Rate; start: number; length: number;
}

/** Measure a video for `plan`. Rejects with the signal's reason once aborted. */
async function measure(source: Blob, signal?: AbortSignal): Promise<Facts> {
  const { track, name, close } = await open(source, signal);
  try {
    const codec = await track.getCodec();
    // Without WebCodecs no codec decodes, and the page says so itself.
    const webCodecs = typeof VideoDecoder !== "undefined";
    if (!codec || (webCodecs && !(await track.canDecode()))) {
      const id = codec ?? await track.getInternalCodecId();  // a number or bytes in some containers
      const kind = typeof id === "string" ? id : "unknown";
      throw new VideoError(`The browser has no decoder for the ${kind} video in ${name}. `
                           + "Convert it to an H.264 MP4 first.");
    }
    const packets = new EncodedPacketSink(track);
    const last = await packets.getPacket(Infinity, { metadataOnly: true });
    // Decoding starts at a key frame, so the clip starts at the first, checked as the decoder
    // checks it.
    const key = await packets.getFirstKeyPacket({ verifyKeyPackets: true });
    if (!last || !key) throw new VideoError(`no frames decoded from ${name}`);
    // Frames that span no time have no rate of their own, so are held as one frame of the last
    // one's duration, or of a second if that is under a millisecond, taken for a common rate it
    // is near.
    const single = 1 / (last.duration >= 1e-3 ? last.duration : 1);
    const rate = last.timestamp <= key.timestamp ? common(single) ?? single
      : heldRate(await track.computeFrameRateMetrics(), await track.getTimeResolution());
    // Denominators up to 1001 keep NTSC rates such as 30000/1001 exact, and up to a frame's
    // length in seconds keep a rate under a frame a second from rounding to none.
    const fps = limitDenominator(rate, Math.max(1001, Math.ceil(1 / rate)));
    // Frames before 0 only lead into an edit list's start, which can fall part way through a
    // frame: the clip starts with the frame on screen at 0, so slots fall mid-frame.
    const first = key.timestamp < 0 ? (await packets.getPacket(0, { metadataOnly: true }))! : key;
    const start = first.timestamp;
    // The last frame lasts as long as it says, and at least a frame.
    const end = last.timestamp + Math.max(last.duration, fps.den / fps.num);
    const width = await track.getDisplayWidth(), height = await track.getDisplayHeight();
    // Letting the file go stops only reads from it, not work on what is already read.
    signal?.throwIfAborted();
    return { name, width, height, fps, start, length: end - start };
  } catch (error) {
    signal?.throwIfAborted();
    if (error instanceof VideoError) throw error;
    throw new VideoError(`can't read a video stream from ${name}`, { cause: error });
  } finally {
    close();
  }
}

/** The rate to hold a clip's frames at, from its first frames' times: the lattice Mediabunny
 * finds they lie on, unless that would mislead. A stand-in is taken for a common rate it is near. */
function heldRate(metrics: FrameRateMetrics, resolution: number): number {
  const { underlyingFrameRate: lattice, medianFrameRate: median, averageFrameRate: mean } = metrics;
  // Times jittered about a steady rate fit no lattice. Their median is taken for a common rate it
  // is near, or failing that their mean, which jitter moves less but dropped frames more, or else
  // is taken as it is.
  if (lattice === null) return common(median) ?? common(mean) ?? median;
  // With most frames two or more steps apart, the clip would hold mostly repeats.
  const sparse = lattice > 1.5 * median;
  // Under two ticks a step, a lattice between ticks, or of the ticks themselves with most frames
  // further apart, is finer than the timestamps can place.
  const ticks = resolution / lattice;
  if (ticks < 2 && (Math.abs(ticks - Math.round(ticks)) > 1e-6 || sparse))
    return common(mean) ?? mean;
  return sparse ? common(median) ?? median : lattice;
}

// The rates cameras and encoders use. A rate measured from rounded or uneven times is taken for
// one within 2.5 %, or for an NTSC rate within 0.05 %, as each lies 0.1 % from a whole rate.
const COMMON = [5, 10, 12, 15, 20, 24, 25, 30, 48, 50, 60, 100, 120, 144, 240];
const NTSC = [24, 30, 60, 120].map((rate) => rate / 1.001);

/** The common rate `rate` is near, if there is one. */
function common(rate: number): number | undefined {
  const off = (to: number) => Math.abs(to / rate - 1);
  const near = COMMON.reduce((a, b) => (off(b) < off(a) ? b : a));
  return NTSC.find((to) => off(to) <= 0.0005) ?? (off(near) <= 0.025 ? near : undefined);
}

/** The frames `load` delivers: their size, rate and number, and the time each is taken at. */
interface Plan extends LoadPlan { times: number[] }

function plan(facts: Facts, options: LoadOptions): Plan {
  const { scale = 1, timeScale = 1, start = 0, duration } = options;
  if (!Number.isFinite(scale)) throw new RangeError(`scale must be a number, not ${scale}`);
  const share = limitDenominator(timeScale, 1000);
  if (share.num <= 0) throw new RangeError(`a timeScale of ${timeScale} keeps no frames`);
  const fps = multiply(facts.fps, share);
  const seconds = clipSeconds(facts.length, start, duration)!;
  if (!(seconds > 0)) throw new VideoError(`no frames decoded from ${facts.name}`);
  // As many slots as the window holds, to the nearest whole.
  const frames = Math.max(1, Math.round(seconds * fps.num / fps.den));
  // Each slot is taken at the middle of the frame, at the clip's rate, nearest the slot's own
  // middle (the earlier where two are as near): as far from where frames change as their timing
  // allows, so timestamps rounded or a little uneven still give evenly spaced frames.
  const { num, den } = facts.fps;
  const times = Array.from({ length: frames }, (_, k) => {
    const middle = start * num / den + (k + 0.5) * share.den / share.num;  // in frames
    return facts.start + (Math.ceil(middle - 1 - 1e-9) + 0.5) * den / num;
  });
  return {
    width: Math.max(1, roundHalfEven(facts.width * scale)),
    height: Math.max(1, roundHalfEven(facts.height * scale)), frames, fps, times,
  };
}

function multiply(a: Rate, b: Rate): Rate {
  const num = a.num * b.num;
  const den = a.den * b.den;
  const divisor = gcd(num, den) || 1;
  return { num: num / divisor, den: den / divisor };
}
