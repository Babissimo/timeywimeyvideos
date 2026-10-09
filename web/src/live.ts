/**
 * The live preview: a copy of the clip at half size in x, y and time, held on the GPU in YUV
 * 4:2:0 and sliced there a frame at a time, as `timeslice.py --preview` renders it.
 */
import { load, planLoad, probe, VideoError, type ClipInfo, type LoadOptions,
         type LoadPlan } from "./decode";
import { Slicer, type SliceOptions } from "./gpu/slicer";
import { Volume, VolumeTooLarge } from "./gpu/volume";
import { looping, type Noise } from "./noise";
import {
  endpoints, fullSize, liveNoise, loopFades, plan, Problem, readOptions, type FullSize,
  type OptionValues, type Options,
} from "./options";
import { PREVIEW_SCALE, type Sweep, type TX } from "./planner";
import { limitDenominator, roundHalfEven } from "./pymath";
import type { Columns, Rate } from "./types";

type Ends = [number, number, number, number];

/** A frame of the live view and the sweep it belongs to. */
export interface FrameInfo {
  width: number;
  height: number;
  frames: number;
  frame: number;
  /** The frame rate the sweep plays at, and how long it lasts in seconds. */
  fps: number;
  seconds: number;
  /** The clip as held, [frames, height, width], and the GPU memory it takes. */
  volume: [number, number, number];
  memory: number;
  loop: boolean;
  sides: boolean;
  /** Where this frame's, the first frame's and the last frame's end columns come from. */
  line: Ends;
  first: Ends;
  last: Ends;
  /** What a full-quality render would make, why it can't, or null if that's unknown. */
  full: FullSize | { error: string } | null;
}

/** Fetches a source by name, "videos/<file>" or "uploads/<file>". */
export type Opener = (source: string, signal: AbortSignal) => Promise<Blob>;

/** The kind and file name of a source, refusing anything outside the server's folders. */
function sourceName(source: string): [string, string] {
  const slash = source.indexOf("/");
  const kind = slash < 0 ? source : source.slice(0, slash);
  const name = slash < 0 ? "" : source.slice(slash + 1);
  if ((kind !== "videos" && kind !== "uploads") || !name || name.includes("/")
      || name.startsWith("."))
    throw new Problem("Pick a video first.");
  return [kind, name];
}

/** Fetch a source from the server that offers it. */
export async function fetchSource(source: string, signal: AbortSignal): Promise<File> {
  const [kind, name] = sourceName(source);
  const res = await fetch(`/media/${kind}/${encodeURIComponent(name)}`, { signal });
  if (res.status === 404) throw new Problem(`${name} isn't in ${kind} any more.`);
  if (!res.ok) throw new Problem(`The server said ${res.status}.`);
  return new File([await res.blob()], name);
}

interface Source { name: string; blob: Blob; info: ClipInfo }
interface Clip { key: string; volume: Volume; fps: Rate; info: ClipInfo; face?: ImageData }

/** What every frame of a sweep shares, for one set of options and the clip they load. */
interface Sweeping {
  opts: Options;
  clip: Clip;
  sweep: Sweep;
  noise: { surface: Noise; push: TX } | null;
  info: Omit<FrameInfo, "frame" | "line">;
}

/** Whether two sets of option values hold the same values. */
function sameValues(a: OptionValues, b: OptionValues): boolean {
  const keys = Object.keys(a);
  return keys.length === Object.keys(b).length
    && keys.every((key) => Object.hasOwn(b, key) && Object.is(a[key], b[key]));
}

const SHRINK = limitDenominator(PREVIEW_SCALE);
const LOST = "The live view lost the GPU, perhaps for want of memory. Set a shorter duration or "
  + "a smaller scale, or reload the page.";

/** The live view on a canvas, which it draws with WebGL2. */
export class Live {
  private readonly gl: WebGL2RenderingContext;
  private readonly open: Opener;
  private slicer: Slicer;
  private source: Source | null = null;
  private clip: Clip | null = null;
  private loading: AbortController | null = null;
  private parsed: { values: OptionValues; opts: Options } | null = null;
  private sweeping: Sweeping | null = null;
  private readonly listening = new AbortController();

  constructor(canvas: HTMLCanvasElement, open: Opener = fetchSource) {
    // The cuboid copies the canvas whenever it redraws, so the drawing has to stay.
    const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, depth: false,
                                             stencil: false, preserveDrawingBuffer: true });
    if (!gl) throw new Problem("The live view needs WebGL2, which this browser doesn't offer.");
    this.gl = gl;
    this.open = open;
    try {
      this.slicer = new Slicer(gl);
    } catch (error) {
      // Losing the context frees whatever the slicer made before it failed.
      gl.getExtension("WEBGL_lose_context")?.loseContext();
      const reason = error instanceof Error ? error.message : String(error);
      throw new Problem(`The live view couldn't start on this GPU: ${reason}`);
    }
    // Running short of memory can cost the GPU context; once it comes back, so does the view.
    const { signal } = this.listening;
    canvas.addEventListener("webglcontextlost", (event) => {
      event.preventDefault();
      this.loading?.abort(new Problem(LOST));
      this.clip = null;
    }, { signal });
    canvas.addEventListener("webglcontextrestored", () => {
      this.slicer = new Slicer(gl);
    }, { signal });
  }

  /**
   * Draw the frame `pos` of the way through (0 to 1) the sweep these options make of a
   * source, and describe it. Loads the clip first when the source, or the options it is
   * loaded with, change, calling `progress` as it decodes. Throws a Problem for the page to
   * show. Make one call at a time.
   */
  async show(source: string, values: OptionValues, pos: number,
             progress?: (done: number, total: number) => void): Promise<FrameInfo> {
    sourceName(source);  // refuse a source outside the server's folders first
    if (!this.parsed || !sameValues(this.parsed.values, values))
      this.parsed = { values: { ...values }, opts: readOptions(values) };
    const { opts } = this.parsed;
    const clip = await this.load(source, opts, progress);
    if (this.sweeping?.opts !== opts || this.sweeping.clip !== clip)
      this.sweeping = this.sweepFor(opts, clip);
    const { sweep, noise, info } = this.sweeping;
    const { volume } = clip;
    const f = roundHalfEven((Math.min(Math.max(pos, 0), 1) || 0) * (sweep.frames - 1));
    const columns = sweep.at(f);
    const options = { nearest: true, wrap: sweep.loop, wrapX: sweep.sides };
    if (noise) {
      this.slice(volume, columns, { ...options, noise: {
        push: noise.push, grid: noise.surface.grid(sweep.width, volume.height, f),
      } });
    } else {
      this.slice(volume, columns, options);
    }
    const canvas = this.gl.canvas;
    if (canvas.width !== sweep.width || canvas.height !== volume.height) {
      canvas.width = sweep.width;
      canvas.height = volume.height;
    }
    this.slicer.draw();
    const { t, x } = columns, last = t.length - 1;
    return { ...info, frame: f, line: [t[0], x[0], t[last], x[last]] };
  }

  /**
   * The loaded clip's first frame, or null before a clip has loaded. It is the same
   * ImageData for as long as the clip is held.
   */
  face(): ImageData | null {
    const { clip } = this;
    if (!clip || this.gl.isContextLost()) return null;
    if (!clip.face) {
      const { volume } = clip;
      const x = Float64Array.from({ length: volume.width }, (_, i) => i);
      this.slice(volume, { t: new Float64Array(volume.width), x }, { nearest: true });
      const pixels = new Uint8ClampedArray(this.slicer.read().buffer as ArrayBuffer);
      clip.face = new ImageData(pixels, volume.width, volume.height);
    }
    return clip.face;
  }

  /** Let go of the clip and its file, stopping any load under way. */
  unload(): void {
    this.loading?.abort();
    this.loading = null;
    this.release();
    this.source = null;
  }

  dispose(): void {
    this.unload();
    this.listening.abort();
    this.slicer.dispose();
  }

  private release(): void {
    this.clip?.volume.dispose();
    this.clip = null;
  }

  /** Slice a frame, or throw the Problem of one larger than the GPU can draw. */
  private slice(volume: Volume, columns: Columns, options: SliceOptions): void {
    try {
      this.slicer.slice(volume, columns, options);
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      throw new Problem(`The live view can't draw frames this large on this GPU: `
                        + `${error.message}. Set a smaller angle, a shorter duration or a `
                        + "smaller scale.");
    }
  }

  /** The sweep these options make of the clip, and what each of its frames shares. */
  private sweepFor(opts: Options, clip: Clip): Sweeping {
    const { volume } = clip;
    const noise = liveNoise(opts);
    const sweep = plan(volume.frames, volume.width, opts, noise);
    const surface = noise && looping(noise, sweep);
    const fps = opts.fps ? { num: opts.fps.num * SHRINK.num, den: opts.fps.den * SHRINK.den }
                         : clip.fps;  // already the rate of the frames kept
    return {
      opts, clip, sweep,
      noise: surface && { surface, push: surface.push(sweep.normal) },
      info: {
        width: sweep.width, height: volume.height, frames: sweep.frames,
        fps: fps.num / fps.den, seconds: sweep.frames * fps.den / fps.num,
        volume: [volume.frames, volume.height, volume.width], memory: volume.bytes,
        loop: sweep.loop, sides: sweep.sides,
        first: endpoints(sweep, 0), last: endpoints(sweep, sweep.frames - 1),
        full: fullSize(clip.info, opts),
      },
    };
  }

  /** The clip for these options, loading it if it changed, with the loop's crossfades done. */
  private async load(source: string, opts: Options,
                     progress?: (done: number, total: number) => void): Promise<Clip> {
    if (this.gl.isContextLost()) throw new Problem(LOST);
    const key = JSON.stringify([source, opts.scale, opts.start, opts.duration, opts.loopFade,
                                opts.sideFade]);
    if (this.clip?.key === key) return this.clip;
    const controller = new AbortController();
    this.loading = controller;
    const { signal } = controller;
    try {
      const { blob, info } = await this.fetch(source, signal);
      const options: LoadOptions = {
        scale: opts.scale * PREVIEW_SCALE, timeScale: PREVIEW_SCALE,
        start: opts.start ?? undefined, duration: opts.duration ?? undefined, fast: true, signal,
      };
      const planned = await planLoad(blob, options);
      // Refuse fades that don't fit before decoding the clip.
      const [frames, columns] = loopFades(planned.frames, planned.width,
                                          planned.fps.num / planned.fps.den, opts, PREVIEW_SCALE);
      const volume = this.allocate(planned);
      try {
        await load(blob, options, (i, frame) => {
          volume.upload(i, frame);
          frame.close();
        }, progress);
        signal.throwIfAborted();
        // The fades work in place, so a change to them loads the clip afresh.
        volume.crossfade(frames, 0);
        volume.crossfade(columns, 2);
      } catch (error) {
        volume.dispose();
        throw error;
      }
      this.clip = { key, volume, fps: planned.fps, info };
      return this.clip;
    } catch (error) {
      if (signal.aborted) throw signal.reason;  // whatever stopped it on the way
      if (error instanceof VideoError) throw new Problem(error.message);
      throw error;
    } finally {
      if (this.loading === controller) this.loading = null;
    }
  }

  /** The source's file and what it holds, fetched once. */
  private async fetch(source: string, signal: AbortSignal): Promise<Source> {
    if (this.source?.name === source) return this.source;
    this.source = null;
    const blob = await this.open(source, signal);
    const info = await probe(blob);
    signal.throwIfAborted();
    this.source = { name: source, blob, info };
    return this.source;
  }

  /**
   * A volume for the planned frames in place of the clip held, or the Problem of one too
   * large for the GPU. A clip refused for the GPU's limits is kept.
   */
  private allocate({ frames, height, width }: LoadPlan): Volume {
    const { gl } = this;
    // The context can be lost before its event arrives, and a lost one gives no limits.
    if (gl.isContextLost()) throw new Problem(LOST);
    const layers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
    const size = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    if (frames > layers) {
      throw new Problem(`The live view can hold at most ${layers} frames on this GPU, and `
                        + `this clip has ${frames} at half its frame rate. Set a shorter `
                        + "duration.");
    }
    if (width > size || height > size) {
      throw new Problem(`The live view can hold frames at most ${size} pixels across on this `
                        + `GPU, and this clip's are ${width}×${height} at half size. Set a `
                        + "smaller scale.");
    }
    this.release();  // let the old clip go before the new one takes its memory
    try {
      return Volume.create(gl, frames, height, width, { format: "yuv420" });
    } catch (error) {
      if (!(error instanceof VolumeTooLarge)) throw error;
      throw new Problem(`The live view would need ${(error.bytes / 1e9).toFixed(1)} GB of GPU `
                        + "memory for this clip, more than the GPU can give it. Set a shorter "
                        + "duration or a smaller scale.");
    }
  }
}
