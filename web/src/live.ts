/**
 * The live preview: a copy of the clip at half size in x, y and time, held on the GPU in YUV
 * 4:2:0 and sliced there a frame at a time, as a render preview makes it.
 */
import { load, planLoad, VideoError, type LoadOptions, type LoadPlan } from "./decode";
import { Slicer, type SliceOptions } from "./gpu/slicer";
import { Volume, VolumeTooLarge } from "./gpu/volume";
import { looping, type Noise } from "./noise";
import {
  endpoints, liveNoise, planFaded, Problem, readOptions, type OptionValues, type Options,
} from "./options";
import { PREVIEW_SCALE, type Line, type Sweep, type TX } from "./planner";
import { limitDenominator } from "./numbers";
import { changed, hasChanged } from "./sources";
import type { Rate } from "./types";

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
  /** The clip as faded, [frames, height, width], and the GPU memory it takes as held. */
  volume: [number, number, number];
  memory: number;
  loop: boolean;
  sides: boolean;
  /** Where this frame's, the first frame's and the last frame's end columns come from. */
  line: Ends;
  first: Ends;
  last: Ends;
}

interface Clip {
  file: File; key: string; volume: Volume; fps: Rate;
  face?: { fade: TX; image: ImageData };  // its first frame with these fades
}

/** What every frame of a sweep shares, for one set of options and the clip they load. */
interface Sweeping {
  opts: Options;
  clip: Clip;
  sweep: Sweep;
  fade: TX;  // the frames and columns faded
  noise: { surface: Noise; push: TX } | null;
  info: Omit<FrameInfo, "frame" | "line">;
}

/** Whether two sets of option values hold the same values. */
function sameValues(a: OptionValues, b: OptionValues): boolean {
  const keys = Object.keys(a);
  return keys.length === Object.keys(b).length
    && keys.every((key) => Object.hasOwn(b, key) && Object.is(a[key], b[key]));
}

/** What the clip loaded for some options depends on: a change to it loads the clip afresh. */
const loadKey = (opts: Options) => JSON.stringify([opts.scale, opts.start, opts.duration]);

const SHRINK = limitDenominator(PREVIEW_SCALE);
const LOST = "The live view lost the GPU, perhaps for want of memory. Set a shorter duration or "
  + "a smaller scale, or reload the page.";

/** The live view on a canvas, which it draws with WebGL2. */
export class Live {
  private readonly gl: WebGL2RenderingContext;
  private slicer: Slicer;
  private clip: Clip | null = null;
  private loading: AbortController | null = null;
  private loadingKey = "";  // the load key of the clip `loading` loads
  private parsed: { values: OptionValues; opts: Options } | null = null;
  private sweeping: Sweeping | null = null;
  private readonly listening = new AbortController();

  constructor(canvas: HTMLCanvasElement) {
    // The cuboid copies the canvas whenever it redraws, so the drawing has to stay.
    const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, depth: false,
                                             stencil: false, preserveDrawingBuffer: true });
    if (!gl) throw new Problem("The live view needs WebGL2, which this browser doesn't offer.");
    this.gl = gl;
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
      this.release();
    }, { signal });
    canvas.addEventListener("webglcontextrestored", () => {
      this.slicer = new Slicer(gl);
    }, { signal });
  }

  /**
   * Draw the frame `pos` of the way through (0 to 1) the sweep these options make of a
   * video file, and describe it. Loads the clip first when the file, or the options it is
   * loaded with, change, calling `progress` as it decodes. Throws a Problem for the page to
   * show. Make one call at a time.
   */
  async show(file: File, values: OptionValues, pos: number,
             progress?: (done: number, total: number) => void): Promise<FrameInfo> {
    if (!this.parsed || !sameValues(this.parsed.values, values))
      this.parsed = { values: { ...values }, opts: readOptions(values) };
    const { opts } = this.parsed;
    const clip = await this.load(file, opts, progress);
    if (this.sweeping?.opts !== opts || this.sweeping.clip !== clip)
      this.sweeping = this.sweepFor(opts, clip);
    const { sweep, fade, noise, info } = this.sweeping;
    const { volume } = clip;
    const f = Math.round((Math.min(Math.max(pos, 0), 1) || 0) * (sweep.frames - 1));
    this.slice(volume, sweep.width, sweep.at(f), {
      nearest: true, wrap: sweep.loop, wrapX: sweep.sides, fade,
      noise: noise ? { ...noise, f } : undefined,
    });
    const canvas = this.gl.canvas;
    if (canvas.width !== sweep.width || canvas.height !== volume.height) {
      canvas.width = sweep.width;
      canvas.height = volume.height;
    }
    this.slicer.draw();
    return { ...info, frame: f, line: endpoints(sweep, f) };
  }

  /**
   * The first frame of the clip held, as faded for the sweep last shown, or null until a sweep of
   * it has been shown. It is the same ImageData for as long as the clip and its fades stay.
   */
  face(): ImageData | null {
    const { clip, sweeping } = this;
    if (!clip || sweeping?.clip !== clip || this.gl.isContextLost()) return null;
    const { fade } = sweeping;
    if (clip.face?.fade[0] !== fade[0] || clip.face.fade[1] !== fade[1]) {
      const { volume } = clip, width = volume.width - fade[1];
      this.slice(volume, width, { origin: [0, 0], step: [0, 1] }, { fade });
      const pixels = new Uint8ClampedArray(this.slicer.read().buffer as ArrayBuffer);
      clip.face = { fade, image: new ImageData(pixels, width, volume.height) };
    }
    return clip.face.image;
  }

  /**
   * Take `to`, the same file opened again, in place of `from` for the clip held, so it serves
   * `to` without loading afresh.
   */
  adopt(from: File, to: File): void {
    if (this.clip?.file === from) this.clip.file = to;
  }

  /**
   * Stop a load under way that these option values don't want, so the next show() can start
   * theirs at once. Values it can't read leave the load be, for show() to say what's wrong.
   */
  supersede(values: OptionValues): void {
    if (!this.loading) return;
    let key: string;
    try {
      key = loadKey(readOptions(values));
    } catch {
      return;
    }
    if (key !== this.loadingKey) this.loading.abort();
  }

  /** Let go of the clip and its file, stopping any load under way. */
  unload(): void {
    this.loading?.abort();
    this.loading = null;
    this.release();
  }

  dispose(): void {
    this.unload();
    this.listening.abort();
    this.slicer.dispose();
  }

  private release(): void {
    this.clip?.volume.dispose();
    this.clip = null;
    this.sweeping = null;
  }

  /** Slice a frame, or throw the Problem of one larger than the GPU can draw. */
  private slice(volume: Volume, width: number, line: Line, options: SliceOptions): void {
    try {
      this.slicer.slice(volume, width, line, options);
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
    const { fade, sweep } = planFaded(volume.frames, volume.width, clip.fps.num / clip.fps.den,
                                      opts, noise, PREVIEW_SCALE);
    const surface = noise && looping(noise, sweep);
    const fps = opts.fps ? { num: opts.fps.num * SHRINK.num, den: opts.fps.den * SHRINK.den }
                         : clip.fps;  // already the rate of the frames kept
    return {
      opts, clip, sweep, fade,
      noise: surface && { surface, push: surface.push(sweep.normal) },
      info: {
        width: sweep.width, height: volume.height, frames: sweep.frames,
        fps: fps.num / fps.den, seconds: sweep.frames * fps.den / fps.num,
        volume: [volume.frames - fade[0], volume.height, volume.width - fade[1]],
        memory: volume.bytes,
        loop: sweep.loop, sides: sweep.sides,
        first: endpoints(sweep, 0), last: endpoints(sweep, sweep.frames - 1),
      },
    };
  }

  /** The clip for these options, loading it if it changed. */
  private async load(file: File, opts: Options,
                     progress?: (done: number, total: number) => void): Promise<Clip> {
    if (this.gl.isContextLost()) throw new Problem(LOST);
    const key = loadKey(opts);
    if (this.clip?.file === file && this.clip.key === key) return this.clip;
    const controller = new AbortController();
    this.loading = controller;
    this.loadingKey = key;
    const { signal } = controller;
    try {
      const options: LoadOptions = {
        scale: opts.scale * PREVIEW_SCALE, timeScale: PREVIEW_SCALE,
        start: opts.start ?? undefined, duration: opts.duration ?? undefined, fast: true, signal,
      };
      const planned = await planLoad(file, options);
      signal.throwIfAborted();
      const volume = this.allocate(planned);
      try {
        await load(file, options, (i, frame) => {
          volume.upload(i, frame);
          frame.close();
        }, progress);
        signal.throwIfAborted();
        // The context can be lost before its event arrives, and uploads to a lost one do nothing.
        if (this.gl.isContextLost()) throw new Problem(LOST);
      } catch (error) {
        volume.dispose();
        throw error;
      }
      this.clip = { file, key, volume, fps: planned.fps };
      return this.clip;
    } catch (error) {
      if (signal.aborted) throw signal.reason;  // whatever stopped it on the way
      if (!(error instanceof Problem) && await hasChanged(file))
        throw new Problem(changed(file.name));
      if (error instanceof VideoError) throw new Problem(error.message);
      throw error;
    } finally {
      if (this.loading === controller) this.loading = null;
    }
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
      return Volume.create(gl, frames, height, width);
    } catch (error) {
      if (!(error instanceof VolumeTooLarge)) throw error;
      throw new Problem(`The live view would need ${(error.bytes / 1e9).toFixed(1)} GB of GPU `
                        + "memory for this clip, more than the GPU can give it. Set a shorter "
                        + "duration or a smaller scale.");
    }
  }
}
