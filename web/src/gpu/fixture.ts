/**
 * The sampler fixture (web/test/fixtures/sampler.py) decoded, and helpers for the GPU
 * tests that check against it.
 */
import raw from "../../test/fixtures/sampler.json?raw";
import type { Columns, NoiseGrid } from "../types";
import { Volume, type VolumeFormat } from "./volume";

/** A clip as numpy holds it: (t, y, x) voxels of 3 channels. */
export interface Clip { frames: number; height: number; width: number; rgb: Uint8Array }

/** An output frame of a sweep: its columns and noise, and what slice_frame makes of them. */
export interface SliceFrame {
  f: number; columns: Columns; noise?: { push: [number, number]; grid: NoiseGrid };
  bilinear: Uint8Array; nearest: Uint8Array;
}

export interface SliceCase {
  name: string; clip: Clip; wrap: boolean; wrapX: boolean; width: number; noisy: boolean;
  fade: [number, number];  // the frames, then the columns, crossfaded before the sweep
  frames: SliceFrame[];
}

/** What timeslice.crossfade makes of a clip: the n slices along the axis it blends. */
export interface CrossfadeCase {
  name: string; clip: Clip; n: number; axis: 0 | 2; faded: Uint8Array;
}

interface RawFrame {
  f: number; t: string; x: string; bilinear: string; nearest: string;
  nodes?: string; nodeRows?: number; nodeCols?: number;
}
interface RawNoise {
  push: [number, number]; cols: string; colW: string; rows: string; rowW: string;
}
interface RawFixture {
  volumes: Record<string, { shape: [number, number, number]; rgb?: string }>;
  slices: { name: string; volume: string; wrap: boolean; wrapX: boolean; width: number;
            fade?: [number, number]; noise?: RawNoise; frames: RawFrame[] }[];
  crossfades: { name: string; volume: string; n: number; axis: 0 | 2; faded: string }[];
}

function bytes(base64: string): Uint8Array {
  const text = atob(base64);
  const out = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) out[i] = text.charCodeAt(i);
  return out;
}

const float64 = (base64: string) => new Float64Array(bytes(base64).buffer);
const int32 = (base64: string) => new Int32Array(bytes(base64).buffer);

/** A fixture clip whose voxels are (t*131 + y*71 + x*37 + c*17 + (t*x) % 23) % 256. */
function formulaClip(frames: number, height: number, width: number): Clip {
  const rgb = new Uint8Array(frames * height * width * 3);
  let i = 0;
  for (let t = 0; t < frames; t++)
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++)
        for (let c = 0; c < 3; c++)
          rgb[i++] = (t * 131 + y * 71 + x * 37 + c * 17 + (t * x) % 23) % 256;
  return { frames, height, width, rgb };
}

const fixture = JSON.parse(raw) as RawFixture;

export const clips: Record<string, Clip> = Object.fromEntries(
  Object.entries(fixture.volumes).map(([name, { shape: [frames, height, width], rgb }]) =>
    [name, rgb ? { frames, height, width, rgb: bytes(rgb) } : formulaClip(frames, height, width)]),
);

function noise(raw: RawNoise, frame: RawFrame): SliceFrame["noise"] {
  return {
    push: raw.push,
    grid: {
      nodes: float64(frame.nodes!), nodeRows: frame.nodeRows!, nodeCols: frame.nodeCols!,
      cols: int32(raw.cols), colW: float64(raw.colW),
      rows: int32(raw.rows), rowW: float64(raw.rowW),
    },
  };
}

export const sliceCases: SliceCase[] = fixture.slices.map((c) => ({
  name: c.name, clip: clips[c.volume], wrap: c.wrap, wrapX: c.wrapX, width: c.width,
  noisy: c.noise !== undefined, fade: c.fade ?? [0, 0],
  frames: c.frames.map((frame) => ({
    f: frame.f,
    columns: { t: float64(frame.t), x: float64(frame.x) },
    noise: c.noise && noise(c.noise, frame),
    bilinear: bytes(frame.bilinear),
    nearest: bytes(frame.nearest),
  })),
}));

export const crossfadeCases: CrossfadeCase[] = fixture.crossfades.map((c) => ({
  name: c.name, clip: clips[c.volume], n: c.n, axis: c.axis, faded: bytes(c.faded),
}));

/** Frame t of a clip as opaque RGBA, rows top-first. */
export function rgba(clip: Clip, t: number): Uint8Array {
  const pixels = clip.height * clip.width;
  const out = new Uint8Array(pixels * 4);
  for (let p = 0; p < pixels; p++) {
    out.set(clip.rgb.subarray((t * pixels + p) * 3, (t * pixels + p + 1) * 3), p * 4);
    out[p * 4 + 3] = 255;
  }
  return out;
}

/** A clip whose every 2×2 block of pixels, from the top left, is one random colour. */
export function blockClip(frames: number, height: number, width: number): Clip {
  let seed = frames * 7919 + height * 104729 + width;
  const random = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 24;
  const [rows, columns] = [Math.ceil(height / 2), Math.ceil(width / 2)];
  const blocks = Array.from({ length: frames * rows * columns * 3 }, random);
  const rgb = new Uint8Array(frames * height * width * 3);
  let i = 0;
  for (let t = 0; t < frames; t++)
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++)
        for (let c = 0; c < 3; c++)
          rgb[i++] = blocks[((t * rows + (y >> 1)) * columns + (x >> 1)) * 3 + c];
  return { frames, height, width, rgb };
}

/** A volume holding the clip. */
export function upload(gl: WebGL2RenderingContext, clip: Clip,
                       format: VolumeFormat = "rgba"): Volume {
  const volume = Volume.create(gl, clip.frames, clip.height, clip.width, { format });
  for (let t = 0; t < clip.frames; t++) volume.upload(t, rgba(clip, t));
  return volume;
}

/** Layer t of a texture array, width × height texels of it, as RGBA rows top-first. */
export function readLayer(gl: WebGL2RenderingContext, texture: WebGLTexture, t: number,
                          width: number, height: number): Uint8Array {
  const framebuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, texture, 0, t);
  const out = new Uint8Array(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, out);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.deleteFramebuffer(framebuffer);
  return out;
}

/** Frame t of an RGBA volume, as many columns as it has now, as RGBA rows top-first. */
export function readFrame(gl: WebGL2RenderingContext, volume: Volume, t: number): Uint8Array {
  return readLayer(gl, volume.planes[0].texture, t, volume.width, volume.height);
}

const percent = (part: number, whole: number) => `${+(100 * part / whole).toFixed(3)}%`;

/** How closely RGBA output matches numpy's RGB, over all the frames added to it. */
export class Tally {
  channels = 0; identical = 0; withinOne = 0;
  pixels = 0; pixelsIdentical = 0;
  maxDiff = 0;
  translucent = 0;  // pixels whose alpha isn't 255
  // Columns of the frames added with their width, and those most of whose pixels are more
  // than 1 off.
  columns = 0; wrongColumns = 0;

  /** Add a frame; given its width, also check it column by column. */
  add(rgbaOut: Uint8Array, rgb: Uint8Array, width?: number): void {
    if (rgbaOut.length / 4 !== rgb.length / 3)
      throw new Error(`${rgbaOut.length / 4} pixels, expected ${rgb.length / 3}`);
    const off = new Uint32Array(width ?? 0);  // each column's pixels more than 1 off
    for (let p = 0; p < rgb.length / 3; p++) {
      let same = true, most = 0;
      for (let c = 0; c < 3; c++) {
        const diff = Math.abs(rgbaOut[p * 4 + c] - rgb[p * 3 + c]);
        this.channels++;
        if (diff === 0) this.identical++;
        else same = false;
        if (diff <= 1) this.withinOne++;
        most = Math.max(most, diff);
      }
      this.maxDiff = Math.max(this.maxDiff, most);
      this.pixels++;
      if (same) this.pixelsIdentical++;
      if (rgbaOut[p * 4 + 3] !== 255) this.translucent++;
      if (width && most > 1) off[p % width]++;
    }
    const rows = width ? rgb.length / 3 / width : 0;
    this.columns += off.length;
    for (const count of off) if (2 * count > rows) this.wrongColumns++;
  }

  get summary(): string {
    return `${percent(this.identical, this.channels)} of ${this.channels} channels identical, ` +
      `${percent(this.withinOne, this.channels)} within 1, ` +
      `${percent(this.pixelsIdentical, this.pixels)} of pixels identical, ` +
      `max diff ${this.maxDiff}` +
      (this.columns ? `, ${this.wrongColumns} of ${this.columns} columns wrong` : "");
  }
}
