/**
 * The sampler fixture (web/test/fixtures/sampler.py) decoded, and helpers for the GPU
 * tests that check against it.
 */
import raw from "../../test/fixtures/sampler.json?raw";
import type { Columns } from "../types";
import { Volume } from "./volume";

/** A clip as numpy holds it: (t, y, x) voxels of 3 channels. */
export interface Clip { frames: number; height: number; width: number; rgb: Uint8Array }

/** An output frame of a sweep: its columns, and what slice_frame makes of them. */
export interface SliceFrame {
  f: number; columns: Columns; bilinear: Uint8Array; nearest: Uint8Array;
}

export interface SliceCase {
  name: string; clip: Clip; wrap: boolean; wrapX: boolean; width: number; frames: SliceFrame[];
}

interface RawFrame { f: number; t: string; x: string; bilinear: string; nearest: string }
interface RawFixture {
  volumes: Record<string, { shape: [number, number, number]; rgb?: string }>;
  slices: { name: string; volume: string; wrap: boolean; wrapX: boolean; width: number;
            frames: RawFrame[] }[];
}

function bytes(base64: string): Uint8Array {
  const text = atob(base64);
  const out = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) out[i] = text.charCodeAt(i);
  return out;
}

const float64 = (base64: string) => new Float64Array(bytes(base64).buffer);

/** The fixture's large clip, whose voxels are (t*131 + y*71 + x*37 + c*17 + (t*x) % 23) % 256. */
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

export const sliceCases: SliceCase[] = fixture.slices.map((c) => ({
  name: c.name, clip: clips[c.volume], wrap: c.wrap, wrapX: c.wrapX, width: c.width,
  frames: c.frames.map((frame) => ({
    f: frame.f,
    columns: { t: float64(frame.t), x: float64(frame.x) },
    bilinear: bytes(frame.bilinear),
    nearest: bytes(frame.nearest),
  })),
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

/** A volume holding the clip. */
export function upload(gl: WebGL2RenderingContext, clip: Clip): Volume {
  const volume = Volume.create(gl, clip.frames, clip.height, clip.width);
  for (let t = 0; t < clip.frames; t++) volume.upload(t, rgba(clip, t));
  return volume;
}

/** Frame t of a volume, as many columns as it has now, as RGBA rows top-first. */
export function readFrame(gl: WebGL2RenderingContext, volume: Volume, t: number): Uint8Array {
  const framebuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, volume.texture, 0, t);
  const out = new Uint8Array(volume.width * volume.height * 4);
  gl.readPixels(0, 0, volume.width, volume.height, gl.RGBA, gl.UNSIGNED_BYTE, out);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.deleteFramebuffer(framebuffer);
  return out;
}

const percent = (part: number, whole: number) => `${+(100 * part / whole).toFixed(3)}%`;

/** How closely RGBA output matches numpy's RGB, over all the frames added to it. */
export class Tally {
  channels = 0; identical = 0; withinOne = 0;
  pixels = 0; pixelsIdentical = 0;
  maxDiff = 0;
  translucent = 0;  // pixels whose alpha isn't 255

  add(rgbaOut: Uint8Array, rgb: Uint8Array): void {
    if (rgbaOut.length / 4 !== rgb.length / 3)
      throw new Error(`${rgbaOut.length / 4} pixels, expected ${rgb.length / 3}`);
    for (let p = 0; p < rgb.length / 3; p++) {
      let same = true;
      for (let c = 0; c < 3; c++) {
        const diff = Math.abs(rgbaOut[p * 4 + c] - rgb[p * 3 + c]);
        this.channels++;
        if (diff === 0) this.identical++;
        else same = false;
        if (diff <= 1) this.withinOne++;
        this.maxDiff = Math.max(this.maxDiff, diff);
      }
      this.pixels++;
      if (same) this.pixelsIdentical++;
      if (rgbaOut[p * 4 + 3] !== 255) this.translucent++;
    }
  }

  get summary(): string {
    return `${percent(this.identical, this.channels)} of ${this.channels} channels identical, ` +
      `${percent(this.withinOne, this.channels)} within 1, ` +
      `${percent(this.pixelsIdentical, this.pixels)} of pixels identical, max diff ${this.maxDiff}`;
  }
}
