/** Clips made up for the GPU tests, and ways to put them on the GPU and read them back. */
import { Volume } from "../src/gpu/volume";

/** A clip as RGB voxels, 3 bytes each, t then y then x. */
export interface Clip { frames: number; height: number; width: number; rgb: Uint8Array }

/** A clip of colours made by colour(t, y, x). */
function clipOf(frames: number, height: number, width: number,
                colour: (t: number, y: number, x: number) => number[]): Clip {
  const rgb = new Uint8Array(frames * height * width * 3);
  let i = 0;
  for (let t = 0; t < frames; t++)
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        rgb.set(colour(t, y, x), i);
        i += 3;
      }
  return { frames, height, width, rgb };
}

/** A clip of greys, whose voxel (t, y, x) has the level level(t, x), whatever its row. */
export function greyClip(frames: number, height: number, width: number,
                         level: (t: number, x: number) => number): Clip {
  return clipOf(frames, height, width, (t, _, x) => Array(3).fill(level(t, x)));
}

/** Random levels, the same for the same arguments. */
function random(...seeds: number[]): () => number {
  let seed = seeds.reduce((a, b) => Math.imul(a, 7919) + b, 1);
  return () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 24;
}

/** A clip of random colours. */
export function noisyClip(frames: number, height: number, width: number): Clip {
  const next = random(frames, height, width);
  return clipOf(frames, height, width, () => [next(), next(), next()]);
}

/** A clip whose every 2×2 block of pixels, from the top left, is one random colour. */
export function blockClip(frames: number, height: number, width: number): Clip {
  const next = random(frames, height, width, 2);
  const [rows, columns] = [Math.ceil(height / 2), Math.ceil(width / 2)];
  const blocks = Array.from({ length: frames * rows * columns }, () => [next(), next(), next()]);
  return clipOf(frames, height, width,
                (t, y, x) => blocks[(t * rows + (y >> 1)) * columns + (x >> 1)]);
}

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
