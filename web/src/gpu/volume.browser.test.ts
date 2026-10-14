import { afterAll, describe, expect, onTestFinished, test } from "vitest";
import { type Clip, noisyClip, readLayer, rgba, upload } from "../../test/gpu";
import { Volume, VolumeTooLarge } from "./volume";

const gl = document.createElement("canvas").getContext("webgl2")!;
afterAll(() => gl.getExtension("WEBGL_lose_context")?.loseContext());

function thrown(fn: () => unknown): unknown {
  try { fn(); } catch (error) { return error; }
  return undefined;
}

const half = (n: number) => Math.ceil(n / 2);
const level = (v: number) => Math.min(255, Math.max(0, Math.floor(v + 0.5)));
const yuvBytes = (frames: number, height: number, width: number) =>
  frames * (height * width + 2 * half(height) * half(width));

/** A frame's planes: luma, then chroma as Cb and Cr in turn, rows top-first. */
interface Planes { luma: Uint8Array; chroma: Uint8Array }

/**
 * Frame t of a clip in YUV 4:2:0, worked out in float64: full-range BT.601 luma, and the
 * chroma of each 2×2 block's mean colour, over the pixels the block has at an odd edge.
 */
function yuv420(clip: Clip, t: number): Planes {
  const { height, width } = clip;
  const at = (y: number, x: number, c: number) =>
    clip.rgb[((t * height + y) * width + x) * 3 + c];
  const luma = new Uint8Array(height * width);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++)
      luma[y * width + x] = level(0.299 * at(y, x, 0) + 0.587 * at(y, x, 1)
                                  + 0.114 * at(y, x, 2));
  const chroma = new Uint8Array(half(height) * half(width) * 2);
  for (let i = 0; i < half(height); i++)
    for (let j = 0; j < half(width); j++) {
      const ys = [...new Set([2 * i, Math.min(2 * i + 1, height - 1)])];
      const xs = [...new Set([2 * j, Math.min(2 * j + 1, width - 1)])];
      const mean = [0, 1, 2].map((c) => ys.flatMap((y) => xs.map((x) => at(y, x, c)))
        .reduce((a, b) => a + b) / (ys.length * xs.length));
      const y = 0.299 * mean[0] + 0.587 * mean[1] + 0.114 * mean[2];
      chroma.set([level((mean[2] - y) / 1.772 + 128), level((mean[0] - y) / 1.402 + 128)],
                 2 * (i * half(width) + j));
    }
  return { luma, chroma };
}

/** Frame t of a volume's planes. */
function readPlanes(volume: Volume, t: number): Planes {
  const [luma, chroma] = volume.planes;
  const y = readLayer(gl, luma.texture, t, luma.width, luma.height);
  const c = readLayer(gl, chroma.texture, t, chroma.width, chroma.height);
  return {
    luma: y.filter((_, i) => i % 4 === 0),
    chroma: c.filter((_, i) => i % 4 < 2),
  };
}

/** How far apart two planes are, at most, and the share of levels the same. */
function compare(got: Uint8Array, expected: Uint8Array): { maxDiff: number; same: number } {
  expect(got.length).toBe(expected.length);
  let maxDiff = 0, same = 0;
  got.forEach((v, i) => {
    maxDiff = Math.max(maxDiff, Math.abs(v - expected[i]));
    if (v === expected[i]) same++;
  });
  return { maxDiff, same: same / got.length };
}

describe("Volume", () => {
  test("takes 1.5 bytes a voxel, rounding chroma up where a size is odd", () => {
    for (const [frames, height, width] of [[3, 4, 6], [2, 5, 7], [1, 1, 1]]) {
      const volume = Volume.create(gl, frames, height, width);
      expect([volume.frames, volume.height, volume.width]).toEqual([frames, height, width]);
      expect(volume.planes.map(({ width, height }) => [width, height]))
        .toEqual([[width, height], [half(width), half(height)]]);
      expect(volume.bytes).toBe(yuvBytes(frames, height, width));
      volume.dispose();
    }
    const volume = Volume.create(gl, 3, 4, 6);
    expect(volume.bytes / (3 * 4 * 6)).toBe(1.5);
    volume.dispose();
    expect(gl.getError()).toBe(gl.NO_ERROR);
  });

  test("holds full-range BT.601 luma, and chroma averaged over 2×2 blocks", () => {
    for (const clip of [noisyClip(7, 5, 11), noisyClip(13, 3, 5), noisyClip(4, 4, 300)]) {
      const volume = upload(gl, clip);  // odd sizes, and even
      for (let t = 0; t < clip.frames; t++) {
        const got = readPlanes(volume, t), expected = yuv420(clip, t);
        // The GPU works in float32, so a level that is a half in float64 may round either way.
        for (const plane of ["luma", "chroma"] as const) {
          const { maxDiff, same } = compare(got[plane], expected[plane]);
          expect(maxDiff, `${plane} of frame ${t}`).toBeLessThanOrEqual(1);
          expect(same, `${plane} of frame ${t}`).toBeGreaterThanOrEqual(0.98);
        }
      }
      volume.dispose();
    }
    expect(gl.getError()).toBe(gl.NO_ERROR);
  });

  test("takes pixels as they are, whatever the unpack state", async () => {
    const clip = noisyClip(7, 5, 11);
    const pixels = rgba(clip, 3);  // random, so a flip would show
    for (let p = 0; p < pixels.length / 4; p += 3) pixels[4 * p + 3] = 96;  // premultiplying
    const image = new ImageData(new Uint8ClampedArray(pixels), clip.width, clip.height);
    const bitmap = await createImageBitmap(image, { premultiplyAlpha: "none" });
    const volume = Volume.create(gl, 3, clip.height, clip.width);
    for (let round = 0; round < 2; round++) {  // so that no upload comes first
      [pixels, image, bitmap].forEach((source, t) => {
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
        volume.upload(t, source);
      });
    }
    bitmap.close();
    const expected = yuv420(clip, 3);
    for (let t = 0; t < 3; t++) {
      const got = readPlanes(volume, t);
      for (const plane of ["luma", "chroma"] as const)
        expect(compare(got[plane], expected[plane]).maxDiff, `${plane} of frame ${t}`)
          .toBeLessThanOrEqual(1);
    }
    volume.dispose();
  });

  test("needs at least one voxel each way, a whole number of them", () => {
    for (const [frames, height, width] of [[0, 2, 2], [2, 0, 2], [2, 2, 0], [2, 1.5, 2]])
      expect(() => Volume.create(gl, frames, height, width)).toThrow(RangeError);
  });

  test("refuses more frames or pixels than the GPU's limits", () => {
    const layers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
    const size = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    const pixels = (width: number, height: number) =>
      `frames of ${width} × ${height} pixels are more than the GPU's limit of ${size} each way`;
    const oversize: [number, number, number, string][] = [
      [layers + 1, 2, 2,
       `a volume of ${layers + 1} frames is more than the GPU's limit of ${layers}`],
      [2, size + 1, 2, pixels(2, size + 1)],
      [2, 2, size + 1, pixels(size + 1, 2)],
    ];
    for (const [frames, height, width, message] of oversize) {
      const error = thrown(() => Volume.create(gl, frames, height, width));
      expect(error).toBeInstanceOf(RangeError);
      expect((error as Error).message).toBe(message);
    }
  });

  test("is too large when the GPU can't allocate it", () => {
    // A context of its own, since a driver may lose it rather than refuse the allocation.
    const own = document.createElement("canvas").getContext("webgl2")!;
    onTestFinished(() => own.getExtension("WEBGL_lose_context")?.loseContext());
    const layers = own.getParameter(own.MAX_ARRAY_TEXTURE_LAYERS) as number;
    const size = own.getParameter(own.MAX_TEXTURE_SIZE) as number;
    const error = thrown(() => Volume.create(own, layers, size, size));
    expect(error).toBeInstanceOf(VolumeTooLarge);
    expect((error as VolumeTooLarge).bytes).toBe(yuvBytes(layers, size, size));
    expect((error as Error).message)
      .toContain(`${(yuvBytes(layers, size, size) / 1e9).toFixed(2)} GB`);
    expect(own.isContextLost()).toBe(false);
    expect(own.getError()).toBe(own.NO_ERROR);
  });
});
