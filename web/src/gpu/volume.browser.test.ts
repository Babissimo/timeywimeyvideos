import { afterAll, describe, expect, onTestFinished, test, vi } from "vitest";
import { blockClip, type Clip, clips, crossfadeCases, readFrame, readLayer, rgba, Tally,
         upload } from "./fixture";
import { Volume, VolumeTooLarge } from "./volume";

const gl = document.createElement("canvas").getContext("webgl2")!;
afterAll(() => gl.getExtension("WEBGL_lose_context")?.loseContext());

function thrown(fn: () => unknown): unknown {
  try { fn(); } catch (error) { return error; }
  return undefined;
}

describe("Volume", () => {
  test("holds a clip's frames, rows top-first", () => {
    const clip = clips.small;
    const volume = upload(gl, clip);
    expect([volume.frames, volume.height, volume.width]).toEqual([7, 5, 11]);
    expect(volume.bytes).toBe(7 * 5 * 11 * 4);
    for (let t = 0; t < clip.frames; t++) expect(readFrame(gl, volume, t)).toEqual(rgba(clip, t));
    volume.dispose();
    expect(gl.getError()).toBe(gl.NO_ERROR);
  });

  test("takes an image as well as bytes", () => {
    const clip = clips.long;
    const volume = Volume.create(gl, 1, clip.height, clip.width);
    const pixels = rgba(clip, 4);
    volume.upload(0, new ImageData(new Uint8ClampedArray(pixels), clip.width, clip.height));
    expect(readFrame(gl, volume, 0)).toEqual(pixels);
    volume.dispose();
  });

  test("takes an image's pixels as they are, whatever the unpack state", async () => {
    const clip = clips.long;
    const pixels = rgba(clip, 4);
    pixels[3] = 128;  // a translucent pixel, which premultiplying would darken
    const image = new ImageData(new Uint8ClampedArray(pixels), clip.width, clip.height);
    const bitmap = await createImageBitmap(new ImageData(new Uint8ClampedArray(rgba(clip, 5)),
                                                         clip.width, clip.height));
    const volume = Volume.create(gl, 2, clip.height, clip.width);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    volume.upload(0, image);
    volume.upload(1, bitmap);
    bitmap.close();
    expect(readFrame(gl, volume, 0)).toEqual(pixels);
    expect(readFrame(gl, volume, 1)).toEqual(rgba(clip, 5));
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
    expect((error as Error).message).toContain(`${(layers * size * size * 4 / 1e9).toFixed(2)} GB`);
    expect(own.isContextLost()).toBe(false);
    expect(own.getError()).toBe(own.NO_ERROR);
  });
});

describe("crossfade matches timeslice.crossfade", () => {
  for (const { name, clip, n, axis, faded } of crossfadeCases) {
    test(name, () => {
      const volume = upload(gl, clip);
      volume.crossfade(n, axis);
      const frames = axis === 0 ? clip.frames - n : clip.frames;
      const width = axis === 2 ? clip.width - n : clip.width;
      expect([volume.frames, volume.height, volume.width]).toEqual([frames, clip.height, width]);
      const blended = new Tally(), kept = new Tally();
      for (let t = 0; t < frames; t++) {
        const out = readFrame(gl, volume, t);
        for (let y = 0; y < clip.height; y++)
          for (let x = 0; x < width; x++) {
            const pixel = out.subarray(4 * (y * width + x), 4 * (y * width + x + 1));
            if (axis === 0 ? t < n : x < n) {
              // faded is (n, H, W) or (T, H, n)
              const q = (t * clip.height + y) * (axis === 0 ? clip.width : n) + x;
              blended.add(pixel, faded.subarray(3 * q, 3 * q + 3));
            } else {
              const q = (t * clip.height + y) * clip.width + x;
              kept.add(pixel, clip.rgb.subarray(3 * q, 3 * q + 3));
            }
          }
      }
      volume.dispose();
      console.log(`${name}\n  blended: ${blended.summary}\n  kept: ${kept.summary}`);
      expect(blended.maxDiff).toBeLessThanOrEqual(1);
      expect(kept.identical).toBe(kept.channels);
      expect(blended.translucent + kept.translucent).toBe(0);
    });
  }

  test("takes a whole number of frames or columns", () => {
    const volume = upload(gl, clips.long);
    const half = () => volume.crossfade(2.5, 0);
    expect(half).toThrow(RangeError);
    expect(half).toThrow("a fade has to be a whole number of frames, not 2.5");
    expect(() => volume.crossfade(0.5, 2))
      .toThrow("a fade has to be a whole number of columns, not 0.5");
    expect([volume.frames, volume.width]).toEqual([13, 5]);
    volume.dispose();
  });

  test("fades through memory, not a pixel buffer", () => {
    // Chrome here has uploaded from a pixel buffer before a read into it landed. That shows
    // only on the first fade of a session ("tall columns" above, run on its own), so this
    // holds the fade to the route that can't race.
    const pixelBuffers: GLenum[] = [gl.PIXEL_PACK_BUFFER, gl.PIXEL_UNPACK_BUFFER];
    for (const format of ["rgba", "yuv420"] as const) {
      const volume = upload(gl, clips.small, format);
      const bindBuffer = vi.spyOn(gl, "bindBuffer");
      volume.crossfade(2, 0);
      volume.crossfade(3, 2);
      expect(bindBuffer.mock.calls.filter(([target]) => pixelBuffers.includes(target)), format)
        .toEqual([]);
      bindBuffer.mockRestore();
      volume.dispose();
    }
  });

  test("leaves no GL error, and uploads working as before", () => {
    const clip = clips.small;
    const volume = upload(gl, clip);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    volume.crossfade(2, 0);
    expect(gl.getError()).toBe(gl.NO_ERROR);
    volume.upload(4, rgba(clip, 0));
    expect(readFrame(gl, volume, 4)).toEqual(rgba(clip, 0));
    volume.dispose();
  });

  test("takes at most half the clip", () => {
    const volume = upload(gl, clips.long);
    expect(() => volume.crossfade(7, 0))
      .toThrow("a fade of 7 doesn't fit in a clip 13 long: it can be at most half");
    expect(() => volume.crossfade(3, 2))
      .toThrow("a fade of 3 doesn't fit in a clip 5 long: it can be at most half");
    expect(() => volume.crossfade(-1, 0))
      .toThrow("a fade of -1 doesn't fit in a clip 13 long: it can be at most half");
    volume.crossfade(0, 0);
    expect([volume.frames, volume.width]).toEqual([13, 5]);
    for (let t = 0; t < 13; t++) expect(readFrame(gl, volume, t)).toEqual(rgba(clips.long, t));
    volume.dispose();
  });
});

const YUV = { format: "yuv420" } as const;
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

/** Frame t of a YUV 4:2:0 volume's planes, all the columns it stores. */
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

describe("Volume in YUV 4:2:0", () => {
  test("takes 1.5 bytes a voxel, rounding chroma up where a size is odd", () => {
    for (const [frames, height, width] of [[3, 4, 6], [2, 5, 7], [1, 1, 1]]) {
      const volume = Volume.create(gl, frames, height, width, YUV);
      expect(volume.format).toBe("yuv420");
      expect([volume.frames, volume.height, volume.width]).toEqual([frames, height, width]);
      expect(volume.planes.map(({ width, height }) => [width, height]))
        .toEqual([[width, height], [half(width), half(height)]]);
      expect(volume.bytes).toBe(yuvBytes(frames, height, width));
      volume.dispose();
    }
    const volume = Volume.create(gl, 3, 4, 6, YUV);
    expect(volume.bytes / (3 * 4 * 6)).toBe(1.5);
    volume.dispose();
    const plain = Volume.create(gl, 1, 1, 1);
    expect(plain.format).toBe("rgba");
    plain.dispose();
    expect(gl.getError()).toBe(gl.NO_ERROR);
  });

  test("holds full-range BT.601 luma, and chroma averaged over 2×2 blocks", () => {
    for (const clip of [clips.small, clips.long, clips.big]) {  // odd sizes, and even
      const volume = upload(gl, clip, "yuv420");
      for (let t = 0; t < Math.min(clip.frames, 20); t++) {
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
    const clip = clips.small;
    const pixels = rgba(clip, 3);  // random, so a flip would show
    for (let p = 0; p < pixels.length / 4; p += 3) pixels[4 * p + 3] = 96;  // premultiplying
    const image = new ImageData(new Uint8ClampedArray(pixels), clip.width, clip.height);
    const bitmap = await createImageBitmap(image, { premultiplyAlpha: "none" });
    const volume = Volume.create(gl, 3, clip.height, clip.width, YUV);
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

  test("refuses more frames or pixels than the GPU's limits", () => {
    const layers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
    const size = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    for (const [frames, height, width] of [[layers + 1, 2, 2], [2, size + 1, 2], [2, 2, size + 1]])
      expect(() => Volume.create(gl, frames, height, width, YUV)).toThrow(RangeError);
  });

  test("is too large when the GPU can't allocate it", () => {
    // A context of its own, since a driver may lose it rather than refuse the allocation.
    const own = document.createElement("canvas").getContext("webgl2")!;
    onTestFinished(() => own.getExtension("WEBGL_lose_context")?.loseContext());
    const layers = own.getParameter(own.MAX_ARRAY_TEXTURE_LAYERS) as number;
    const size = own.getParameter(own.MAX_TEXTURE_SIZE) as number;
    const error = thrown(() => Volume.create(own, layers, size, size, YUV));
    expect(error).toBeInstanceOf(VolumeTooLarge);
    expect((error as VolumeTooLarge).bytes).toBe(yuvBytes(layers, size, size));
    expect(own.isContextLost()).toBe(false);
    expect(own.getError()).toBe(own.NO_ERROR);
  });
});

/**
 * What a crossfade makes of a YUV 4:2:0 volume's planes, worked out in float64: luma as an
 * RGBA volume's channels, and each chroma column the mean, over the picture's columns it
 * covers, of the chroma each takes, blended where that column is faded and kept where not.
 */
function fadePlanes(before: Planes[], width: number, n: number, axis: 0 | 2): Planes[] {
  const length = (axis === 0 ? before.length : width) - n;
  const w = (i: number) => Math.fround((i + 1) / (n + 1));
  const mix = (a: number, b: number, i: number) => a * w(i) + b * Math.fround(1 - w(i));
  const cw = half(width);
  if (axis === 0) {
    return before.slice(0, length).map((planes, t) => t >= n ? planes : {
      luma: planes.luma.map((v, i) => level(mix(v, before[t + length].luma[i], t))),
      chroma: planes.chroma.map((v, i) => level(mix(v, before[t + length].chroma[i], t))),
    });
  }
  return before.map(({ luma, chroma }) => ({
    luma: luma.map((v, i) => i % width < n ? level(mix(v, luma[i + length], i % width)) : v),
    chroma: chroma.map((v, i) => {
      const j = (i >> 1) % cw, row = Math.floor((i >> 1) / cw), k = i & 1;
      if (2 * j >= n) return v;
      const xs = [2 * j, 2 * j + 1].filter((x) => x < length);
      const sum = xs.reduce((s, x) =>
        s + (x < n ? mix(v, chroma[2 * (row * cw + ((x + length) >> 1)) + k], x) : v), 0);
      return level(sum / xs.length);
    }),
  }));
}

describe("crossfade in YUV 4:2:0 fades luma and chroma alike", () => {
  const cases: [string, Clip, number, 0 | 2][] = [
    ...crossfadeCases.map(({ name, clip, n, axis }) =>
      [name, clip, n, axis] as [string, Clip, number, 0 | 2]),
    ["small, 3 of 11 columns", clips.small, 3, 2],  // an odd fade, an even remainder
    ["small, 4 of 11 columns", clips.small, 4, 2],  // an even fade, an odd remainder
    ["long, 1 of 5 columns", clips.long, 1, 2],
    ["big, 11 of 300 columns", clips.big, 11, 2],   // an odd fade, an odd remainder
    ["3 of 6 columns", blockClip(3, 3, 6), 3, 2],   // the last chroma column half kept
  ];
  for (const [name, clip, n, axis] of cases) {
    test(name, () => {
      const volume = upload(gl, clip, "yuv420");
      const before = Array.from({ length: clip.frames }, (_, t) => readPlanes(volume, t));
      volume.crossfade(n, axis);
      const frames = axis === 0 ? clip.frames - n : clip.frames;
      const width = axis === 2 ? clip.width - n : clip.width;
      expect([volume.frames, volume.height, volume.width]).toEqual([frames, clip.height, width]);
      expect(volume.bytes).toBe(yuvBytes(clip.frames, clip.height, clip.width));
      const expected = fadePlanes(before, clip.width, n, axis);
      for (let t = 0; t < frames; t++) {
        const got = readPlanes(volume, t);
        for (const plane of ["luma", "chroma"] as const)
          expect(compare(got[plane], expected[t][plane]).maxDiff, `${plane} of frame ${t}`)
            .toBeLessThanOrEqual(1);
      }
      volume.dispose();
      expect(gl.getError()).toBe(gl.NO_ERROR);
    });
  }
});
