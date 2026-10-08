import { afterAll, describe, expect, onTestFinished, test, vi } from "vitest";
import { clips, crossfadeCases, readFrame, rgba, Tally, upload } from "./fixture";
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
    const volume = upload(gl, clips.small);
    const bindBuffer = vi.spyOn(gl, "bindBuffer");
    volume.crossfade(3, 2);
    const pixelBuffers: GLenum[] = [gl.PIXEL_PACK_BUFFER, gl.PIXEL_UNPACK_BUFFER];
    expect(bindBuffer.mock.calls.filter(([target]) => pixelBuffers.includes(target))).toEqual([]);
    bindBuffer.mockRestore();
    volume.dispose();
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
