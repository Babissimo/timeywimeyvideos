import { afterAll, describe, expect, onTestFinished, test } from "vitest";
import { clips, readFrame, rgba, upload } from "./fixture";
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
