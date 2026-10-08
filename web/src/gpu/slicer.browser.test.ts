import { afterAll, describe, expect, test } from "vitest";
import { clips, rgba, sliceCases, Tally, upload } from "./fixture";
import { Slicer } from "./slicer";
import { Volume } from "./volume";

const canvas = document.createElement("canvas");
const gl = canvas.getContext("webgl2")!;
const slicer = new Slicer(gl);
afterAll(() => {
  slicer.dispose();
  gl.getExtension("WEBGL_lose_context")?.loseContext();
});

describe("flat slices match timeslice.slice_frame", () => {
  for (const c of sliceCases) {
    test(c.name, () => {
      const volume = upload(gl, c.clip);
      const nearest = new Tally(), bilinear = new Tally();
      for (const frame of c.frames) {
        slicer.slice(volume, frame.columns, { nearest: true, wrap: c.wrap, wrapX: c.wrapX });
        nearest.add(slicer.read(), frame.nearest);
        slicer.slice(volume, frame.columns, { wrap: c.wrap, wrapX: c.wrapX });
        bilinear.add(slicer.read(), frame.bilinear);
      }
      volume.dispose();
      console.log(`${c.name}\n  nearest: ${nearest.summary}\n  bilinear: ${bilinear.summary}`);
      expect(nearest.identical).toBe(nearest.channels);
      expect(bilinear.maxDiff).toBeLessThanOrEqual(1);
      expect(bilinear.identical / bilinear.channels).toBeGreaterThanOrEqual(0.999);
      expect(nearest.translucent + bilinear.translucent).toBe(0);
    });
  }
});

describe("Slicer", () => {
  test("matches timeslice's float64 blend at random positions", () => {
    let seed = 1;
    const random = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 2 ** 32;
    const [frames, height, width, n] = [20, 64, 256, 1000];
    const voxels = Uint8Array.from({ length: frames * height * width * 4 },
                                   (_, i) => i % 4 === 3 ? 255 : random() * 256);
    const volume = Volume.create(gl, frames, height, width);
    const layer = height * width * 4;
    for (let t = 0; t < frames; t++) volume.upload(t, voxels.subarray(t * layer, (t + 1) * layer));
    const tally = new Tally();
    for (let rep = 0; rep < 4; rep++) {
      const t = Float64Array.from({ length: n }, () => random() * (frames - 1));
      const x = Float64Array.from({ length: n }, () => random() * (width - 1));
      const expected = new Uint8Array(height * n * 3);
      for (let j = 0; j < n; j++) {
        const t0 = Math.floor(t[j]), x0 = Math.floor(x[j]), ft = t[j] - t0, fx = x[j] - x0;
        for (let row = 0; row < height; row++)
          for (let c = 0; c < 3; c++) {
            const at = (dt: number, dx: number) =>
              voxels[(((t0 + dt) * height + row) * width + x0 + dx) * 4 + c];
            const v = ((at(0, 0) * (1 - fx) + at(0, 1) * fx) * (1 - ft)
                       + (at(1, 0) * (1 - fx) + at(1, 1) * fx) * ft);  // as _sample_columns
            expected[(row * n + j) * 3 + c] = Math.min(255, Math.trunc(v + 0.5));
          }
      }
      slicer.slice(volume, { t, x });
      tally.add(slicer.read(), expected);
    }
    volume.dispose();
    console.log(`random positions\n  bilinear: ${tally.summary}`);
    expect(tally.identical).toBe(tally.channels);
  });

  test("a 0° slice reproduces the volume", () => {
    const clip = clips.big;
    const volume = upload(gl, clip);
    const x = Float64Array.from({ length: clip.width }, (_, i) => i);
    for (const f of [0, 57, clip.frames - 1]) {
      const columns = { t: new Float64Array(clip.width).fill(f), x };
      for (const nearest of [false, true]) {
        slicer.slice(volume, columns, { nearest });
        expect(slicer.read()).toEqual(rgba(clip, f));
      }
    }
    volume.dispose();
  });

  test("makes a frame as wide as the columns and as high as the volume", () => {
    const volume = upload(gl, clips.small);
    slicer.slice(volume, { t: Float64Array.of(1, 2, 3), x: Float64Array.of(4, 5, 6) });
    expect(slicer.read().length).toBe(3 * 5 * 4);
    volume.dispose();
  });

  test("needs an x for each t, and a frame no larger than the GPU can draw", () => {
    const volume = upload(gl, clips.small);
    const short = () =>
      slicer.slice(volume, { t: Float64Array.of(1, 2, 3), x: Float64Array.of(4, 5) });
    expect(short).toThrow(RangeError);
    expect(short).toThrow("columns need an x for each t, not 2 for 3");
    const size = Math.min(gl.getParameter(gl.MAX_TEXTURE_SIZE) as number,
                          ...(gl.getParameter(gl.MAX_VIEWPORT_DIMS) as Int32Array));
    const columns = new Float64Array(size + 1);
    const wide = () => slicer.slice(volume, { t: columns, x: columns });
    expect(wide).toThrow(RangeError);
    expect(wide).toThrow(`a slice of ${size + 1} × 5 pixels is more than the GPU's limit of `
                         + `${size} each way`);
    volume.dispose();
  });

  test("draws the last slice upright, leaving no GL error", () => {
    const clip = clips.small;
    const volume = upload(gl, clip);
    const x = Float64Array.from({ length: clip.width }, (_, i) => i);
    slicer.slice(volume, { t: new Float64Array(clip.width).fill(3), x });
    canvas.width = clip.width;
    canvas.height = clip.height;
    slicer.draw();
    expect(gl.getError()).toBe(gl.NO_ERROR);
    const drawn = new Uint8Array(clip.width * clip.height * 4);
    gl.readPixels(0, 0, clip.width, clip.height, gl.RGBA, gl.UNSIGNED_BYTE, drawn);
    // The canvas's bottom row is its last.
    const frame = rgba(clip, 3), row = clip.width * 4;
    for (let y = 0; y < clip.height; y++)
      expect(drawn.subarray(y * row, (y + 1) * row))
        .toEqual(frame.subarray((clip.height - 1 - y) * row, (clip.height - y) * row));
    volume.dispose();
  });
});
