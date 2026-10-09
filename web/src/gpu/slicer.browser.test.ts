import { afterAll, describe, expect, test } from "vitest";
import { Noise } from "../noise";
import { blockClip, type Clip, clips, crossfadeCases, readFrame, rgba, type SliceCase,
         sliceCases, smoothClip, Tally, upload } from "./fixture";
import { Slicer } from "./slicer";
import { Volume } from "./volume";

const canvas = document.createElement("canvas");
const gl = canvas.getContext("webgl2")!;
const slicer = new Slicer(gl);
afterAll(() => {
  slicer.dispose();
  gl.getExtension("WEBGL_lose_context")?.loseContext();
});

/** Slice every frame of a case both ways, tallying how closely each matches. */
function tally(c: SliceCase): { nearest: Tally; bilinear: Tally } {
  const volume = upload(gl, c.clip);
  volume.crossfade(c.fade[0], 0);
  volume.crossfade(c.fade[1], 2);
  const nearest = new Tally(), bilinear = new Tally();
  for (const frame of c.frames) {
    const options = { wrap: c.wrap, wrapX: c.wrapX, noise: frame.noise };
    slicer.slice(volume, frame.columns, { ...options, nearest: true });
    nearest.add(slicer.read(), frame.nearest, c.width);
    slicer.slice(volume, frame.columns, options);
    bilinear.add(slicer.read(), frame.bilinear, c.width);
  }
  volume.dispose();
  console.log(`${c.name}\n  nearest: ${nearest.summary}\n  bilinear: ${bilinear.summary}`);
  expect(nearest.translucent + bilinear.translucent).toBe(0);
  return { nearest, bilinear };
}

describe("flat slices match timeslice.slice_frame", () => {
  for (const c of sliceCases.filter((c) => !c.noisy)) {
    test(c.name, () => {
      const { nearest, bilinear } = tally(c);
      expect(nearest.identical).toBe(nearest.channels);
      expect(bilinear.maxDiff).toBeLessThanOrEqual(1);
      expect(bilinear.identical / bilinear.channels).toBeGreaterThanOrEqual(0.999);
    });
  }
});

describe("noisy slices match timeslice.slice_frame", () => {
  for (const c of sliceCases.filter((c) => c.noisy)) {
    test(c.name, () => {
      const { nearest, bilinear } = tally(c);
      expect(nearest.pixelsIdentical / nearest.pixels).toBeGreaterThanOrEqual(0.995);
      expect(bilinear.withinOne / bilinear.channels).toBeGreaterThanOrEqual(0.995);
      // Float32 may put a pixel off here and there, but not most of a column.
      expect(nearest.wrongColumns + bilinear.wrongColumns).toBe(0);
    });
  }
});

/** RGBA pixels as RGB, to tally against. */
const rgb = (pixels: Uint8Array) => pixels.filter((_, i) => i % 4 !== 3);

/**
 * Slice every frame of a case both ways from a clip of its volume's shape, made by `make`,
 * held in YUV 4:2:0, tallying how far each is from the same slice of the clip held as RGBA.
 */
function yuvTally(c: SliceCase, make: (frames: number, height: number, width: number) => Clip):
    { nearest: Tally; bilinear: Tally } {
  const clip = make(c.clip.frames, c.clip.height, c.clip.width);
  const exact = upload(gl, clip), yuv = upload(gl, clip, "yuv420");
  for (const volume of [exact, yuv]) {
    volume.crossfade(c.fade[0], 0);
    volume.crossfade(c.fade[1], 2);
  }
  const nearest = new Tally(), bilinear = new Tally();
  for (const frame of c.frames) {
    for (const [tally, near] of [[nearest, true], [bilinear, false]] as const) {
      const options = { wrap: c.wrap, wrapX: c.wrapX, noise: frame.noise, nearest: near };
      slicer.slice(exact, frame.columns, options);
      const expected = rgb(slicer.read());
      slicer.slice(yuv, frame.columns, options);
      tally.add(slicer.read(), expected);
    }
  }
  exact.dispose();
  yuv.dispose();
  console.log(`${c.name}, ${make.name}\n  nearest: ${nearest.summary}\n`
              + `  bilinear: ${bilinear.summary}`);
  expect(nearest.translucent + bilinear.translucent).toBe(0);
  return { nearest, bilinear };
}

describe("YUV 4:2:0 slices match RGBA ones to rounding where 2×2 blocks are one colour", () => {
  // A fade of columns blends blocks that 4:2:0 then can't hold apart.
  for (const c of sliceCases.filter((c) => c.fade[1] === 0)) {
    test(c.name, () => {
      const { nearest, bilinear } = yuvTally(c, blockClip);
      expect(nearest.maxDiff).toBeLessThanOrEqual(1);
      expect(bilinear.maxDiff).toBeLessThanOrEqual(2);
    });
  }
});

describe("YUV 4:2:0 slices of smooth content match RGBA ones, mean 2 and max 12 levels", () => {
  for (const c of sliceCases) {
    test(c.name, () => {
      for (const tally of Object.values(yuvTally(c, smoothClip))) {
        expect(tally.meanDiff).toBeLessThanOrEqual(2);
        expect(tally.maxDiff).toBeLessThanOrEqual(12);
      }
    });
  }
});

/** Frame t of a volume, sliced at 0°. */
function frameOf(volume: Volume, t: number): Uint8Array {
  const x = Float64Array.from({ length: volume.width }, (_, i) => i);
  slicer.slice(volume, { t: new Float64Array(volume.width).fill(t), x }, { nearest: true });
  return slicer.read();
}

describe("YUV 4:2:0 crossfades of smooth content are close to RGBA ones held in YUV 4:2:0", () => {
  const cases: [string, Clip, number, 0 | 2][] = [
    ...crossfadeCases.map(({ name, clip, n, axis }) =>
      [name, clip, n, axis] as [string, Clip, number, 0 | 2]),
    ["small, 3 of 11 columns", clips.small, 3, 2],
    ["small, 4 of 11 columns", clips.small, 4, 2],
    ["big, 11 of 300 columns", clips.big, 11, 2],
  ];
  for (const [name, { frames, height, width }, n, axis] of cases) {
    test(name, () => {
      const clip = smoothClip(frames, height, width);
      const exact = upload(gl, clip), yuv = upload(gl, clip, "yuv420");
      exact.crossfade(n, axis);
      yuv.crossfade(n, axis);
      expect([yuv.frames, yuv.width]).toEqual([exact.frames, exact.width]);
      // A fade of columns far apart changes colour steeply across them, more than chroma at
      // half size can follow, so the YUV fade is held to what 4:2:0 makes of the RGBA one.
      const held = Volume.create(gl, exact.frames, exact.height, exact.width,
                                 { format: "yuv420" });
      for (let t = 0; t < exact.frames; t++) held.upload(t, readFrame(gl, exact, t));
      const fromExact = new Tally(), fromHeld = new Tally();
      for (let t = 0; t < yuv.frames; t++) {
        const faded = frameOf(yuv, t);
        fromExact.add(faded, rgb(frameOf(exact, t)));
        fromHeld.add(faded, rgb(frameOf(held, t)));
      }
      for (const volume of [exact, yuv, held]) volume.dispose();
      console.log(`${name}\n  from RGBA: ${fromExact.summary}\n`
                  + `  from RGBA held in YUV: ${fromHeld.summary}`);
      // 2.3 where the fade is a tenth of the frame, more of it steep than elsewhere.
      expect(fromExact.meanDiff).toBeLessThanOrEqual(2.5);
      expect(fromHeld.meanDiff).toBeLessThanOrEqual(1);
      expect(fromHeld.maxDiff).toBeLessThanOrEqual(5);
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

  test("wraps round in x between columns that share no chroma", () => {
    for (const width of [11, 10]) {
      const clip = blockClip(3, 5, width);
      const exact = upload(gl, clip), yuv = upload(gl, clip, "yuv420");
      // From the second-last column, past the last and round to the second.
      const x = Float64Array.from({ length: 64 }, (_, j) => width - 2 + j * 4 / 64);
      const columns = { t: new Float64Array(64).fill(1.25), x };
      for (const nearest of [true, false]) {
        const tally = new Tally();
        slicer.slice(exact, columns, { nearest, wrapX: true });
        const expected = rgb(slicer.read());
        slicer.slice(yuv, columns, { nearest, wrapX: true });
        tally.add(slicer.read(), expected);
        expect(tally.maxDiff, `${width} wide, nearest ${nearest}`).toBeLessThanOrEqual(2);
      }
      exact.dispose();
      yuv.dispose();
    }
  });

  test("slices volumes of either format in turn", () => {
    const clip = blockClip(4, 6, 9);
    const exact = upload(gl, clip), yuv = upload(gl, clip, "yuv420");
    canvas.width = clip.width;
    canvas.height = clip.height;
    const x = Float64Array.from({ length: clip.width }, (_, i) => i);
    const columns = { t: new Float64Array(clip.width).fill(2), x };
    for (const nearest of [true, false]) {
      for (let rep = 0; rep < 2; rep++) {
        slicer.slice(exact, columns, { nearest });
        expect(slicer.read()).toEqual(rgba(clip, 2));
        slicer.draw();
        slicer.slice(yuv, columns, { nearest });
        const tally = new Tally();
        tally.add(slicer.read(), rgb(rgba(clip, 2)));
        expect(tally.maxDiff).toBeLessThanOrEqual(1);
        slicer.draw();
      }
    }
    expect(gl.getError()).toBe(gl.NO_ERROR);
    exact.dispose();
    yuv.dispose();
  });

  test("takes noise only for a frame the size of its grid", () => {
    const c = sliceCases.find((c) => c.noisy)!;
    const volume = upload(gl, c.clip);
    const { columns: { t, x }, noise } = c.frames[0];
    expect(() => slicer.slice(volume, { t: t.subarray(1), x: x.subarray(1) }, { noise }))
      .toThrow(RangeError);
    volume.dispose();
  });

  test("takes noise only with no more nodes each way than the GPU can hold", () => {
    const size = Math.min(gl.getParameter(gl.MAX_TEXTURE_SIZE) as number,
                          ...(gl.getParameter(gl.MAX_VIEWPORT_DIMS) as Int32Array));
    const volume = Volume.create(gl, 2, 1, 4);
    const width = size - 1;
    const grid = new Noise(1, { size: 3 }).grid(width, 1, 0);  // a node a pixel, and 3 more
    const columns = { t: new Float64Array(width), x: new Float64Array(width) };
    const noisy = () => slicer.slice(volume, columns, { noise: { push: [1, 0], grid } });
    expect(noisy).toThrow(RangeError);
    expect(noisy).toThrow(`a noise grid of ${size + 2} × ${grid.nodeRows} nodes is more than `
                          + `the GPU's limit of ${size} each way`);
    volume.dispose();
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

  test("has nothing to read or draw before it slices", () => {
    const fresh = new Slicer(gl);
    expect(() => fresh.read()).toThrow("nothing has been sliced yet");
    expect(() => fresh.draw()).toThrow("nothing has been sliced yet");
    fresh.dispose();
  });

  test("slices again after drawing, as the live view does every frame", () => {
    const clip = clips.small;
    const volume = upload(gl, clip);
    canvas.width = clip.width;
    canvas.height = clip.height;
    const x = Float64Array.from({ length: clip.width }, (_, i) => i);
    const c = sliceCases.find((c) => c.noisy && c.clip === clip)!;
    const tally = new Tally();
    for (const [i, frame] of c.frames.entries()) {
      const t = i % clip.frames;
      slicer.slice(volume, { t: new Float64Array(clip.width).fill(t), x }, { nearest: true });
      expect(slicer.read()).toEqual(rgba(clip, t));
      slicer.draw();
      const options = { wrap: c.wrap, wrapX: c.wrapX, noise: frame.noise, nearest: true };
      slicer.slice(volume, frame.columns, options);
      tally.add(slicer.read(), frame.nearest);
      slicer.draw();
    }
    expect(tally.pixelsIdentical / tally.pixels).toBeGreaterThanOrEqual(0.995);
    expect(gl.getError()).toBe(gl.NO_ERROR);
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
