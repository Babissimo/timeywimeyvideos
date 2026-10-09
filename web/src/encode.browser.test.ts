import { afterEach, describe, expect, test, vi } from "vitest";
import { codecString, Encoder, settings, Unencodable } from "./encode";
import { framesOf, readBack } from "./readback";
import type { Rate } from "./types";

// Well-separated colours, one a frame, so each survives lossy coding and shows which frame it is.
const COLOURS = [[200, 40, 40], [40, 200, 40], [40, 40, 200], [200, 200, 40], [40, 200, 200],
                 [200, 40, 200], [220, 220, 220], [60, 60, 60]];

/** A WebGL2 canvas this size. */
function canvas(width: number, height: number): WebGL2RenderingContext {
  return new OffscreenCanvas(width, height).getContext("webgl2", { alpha: false })!;
}

/** Encode n frames of a canvas this size, each one colour over all but its bottom right. */
async function encode(width: number, height: number, fps: Rate, n: number,
                      preview = false) {
  const gl = canvas(width, height);
  const encoder = await Encoder.open(gl, fps, settings(width, height, fps, preview));
  for (let i = 0; i < n; i++) {
    const [r, g, b] = COLOURS[i % COLOURS.length];
    gl.disable(gl.SCISSOR_TEST);
    gl.clearColor(r / 255, g / 255, b / 255, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.SCISSOR_TEST);  // the bottom right quarter, as the canvas's y runs up
    gl.scissor(width >> 1, 0, width - (width >> 1), height - (height >> 1));
    gl.clearColor(1, 1, 1, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    await encoder.add();
  }
  return encoder.finish();
}

/** The mean colour of a frame's pixels in columns x0 to x1 and rows y0 to y1. */
function mean(frame: Uint8ClampedArray, width: number, [x0, x1]: [number, number],
              [y0, y1]: [number, number]): number[] {
  const sum = [0, 0, 0];
  for (let y = y0; y < y1; y++)
    for (let x = x0; x < x1; x++)
      for (let k = 0; k < 3; k++) sum[k] += frame[(y * width + x) * 4 + k];
  return sum.map((s) => s / ((x1 - x0) * (y1 - y0)));
}

const BT709 = { primaries: "bt709", transfer: "bt709", matrix: "bt709", fullRange: false };

const luma = ([r, g, b]: number[]) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

const near = (got: number[], expected: number[], within: number) =>
  got.forEach((v, k) => expect(Math.abs(v - expected[k]), `${got} against ${expected}`)
    .toBeLessThanOrEqual(within));

afterEach(() => {
  vi.restoreAllMocks();
});

describe("Encoder", () => {
  test("pads odd sizes with black, as ffmpeg's pad filter does", async () => {
    const fps = { num: 30000, den: 1001 };
    const encoded = await encode(33, 25, fps, 12);
    expect([encoded.width, encoded.height]).toEqual([34, 26]);
    expect(await readBack(encoded.video))
      .toEqual({ codec: "avc", width: 34, height: 26, frames: 12, fps, colour: BT709 });
    const frames = await framesOf(encoded.video, [0, 5, 11]);
    for (const [n, frame] of frames.entries()) {
      const i = [0, 5, 11][n];
      near(mean(frame, 34, [0, 14], [0, 10]), COLOURS[i % COLOURS.length], 12);
      near(mean(frame, 34, [18, 32], [14, 24]), [255, 255, 255], 12);
      // The padding, the last column and the bottom row, is black, though it shares its
      // colour with the picture's last column and row, at half resolution as 4:2:0 holds it.
      expect(luma(mean(frame, 34, [33, 34], [0, 10]))).toBeLessThanOrEqual(24);
      expect(luma(mean(frame, 34, [0, 14], [25, 26]))).toBeLessThanOrEqual(24);
    }
  });

  test("keeps even sizes and the frame rate", async () => {
    const fps = { num: 12, den: 1 };
    const encoded = await encode(64, 48, fps, 5, true);
    expect(await readBack(encoded.video))
      .toEqual({ codec: "avc", width: 64, height: 48, frames: 5, fps, colour: BT709 });
    const [last] = await framesOf(encoded.video, [4]);
    near(mean(last, 64, [0, 30], [0, 22]), COLOURS[4], 12);
  });

  test("uses a quantizer where the encoder takes one, else a variable bitrate", async () => {
    const fps = { num: 30, den: 1 };
    for (const [width, height] of [[1280, 720], [640, 360], [64, 48]]) {
      const quantizer = await VideoEncoder.isConfigSupported({
        codec: codecString(width, height, fps), width, height, framerate: 30,
        bitrateMode: "quantizer", avc: { format: "avc" }, latencyMode: "quality",
      });
      const { video, rateControl } = await encode(width, height, fps, 3);
      console.log(`${width}×${height}: ${rateControl}`);
      if (quantizer.supported) expect(rateControl).toBe("quantizer 18");
      else expect(rateControl).toMatch(/^variable bitrate, \d+ [kM]bit\/s$/);
      // Whichever encoder it was, the colours come back as they went in, give or take what a
      // low bitrate loses on a tiny frame.
      expect((await readBack(video)).colour).toEqual(BT709);
      const frames = await framesOf(video, [0, 1, 2]);
      frames.forEach((frame, i) => {
        near(mean(frame, width, [0, width >> 2], [0, height >> 2]), COLOURS[i],
             quantizer.supported ? 6 : 12);
      });
    }
  });

  test("sizes the variable bitrate from the frame size and rate", () => {
    expect(settings(1920, 1080, { num: 30, den: 1 }, false))
      .toEqual({ quantizer: 18, bitrate: 15552000 });
    expect(settings(1920, 1080, { num: 15, den: 1 }, true))
      .toEqual({ quantizer: 28, bitrate: 2488320 });
    expect(settings(64, 48, { num: 30000, den: 1001 }, false).bitrate).toBe(500000);
    expect(settings(64, 48, { num: 15, den: 1 }, true).bitrate).toBe(500000);
  });

  test("refuses frames the browser can't encode", async () => {
    const fps = { num: 30, den: 1 };
    await expect(Encoder.open(canvas(30000, 16), fps, settings(30000, 16, fps, false)))
      .rejects.toThrow(new Unencodable("The browser can't encode video 30000×16."));
  });

  test("says when the browser can't encode H.264 at all", async () => {
    vi.spyOn(VideoEncoder, "isConfigSupported").mockResolvedValue({ supported: false });
    const fps = { num: 30, den: 1 };
    await expect(Encoder.open(canvas(64, 48), fps, settings(64, 48, fps, false)))
      .rejects.toThrow(new Unencodable("The browser can't encode H.264 video."));
  });

  test("says when the encoder fails partway", async () => {
    const fps = { num: 30, den: 1 };
    const encoder = await Encoder.open(canvas(64, 48), fps, settings(64, 48, fps, false));
    await encoder.add();
    vi.spyOn(VideoEncoder.prototype, "encode").mockImplementation(() => {
      throw new DOMException("Encoding error.", "EncodingError");
    });
    const error = await encoder.add().then(() => null, (error: unknown) => error);
    expect(error).toEqual(new Unencodable("The browser's encoder failed: Encoding error."));
    await expect(encoder.finish()).rejects.toBe(error);
  });
});
