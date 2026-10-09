/**
 * What a rendered MP4 holds, read back for the tests: its coded size, frame count and rate, and
 * small thumbnails of chosen frames, made the way web/test/fixtures/render.py makes them.
 */
import { ALL_FORMATS, BlobSource, EncodedPacketSink, Input, VideoSampleSink } from "mediabunny";
import { probe } from "./decode";
import type { Rate } from "./types";

export interface Readback {
  codec: string | null; width: number; height: number; frames: number; fps: Rate;
  colour: VideoColorSpaceInit;
}

/**
 * The video's codec, coded size, number of frames, frame rate (ffprobe's r_frame_rate, which
 * probe reports for a constant-rate video) and the colour space it is marked with.
 */
export async function readBack(video: Blob): Promise<Readback> {
  const input = new Input({ source: new BlobSource(video), formats: ALL_FORMATS });
  try {
    const track = (await input.getPrimaryVideoTrack())!;
    let frames = 0;
    for await (const _ of new EncodedPacketSink(track).packets(undefined, undefined,
                                                                { metadataOnly: true }))
      frames++;
    return { codec: await track.getCodec(), width: await track.getCodedWidth(),
             height: await track.getCodedHeight(), frames, fps: (await probe(video)).fps,
             colour: await track.getColorSpace() };
  } finally {
    input.dispose();
  }
}

/** The frames at these indices, decoded to RGBA rows top-first at the coded size. */
export async function framesOf(video: Blob, indices: number[]): Promise<Uint8ClampedArray[]> {
  const input = new Input({ source: new BlobSource(video), formats: ALL_FORMATS });
  try {
    const track = (await input.getPrimaryVideoTrack())!;
    const [width, height] = [await track.getCodedWidth(), await track.getCodedHeight()];
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext("2d", { willReadFrequently: true })!;
    const found = new Map<number, Uint8ClampedArray>();
    let i = 0;
    for await (const sample of new VideoSampleSink(track).samples()) {
      if (indices.includes(i)) {
        sample.draw(context, 0, 0, width, height);
        found.set(i, context.getImageData(0, 0, width, height).data);
      }
      sample.close();
      i++;
    }
    return indices.map((index) => {
      const frame = found.get(index);
      if (!frame) throw new Error(`the video has no frame ${index}`);
      return frame;
    });
  } finally {
    input.dispose();
  }
}

/**
 * A frame shrunk to cols × rows blocks, each the mean red, green and blue of the pixels it
 * covers, rounded: block (i, j) covers columns floor(i W / cols) to floor((i + 1) W / cols).
 * RGBA pixels give RGB blocks, row by row.
 */
export function thumbnail(pixels: ArrayLike<number>, width: number, height: number, cols: number,
                          rows: number, channels = 4): number[] {
  const out: number[] = [];
  for (let j = 0; j < rows; j++) {
    const [y0, y1] = [Math.floor(j * height / rows), Math.floor((j + 1) * height / rows)];
    for (let i = 0; i < cols; i++) {
      const [x0, x1] = [Math.floor(i * width / cols), Math.floor((i + 1) * width / cols)];
      for (let k = 0; k < 3; k++) {
        let sum = 0;
        for (let y = y0; y < y1; y++)
          for (let x = x0; x < x1; x++) sum += pixels[(y * width + x) * channels + k];
        out.push(Math.round(sum / ((y1 - y0) * (x1 - x0))));
      }
    }
  }
  return out;
}
