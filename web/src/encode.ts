/**
 * Writing a render: frames drawn on a WebGL2 canvas, encoded as H.264 by WebCodecs and muxed
 * into an MP4 by Mediabunny, as timeslice.open_writer has ffmpeg do with x264.
 */
import {
  EncodedPacket, EncodedVideoPacketSource, Mp4OutputFormat, Output, StreamTarget,
  type StreamTargetChunk,
} from "mediabunny";
import { BT709, markColour } from "./avc";
import { bind, dataTexture, fill, framebuffer, program, type Program } from "./gpu/gl";
import { COVER } from "./gpu/shaders";
import type { Rate } from "./types";

/** H.264 needs even dimensions, so an odd one gains a black column or row. */
export const even = (n: number): number => n + (n % 2);

// The quantizer (QP) where the encoder takes one, lower for better quality: 18 for full quality,
// the number timeslice.py gives x264 as its CRF, though a fixed QP spends more bits than that
// CRF does. Where it takes none, a variable bitrate of so many bits a pixel a frame, and at
// least so many a second, as small frames take more a pixel.
const QUANTIZER = { full: 18, preview: 28 };
const BITS_PER_PIXEL = { full: 0.25, preview: 0.08 };
const LEAST_BITRATE = 500_000;
// x264's longest gap between key frames, in frames.
const KEY_INTERVAL = 250;

/** How well to encode: a quantizer, and the variable bitrate in its place where there is none. */
export interface Settings { quantizer: number; bitrate: number }

/** The settings for frames of this size at this rate, for a full render or a preview. */
export function settings(width: number, height: number, fps: Rate, preview: boolean): Settings {
  const kind = preview ? "preview" : "full";
  const bitrate = Math.round(BITS_PER_PIXEL[kind] * width * height * fps.num / fps.den);
  return { quantizer: QUANTIZER[kind], bitrate: Math.max(bitrate, LEAST_BITRATE) };
}

/** The browser can't encode the video: it has no H.264 encoder for it, or its encoder failed. */
export class Unencodable extends Error {
  override name = "Unencodable";
}

/** An encoded video, and how its bitrate was controlled. */
export interface Encoded { video: Blob; width: number; height: number; rateControl: string }

/**
 * The bytes Mediabunny writes, kept as the parts it wrote them in. A write before the end
 * overwrites what is there, as the muxer does to fill in sizes once it knows them.
 */
export class Gathered {
  private readonly parts: { at: number; data: Uint8Array<ArrayBuffer> }[] = [];
  private end = 0;

  write({ data, position }: StreamTargetChunk): void {
    let done = 0;  // how much of data is in place
    for (const part of this.parts) {  // in order, end to end from 0
      if (done === data.length || position + done >= this.end) break;
      const from = position + done - part.at;
      if (from >= part.data.length) continue;
      const n = Math.min(part.data.length - from, data.length - done);
      part.data.set(data.subarray(done, done + n), from);
      done += n;
    }
    if (done === data.length) return;
    if (position + done > this.end) this.append(new Uint8Array(position + done - this.end));
    this.append(done ? data.slice(done) : data);
  }

  blob(): Blob {
    return new Blob(this.parts.map(({ data }) => data), { type: "video/mp4" });
  }

  private append(data: Uint8Array<ArrayBuffer>): void {
    this.parts.push({ at: this.end, data });
    this.end += data.length;
  }
}

// Frames go to the encoder in BT.709 at limited range, as HD video is, and the video is marked
// so: in its MP4, and in each SPS, by markColour.
const COLOUR_SPACE: VideoColorSpaceInit = {
  primaries: "bt709", transfer: "bt709", matrix: "bt709", fullRange: false,
};
const I420 = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D picture;  // as drawn on the canvas, row 0 its bottom
uniform int plane;                // 0 for Y, 1 for Cb, 2 for Cr
out vec4 levels;

const vec3 KY = vec3(0.2126, 0.7152, 0.0722);

// Pixel (x, y) of the picture, y from the top, padded with black to even sizes.
vec3 pixel(int x, int y) {
  ivec2 size = textureSize(picture, 0);
  return x < size.x && y < size.y ? texelFetch(picture, ivec2(x, size.y - 1 - y), 0).rgb
                                  : vec3(0.0);
}

// The plane's level at (x, y): luma for each pixel, or chroma for each 2×2 block's mean.
float level(int x, int y) {
  if (plane == 0) return 16.0 + 219.0 * dot(pixel(x, y), KY);
  vec3 rgb = (pixel(2 * x, 2 * y) + pixel(2 * x + 1, 2 * y) + pixel(2 * x, 2 * y + 1)
              + pixel(2 * x + 1, 2 * y + 1)) / 4.0;
  float luma = dot(rgb, KY);
  float c = plane == 1 ? (rgb.b - luma) / (2.0 * (1.0 - KY.b))
                       : (rgb.r - luma) / (2.0 * (1.0 - KY.r));
  return 128.0 + 224.0 * c;
}

// Each texel packs four of the plane's levels in a row, and row r of the target is row r of
// the plane, so the rows read back top first.
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 four = vec4(level(4 * p.x, p.y), level(4 * p.x + 1, p.y), level(4 * p.x + 2, p.y),
                   level(4 * p.x + 3, p.y));
  levels = clamp(floor(four + 0.5), 0.0, 255.0) / 255.0;
}`;

/**
 * Reads what a WebGL2 canvas shows as an I420 frame of even size, the picture in its top left
 * corner and black beyond it, as ffmpeg's pad filter makes it: Y, then Cb and Cr at half size
 * each way, worked out on the GPU.
 */
class Planes {
  readonly data: Uint8Array<ArrayBuffer>;
  readonly layout: PlaneLayout[];
  private readonly gl: WebGL2RenderingContext;
  private readonly picture: WebGLTexture;
  private readonly convert: Program;
  private readonly vao: WebGLVertexArrayObject;
  private readonly targets: { texture: WebGLTexture; framebuffer: WebGLFramebuffer;
                              width: number; height: number; offset: number }[] = [];

  constructor(gl: WebGL2RenderingContext, width: number, height: number) {
    this.gl = gl;
    this.picture = dataTexture(gl);
    // A subset of the canvas's channels, as copying from it requires.
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGB8, width, height);
    this.convert = program(gl, COVER, I420, { picture: 0 }, ["plane"]);
    this.vao = gl.createVertexArray();
    let offset = 0;
    this.layout = [];
    for (const [w, h] of [[even(width), even(height)], [even(width) / 2, even(height) / 2],
                          [even(width) / 2, even(height) / 2]]) {
      const texels = Math.ceil(w / 4);
      const texture = dataTexture(gl);
      fill(gl, texture, gl.RGBA8, texels, h, null);
      this.targets.push({ texture, framebuffer: framebuffer(gl, texture), width: texels,
                          height: h, offset });
      this.layout.push({ offset, stride: 4 * texels });
      offset += 4 * texels * h;
    }
    this.data = new Uint8Array(offset);
  }

  /** Fill data with the planes of what the canvas shows now. Leaves the scissor test off. */
  read(): void {
    const { gl } = this;
    const { width, height } = gl.canvas;
    gl.disable(gl.SCISSOR_TEST);
    bind(gl, 0, this.picture);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    gl.copyTexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 0, 0, width, height);
    gl.useProgram(this.convert.program);
    gl.bindVertexArray(this.vao);
    this.targets.forEach(({ framebuffer, width, height, offset }, plane) => {
      gl.uniform1i(this.convert.uniforms.plane, plane);
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.viewport(0, 0, width, height);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE,
                    this.data.subarray(offset, offset + 4 * width * height));
    });
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  dispose(): void {
    const { gl } = this;
    gl.deleteTexture(this.picture);
    gl.deleteProgram(this.convert.program);
    gl.deleteVertexArray(this.vao);
    for (const { texture, framebuffer } of this.targets) {
      gl.deleteTexture(texture);
      gl.deleteFramebuffer(framebuffer);
    }
  }
}

// H.264 levels, as the standard's Table A-1 has them: level_idc, the most macroblocks a
// second and the most in a frame.
const LEVELS: [number, number, number][] = [
  [31, 108000, 3600], [32, 216000, 5120], [40, 245760, 8192], [41, 245760, 8192],
  [42, 522240, 8704],
  [50, 589824, 22080], [51, 983040, 36864], [52, 2073600, 36864], [60, 4177920, 139264],
  [61, 8355840, 139264], [62, 16711680, 139264],
];

/**
 * The codec string for H.264's High profile at the lowest level, from 3.1, that holds frames
 * of this size at this rate.
 */
export function codecString(width: number, height: number, fps: Rate): string {
  const [across, down] = [Math.ceil(width / 16), Math.ceil(height / 16)];
  const fits = ([, rate, frame]: [number, number, number]) =>
    across * down <= frame && across * down * fps.num / fps.den <= rate
    && Math.max(across, down) ** 2 <= 8 * frame;
  const [level] = LEVELS.find(fits) ?? LEVELS[LEVELS.length - 1];
  return `avc1.6400${level.toString(16)}`;
}

async function supported(config: VideoEncoderConfig): Promise<boolean> {
  try {
    return Boolean((await VideoEncoder.isConfigSupported(config)).supported);
  } catch {
    return false;  // a config it can't even consider
  }
}

// Frames any H.264 encoder takes.
const SMALL: VideoEncoderConfig = {
  codec: "avc1.64001f", width: 320, height: 240, framerate: 30, bitrate: 1_000_000,
  avc: { format: "avc" },
};

/**
 * How to configure the encoder for frames of this size: with a quantizer where it takes one,
 * else a variable bitrate. Throws Unencodable where it takes neither.
 */
async function choose(width: number, height: number, fps: Rate, settings: Settings):
    Promise<{ config: VideoEncoderConfig; quantizer: number | null }> {
  if (typeof VideoEncoder === "undefined")
    throw new Unencodable("The browser can't encode video.");
  const base: VideoEncoderConfig = {
    codec: codecString(width, height, fps), width, height, framerate: fps.num / fps.den,
    avc: { format: "avc" }, latencyMode: "quality",
  };
  const quantizer: VideoEncoderConfig = { ...base, bitrateMode: "quantizer" };
  if (await supported(quantizer)) return { config: quantizer, quantizer: settings.quantizer };
  const variable: VideoEncoderConfig = { ...base, bitrateMode: "variable",
                                         bitrate: settings.bitrate };
  if (await supported(variable)) return { config: variable, quantizer: null };
  if (!await supported(SMALL)) throw new Unencodable("The browser can't encode H.264 video.");
  throw new Unencodable(`The browser can't encode video ${width}×${height}.`);
}

// How many frames may wait for the encoder before add() waits too.
const QUEUE = 3;

/**
 * Encodes what a WebGL2 canvas shows, a frame at a time, into an MP4 at a constant frame rate.
 * The canvas keeps its size throughout. Frames go to the encoder as I420, so that the colour
 * matrix is the one the video is marked with: given RGB frames, Chrome's encoder doesn't always
 * use the one it marks. Each SPS is marked with it too, as some encoders leave it out.
 */
export class Encoder {
  readonly width: number;
  readonly height: number;
  /** How the bitrate is controlled: by a quantizer, or a variable bitrate. */
  readonly rateControl: string;
  private readonly planes: Planes;
  private readonly encoder: VideoEncoder;
  private readonly options: VideoEncoderEncodeOptions;
  private readonly output: Output;
  private readonly source: EncodedVideoPacketSource;
  private readonly gathered = new Gathered();
  private readonly fps: Rate;
  private muxed: Promise<void> = Promise.resolve();  // the packets handed to the muxer
  private failure: Unencodable | null = null;
  private wake: (() => void) | null = null;  // stops a wait for the encoder
  private frames = 0;

  private constructor(gl: WebGL2RenderingContext, fps: Rate, config: VideoEncoderConfig,
                      quantizer: number | null) {
    this.fps = fps;
    this.width = config.width;
    this.height = config.height;
    this.options = quantizer === null ? {} : { avc: { quantizer } } as VideoEncoderEncodeOptions;
    this.rateControl = quantizer === null
      ? `variable bitrate, ${formatBitrate(config.bitrate!)}` : `quantizer ${quantizer}`;
    this.planes = new Planes(gl, gl.canvas.width, gl.canvas.height);
    this.output = new Output({
      format: new Mp4OutputFormat(),
      target: new StreamTarget(new WritableStream({ write: (chunk) => this.gathered.write(chunk) }),
                               { chunked: true }),
    });
    this.source = new EncodedVideoPacketSource("avc");
    this.output.addVideoTrack(this.source, { frameRate: fps.num / fps.den });
    this.encoder = new VideoEncoder({
      output: (chunk, meta) => this.mux(chunk, meta),
      error: (error) => this.fail(error),
    });
    this.encoder.configure(config);
  }

  /**
   * An encoder for what the context's canvas shows, at `fps` frames a second. Throws
   * Unencodable if the browser can't encode frames that size; add and finish throw it if the
   * encoder fails later.
   */
  static async open(gl: WebGL2RenderingContext, fps: Rate, settings: Settings): Promise<Encoder> {
    const { config, quantizer } = await choose(even(gl.canvas.width), even(gl.canvas.height), fps,
                                               settings);
    const encoder = new Encoder(gl, fps, config, quantizer);
    await encoder.output.start();
    return encoder;
  }

  /** Encode what the canvas shows now as the next frame; resolves when there's room for more. */
  async add(): Promise<void> {
    this.check();
    this.planes.read();
    const { num, den } = this.fps;
    const frame = new VideoFrame(this.planes.data, {
      format: "I420", codedWidth: this.width, codedHeight: this.height,
      layout: this.planes.layout, colorSpace: COLOUR_SPACE,
      timestamp: Math.round(this.frames * den / num * 1e6), duration: Math.round(den / num * 1e6),
    });
    try {
      this.encoder.encode(frame, { ...this.options, keyFrame: this.frames % KEY_INTERVAL === 0 });
    } catch (error) {
      this.fail(error);
    } finally {
      frame.close();
    }
    this.frames++;
    while (this.encoder.encodeQueueSize > QUEUE && !this.failure) {
      await new Promise<void>((resolve) => {
        this.wake = resolve;
        this.encoder.addEventListener("dequeue", () => resolve(), { once: true });
      });
    }
    await this.muxed;
    this.check();
  }

  /** Finish the video. */
  async finish(): Promise<Encoded> {
    this.check();
    await this.encoder.flush().catch((error: unknown) => this.fail(error));
    await this.muxed;
    this.check();
    this.encoder.close();
    this.source.close();
    await this.output.finalize();
    this.planes.dispose();
    return { video: this.gathered.blob(), width: this.width, height: this.height,
             rateControl: this.rateControl };
  }

  /** Hand a packet to the muxer, its decoder config marked with the frames' colour. */
  private mux(chunk: EncodedVideoChunk, meta?: EncodedVideoChunkMetadata): void {
    const packet = EncodedPacket.fromEncodedChunk(chunk);
    const config = meta?.decoderConfig;
    if (config?.description) {
      const description = ArrayBuffer.isView(config.description)
        ? new Uint8Array(config.description.buffer, config.description.byteOffset,
                         config.description.byteLength)
        : new Uint8Array(config.description);
      let marked: Uint8Array = description;
      try {
        marked = markColour(description, BT709);
      } catch {
        // An SPS it can't read keeps what the encoder wrote; the MP4 still says BT.709.
      }
      meta = { ...meta, decoderConfig: { ...config, colorSpace: COLOUR_SPACE,
                                         description: marked } };
    }
    this.muxed = this.muxed.then(() => this.source.add(packet, meta))
      .catch((error: unknown) => this.fail(error));
  }

  /** Note the first failure, of the encoder or the muxer, for check to throw. */
  private fail(error: unknown): void {
    const reason = (error instanceof Error ? error.message : String(error)).replace(/\.$/, "");
    this.failure ??= new Unencodable(`The browser's encoder failed: ${reason}.`, { cause: error });
    this.wake?.();
  }

  private check(): void {
    if (this.failure) throw this.failure;
  }
}

function formatBitrate(bits: number): string {
  return bits >= 1e6 ? `${(bits / 1e6).toFixed(1)} Mbit/s` : `${Math.round(bits / 1e3)} kbit/s`;
}
