/**
 * Writing a render: pictures in WebGL2 textures, encoded as H.264 by WebCodecs and muxed into
 * an MP4 by Mediabunny.
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

// The quantizer (QP) where the encoder takes one, lower for better quality. Where it takes none,
// a variable bitrate of so many bits a pixel a frame, and at least so many a second, as small
// frames take more a pixel.
const QUANTIZER = { full: 18, preview: 28 };
const BITS_PER_PIXEL = { full: 0.25, preview: 0.08 };
const LEAST_BITRATE = 500_000;
// The longest gap between key frames, in frames.
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
uniform highp sampler2D picture;  // row 0 its top
uniform ivec2 size;               // the frame's, the picture's rounded up to even
uniform int across;               // the target's texels a row
out vec4 levels;

const vec3 KY = vec3(0.2126, 0.7152, 0.0722);

// Pixel (x, y) of the picture, padded with black to the frame's size.
vec3 pixel(int x, int y) {
  ivec2 drawn = textureSize(picture, 0);
  return x < drawn.x && y < drawn.y ? texelFetch(picture, ivec2(x, y), 0).rgb : vec3(0.0);
}

// Byte i of the frame: Y, then Cb and Cr at half size each way, every row packed tight.
float level(int i) {
  int pixels = size.x * size.y, quarter = pixels / 4;
  if (i < pixels) return 16.0 + 219.0 * dot(pixel(i % size.x, i / size.x), KY);
  int plane = (i - pixels) / quarter, j = (i - pixels) % quarter;  // 0 for Cb, 1 for Cr
  if (plane > 1) return 0.0;  // past the end
  int x = 2 * (j % (size.x / 2)), y = 2 * (j / (size.x / 2));
  vec3 rgb = (pixel(x, y) + pixel(x + 1, y) + pixel(x, y + 1) + pixel(x + 1, y + 1)) / 4.0;
  float luma = dot(rgb, KY);
  float c = plane == 0 ? (rgb.b - luma) / (2.0 * (1.0 - KY.b))
                       : (rgb.r - luma) / (2.0 * (1.0 - KY.r));
  return 128.0 + 224.0 * c;
}

// Texel (x, y) holds bytes 4n to 4n + 3 of the frame, n = y across + x, so the target reads
// back as the frame.
void main() {
  int i = 4 * (int(gl_FragCoord.y) * across + int(gl_FragCoord.x));
  vec4 four = vec4(level(i), level(i + 1), level(i + 2), level(i + 3));
  levels = clamp(floor(four + 0.5), 0.0, 255.0) / 255.0;
}`;

/**
 * Turns a picture, an RGBA8 texture, into an I420 frame of even size, the picture in its top
 * left corner and black beyond it: Y, then Cb and Cr at half size each way, worked out on the
 * GPU into a target that reads back as the frame's bytes. Each plane's rows are packed tight,
 * as Firefox's VideoFrame reads them whatever stride a layout gives.
 */
class Planes {
  /** The frame, once read. */
  readonly data: Uint8Array<ArrayBuffer>;
  private readonly target: Uint8Array<ArrayBuffer>;  // the target's bytes, the frame's first
  private readonly gl: WebGL2RenderingContext;
  private readonly convert: Program;
  private readonly vao: WebGLVertexArrayObject;
  private readonly texture: WebGLTexture;
  private readonly framebuffer: WebGLFramebuffer;
  private readonly across: number;  // the target's texels a row
  private readonly down: number;    // and its rows

  constructor(gl: WebGL2RenderingContext, width: number, height: number) {
    this.gl = gl;
    const [w, h] = [even(width), even(height)];
    // Two rows of the frame's luma a row of the target, and as many rows as hold 1.5 bytes a
    // pixel.
    this.across = w / 2;
    this.down = Math.ceil(3 * h / 4);
    this.convert = program(gl, COVER, I420, { picture: 0 }, ["size", "across"]);
    gl.uniform2i(this.convert.uniforms.size, w, h);
    gl.uniform1i(this.convert.uniforms.across, this.across);
    this.vao = gl.createVertexArray();
    this.texture = dataTexture(gl);
    fill(gl, this.texture, gl.RGBA8, this.across, this.down, null);
    this.framebuffer = framebuffer(gl, this.texture);
    this.target = new Uint8Array(4 * this.across * this.down);
    this.data = this.target.subarray(0, w * h * 3 / 2);
  }

  /** Fill data with the planes of a picture. */
  read(picture: WebGLTexture): void {
    const { gl } = this;
    bind(gl, 0, picture);
    gl.useProgram(this.convert.program);
    gl.bindVertexArray(this.vao);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.viewport(0, 0, this.across, this.down);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.readPixels(0, 0, this.across, this.down, gl.RGBA, gl.UNSIGNED_BYTE, this.target);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindVertexArray(null);
  }

  dispose(): void {
    const { gl } = this;
    gl.deleteProgram(this.convert.program);
    gl.deleteVertexArray(this.vao);
    gl.deleteTexture(this.texture);
    gl.deleteFramebuffer(this.framebuffer);
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
 * How to configure the encoder for frames of this size: with a quantizer where it takes and
 * honours one, else a variable bitrate. Throws Unencodable where it takes neither.
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
  // Firefox's encoder takes the quantizer mode but ignores the quantizer, coding a preview as a
  // full render, and in that mode garbles the luma of small frames' P-frames the more often.
  if (!navigator.userAgent.includes("Firefox/") && await supported(quantizer))
    return { config: quantizer, quantizer: settings.quantizer };
  const variable: VideoEncoderConfig = { ...base, bitrateMode: "variable",
                                         bitrate: settings.bitrate };
  if (await supported(variable)) return { config: variable, quantizer: null };
  if (!await supported(SMALL)) throw new Unencodable("The browser can't encode H.264 video.");
  throw new Unencodable(`The browser can't encode video ${width}×${height}.`);
}

// How many frames may wait for the encoder before add() waits too.
const QUEUE = 3;

/**
 * Encodes pictures, a frame at a time, into an MP4 at a constant frame rate. Each picture is an
 * RGBA8 texture of the size the encoder was opened with, row 0 its top. Frames go to the
 * encoder as I420, so that the colour matrix is the one the video is marked with: given RGB
 * frames, Chrome's encoder doesn't always use the one it marks. Each SPS is marked with it too,
 * as some encoders leave it out.
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
    this.planes = new Planes(gl, config.width, config.height);
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
   * An encoder for pictures width × height, at `fps` frames a second. Throws Unencodable if the
   * browser can't encode frames that size; add and finish throw it if the encoder fails later.
   */
  static async open(gl: WebGL2RenderingContext, width: number, height: number, fps: Rate,
                    settings: Settings): Promise<Encoder> {
    const { config, quantizer } = await choose(even(width), even(height), fps, settings);
    const encoder = new Encoder(gl, fps, config, quantizer);
    await encoder.output.start();
    return encoder;
  }

  /** Encode a picture as the next frame; resolves when there's room for more. */
  async add(picture: WebGLTexture): Promise<void> {
    this.check();
    this.planes.read(picture);
    const { num, den } = this.fps;
    const frame = new VideoFrame(this.planes.data, {
      format: "I420", codedWidth: this.width, codedHeight: this.height, colorSpace: COLOUR_SPACE,
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
