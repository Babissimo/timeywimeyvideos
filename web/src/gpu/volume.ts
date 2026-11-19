/**
 * A clip held on the GPU in YUV 4:2:0, in TEXTURE_2D_ARRAYs: full-range BT.601 luma in an R8
 * plane, texel (x, y, layer) voxel (t = layer, y, x), and in an RG8 plane half its size each
 * way (rounded up), Cb and Cr of each 2×2 block's mean colour. 1.5 bytes a voxel. Making one
 * refuses, with the page's advice, a clip the GPU can't hold.
 */
import { load, type LoadOptions } from "../decode";
import { Problem } from "../options";
import { bind, dataTexture, fill, program, type Program, unpackAsIs } from "./gl";
import { CHROMA, COVER, LUMA } from "./shaders";

/** One of a volume's texture arrays, and the size of each of its layers. */
export interface Plane {
  readonly texture: WebGLTexture;
  readonly width: number;
  readonly height: number;
}

/** What the page says once the GPU's context is lost. */
export const LOST = "The GPU gave out, perhaps for want of memory. Set a shorter duration or a "
  + "smaller scale.";

const half = (n: number) => Math.ceil(n / 2);

/** Turns RGBA frames into a volume's planes, through a texture the frame's size. */
class Converter {
  private readonly gl: WebGL2RenderingContext;
  private readonly frame: WebGLTexture;
  private readonly luma: Program;
  private readonly chroma: Program;
  private readonly target: WebGLFramebuffer;
  private readonly vao: WebGLVertexArrayObject;

  constructor(gl: WebGL2RenderingContext, width: number, height: number) {
    this.gl = gl;
    this.frame = dataTexture(gl);
    fill(gl, this.frame, gl.RGBA8, width, height, null);
    this.luma = program(gl, COVER, LUMA, { frame: 0 });
    this.chroma = program(gl, COVER, CHROMA, { frame: 0 });
    this.target = gl.createFramebuffer();
    this.vao = gl.createVertexArray();
  }

  /** Set layer t of the planes from a frame. */
  convert(t: number, pixels: Uint8Array | TexImageSource, planes: readonly Plane[]): void {
    const { gl } = this;
    const [{ width, height }] = planes;
    bind(gl, 0, this.frame);
    unpackAsIs(gl);
    // Two calls, one for each of texSubImage2D's overloads.
    if (ArrayBuffer.isView(pixels))
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    else
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.target);
    gl.bindVertexArray(this.vao);
    planes.forEach((plane, i) => {
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, plane.texture, 0, t);
      gl.viewport(0, 0, plane.width, plane.height);
      gl.useProgram((i === 0 ? this.luma : this.chroma).program);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    });
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  dispose(): void {
    const { gl } = this;
    gl.deleteTexture(this.frame);
    gl.deleteProgram(this.luma.program);
    gl.deleteProgram(this.chroma.program);
    gl.deleteFramebuffer(this.target);
    gl.deleteVertexArray(this.vao);
  }
}

export class Volume {
  private readonly gl: WebGL2RenderingContext;
  /** The luma plane, then the chroma. */
  readonly planes: readonly [Plane, Plane];
  readonly frames: number;
  readonly height: number;
  readonly width: number;
  readonly bytes: number;
  private converter: Converter | null = null;  // made by the first upload

  private constructor(gl: WebGL2RenderingContext, planes: [Plane, Plane], frames: number,
                      bytes: number) {
    this.gl = gl;
    this.planes = planes;
    this.frames = frames;
    this.height = planes[0].height;
    this.width = planes[0].width;
    this.bytes = bytes;
  }

  /**
   * Allocate a volume of a whole number of voxels each way, at least one, or throw the Problem of
   * a clip the GPU can't hold, as a preview's at half size and frame rate. Once the clip is within
   * the GPU's limits, `free` lets go of memory the volume can then take.
   */
  static create(gl: WebGL2RenderingContext,
                { frames, height, width }: { frames: number; height: number; width: number },
                { preview = false, free }: { preview?: boolean; free?: () => void } = {}): Volume {
    if (![frames, height, width].every((n) => Number.isInteger(n) && n >= 1))
      throw new RangeError(`can't make a volume of ${frames} × ${height} × ${width} voxels`);
    if (gl.isContextLost()) throw new Problem(LOST);  // its limits would read as 0
    const layers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
    if (frames > layers) {
      throw new Problem(`The GPU can hold at most ${layers} frames, and this clip has `
                        + `${frames}${preview ? " at half its frame rate" : ""}. Set a shorter `
                        + "duration.");
    }
    const size = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    if (height > size || width > size) {
      throw new Problem(`The GPU can hold frames at most ${size} pixels across, and this clip's `
                        + `are ${width}×${height}${preview ? " at half size" : ""}. Set a smaller `
                        + "scale.");
    }
    free?.();
    const sizes: [GLenum, number, number][] = [[gl.R8, width, height],
                                               [gl.RG8, half(width), half(height)]];
    const bytes = frames * (width * height + 2 * half(width) * half(height));
    while (gl.getError() !== gl.NO_ERROR) {
      // Clear errors left by earlier calls, so the checks below see only these allocations'.
      // Any the caller left pending are lost.
    }
    const planes: Plane[] = [];
    for (const [internalFormat, w, h] of sizes) {
      const texture = gl.createTexture();
      planes.push({ texture, width: w, height: h });
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
      gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, internalFormat, w, h, frames);
      if (gl.getError() !== gl.NO_ERROR) {
        for (const plane of planes) gl.deleteTexture(plane.texture);
        throw new Problem(`This clip needs ${(bytes / 1e9).toFixed(1)} GB of GPU memory, more `
                          + "than the GPU can give it. Set a shorter duration or a smaller scale.");
      }
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    }
    return new Volume(gl, planes as [Plane, Plane], frames, bytes);
  }

  /**
   * Set frame t from RGBA bytes, rows top-first, or from an image, its top row first.
   * Either is the size the volume was created with.
   */
  upload(t: number, pixels: Uint8Array | TexImageSource): void {
    this.converter ??= new Converter(this.gl, this.width, this.height);
    this.converter.convert(t, pixels, this.planes);
  }

  /**
   * Decode into the volume the frames a load plans, closing each. Rejects with the Problem of a
   * lost GPU, or the signal's reason once aborted, and lets the volume go if it fails.
   */
  async decode(source: Blob, options: LoadOptions,
               progress?: (done: number, total: number) => void): Promise<void> {
    // The context can be lost before its event arrives, and uploads to a lost one do nothing.
    const lost = () => this.gl.isContextLost();
    try {
      await load(source, options, (t, frame) => {
        try {
          this.upload(t, frame);
        } finally {
          frame.close();
        }
        if (lost()) throw new Problem(LOST);
      }, progress);
      options.signal?.throwIfAborted();
      if (lost()) throw new Problem(LOST);
    } catch (error) {
      this.dispose();
      options.signal?.throwIfAborted();
      // The first upload to a lost context fails as shaders that won't compile.
      throw lost() && !(error instanceof Problem) ? new Problem(LOST) : error;
    }
  }

  dispose(): void {
    for (const { texture } of this.planes) this.gl.deleteTexture(texture);
    this.converter?.dispose();
    this.converter = null;
  }
}
