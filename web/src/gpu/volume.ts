/**
 * A clip held on the GPU in YUV 4:2:0, in TEXTURE_2D_ARRAYs: full-range BT.601 luma in an R8
 * plane, texel (x, y, layer) voxel (t = layer, y, x), and in an RG8 plane half its size each
 * way (rounded up), Cb and Cr of each 2×2 block's mean colour. 1.5 bytes a voxel.
 */
import { bind, dataTexture, fill, program, type Program, unpackAsIs } from "./gl";
import { CHROMA, COVER, LUMA } from "./shaders";

/** One of a volume's texture arrays, and the size of each of its layers. */
export interface Plane {
  readonly texture: WebGLTexture;
  readonly width: number;
  readonly height: number;
}

/** The clip needs more of the GPU than it can give. */
export class VolumeTooLarge extends Error {
  readonly bytes: number;

  constructor(bytes: number) {
    super(`a ${(bytes / 1e9).toFixed(2)} GB clip is more than the GPU can hold`);
    this.name = "VolumeTooLarge";
    this.bytes = bytes;
  }
}

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
   * Allocate a volume. Throws RangeError past the GPU's limits on frames or on pixels each
   * way, and VolumeTooLarge if the GPU can't give it the memory.
   */
  static create(gl: WebGL2RenderingContext, frames: number, height: number,
                width: number): Volume {
    if (![frames, height, width].every((n) => Number.isInteger(n) && n >= 1))
      throw new RangeError(`can't make a volume of ${frames} × ${height} × ${width} voxels`);
    const layers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
    if (frames > layers)
      throw new RangeError(`a volume of ${frames} frames is more than the GPU's limit of `
                           + `${layers}`);
    const size = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    if (height > size || width > size)
      throw new RangeError(`frames of ${width} × ${height} pixels are more than the GPU's limit `
                           + `of ${size} each way`);
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
        throw new VolumeTooLarge(bytes);
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

  dispose(): void {
    for (const { texture } of this.planes) this.gl.deleteTexture(texture);
    this.converter?.dispose();
    this.converter = null;
  }
}
