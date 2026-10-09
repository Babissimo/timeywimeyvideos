/**
 * A clip held on the GPU in TEXTURE_2D_ARRAYs: texel (x, y, layer) is voxel (t = layer, y, x),
 * or in a chroma plane the 2×2 block of voxels from (t, 2y, 2x).
 */
import { bind, dataTexture, fill, framebuffer, program, type Program, unpackAsIs } from "./gl";
import { CHROMA, CHROMA_CROSSFADE, COVER, CROSSFADE, LUMA } from "./shaders";

/**
 * How a volume holds its voxels: "rgba" as RGBA8, 4 bytes a voxel; "yuv420" as full-range
 * BT.601 luma in an R8 plane and, in an RG8 plane half its size each way (rounded up), Cb and
 * Cr of each 2×2 block's mean colour, 1.5 bytes a voxel.
 */
export type VolumeFormat = "rgba" | "yuv420";

/** One of a volume's texture arrays, and the size of each of its layers. */
export interface Plane {
  readonly texture: WebGLTexture;
  readonly channels: number;  // a byte each
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

/** The internal format, bytes a texel, width and height of each plane of a volume. */
function layout(gl: WebGL2RenderingContext, format: VolumeFormat, height: number,
                width: number): [GLenum, number, number, number][] {
  return format === "rgba" ? [[gl.RGBA8, 4, width, height]]
                           : [[gl.R8, 1, width, height], [gl.RG8, 2, half(width), half(height)]];
}

/** Turns RGBA frames into a YUV 4:2:0 volume's planes, through a texture the frame's size. */
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
  readonly format: VolumeFormat;
  /** For the slicer to read: the RGBA plane, or the luma plane then the chroma. */
  readonly planes: readonly Plane[];
  readonly height: number;
  readonly bytes: number;
  private size: { frames: number; width: number };  // a crossfade uses fewer than are stored
  private converter: Converter | null = null;  // made by the first upload of a YUV volume

  private constructor(gl: WebGL2RenderingContext, format: VolumeFormat, planes: Plane[],
                      frames: number, bytes: number) {
    this.gl = gl;
    this.format = format;
    this.planes = planes;
    this.height = planes[0].height;
    this.bytes = bytes;
    this.size = { frames, width: planes[0].width };
  }

  get frames(): number { return this.size.frames; }
  get width(): number { return this.size.width; }

  /**
   * Allocate a volume. Throws RangeError past the GPU's limits on frames or on pixels each
   * way, and VolumeTooLarge if the GPU can't give it the memory.
   */
  static create(gl: WebGL2RenderingContext, frames: number, height: number, width: number,
                { format = "rgba" }: { format?: VolumeFormat } = {}): Volume {
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
    const sizes = layout(gl, format, height, width);
    const bytes = frames * sizes.reduce((sum, [, texel, w, h]) => sum + texel * w * h, 0);
    while (gl.getError() !== gl.NO_ERROR) {
      // Clear errors left by earlier calls, so the checks below see only these allocations'.
      // Any the caller left pending are lost.
    }
    const planes: Plane[] = [];
    for (const [internalFormat, channels, w, h] of sizes) {
      const texture = gl.createTexture();
      planes.push({ texture, channels, width: w, height: h });
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
      gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, internalFormat, w, h, frames);
      if (gl.getError() !== gl.NO_ERROR) {
        for (const plane of planes) gl.deleteTexture(plane.texture);
        throw new VolumeTooLarge(bytes);
      }
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    }
    return new Volume(gl, format, planes, frames, bytes);
  }

  /**
   * Set frame t from RGBA bytes, rows top-first, or from an image, its top row first.
   * Either is the size the volume was created with.
   */
  upload(t: number, pixels: Uint8Array | TexImageSource): void {
    const { gl, planes: [{ texture, width }], height } = this;
    if (this.format === "yuv420") {
      this.converter ??= new Converter(gl, width, height);
      this.converter.convert(t, pixels, this.planes);
      return;
    }
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
    unpackAsIs(gl);  // a 3D upload of bytes even fails if flipped or premultiplied
    // Two calls, one for each of texSubImage3D's overloads.
    if (ArrayBuffer.isView(pixels))
      gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, t, width, height, 1, gl.RGBA,
                       gl.UNSIGNED_BYTE, pixels);
    else
      gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, t, width, height, 1, gl.RGBA,
                       gl.UNSIGNED_BYTE, pixels);
  }

  /**
   * Blend the last n frames into the first, in place, and drop them, as timeslice.crossfade
   * does: a clip that runs on smoothly from its end into its start when time runs round in
   * a ring. With axis 2 it blends the last n columns into the first instead, for when x
   * wraps round too. The fade can be at most half the clip.
   *
   * In YUV 4:2:0 luma fades as a colour channel does, and chroma by frames alike. By columns,
   * each chroma column takes the mean of the chroma of the (one or two) columns it covers in
   * the faded picture, so with n odd the column covering the picture's columns n - 1 and n is
   * half faded; and with an odd number of columns left, the last keeps chroma it shares with
   * the first column dropped.
   */
  crossfade(n: number, axis: 0 | 2): void {
    const size = axis === 0 ? this.frames : this.width;
    const length = size - n;
    if (!Number.isInteger(n))
      throw new RangeError(`a fade has to be a whole number of `
                           + `${axis === 0 ? "frames" : "columns"}, not ${n}`);
    if (!(0 <= n && n <= length))
      throw new Error(`a fade of ${n} doesn't fit in a clip ${size} long: it can be at most half`);
    if (n > 0) this.fade(n, length, axis);
    this.size = axis === 0 ? { frames: length, width: this.width }
                           : { frames: this.frames, width: length };
  }

  private fade(n: number, length: number, axis: 0 | 2): void {
    const { gl } = this;
    const weights = new Float32Array(2 * n);
    for (let i = 0; i < n; i++) {
      const w = Math.fround((i + 1) / (n + 1));  // as numpy rounds them
      weights.set([w, Math.fround(1 - w)], 2 * i);
    }
    const weighting = dataTexture(gl);
    fill(gl, weighting, gl.RG32F, n, 1, weights);
    const blend = program(gl, COVER, CROSSFADE, { volume: 0, weights: 1 },
                          ["layer", "past", "across", "zero", "width", "channels"]);
    gl.useProgram(blend.program);
    gl.uniform2i(blend.uniforms.past, axis === 0 ? length : 0, axis === 2 ? length : 0);
    gl.uniform1i(blend.uniforms.across, Number(axis === 2));
    gl.uniform1i(blend.uniforms.zero, 0);
    const [first, chroma] = this.planes;
    if (axis === 0) {
      // Each chroma column covers two of the picture's.
      for (const plane of this.planes)
        this.redraw(plane, blend, plane === first ? this.width : half(this.width), n, weighting);
    } else {
      this.redraw(first, blend, n, this.frames, weighting);
      if (chroma) {
        const pairs = program(gl, COVER, CHROMA_CROSSFADE, { volume: 0, weights: 1 },
                              ["layer", "n", "kept", "width", "channels"]);
        gl.useProgram(pairs.program);
        gl.uniform1i(pairs.uniforms.n, n);
        gl.uniform1i(pairs.uniforms.kept, length);
        this.redraw(chroma, pairs, half(n), this.frames, weighting);
        gl.deleteProgram(pairs.program);
      }
    }
    gl.deleteTexture(weighting);
    gl.deleteProgram(blend.program);
  }

  /**
   * Draw each of a plane's first `layers` layers, `width` columns of it, anew with a fade's
   * shader, which reads the plane on unit 0 and the weights on unit 1.
   */
  private redraw(plane: Plane, shader: Program, width: number, layers: number,
                 weighting: WebGLTexture): void {
    const { gl } = this;
    const { height, channels } = plane;
    // A texture can't be drawn into while it is read, so each layer is drawn into a scratch
    // texture, read out and uploaded back. The scratch is an integer one, which keeps each
    // level as the shader rounded it, where a normalised one would convert it from a float that
    // GLES lets round to either neighbour. Each of its RGBA32UI texels holds 16 bytes of a row of
    // the plane, as every implementation reads unsigned integers out as 32 bits. The levels pass
    // through memory, not a pixel buffer: Chrome here can upload from a pixel buffer before a
    // read into it has landed.
    const texels = Math.ceil(width * channels / 16);
    const scratch = dataTexture(gl);
    fill(gl, scratch, gl.RGBA32UI, texels, height, null);
    const target = framebuffer(gl, scratch);
    const levels = new Uint32Array(4 * texels * height);
    // The same memory as bytes, each uint's lowest first, as on every platform WebGL runs on.
    const bytes = new Uint8Array(levels.buffer);
    const vao = gl.createVertexArray();
    gl.useProgram(shader.program);
    gl.uniform1i(shader.uniforms.width, width);
    gl.uniform1i(shader.uniforms.channels, channels);
    bind(gl, 1, weighting);  // after the fill, which binds to the active unit
    bind(gl, 0, plane.texture, gl.TEXTURE_2D_ARRAY);  // left active, for the upload
    gl.bindFramebuffer(gl.FRAMEBUFFER, target);
    gl.viewport(0, 0, texels, height);
    gl.bindVertexArray(vao);
    unpackAsIs(gl);  // a 3D upload of bytes even fails if flipped or premultiplied
    gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 16 * texels / channels);
    const format = [gl.RED, gl.RG, gl.RGB, gl.RGBA][channels - 1];  // a byte a channel
    for (let layer = 0; layer < layers; layer++) {
      gl.uniform1i(shader.uniforms.layer, layer);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.readPixels(0, 0, texels, height, gl.RGBA_INTEGER, gl.UNSIGNED_INT, levels);
      gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, layer, width, height, 1, format,
                       gl.UNSIGNED_BYTE, bytes);
    }
    gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 0);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteVertexArray(vao);
    gl.deleteFramebuffer(target);
    gl.deleteTexture(scratch);
  }

  dispose(): void {
    for (const { texture } of this.planes) this.gl.deleteTexture(texture);
    this.converter?.dispose();
    this.converter = null;
  }
}
