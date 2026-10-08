/** A clip held on the GPU as one RGBA8 TEXTURE_2D_ARRAY: texel (x, y, layer) is voxel (t, y, x). */
import { bind, dataTexture, fill, framebuffer, program, unpackAsIs } from "./gl";
import { COVER, CROSSFADE } from "./shaders";

/** The clip needs more of the GPU than it can give. */
export class VolumeTooLarge extends Error {
  readonly bytes: number;

  constructor(bytes: number) {
    super(`a ${(bytes / 1e9).toFixed(2)} GB clip is more than the GPU can hold`);
    this.name = "VolumeTooLarge";
    this.bytes = bytes;
  }
}

export class Volume {
  private readonly gl: WebGL2RenderingContext;
  readonly texture: WebGLTexture;  // for the slicer to read
  readonly height: number;
  readonly bytes: number;
  private readonly storedWidth: number;
  private size: { frames: number; width: number };  // a crossfade uses fewer than are stored

  private constructor(gl: WebGL2RenderingContext, texture: WebGLTexture,
                      frames: number, height: number, width: number) {
    this.gl = gl;
    this.texture = texture;
    this.height = height;
    this.bytes = frames * height * width * 4;
    this.storedWidth = width;
    this.size = { frames, width };
  }

  get frames(): number { return this.size.frames; }
  get width(): number { return this.size.width; }

  /**
   * Allocate a volume. Throws RangeError past the GPU's limits on frames or on pixels each
   * way, and VolumeTooLarge if the GPU can't give it the memory.
   */
  static create(gl: WebGL2RenderingContext, frames: number, height: number, width: number): Volume {
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
    const bytes = frames * height * width * 4;
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
    while (gl.getError() !== gl.NO_ERROR) {
      // Clear errors left by earlier calls, so the next check sees only this allocation's.
      // Any the caller left pending are lost.
    }
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.RGBA8, width, height, frames);
    if (gl.getError() !== gl.NO_ERROR) {
      gl.deleteTexture(texture);
      throw new VolumeTooLarge(bytes);
    }
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    return new Volume(gl, texture, frames, height, width);
  }

  /**
   * Set frame t from RGBA bytes, rows top-first, or from an image, its top row first.
   * Either is the size the volume was created with.
   */
  upload(t: number, pixels: Uint8Array | TexImageSource): void {
    const { gl, storedWidth: width, height } = this;
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.texture);
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
    const { gl, height } = this;
    const weights = new Float32Array(2 * n);
    for (let i = 0; i < n; i++) {
      const w = Math.fround((i + 1) / (n + 1));  // as numpy rounds them
      weights.set([w, Math.fround(1 - w)], 2 * i);
    }
    const shader = program(gl, COVER, CROSSFADE, { volume: 0, weights: 1 },
                           ["layer", "past", "across", "width", "zero"]);
    const weighting = dataTexture(gl);
    fill(gl, weighting, gl.RG32F, n, 1, weights);
    // A texture can't be drawn into while it is read, so each frame is drawn into a scratch
    // texture, read out and uploaded back. The scratch is an integer one, which keeps each
    // level as the shader rounded it, where a normalised one would convert it from a float that
    // GLES lets round to either neighbour. It holds four voxels to an RGBA32UI texel, as every
    // implementation reads unsigned integers out as 32 bits. The levels pass through memory,
    // not a pixel buffer: Chrome here can upload from a pixel buffer before a read into it
    // has landed.
    const [width, frames] = axis === 0 ? [this.width, n] : [n, this.frames];
    const texels = Math.ceil(width / 4);
    const scratch = dataTexture(gl);
    fill(gl, scratch, gl.RGBA32UI, texels, height, null);
    const target = framebuffer(gl, scratch);
    const levels = new Uint32Array(4 * texels * height);
    // The same memory as bytes, each uint's lowest first, as on every platform WebGL runs on.
    const bytes = new Uint8Array(levels.buffer);
    const vao = gl.createVertexArray();
    gl.useProgram(shader.program);
    gl.uniform2i(shader.uniforms.past, axis === 0 ? length : 0, axis === 2 ? length : 0);
    gl.uniform1i(shader.uniforms.across, Number(axis === 2));
    gl.uniform1i(shader.uniforms.width, width);
    gl.uniform1i(shader.uniforms.zero, 0);
    bind(gl, 1, weighting);
    bind(gl, 0, this.texture, gl.TEXTURE_2D_ARRAY);  // left active, for the upload
    gl.bindFramebuffer(gl.FRAMEBUFFER, target);
    gl.viewport(0, 0, texels, height);
    gl.bindVertexArray(vao);
    unpackAsIs(gl);  // a 3D upload of bytes even fails if flipped or premultiplied
    gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 4 * texels);
    for (let layer = 0; layer < frames; layer++) {
      gl.uniform1i(shader.uniforms.layer, layer);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.readPixels(0, 0, texels, height, gl.RGBA_INTEGER, gl.UNSIGNED_INT, levels);
      gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, layer, width, height, 1, gl.RGBA,
                       gl.UNSIGNED_BYTE, bytes);
    }
    gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 0);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteVertexArray(vao);
    gl.deleteFramebuffer(target);
    gl.deleteTexture(scratch);
    gl.deleteTexture(weighting);
    gl.deleteProgram(shader.program);
  }

  dispose(): void {
    this.gl.deleteTexture(this.texture);
  }
}
