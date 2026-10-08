/** A clip held on the GPU as one RGBA8 TEXTURE_2D_ARRAY: texel (x, y, layer) is voxel (t, y, x). */

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
  readonly gl: WebGL2RenderingContext;
  readonly texture: WebGLTexture;
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
    // Two calls, one for each of texSubImage3D's overloads.
    if (ArrayBuffer.isView(pixels))
      gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, t, width, height, 1, gl.RGBA,
                       gl.UNSIGNED_BYTE, pixels);
    else
      gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, t, width, height, 1, gl.RGBA,
                       gl.UNSIGNED_BYTE, pixels);
  }

  dispose(): void {
    this.gl.deleteTexture(this.texture);
  }
}
