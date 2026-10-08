/** Slices a volume on the GPU as timeslice.sample_columns and sample_noisy_columns do. */
import { floorMod } from "../pymath";
import type { Columns, NoiseGrid } from "../types";
import { bind, DataTexture, dataTexture, fill, framebuffer, program, type Program } from "./gl";
import { COVER, FLAT, NOISY, SHOW } from "./shaders";
import type { Volume } from "./volume";

export interface SliceOptions {
  /** Copy the voxel closest to each point rather than blend the four around it. */
  nearest?: boolean;
  /** Time runs round in a ring, the first frame after the last; without it, black outside. */
  wrap?: boolean;
  /** As wrap, for x: the first column after the last. */
  wrapX?: boolean;
  /**
   * Push each pixel off the plane by the noise there, worked out from grid, times push: the
   * (t, x) move where the noise is at full strength.
   */
  noise?: { push: [number, number]; grid: NoiseGrid };
}

// As timeslice._neighbours: for a coordinate floored to lo on an axis of n, the voxel at or
// before it and the one after. With ring the axis wraps round.
const before = (lo: number, n: number, ring: boolean) => ring ? floorMod(lo, n) : Math.max(lo, 0);
const after = (lo: number, n: number, ring: boolean) =>
  ring ? floorMod(lo + 1, n) : Math.min(lo + 1, n - 1);

const within = (coord: number, n: number) => -0.5 <= coord && coord < n - 0.5;

/** Write a weight into channel k of column j's texels in rows 1 to 3, as whole 2^-36ths. */
function split(out: Int32Array, n: number, j: number, k: number, weight: number): void {
  let rest = weight;
  for (let part = 1; part <= 3; part++) {
    rest *= 4096;  // exact, as are floor and the subtraction
    const whole = Math.floor(rest);
    out[4 * (part * n + j) + k] = whole;
    rest -= whole;
  }
}

/**
 * What each output column reads, worked out in float64 as timeslice._sample_columns does,
 * into four texels a column: voxels t0, t1, x0, x1, with t0 -1 outside the video and, with
 * nearest, the closest in t0 and x0; then fx, ft and fx ft as whole numbers of 2^-36, a
 * 12-bit part of each per texel, most significant first, for the shader's blend.
 */
function flatColumns(columns: Columns, frames: number, width: number, nearest: boolean,
                     wrap: boolean, wrapX: boolean, out: Int32Array): void {
  const n = columns.t.length;
  for (let j = 0; j < n; j++) {
    const t = columns.t[j], x = columns.x[j];
    if (!((wrap || within(t, frames)) && (wrapX || within(x, width)))) {
      out[4 * j] = -1;
      continue;
    }
    const lt = Math.floor(t), lx = Math.floor(x);
    const ft = t - lt, fx = x - lx;
    const t0 = before(lt, frames, wrap), t1 = after(lt, frames, wrap);
    const x0 = before(lx, width, wrapX), x1 = after(lx, width, wrapX);
    out[4 * j] = nearest && ft >= 0.5 ? t1 : t0;
    out[4 * j + 1] = t1;
    out[4 * j + 2] = nearest && fx >= 0.5 ? x1 : x0;
    out[4 * j + 3] = x1;
    split(out, n, j, 0, fx);
    split(out, n, j, 1, ft);
    split(out, n, j, 2, fx * ft);
  }
}

/**
 * Where each output column of a noisy slice reads before the noise moves it: t[j] and x[j]
 * floored, with the column's first node column, into floors; then t[j] and x[j] past those
 * into offsets, and in its second row the column's node weights. The shader finishes each
 * pixel.
 */
function noisyColumns(columns: Columns, grid: NoiseGrid, floors: Int32Array,
                      offsets: Float32Array): void {
  const n = columns.t.length;
  for (let j = 0; j < n; j++) {
    const t = Math.floor(columns.t[j]), x = Math.floor(columns.x[j]);
    floors[4 * j] = t;
    floors[4 * j + 1] = x;
    floors[4 * j + 2] = grid.cols[j];
    offsets[4 * j] = columns.t[j] - t;
    offsets[4 * j + 1] = columns.x[j] - x;
  }
  offsets.set(grid.colW, 4 * n);
}

/** A typed array of this length, the one given if it already is. */
function sized<T extends Int32Array | Float32Array>(array: T, length: number,
                                                     make: new (length: number) => T): T {
  return array.length === length ? array : new make(length);
}

export class Slicer {
  private readonly gl: WebGL2RenderingContext;
  private readonly flat: Program;
  private readonly noisy: Program;
  private readonly show: Program;
  private readonly vao: WebGLVertexArrayObject;
  private readonly columns: DataTexture;
  private readonly offsets: DataTexture;
  private readonly rows: DataTexture;
  private readonly rowWeights: DataTexture;
  private readonly nodes: DataTexture;
  // What goes into those, kept from one slice to the next.
  private columnData = new Int32Array(0);
  private offsetData = new Float32Array(0);
  private rowWeightData = new Float32Array(0);
  private nodeData = new Float32Array(0);
  // The slice goes into an RGBA8UI texture, row 0 its top. The shaders round each level
  // as timeslice does, and an integer target keeps that level as is, where a normalised
  // one would convert it back from a float, which GLES lets round to either neighbour.
  private readonly output: WebGLTexture;
  private readonly framebuffer: WebGLFramebuffer;
  private width = 0;  // of the last slice, 0 before the first
  private height = 0;
  private readonly maxSize: number;  // the most pixels a slice can have each way

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.maxSize = Math.min(gl.getParameter(gl.MAX_TEXTURE_SIZE) as number,
                            ...(gl.getParameter(gl.MAX_VIEWPORT_DIMS) as Int32Array));
    this.flat = program(gl, COVER, FLAT, { volume: 0, columns: 1 }, ["nearest"]);
    this.noisy = program(gl, COVER, NOISY,
                         { volume: 0, columns: 1, offsets: 2, rows: 3, rowWeights: 4, nodes: 5 },
                         ["push", "size", "nearest", "wrap", "wrapX"]);
    this.show = program(gl, COVER, SHOW, { slice: 0 }, ["view"]);
    this.vao = gl.createVertexArray();
    this.columns = new DataTexture(gl);
    this.offsets = new DataTexture(gl);
    this.rows = new DataTexture(gl);
    this.rowWeights = new DataTexture(gl);
    this.nodes = new DataTexture(gl);
    this.output = dataTexture(gl);
    this.framebuffer = framebuffer(gl, this.output);
  }

  /**
   * Slice a frame columns.t.length wide and volume.height high, for read or draw. With
   * noise, its grid is for a frame that size. Throws RangeError for a frame, or a grid of
   * noise nodes, larger than the GPU can take.
   */
  slice(volume: Volume, columns: Columns, options: SliceOptions = {}): void {
    const { gl } = this;
    const { nearest = false, wrap = false, wrapX = false, noise } = options;
    const width = columns.t.length, height = volume.height;
    if (columns.x.length !== width)
      throw new RangeError(`columns need an x for each t, not ${columns.x.length} for ${width}`);
    if (width > this.maxSize || height > this.maxSize)
      throw new RangeError(`a slice of ${width} × ${height} pixels is more than the GPU's `
                           + `limit of ${this.maxSize} each way`);
    if (noise && (noise.grid.cols.length !== width || noise.grid.rows.length !== height))
      throw new RangeError(`a noise grid for ${noise.grid.cols.length} × `
                           + `${noise.grid.rows.length} pixels can't push a slice of `
                           + `${width} × ${height}`);
    if (noise && Math.max(noise.grid.nodeCols, noise.grid.nodeRows) > this.maxSize)
      throw new RangeError(`a noise grid of ${noise.grid.nodeCols} × ${noise.grid.nodeRows} `
                           + `nodes is more than the GPU's limit of ${this.maxSize} each way`);
    if (width !== this.width || height !== this.height) {
      fill(gl, this.output, gl.RGBA8UI, width, height, null);
      this.width = width;
      this.height = height;
    }
    if (noise) {
      const { grid, push } = noise;
      this.columnData = sized(this.columnData, 4 * width, Int32Array);
      this.offsetData = sized(this.offsetData, 8 * width, Float32Array);
      noisyColumns(columns, grid, this.columnData, this.offsetData);
      this.rowWeightData = sized(this.rowWeightData, grid.rowW.length, Float32Array);
      this.rowWeightData.set(grid.rowW);
      this.nodeData = sized(this.nodeData, grid.nodes.length, Float32Array);
      this.nodeData.set(grid.nodes);
      this.columns.put(gl.RGBA32I, width, 1, this.columnData);
      this.offsets.put(gl.RGBA32F, width, 2, this.offsetData);
      this.rows.put(gl.R32I, height, 1, grid.rows);
      this.rowWeights.put(gl.RGBA32F, height, 1, this.rowWeightData);
      this.nodes.put(gl.R32F, grid.nodeCols, grid.nodeRows, this.nodeData);
      const { program, uniforms } = this.noisy;
      gl.useProgram(program);
      gl.uniform2f(uniforms.push, push[0], push[1]);
      gl.uniform2i(uniforms.size, volume.frames, volume.width);
      gl.uniform1i(uniforms.nearest, Number(nearest));
      gl.uniform1i(uniforms.wrap, Number(wrap));
      gl.uniform1i(uniforms.wrapX, Number(wrapX));
      bind(gl, 2, this.offsets.texture);
      bind(gl, 3, this.rows.texture);
      bind(gl, 4, this.rowWeights.texture);
      bind(gl, 5, this.nodes.texture);
    } else {
      this.columnData = sized(this.columnData, 16 * width, Int32Array);
      flatColumns(columns, volume.frames, volume.width, nearest, wrap, wrapX, this.columnData);
      this.columns.put(gl.RGBA32I, width, 4, this.columnData);
      gl.useProgram(this.flat.program);
      gl.uniform1i(this.flat.uniforms.nearest, Number(nearest));
    }
    bind(gl, 0, volume.texture, gl.TEXTURE_2D_ARRAY);
    bind(gl, 1, this.columns.texture);
    this.cover(this.framebuffer, width, height);
  }

  /** The last slice as RGBA bytes, rows top-first. */
  read(): Uint8Array {
    const { gl, width, height } = this;
    this.sliced();
    const levels = new Uint32Array(width * height * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    // The one way every implementation reads an unsigned integer buffer.
    gl.readPixels(0, 0, width, height, gl.RGBA_INTEGER, gl.UNSIGNED_INT, levels);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return new Uint8Array(levels);
  }

  /** Draw the last slice upright on the canvas, scaled to fill it. */
  draw(): void {
    const { gl } = this;
    this.sliced();
    gl.useProgram(this.show.program);
    gl.uniform2f(this.show.uniforms.view, gl.drawingBufferWidth, gl.drawingBufferHeight);
    bind(gl, 0, this.output);
    this.cover(null, gl.drawingBufferWidth, gl.drawingBufferHeight);
  }

  dispose(): void {
    const { gl } = this;
    for (const { program } of [this.flat, this.noisy, this.show]) gl.deleteProgram(program);
    for (const data of [this.columns, this.offsets, this.rows, this.rowWeights, this.nodes])
      data.dispose();
    gl.deleteTexture(this.output);
    gl.deleteFramebuffer(this.framebuffer);
    gl.deleteVertexArray(this.vao);
  }

  /** Throw unless there is a slice to read or draw. */
  private sliced(): void {
    if (!this.width) throw new Error("nothing has been sliced yet");
  }

  /** Run the current program over width × height pixels of a framebuffer, null the canvas. */
  private cover(target: WebGLFramebuffer | null, width: number, height: number): void {
    const { gl } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target);
    gl.viewport(0, 0, width, height);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
}
