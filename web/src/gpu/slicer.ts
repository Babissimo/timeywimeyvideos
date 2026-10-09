/** Slices a volume on the GPU as timeslice.sample_columns and sample_noisy_columns do. */
import { floorMod } from "../pymath";
import type { Columns, NoiseGrid } from "../types";
import { bind, dataTexture, fill, framebuffer, program, type Program } from "./gl";
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

/**
 * As timeslice._neighbours: the voxels either side of coord along an axis of n, and how
 * far coord is past the first. With wrap the axis is a ring.
 */
function neighbours(coord: number, n: number, wrap: boolean): [number, number, number] {
  const lo = Math.floor(coord);
  if (wrap) return [floorMod(lo, n), floorMod(lo + 1, n), coord - lo];
  return [Math.max(lo, 0), Math.min(lo + 1, n - 1), coord - lo];
}

const within = (coord: number, n: number) => -0.5 <= coord && coord < n - 0.5;

/**
 * What each output column reads, worked out in float64 as timeslice._sample_columns does,
 * four texels a column: voxels t0, t1, x0, x1, with t0 -1 outside the video and, with
 * nearest, the closest in t0 and x0; then fx, ft and fx ft as whole numbers of 2^-36, a
 * 12-bit part of each per texel, most significant first, for the shader's blend.
 */
function flatColumns(columns: Columns, frames: number, width: number, nearest: boolean,
                     wrap: boolean, wrapX: boolean): Int32Array {
  const n = columns.t.length;
  const out = new Int32Array(16 * n);
  for (let j = 0; j < n; j++) {
    const t = columns.t[j], x = columns.x[j];
    if (!((wrap || within(t, frames)) && (wrapX || within(x, width)))) {
      out[4 * j] = -1;
      continue;
    }
    const [t0, t1, ft] = neighbours(t, frames, wrap);
    const [x0, x1, fx] = neighbours(x, width, wrapX);
    out.set(nearest ? [ft >= 0.5 ? t1 : t0, t1, fx >= 0.5 ? x1 : x0, x1] : [t0, t1, x0, x1],
            4 * j);
    [fx, ft, fx * ft].forEach((weight, k) => {
      let rest = weight;
      for (let part = 1; part <= 3; part++) {
        rest *= 4096;  // exact, as are floor and the subtraction
        out[4 * (part * n + j) + k] = Math.floor(rest);
        rest -= Math.floor(rest);
      }
    });
  }
  return out;
}

/**
 * Where each output column of a noisy slice reads before the noise moves it: t[j] and x[j]
 * floored, with the column's first node column; then t[j] and x[j] past those, and in a
 * second row the column's node weights. The shader finishes each pixel.
 */
function noisyColumns(columns: Columns, grid: NoiseGrid) {
  const n = columns.t.length;
  const floors = new Int32Array(4 * n);
  const offsets = new Float32Array(8 * n);
  for (let j = 0; j < n; j++) {
    const t = Math.floor(columns.t[j]), x = Math.floor(columns.x[j]);
    floors.set([t, x, grid.cols[j]], 4 * j);
    offsets.set([columns.t[j] - t, columns.x[j] - x], 4 * j);
  }
  offsets.set(grid.colW, 4 * n);
  return { floors, offsets };
}

export class Slicer {
  private readonly gl: WebGL2RenderingContext;
  private readonly flat: Program;
  private readonly noisy: Program;
  private readonly show: Program;
  private readonly vao: WebGLVertexArrayObject;
  private readonly columns: WebGLTexture;
  private readonly offsets: WebGLTexture;
  private readonly rows: WebGLTexture;
  private readonly rowWeights: WebGLTexture;
  private readonly nodes: WebGLTexture;
  // The slice goes into an RGBA8UI texture, row 0 its top. The shaders round each level
  // as timeslice does, and an integer target keeps that level as is, where a normalised
  // one would convert it back from a float, which GLES lets round to either neighbour.
  private readonly output: WebGLTexture;
  private readonly framebuffer: WebGLFramebuffer;
  private width = 0;
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
    [this.columns, this.offsets, this.rows, this.rowWeights, this.nodes] =
      Array.from({ length: 5 }, () => dataTexture(gl));
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
      const { floors, offsets } = noisyColumns(columns, grid);
      fill(gl, this.columns, gl.RGBA32I, width, 1, floors);
      fill(gl, this.offsets, gl.RGBA32F, width, 2, offsets);
      fill(gl, this.rows, gl.R32I, height, 1, grid.rows);
      fill(gl, this.rowWeights, gl.RGBA32F, height, 1, new Float32Array(grid.rowW));
      fill(gl, this.nodes, gl.R32F, grid.nodeCols, grid.nodeRows, new Float32Array(grid.nodes));
      const { program, uniforms } = this.noisy;
      gl.useProgram(program);
      gl.uniform2f(uniforms.push, push[0], push[1]);
      gl.uniform2i(uniforms.size, volume.frames, volume.width);
      gl.uniform1i(uniforms.nearest, Number(nearest));
      gl.uniform1i(uniforms.wrap, Number(wrap));
      gl.uniform1i(uniforms.wrapX, Number(wrapX));
      [this.offsets, this.rows, this.rowWeights, this.nodes]
        .forEach((texture, i) => bind(gl, 2 + i, texture));
    } else {
      fill(gl, this.columns, gl.RGBA32I, width, 4,
           flatColumns(columns, volume.frames, volume.width, nearest, wrap, wrapX));
      gl.useProgram(this.flat.program);
      gl.uniform1i(this.flat.uniforms.nearest, Number(nearest));
    }
    bind(gl, 0, volume.texture, gl.TEXTURE_2D_ARRAY);
    bind(gl, 1, this.columns);
    this.cover(this.framebuffer, width, height);
  }

  /** The last slice as RGBA bytes, rows top-first. */
  read(): Uint8Array {
    const { gl, width, height } = this;
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
    gl.useProgram(this.show.program);
    gl.uniform2f(this.show.uniforms.view, gl.drawingBufferWidth, gl.drawingBufferHeight);
    bind(gl, 0, this.output);
    this.cover(null, gl.drawingBufferWidth, gl.drawingBufferHeight);
  }

  dispose(): void {
    const { gl } = this;
    for (const { program } of [this.flat, this.noisy, this.show]) gl.deleteProgram(program);
    for (const texture of [this.columns, this.offsets, this.rows, this.rowWeights, this.nodes,
                           this.output])
      gl.deleteTexture(texture);
    gl.deleteFramebuffer(this.framebuffer);
    gl.deleteVertexArray(this.vao);
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
