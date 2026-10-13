/** Slices a volume on the GPU, an output frame at a time. */
import { type Noise, permutation } from "../noise";
import { floorMod } from "../numbers";
import type { Line, TX } from "../planner";
import { bind, dataTexture, fill, framebuffer, program, type Program } from "./gl";
import { COVER, SHOW, SLICE } from "./shaders";
import type { Volume } from "./volume";

export interface SliceOptions {
  /** Copy the voxel closest to each point rather than blend the four around it. */
  nearest?: boolean;
  /** Time runs round in a ring, the first frame after the last; without it, black outside. */
  wrap?: boolean;
  /** As wrap, for x: the first column after the last. */
  wrapX?: boolean;
  /**
   * How many of the clip's last frames to blend into its first, and of the picture's last
   * columns into its first, fewer than it has. Each fade drops those it blends in.
   */
  fade?: Readonly<TX>;
  /**
   * Push each pixel off the line by the noise over output frame f, as far as push, a (t, x)
   * move, where the noise is strongest.
   */
  noise?: { surface: Noise; push: Readonly<TX>; f: number };
}

export class Slicer {
  private readonly gl: WebGL2RenderingContext;
  private readonly slicing: Program;
  private readonly show: Program;
  private readonly vao: WebGLVertexArrayObject;
  private readonly hash: WebGLTexture;  // the noise's lattice hash
  private seed = 0;                     // whose hash it holds
  // The slice goes into an RGBA8 texture, row 0 its top.
  private readonly output: WebGLTexture;
  private readonly framebuffer: WebGLFramebuffer;
  private width = 0;  // of the last slice, 0 before the first
  private height = 0;
  private readonly maxSize: number;  // the most pixels a slice can have each way

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.maxSize = Math.min(gl.getParameter(gl.MAX_TEXTURE_SIZE) as number,
                            ...(gl.getParameter(gl.MAX_VIEWPORT_DIMS) as Int32Array));
    this.slicing = program(gl, COVER, SLICE, { luma: 0, chroma: 1, hash: 2 },
                           ["origin", "along", "size", "fade", "wrap", "wrapX", "nearest", "push",
                            "scale", "depth", "ring"]);
    this.show = program(gl, COVER, SHOW, { slice: 0 }, ["view"]);
    this.vao = gl.createVertexArray();
    this.hash = dataTexture(gl);
    // Filled now, so that the sampler always has a whole texture behind it.
    fill(gl, this.hash, gl.R32I, 512, 1, permutation(this.seed));
    this.output = dataTexture(gl);
    this.framebuffer = framebuffer(gl, this.output);
  }

  /**
   * Slice a frame `width` columns wide and volume.height high along a line, for read or draw.
   * Throws RangeError for a frame larger than the GPU can take.
   */
  slice(volume: Volume, width: number, { origin, step }: Line, options: SliceOptions = {}):
      void {
    const { gl } = this;
    const { nearest = false, wrap = false, wrapX = false, fade = [0, 0], noise } = options;
    const { height } = volume;
    if (width > this.maxSize || height > this.maxSize)
      throw new RangeError(`a slice of ${width} × ${height} pixels is more than the GPU's `
                           + `limit of ${this.maxSize} each way`);
    if (width !== this.width || height !== this.height) {
      fill(gl, this.output, gl.RGBA8, width, height, null);
      this.width = width;
      this.height = height;
    }
    const frames = volume.frames - fade[0], columns = volume.width - fade[1];
    const { program, uniforms } = this.slicing;
    gl.useProgram(program);
    // A ring's origin is taken round it here, where float64 keeps the fraction.
    gl.uniform2f(uniforms.origin, wrap ? floorMod(origin[0], frames) : origin[0],
                 wrapX ? floorMod(origin[1], columns) : origin[1]);
    gl.uniform2f(uniforms.along, step[0], step[1]);
    gl.uniform2i(uniforms.size, frames, columns);
    gl.uniform2i(uniforms.fade, fade[0], fade[1]);
    gl.uniform1i(uniforms.wrap, Number(wrap));
    gl.uniform1i(uniforms.wrapX, Number(wrapX));
    gl.uniform1i(uniforms.nearest, Number(nearest));
    gl.uniform2f(uniforms.push, ...(noise?.push ?? [0, 0]));
    if (noise) {
      const { surface, f } = noise;
      if (surface.seed !== this.seed) {
        fill(gl, this.hash, gl.R32I, 512, 1, permutation(surface.seed));
        this.seed = surface.seed;
      }
      const { z, ring } = surface.depth(f);
      gl.uniform1f(uniforms.scale, 1 / surface.size);
      gl.uniform1f(uniforms.depth, z);
      gl.uniform1i(uniforms.ring, ring);
    }
    // Bound after the fills above, which bind to the active unit.
    volume.planes.forEach(({ texture }, unit) => bind(gl, unit, texture, gl.TEXTURE_2D_ARRAY));
    bind(gl, 2, this.hash);
    this.cover(this.framebuffer, width, height);
  }

  /** The last slice as RGBA bytes, rows top-first. */
  read(): Uint8Array {
    const { gl, width, height } = this;
    this.sliced();
    const pixels = new Uint8Array(width * height * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return pixels;
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
    gl.deleteProgram(this.slicing.program);
    gl.deleteProgram(this.show.program);
    gl.deleteTexture(this.hash);
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
