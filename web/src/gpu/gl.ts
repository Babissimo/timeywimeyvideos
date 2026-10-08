/**
 * Small WebGL2 helpers: programs, data textures and framebuffers. dataTexture, fill and
 * DataTexture.put bind to whichever texture unit is active, and program leaves its program
 * in use, so make a draw's textures and programs first, then bind every unit it samples.
 */

/** A linked program and its uniforms' locations. */
export interface Program {
  program: WebGLProgram;
  uniforms: Record<string, WebGLUniformLocation | null>;
}

function compile(gl: WebGL2RenderingContext, type: GLenum, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`shader didn't compile: ${log}`);
  }
  return shader;
}

/**
 * Compile and link a program, throwing with the info log if that fails, and point its
 * samplers at texture units: `samplers` maps each sampler's name to its unit.
 */
export function program(gl: WebGL2RenderingContext, vertex: string, fragment: string,
                        samplers: Record<string, number>, uniforms: string[] = []): Program {
  const shaders = [compile(gl, gl.VERTEX_SHADER, vertex)];
  try {
    shaders.push(compile(gl, gl.FRAGMENT_SHADER, fragment));
  } catch (error) {
    gl.deleteShader(shaders[0]);
    throw error;
  }
  const linked = gl.createProgram();
  for (const shader of shaders) gl.attachShader(linked, shader);
  gl.linkProgram(linked);
  for (const shader of shaders) gl.deleteShader(shader);  // freed with the program
  if (!gl.getProgramParameter(linked, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(linked);
    gl.deleteProgram(linked);
    throw new Error(`program didn't link: ${log}`);
  }
  gl.useProgram(linked);
  for (const [name, unit] of Object.entries(samplers))
    gl.uniform1i(gl.getUniformLocation(linked, name), unit);
  return {
    program: linked,
    uniforms: Object.fromEntries(uniforms.map((name) =>
      [name, gl.getUniformLocation(linked, name)])),
  };
}

/** A 2D texture for numbers read with texelFetch, filled by `fill`. */
export function dataTexture(gl: WebGL2RenderingContext): WebGLTexture {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  return texture;
}

/** The format and type texImage2D takes with each internal format used here. */
function layout(gl: WebGL2RenderingContext, internalFormat: GLenum): [GLenum, GLenum] {
  switch (internalFormat) {
    case gl.R32F: return [gl.RED, gl.FLOAT];
    case gl.RG32F: return [gl.RG, gl.FLOAT];
    case gl.RGBA32F: return [gl.RGBA, gl.FLOAT];
    case gl.RGBA32UI: return [gl.RGBA_INTEGER, gl.UNSIGNED_INT];
    case gl.R32I: return [gl.RED_INTEGER, gl.INT];
    case gl.RGBA32I: return [gl.RGBA_INTEGER, gl.INT];
    case gl.RGBA8: return [gl.RGBA, gl.UNSIGNED_BYTE];
    case gl.RGBA8UI: return [gl.RGBA_INTEGER, gl.UNSIGNED_BYTE];
  }
  throw new Error(`no layout for internal format ${internalFormat}`);
}

/**
 * Have uploads take pixels as they are: not flipped, premultiplied or colour-converted.
 * These settings belong to the context, not to a texture.
 */
export function unpackAsIs(gl: WebGL2RenderingContext): void {
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
}

/** Give a texture new contents, or with data null, new storage of that size. */
export function fill(gl: WebGL2RenderingContext, texture: WebGLTexture, internalFormat: GLenum,
                     width: number, height: number, data: ArrayBufferView | null): void {
  const [format, type] = layout(gl, internalFormat);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  unpackAsIs(gl);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, data);
}

/** A data texture that keeps its storage while its size and format stay the same. */
export class DataTexture {
  readonly texture: WebGLTexture;
  private readonly gl: WebGL2RenderingContext;
  private width = 0;
  private height = 0;
  private format: GLenum = 0;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.texture = dataTexture(gl);
  }

  /** Give it these contents, reallocating only for a new size or format. */
  put(internalFormat: GLenum, width: number, height: number, data: ArrayBufferView): void {
    const { gl } = this;
    if (width !== this.width || height !== this.height || internalFormat !== this.format) {
      fill(gl, this.texture, internalFormat, width, height, data);
      this.width = width;
      this.height = height;
      this.format = internalFormat;
      return;
    }
    const [format, type] = layout(gl, internalFormat);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    unpackAsIs(gl);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, format, type, data);
  }

  dispose(): void {
    this.gl.deleteTexture(this.texture);
  }
}

/** Bind a texture to a unit. */
export function bind(gl: WebGL2RenderingContext, unit: number, texture: WebGLTexture,
                     target: GLenum = gl.TEXTURE_2D): void {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(target, texture);
}

/** A framebuffer that draws into a 2D texture. */
export function framebuffer(gl: WebGL2RenderingContext, texture: WebGLTexture): WebGLFramebuffer {
  const made = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, made);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return made;
}
