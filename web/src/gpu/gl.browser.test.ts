import { afterAll, expect, test, vi } from "vitest";
import { program } from "./gl";
import { COVER } from "./shaders";

const gl = document.createElement("canvas").getContext("webgl2")!;
afterAll(() => gl.getExtension("WEBGL_lose_context")?.loseContext());

test("a program that doesn't compile leaves no shader behind", () => {
  const made: WebGLShader[] = [];
  const create = gl.createShader.bind(gl);
  vi.spyOn(gl, "createShader").mockImplementation((type) => {
    const shader = create(type)!;
    made.push(shader);
    return shader;
  });
  expect(() => program(gl, COVER, "#version 300 es\nnot glsl", {})).toThrow("didn't compile");
  expect(made).toHaveLength(2);
  for (const shader of made) expect(gl.isShader(shader)).toBe(false);
});
