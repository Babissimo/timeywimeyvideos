import { expect, test } from "vitest";

test("WebGL2 is available", () => {
  expect(document.createElement("canvas").getContext("webgl2")).not.toBeNull();
});

test("WebCodecs VideoDecoder is available", () => {
  expect(typeof VideoDecoder).toBe("function");
});
