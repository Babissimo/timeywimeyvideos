import { describe, expect, test } from "vitest";
import raw from "../test/fixtures/render.json?raw";
import { readOptions } from "./options";
import { outputName } from "./render";

interface Named {
  name: string; values: Record<string, string>; preview: boolean; output: string;
}

const { names } = JSON.parse(raw) as { names: Named[] };

describe("names a render", () => {
  for (const c of names) {
    test(`${c.name} ${JSON.stringify(c.values)}${c.preview ? " preview" : ""}`, () => {
      const opts = readOptions(c.values);
      const fps = opts.fps && String(c.values.fps);
      expect(outputName(c.name, opts, fps, c.preview)).toBe(c.output);
    });
  }
});
