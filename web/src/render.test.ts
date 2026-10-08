import { describe, expect, test } from "vitest";
import raw from "../test/fixtures/render.json?raw";
import { readOptions } from "./options";
import { handOver, outputName, renderCommand, shellJoin, shellQuote } from "./render";
import { Unrenderable } from "./unrenderable";

interface PyName {
  name: string; values: Record<string, string>; preview: boolean; output: string;
  command: string[]; line: string;
}

const { names } = JSON.parse(raw) as { names: PyName[] };

describe("names a render and its command as render.py does", () => {
  for (const c of names) {
    test(`${c.name} ${JSON.stringify(c.values)}${c.preview ? " preview" : ""}`, () => {
      const opts = readOptions(c.values);
      const fps = opts.fps && String(c.values.fps);
      const output = outputName(c.name, opts, fps, c.preview);
      expect(output).toBe(c.output);
      const command = renderCommand(c.name, output, opts, fps, c.preview);
      expect(command).toEqual(c.command);
      expect(shellJoin(command)).toBe(c.line);
    });
  }
});

test("quotes arguments as shlex.quote does", () => {
  expect(["", "abc", "a b", "it's", "a=b,c:d/e.f+g@h%i-j_k", "ü", "--x=1", "$HOME"]
    .map(shellQuote))
    .toEqual(["''", "abc", "'a b'", `'it'"'"'s'`, "a=b,c:d/e.f+g@h%i-j_k", "'ü'", "--x=1",
              "'$HOME'"]);
});

test("says how timeslice.py takes over a clip the browser can't render", () => {
  const error = new Unrenderable("The GPU gave out, perhaps for want of memory.", "uv run ...");
  expect(handOver(error)).toBe("The GPU gave out, perhaps for want of memory. timeslice.py can "
                               + "render it: run this beside it, with the video's path in place "
                               + "of its name.");
});
