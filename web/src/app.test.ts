import { expect, test } from "vitest";
import script from "../app.js?raw";
import page from "../index.html?raw";

/** The tag of each element in index.html that has an id, by id. */
const tags = new Map([...page.matchAll(/<(\w+)\b[^>]*\bid="([\w-]+)"/g)]
  .map(([, tag, id]) => [id, tag]));

/** The tag of the elements each of $'s return types stands for, where it is one. */
const TAGS: Record<string, string | null> = {
  HTMLInputElement: "input", HTMLSelectElement: "select", HTMLButtonElement: "button",
  HTMLCanvasElement: "canvas", HTMLVideoElement: "video", HTMLElement: null,
};

// The type check holds app.js to the ids $'s overloads list; this holds them to the page.
test("app.js types each element it looks up as index.html has it", () => {
  const overloads = script.slice(0, script.indexOf("function $("));
  const typed = [...overloads.matchAll(/@param \{([^}]*)\} id\s*\*\s*@returns \{(\w+)\}/g)];
  expect(typed.map(([, , type]) => type).sort()).toEqual(Object.keys(TAGS).sort());
  for (const [, ids, type] of typed) {
    for (const [, id] of ids.matchAll(/"([\w-]+)"/g)) {
      expect(tags.has(id), `#${id} is in index.html`).toBe(true);
      if (TAGS[type]) expect(tags.get(id), `#${id}`).toBe(TAGS[type]);
    }
  }
});
