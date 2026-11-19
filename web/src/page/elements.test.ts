import { expect, test } from "vitest";
import page from "../../index.html?raw";
import { TAGS } from "./elements";

/** The tag of each element in index.html that has an id, by id. */
const tags = new Map([...page.matchAll(/<(\w+)\b[^>]*\bid="([\w-]+)"/g)]
  .map(([, tag, id]) => [id, tag]));

// The type check holds the page to the ids TAGS lists; this holds them to index.html.
test("each element the page looks up is in index.html with the tag it is typed by", () => {
  for (const [id, tag] of Object.entries(TAGS)) expect(tags.get(id), `#${id}`).toBe(tag);
});
