import { expect, test } from "vitest";
import page from "../../index.html?raw";

test("the page's script starts on index.html", async () => {
  document.body.innerHTML = new DOMParser().parseFromString(page, "text/html").body.innerHTML;
  await import("./main");
  const { NO_SOURCE } = await import("./source");
  expect(document.getElementById("unserved")).toBeNull();
  expect(document.getElementById("slice-note")?.textContent).not.toBe("");
  expect(document.getElementById("live-text")?.textContent).toBe(NO_SOURCE);
});
