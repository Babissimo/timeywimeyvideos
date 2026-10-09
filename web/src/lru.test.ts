import { expect, test } from "vitest";
import { LruCache } from "./lru";

test("drops the entry used least recently", () => {
  const cache = new LruCache<string, number>(2);
  cache.set("a", 1);
  cache.set("b", 2);
  cache.get("a");  // so b is the older
  cache.set("c", 3);
  expect([cache.get("a"), cache.get("b"), cache.get("c")]).toEqual([1, undefined, 3]);
});

test("setting a key again keeps it and refreshes it", () => {
  const cache = new LruCache<string, number>(2);
  cache.set("a", 1);
  cache.set("b", 2);
  cache.set("a", 10);
  cache.set("c", 3);
  expect([cache.get("a"), cache.get("b"), cache.get("c")]).toEqual([10, undefined, 3]);
});
