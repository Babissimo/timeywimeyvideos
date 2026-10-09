import { readFileSync } from "node:fs";
import { expect, test } from "vitest";
import type { Rate } from "./types";

test("a Rate holds a numerator and denominator", () => {
  const ntsc: Rate = { num: 30000, den: 1001 };
  expect(ntsc.num / ntsc.den).toBeCloseTo(29.97, 2);
});

test("a fixture reads from disk beside the source", () => {
  const rate: Rate = JSON.parse(readFileSync(new URL("../test/fixtures/smoke.json", import.meta.url), "utf8"));
  expect(rate).toEqual({ num: 30000, den: 1001 });
});
