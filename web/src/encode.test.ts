import { expect, test } from "vitest";
import { codecString, even } from "./encode";

test("even rounds up to an even size", () => {
  expect([1, 2, 33, 34].map(even)).toEqual([2, 2, 34, 34]);
});

test("codecString gives the lowest level, from 3.1, that holds the frames", () => {
  const fps = (num: number) => ({ num, den: 1 });
  expect(codecString(64, 48, fps(30))).toBe("avc1.64001f");
  expect(codecString(1920, 1080, fps(30))).toBe("avc1.640028");
  expect(codecString(1920, 1080, fps(60))).toBe("avc1.64002a");
  expect(codecString(3840, 2160, fps(30))).toBe("avc1.640033");
  expect(codecString(3840, 2160, fps(60))).toBe("avc1.640034");
  expect(codecString(16384, 64, fps(30))).toBe("avc1.64003c");
  expect(codecString(30000, 16384, fps(30))).toBe("avc1.64003e");
});
