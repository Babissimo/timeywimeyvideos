import { describe, expect, test } from "vitest";
import { codecString, even, Gathered } from "./encode";

const bytes = (...values: number[]) => new Uint8Array(values);

async function gathered(writes: [number, number[]][]): Promise<number[]> {
  const out = new Gathered();
  for (const [position, data] of writes)
    out.write({ type: "write", data: bytes(...data), position });
  const blob = out.blob();
  expect(blob.type).toBe("video/mp4");
  return [...new Uint8Array(await blob.arrayBuffer())];
}

describe("Gathered", () => {
  test("appends writes at the end", async () => {
    expect(await gathered([[0, [1, 2]], [2, [3]], [3, [4, 5, 6]]])).toEqual([1, 2, 3, 4, 5, 6]);
  });

  test("overwrites what is there, across parts, and runs on past the end", async () => {
    expect(await gathered([[0, [1, 2, 3]], [3, [4, 5]], [1, [9, 9]]])).toEqual([1, 9, 9, 4, 5]);
    expect(await gathered([[0, [1, 2, 3]], [3, [4, 5]], [2, [7, 8, 9]]])).toEqual([1, 2, 7, 8, 9]);
    expect(await gathered([[0, [1, 2]], [2, [3, 4]], [3, [7, 8, 9]]])).toEqual([1, 2, 3, 7, 8, 9]);
  });

  test("fills a gap before a write past the end with zeros", async () => {
    expect(await gathered([[0, [1]], [3, [4]]])).toEqual([1, 0, 0, 4]);
  });
});

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
