import { expect, test } from "vitest";
import {
  bytes, decimal, floorMod, gcd, limitDenominator, roundHalfEven, seconds,
} from "./numbers";

test("floorMod wraps into 0 up to n", () => {
  expect([7, -1, -7, 0, 2.5, -0.5].map((a) => floorMod(a, 7))).toEqual([0, 6, 0, 0, 2.5, 6.5]);
});

test("gcd", () => {
  expect([gcd(12, 18), gcd(-12, 18), gcd(0, 5), gcd(0, 0)]).toEqual([6, 6, 5, 0]);
});

test("decimal writes a number to so many significant digits", () => {
  expect([decimal(36.8), decimal(2 / 3, 3), decimal(45), decimal(-30.5), decimal(1.5, 3)])
    .toEqual(["36.8", "0.667", "45", "-30.5", "1.5"]);
});

test("seconds writes tenths under a minute and minutes:seconds from one", () => {
  expect([0, 4.24, 59.94, 59.96, 61, 119.6, 3600].map(seconds))
    .toEqual(["0.0 s", "4.2 s", "59.9 s", "1:00", "1:01", "2:00", "60:00"]);
});

test("bytes writes a size in KB, MB or GB", () => {
  expect([300, 1499, 640_000, 345.6e6, 1e9, 1.234e9].map(bytes))
    .toEqual(["1 KB", "1 KB", "640 KB", "346 MB", "1.00 GB", "1.23 GB"]);
});

test("bytes moves up a unit when a size rounds to 1000 of the one below", () => {
  expect([999_499, 999_500, 999_600, 999.4e6, 999.5e6].map(bytes))
    .toEqual(["999 KB", "1 MB", "1 MB", "999 MB", "1.00 GB"]);
});

test("roundHalfEven", () => {
  expect([0.5, 1.5, 2.5, -0.5, -1.5, 2.4, 2.6].map(roundHalfEven))
    .toEqual([0, 2, 2, 0, -2, 2, 3]);
  expect(Object.is(roundHalfEven(-0.4), 0)).toBe(true);
});

test("limitDenominator finds the closest fraction with a small enough denominator", () => {
  expect(limitDenominator(0.5)).toEqual({ num: 1, den: 2 });
  expect(limitDenominator(30000 / 1001, 1001)).toEqual({ num: 30000, den: 1001 });
  expect(limitDenominator(Math.PI, 1000)).toEqual({ num: 355, den: 113 });
  expect(limitDenominator(-0.75)).toEqual({ num: -3, den: 4 });
});
