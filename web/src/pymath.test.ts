import { describe, expect, test } from "vitest";
import raw from "../test/fixtures/pymath.json?raw";
import {
  copysign, degrees, floorMod, formatG, gcd, hypot, ieeeRemainder, limitDenominator, pyRepr,
  radians, roundHalfEven,
} from "./pymath";

type Row = [number, number | null, number];
const py = JSON.parse(raw) as {
  round: Row[]; mod: [number, number, number][]; remainder: [number, number, number][];
  gcd: [number, number, number][]; limitDenominator: [number, number, [number, number]][];
  hypot: [number, number, number][]; radians: [number, number][]; degrees: [number, number][];
  formatG: [number, number | null, string][]; repr: [string, string][];
  copysign: [number, number, number][];
};

describe("matches Python", () => {
  test("round", () => {
    for (const [x, digits, expected] of py.round) {
      const got = digits === null ? roundHalfEven(x) : roundHalfEven(x, digits);
      expect(got, `round(${x}, ${digits})`).toBe(expected);
    }
  });

  test("%", () => {
    for (const [a, n, expected] of py.mod) expect(floorMod(a, n), `${a} % ${n}`).toBe(expected);
  });

  test("math.remainder", () => {
    for (const [x, y, expected] of py.remainder) {
      expect(ieeeRemainder(x, y), `remainder(${x}, ${y})`).toBe(expected);
    }
  });

  test("math.gcd", () => {
    for (const [a, b, expected] of py.gcd) expect(gcd(a, b), `gcd(${a}, ${b})`).toBe(expected);
  });

  test("Fraction.limit_denominator", () => {
    for (const [x, max, [num, den]] of py.limitDenominator) {
      expect(limitDenominator(x, max), `${x} within ${max}`).toEqual({ num, den });
    }
  });

  test("math.hypot", () => {
    for (const [x, y, expected] of py.hypot) {
      expect(hypot(x, y), `hypot(${x}, ${y})`).toBe(expected);
    }
  });

  test("math.radians and math.degrees", () => {
    for (const [x, expected] of py.radians) expect(radians(x), `radians(${x})`).toBe(expected);
    for (const [x, expected] of py.degrees) expect(degrees(x), `degrees(${x})`).toBe(expected);
  });

  test("format with g", () => {
    for (const [x, precision, expected] of py.formatG) {
      const got = precision === null ? formatG(x) : formatG(x, precision);
      expect(got, `format(${x}, '.${precision}g')`).toBe(expected);
    }
  });

  test("repr of a string", () => {
    for (const [s, expected] of py.repr) expect(pyRepr(s)).toBe(expected);
  });

  test("math.copysign", () => {
    for (const [x, y, expected] of py.copysign) expect(copysign(x, y)).toBe(expected);
  });
});

test("halves round to the even neighbour", () => {
  expect([0.5, 1.5, 2.5, -0.5, -2.5].map((x) => roundHalfEven(x))).toEqual([0, 2, 2, 0, -2]);
  expect(roundHalfEven(0.125, 2)).toBe(0.12);   // exactly half way in binary
  expect(roundHalfEven(2.675, 2)).toBe(2.67);   // just under half way in binary
});

test("% takes the sign of the divisor", () => {
  expect(floorMod(-1, 256)).toBe(255);
  expect(floorMod(7, -3)).toBe(-2);
});

test("a fraction with a small denominator", () => {
  expect(limitDenominator(1 / 3, 1000)).toEqual({ num: 1, den: 3 });
  expect(limitDenominator(0.1, 1000)).toEqual({ num: 1, den: 10 });
  expect(() => limitDenominator(0.5, 0)).toThrow("max_denominator should be at least 1");
});
