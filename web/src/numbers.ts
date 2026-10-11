// Arithmetic the page shares.

import type { Rate } from "./types";

/** a modulo n for n > 0: from 0 up to n, where JavaScript's % keeps the sign of a. */
export function floorMod(a: number, n: number): number {
  return ((a % n) + n) % n;
}

/** The greatest common divisor of two integers, never negative. */
export function gcd(a: number, b: number): number {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b) [a, b] = [b, a % b];
  return a;
}

/** x to at most `digits` significant digits, without trailing zeros: 36.8, 0.667, 45. */
export function decimal(x: number, digits = 6): string {
  return String(Number(x.toPrecision(digits)));
}

// The decoder times frames by ffmpeg's rules, which round halves to even and turn rates into
// fractions as Python's Fraction.limit_denominator does.

/** x to the nearest integer, halves to even, and never -0. */
export function roundHalfEven(x: number): number {
  if (!Number.isFinite(x)) return x;
  let r = Math.round(x);
  if (Math.abs(x - r) === 0.5) r = 2 * Math.round(x / 2);
  return r === 0 ? 0 : r;
}

const scratch = new DataView(new ArrayBuffer(8));  // a double's bytes, read back as integers

/** A finite double's exact value as digits × 10^exp, with exp ≤ 0. */
function exactDecimal(x: number): { digits: bigint; exp: number } {
  scratch.setFloat64(0, Math.abs(x));
  const bits = scratch.getBigUint64(0);
  const biased = Number(bits >> 52n);
  let mantissa = bits & 0xfffffffffffffn;
  let twos = -1074;  // subnormal
  if (biased !== 0) {
    mantissa |= 1n << 52n;
    twos = biased - 1075;
  }
  if (twos >= 0) return { digits: mantissa << BigInt(twos), exp: 0 };
  // m / 2^k = m × 5^k / 10^k
  return { digits: mantissa * 5n ** BigInt(-twos), exp: twos };
}

function floorDiv(n: bigint, d: bigint): bigint {
  const q = n / d;
  return (n % d !== 0n) && ((n < 0n) !== (d < 0n)) ? q - 1n : q;
}

function bigGcd(a: bigint, b: bigint): bigint {
  while (b) [a, b] = [b, a % b];
  return a;
}

/** The closest fraction to x with a denominator of at most max. */
export function limitDenominator(x: number, max = 1000000): Rate {
  if (max < 1) throw new RangeError(`max must be at least 1, not ${max}`);
  if (!Number.isFinite(x)) throw new RangeError(`${x} has no fraction`);
  // x exactly, as numerator / denominator in lowest terms (the denominator a power of 2).
  const { digits, exp } = exactDecimal(x);
  let num = x < 0 ? -digits : digits;
  let den = 10n ** BigInt(-exp);
  const divisor = bigGcd(num < 0n ? -num : num, den);
  num /= divisor;
  den /= divisor;
  const limit = BigInt(max);
  if (den <= limit) return { num: Number(num), den: Number(den) };

  let [p0, q0, p1, q1] = [0n, 1n, 1n, 0n];
  let [n, d] = [num, den];
  for (;;) {
    const a = floorDiv(n, d);
    const q2 = q0 + a * q1;
    if (q2 > limit) break;
    [p0, q0, p1, q1] = [p1, q1, p0 + a * p1, q2];
    [n, d] = [d, n - a * d];
  }
  const k = floorDiv(limit - q0, q1);
  // The nearer of (p0 + k p1) / (q0 + k q1) and p1 / q1.
  if (2n * d * (q0 + k * q1) <= den) return { num: Number(p1), den: Number(q1) };
  return { num: Number(p0 + k * p1), den: Number(q0 + k * q1) };
}
